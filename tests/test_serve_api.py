"""HTTP API and batch CLI with a deterministic backend; every request uploads a real image file."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sitesight.serve.api import Limits, create_app
from sitesight.serve.cli import describe_batch, main

TEXTS = {0: ("Ten foundations are installed", 0.9), 1: ("blah blah", 0.8)}


class StubBackend:
    """Caption depends on the box x position, so tests can tell the regions apart."""

    def __init__(self) -> None:
        self.calls: list[list[float]] = []

    def describe_region(self, image: Image.Image, bbox: list[float]) -> tuple[str, float]:
        self.calls.append(bbox)
        if bbox[0] >= 50:
            return "Ten foundations are installed", 0.2  # parsed, but low confidence
        return TEXTS[int(bbox[1] > 40)]

    def ground(self, image: Image.Image, phrase: str) -> list[tuple[list[float], float]]:
        return [([1.0, 2.0, 30.0, 40.0], 0.77)] if phrase == "rebar" else []

    def info(self) -> dict[str, str]:
        return {"engine": "stub"}


def png_bytes(size: tuple[int, int] = (120, 80), fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (30, 90, 160)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture()
def backend() -> StubBackend:
    return StubBackend()


@pytest.fixture()
def client(backend: StubBackend) -> TestClient:
    return TestClient(create_app(backend, Limits(max_upload_bytes=50_000, max_boxes=3)))


def post_describe(
    c: TestClient, boxes: object, data: bytes | None = None, name: str = "a.png"
) -> Any:
    return c.post(
        "/v1/describe",
        files={"image": (name, data if data is not None else png_bytes(), "image/png")},
        data={"boxes": boxes if isinstance(boxes, str) else json.dumps(boxes)},
    )


def test_health_reports_the_backend(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "backend": {"engine": "stub"}}


def test_describe_returns_fields_confidence_review_and_summary(client: TestClient) -> None:
    r = post_describe(client, [[0, 0, 20, 20], [0, 50, 20, 20], [60, 0, 20, 20]])
    assert r.status_code == 200
    body = r.json()
    assert (body["image_width"], body["image_height"]) == (120, 80)
    ok, unparsed, low = body["regions"]
    assert ok["caption"] == "Ten foundations are installed" and ok["confidence"] == 0.9
    assert ok["fields"] == {
        "resource": "foundation",
        "count": 10,
        "colour": None,
        "state": "installed",
    }
    assert ok["review"] is False and ok["review_reasons"] == []
    assert unparsed["fields"] is None and unparsed["review_reasons"] == ["caption_not_parsed"]
    assert low["fields"]["resource"] == "foundation" and low["review_reasons"] == ["low_confidence"]
    # the tally counts described regions that parsed (two of three), not objects
    assert body["summary"] == {"foundation": 2}
    assert body["warnings"] == []


def test_boxes_overhanging_the_image_are_clamped_and_reported(
    client: TestClient, backend: StubBackend
) -> None:
    r = post_describe(client, [[-10, 0, 30, 20], [100, 60, 50, 50]])
    assert r.status_code == 200
    assert backend.calls == [[0.0, 0.0, 20.0, 20.0], [100.0, 60.0, 20.0, 20.0]]
    assert r.json()["warnings"] == [
        "box 0 was clamped to the image",
        "box 1 was clamped to the image",
    ]


@pytest.mark.parametrize(
    ("boxes", "fragment"),
    [
        ("not json", "not valid JSON"),
        ("[]", "non-empty list"),
        ("{}", "non-empty list"),
        ([[0, 0, 5, 5]] * 4, "exceeds the limit of 3"),
        ([[0, 0, 5]], "four numbers"),
        ([[0, 0, "5", 5]], "four numbers"),
        ([[0, 0, True, 5]], "four numbers"),
        ([[0, 0, 0, 5]], "empty or outside"),
        ([[500, 500, 10, 10]], "empty or outside"),
        ([[120, 0, 10, 10]], "empty or outside"),  # touches the right edge: zero width once clamped
    ],
)
def test_invalid_boxes_are_rejected_with_a_reason(
    client: TestClient, backend: StubBackend, boxes: object, fragment: str
) -> None:
    r = post_describe(client, boxes)
    assert r.status_code == 422 and fragment in r.json()["detail"]
    assert backend.calls == []  # nothing was sent to the model


def test_uploads_that_are_not_images_or_too_big_are_rejected(client: TestClient) -> None:
    r = post_describe(client, [[0, 0, 5, 5]], data=b"this is not an image")
    assert r.status_code == 415
    r = post_describe(client, [[0, 0, 5, 5]], data=b"\x00" * 50_001)
    assert r.status_code == 413
    pixel_capped = TestClient(create_app(StubBackend(), Limits(max_pixels=1000)))
    r = post_describe(pixel_capped, [[0, 0, 5, 5]])  # 120 x 80 = 9,600 pixels
    assert r.status_code == 413 and "pixels" in r.json()["detail"]


def test_jpeg_uploads_work_too(client: TestClient) -> None:
    r = post_describe(client, [[0, 0, 20, 20]], data=png_bytes(fmt="JPEG"), name="a.jpg")
    assert r.status_code == 200 and r.json()["regions"][0]["fields"]["count"] == 10


def test_ground_returns_boxes_and_validates_the_phrase(client: TestClient) -> None:
    files = {"image": ("a.png", png_bytes(), "image/png")}
    r = client.post("/v1/ground", files=files, data={"phrase": " rebar "})
    assert r.status_code == 200
    assert r.json()["boxes"] == [{"bbox": [1.0, 2.0, 30.0, 40.0], "confidence": 0.77}]
    assert r.json()["phrase"] == "rebar"
    assert client.post("/v1/ground", files=files, data={"phrase": "  "}).status_code == 422
    assert client.post("/v1/ground", files=files, data={"phrase": "x" * 81}).status_code == 422
    assert client.post("/v1/ground", files=files, data={"phrase": "crane"}).json()["boxes"] == []


def test_limits_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SITESIGHT_MAX_BOXES", "7")
    monkeypatch.setenv("SITESIGHT_REVIEW_THRESHOLD", "0.9")
    lim = Limits.from_env()
    assert lim.max_boxes == 7 and lim.review_threshold == 0.9 and lim.max_pixels == 40_000_000


def test_batch_cli_writes_one_line_per_image_and_reports_failures(tmp_path: Path) -> None:
    (tmp_path / "ok.png").write_bytes(png_bytes())
    (tmp_path / "bad.png").write_bytes(b"junk")
    boxes = {
        "ok.png": [[0, 0, 20, 20]],
        "bad.png": [[0, 0, 5, 5]],
        "missing.png": [[0, 0, 5, 5]],
        "badbox.png": [[0, 0, 0, 0]],
    }
    (tmp_path / "badbox.png").write_bytes(png_bytes())
    out = tmp_path / "out.jsonl"
    ok, failed = describe_batch(StubBackend(), tmp_path, boxes, out, Limits())
    assert (ok, failed) == (1, 3)
    rows = {r["file_name"]: r for r in map(json.loads, out.read_text().splitlines())}
    assert rows["ok.png"]["summary"] == {"foundation": 1}
    assert rows["ok.png"]["regions"][0]["fields"]["resource"] == "foundation"
    assert "not a readable image" in rows["bad.png"]["error"]
    assert "not found" in rows["missing.png"]["error"]
    assert "empty or outside" in rows["badbox.png"]["error"]


def test_cli_main_exit_status_and_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "ok.png").write_bytes(png_bytes())
    (tmp_path / "boxes.json").write_text(json.dumps({"ok.png": [[0, 0, 20, 20]]}))
    monkeypatch.setattr("sitesight.serve.app.load_backend", lambda d, t=None: StubBackend())
    argv = ["describe", "--model-dir", str(tmp_path), "--images", str(tmp_path)]
    argv += ["--boxes", str(tmp_path / "boxes.json"), "--out", str(tmp_path / "o.jsonl")]
    assert main(argv) == 0
    assert json.loads((tmp_path / "o.jsonl").read_text())["file_name"] == "ok.png"
    (tmp_path / "boxes.json").write_text(json.dumps({"nope.png": [[0, 0, 5, 5]]}))
    assert main(argv) == 1


def test_confidence_equal_to_the_threshold_is_not_flagged() -> None:
    class Exactly(StubBackend):
        def describe_region(self, image: Image.Image, bbox: list[float]) -> tuple[str, float]:
            return "Ten foundations are installed", 0.5

    c = TestClient(create_app(Exactly(), Limits(review_threshold=0.5)))
    assert post_describe(c, [[0, 0, 5, 5]]).json()["regions"][0]["review"] is False
    c = TestClient(create_app(Exactly(), Limits(review_threshold=0.5000001)))
    assert post_describe(c, [[0, 0, 5, 5]]).json()["regions"][0]["review"] is True
