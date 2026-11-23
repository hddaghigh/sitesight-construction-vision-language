"""Monitoring: quality statistics, drift tests, request log, report and CLI (real image files)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from sitesight.monitor import perturb
from sitesight.monitor.cli import main as monitor_main
from sitesight.monitor.drift import (
    detect_drift,
    kolmogorov_sf,
    ks_2samp,
    mmd2_unbiased,
    mmd_permutation_test,
)
from sitesight.monitor.quality import QualityReference, image_stats, laplacian_variance, stack
from sitesight.monitor.report import render_report
from sitesight.monitor.request_log import RequestLog, error_rate, log_features, read_log
from sitesight.serve.api import Limits, create_app

# ---------------------------------------------------------------- quality statistics


def test_laplacian_variance_by_hand() -> None:
    assert laplacian_variance(np.full((5, 5), 0.7)) == pytest.approx(0.0, abs=1e-12)
    y = np.zeros((4, 4))
    y[1, 1] = 1.0
    # interior (2 x 2) responses are 4, -1, -1, 0: mean 0.5, variance (12.25 + 2*2.25 + 0.25) / 4
    assert laplacian_variance(y) == pytest.approx(4.25)
    assert laplacian_variance(np.zeros((2, 9))) == 0.0  # too small for an interior


def test_image_stats_on_solid_images() -> None:
    red = image_stats(Image.new("RGB", (10, 6), (255, 0, 0)))
    assert (red["width"], red["height"]) == (10.0, 6.0)
    assert red["brightness"] == pytest.approx(76 / 255)  # PIL luma of pure red is 76
    assert red["contrast"] == pytest.approx(0.0, abs=1e-12) and red["sharpness"] == 0.0
    white = image_stats(Image.new("RGB", (8, 8), (255, 255, 255)))
    black = image_stats(Image.new("RGB", (8, 8), (0, 0, 0)))
    assert (white["bright_frac"], white["dark_frac"]) == (1.0, 0.0)
    assert (black["bright_frac"], black["dark_frac"]) == (0.0, 1.0)
    # other modes are converted; size is reported for the original, analysis is reduced
    assert image_stats(Image.new("L", (2000, 1000), 128))["width"] == 2000.0


def test_analysis_is_done_on_the_reduced_image() -> None:
    # a 1-pixel checkerboard averages to flat grey once reduced 2:1 (area average), so the
    # reduced image has no detail; analysing the full-size image would give a large value
    yy, xx = np.indices((1024, 1024))
    board = Image.fromarray(((xx + yy) % 2 * 255).astype(np.uint8))
    assert image_stats(board)["sharpness"] == pytest.approx(0.0, abs=1e-12)
    small = Image.fromarray(((xx[:512, :512] + yy[:512, :512]) % 2 * 255).astype(np.uint8))
    assert image_stats(small)["sharpness"] == pytest.approx(16.0)  # +-4 everywhere, not reduced


def texture(seed: int, size: tuple[int, int] = (96, 64)) -> Image.Image:
    """A synthetic photo-like image: random base level and random texture strength."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(70, 170)
    amp = rng.uniform(15, 45)
    a = base + rng.normal(0, amp, (size[1], size[0], 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def test_blur_lowers_sharpness_and_darkening_lowers_brightness() -> None:
    im = texture(1)
    base = image_stats(im)
    assert image_stats(perturb.blur(im, 2.0))["sharpness"] < 0.2 * base["sharpness"]
    assert image_stats(perturb.scale_brightness(im, 0.5))["brightness"] == pytest.approx(
        base["brightness"] * 0.5, rel=0.02
    )
    assert image_stats(perturb.add_noise(im, 30.0))["sharpness"] > base["sharpness"]


def test_perturbation_helpers() -> None:
    im = Image.new("RGB", (40, 20), (200, 100, 50))
    assert perturb.scale_brightness(im, 2.0).getpixel((0, 0)) == (255, 200, 100)  # clipped at 255
    assert perturb.downscale(im, 0.5).size == (20, 10)
    assert perturb.downscale_then_restore(im, 0.5).size == (40, 20)
    assert perturb.jpeg(texture(2), 5).size == (96, 64)
    assert not np.array_equal(
        np.asarray(perturb.jpeg(texture(2), 5)), np.asarray(texture(2).convert("RGB"))
    )
    noisy = np.asarray(perturb.add_noise(Image.new("RGB", (200, 200), (128, 128, 128)), 10.0))
    assert noisy.std() == pytest.approx(10.0, rel=0.05)
    assert set(perturb.PERTURBATIONS) >= {"blur_r2", "darker_x0.5", "rescaled_x0.5"}


def reference_rows(n: int = 60) -> list[dict[str, float]]:
    return [image_stats(texture(i)) for i in range(n)]


def test_reference_flags_unusual_images_and_needs_enough_data() -> None:
    ref = QualityReference.fit(reference_rows())
    ok = image_stats(texture(1000))
    assert ref.flag(ok) == []
    blurry = image_stats(perturb.blur(texture(1000), 3.0))
    assert "blurry" in ref.flag(blurry)
    dark = image_stats(perturb.scale_brightness(texture(1000), 0.2))
    assert "too_dark" in ref.flag(dark)
    bright = image_stats(perturb.scale_brightness(texture(1000), 2.5))
    assert "too_bright" in ref.flag(bright)
    assert "width_outside_reference" in ref.flag(image_stats(Image.new("RGB", (48, 64))))
    with pytest.raises(ValueError, match="at least 30"):
        QualityReference.fit(reference_rows(29))
    again = QualityReference.from_dict(json.loads(json.dumps(ref.to_dict())))
    assert again.band("sharpness") == ref.band("sharpness")


# ---------------------------------------------------------------- drift tests


def test_kolmogorov_survival_matches_the_published_critical_values() -> None:
    # c(alpha) for alpha = 0.10, 0.05, 0.01 are 1.2238, 1.3581, 1.6276
    assert kolmogorov_sf(1.2238) == pytest.approx(0.10, abs=1e-3)
    assert kolmogorov_sf(1.3581) == pytest.approx(0.05, abs=1e-3)
    assert kolmogorov_sf(1.6276) == pytest.approx(0.01, abs=1e-3)
    assert kolmogorov_sf(0.1) == 1.0 and kolmogorov_sf(10.0) < 1e-12
    # tabulated values away from the tail, where the alternating signs matter
    assert kolmogorov_sf(0.5) == pytest.approx(0.9639, abs=1e-3)
    assert kolmogorov_sf(0.8) == pytest.approx(0.5441, abs=1e-3)


def test_ks_statistic_by_hand() -> None:
    d, p = ks_2samp(np.array([1, 2, 3, 4]), np.array([3, 4, 5, 6]))
    assert d == pytest.approx(0.5) and 0 < p < 1
    d0, p0 = ks_2samp(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]))
    assert d0 == 0.0 and p0 == 1.0
    d1, p1 = ks_2samp(np.arange(50.0), np.arange(50.0) + 100)
    assert d1 == 1.0 and p1 < 1e-10  # disjoint samples
    with pytest.raises(ValueError, match="non-empty"):
        ks_2samp(np.array([]), np.array([1.0]))


def test_unbiased_mmd_by_hand() -> None:
    kxx = np.array([[1.0, 0.5], [0.5, 1.0]])
    kyy = np.array([[1.0, 0.2], [0.2, 1.0]])
    kxy = np.full((2, 2), 0.3)
    # (3 - 2) / 2 + (2.4 - 2) / 2 - 2 * 0.3 = 0.5 + 0.2 - 0.6
    assert mmd2_unbiased(kxx, kyy, kxy) == pytest.approx(0.1)


def test_mmd_permutation_test_separates_shifted_from_same() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (150, 3))
    same = rng.normal(0, 1, (40, 3))
    shifted = rng.normal(1.0, 1, (40, 3))
    _, p_same = mmd_permutation_test(x, same, n_perm=200, seed=1)
    stat, p_shift = mmd_permutation_test(x, shifted, n_perm=200, seed=1)
    assert p_same > 0.05 and p_shift <= 0.01 and stat > 0
    assert mmd_permutation_test(x, shifted, 200, 1) == (stat, p_shift)  # deterministic


def gaussian_cols(
    n: int, seed: int, shift: dict[str, float] | None = None
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shift = shift or {}
    return {f: rng.normal(shift.get(f, 0.0), 1.0, n) for f in ("a", "b", "c")}


def test_detect_drift_quiet_on_same_distribution_and_loud_on_a_shifted_feature() -> None:
    ref = gaussian_cols(400, 0)
    ok = detect_drift(ref, gaussian_cols(40, 1))
    assert ok.status == "ok" and ok.alerts == []
    hit = detect_drift(ref, gaussian_cols(40, 2, {"b": 2.0}))
    assert hit.status == "alert"
    alerted = {(t.name, t.feature) for t in hit.alerts}
    assert ("ks", "b") in alerted and ("ks", "a") not in alerted
    # Bonferroni: three features, half of alpha for the KS family -> 0.01 / 2 / 3 each
    assert [t.threshold for t in hit.tests if t.name == "ks"] == pytest.approx([0.01 / 6] * 3)
    assert [t.threshold for t in hit.tests if t.name == "mmd"] == [0.005]


def test_detect_drift_refuses_small_batches_and_bad_settings() -> None:
    ref = gaussian_cols(100, 0)
    small = detect_drift(ref, gaussian_cols(19, 1))
    assert small.status == "too_small" and small.tests == [] and small.n_batch == 19
    assert detect_drift(ref, gaussian_cols(20, 1)).status != "too_small"
    with pytest.raises(ValueError, match="alpha"):
        detect_drift(ref, ref, alpha=1.5)
    with pytest.raises(ValueError, match="n_perm"):
        detect_drift(ref, ref, n_perm=100)  # 1/101 > 0.005
    with pytest.raises(ValueError, match="no features"):
        detect_drift({"a": np.ones(30)}, {"z": np.ones(30)})


def test_a_joint_shift_no_single_feature_shows_is_caught_by_mmd() -> None:
    rng = np.random.default_rng(3)
    xa = rng.normal(0, 1, 600)
    ref = {"a": xa, "b": xa + rng.normal(0, 0.15, 600)}  # a and b are almost identical
    ya = rng.normal(0, 1, 60)
    batch = {"a": ya, "b": -ya + rng.normal(0, 0.15, 60)}  # same marginals, correlation flipped
    res = detect_drift(ref, batch)
    assert res.status == "alert"
    assert [t.name for t in res.alerts] == ["mmd"]


# ---------------------------------------------------------------- request log


def test_log_round_trip_skips_bad_lines_and_stores_no_identifying_data(tmp_path: Path) -> None:
    log = RequestLog(tmp_path / "sub" / "r.jsonl", clock=lambda: 0.0)
    stats = image_stats(texture(5))
    log.describe_ok(
        latency_ms=12.3456,
        image=stats,
        n_boxes=2,
        confidences=[0.5, 0.9],
        n_parsed=1,
        n_review=1,
        summary={"rebar": 1},
    )
    log.failure("describe", 415, "not a readable image", 1.0)
    with (tmp_path / "sub" / "r.jsonl").open("a") as f:
        f.write("{broken\n\n[1, 2]\n")
    records, bad = read_log(tmp_path / "sub" / "r.jsonl")
    assert bad == 2 and len(records) == 2
    ok = records[0]
    assert ok["ts"] == "1970-01-01T00:00:00+00:00" and ok["latency_ms"] == 12.346
    assert ok["mean_confidence"] == pytest.approx(0.7) and ok["parsed_rate"] == 0.5
    assert ok["review_rate"] == 0.5 and ok["summary"] == {"rebar": 1}
    assert not {"caption", "file_name", "bbox", "image_bytes"} & set(ok)
    cols = log_features(records)  # the 415 has no image or outputs
    assert len(cols["sharpness"]) == 1 and cols["mean_confidence"][0] == pytest.approx(0.7)
    assert error_rate(records) == 0.5 and error_rate([]) is None


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class Stub:
    def describe_region(self, image: Image.Image, bbox: list[float]) -> tuple[str, float]:
        return ("Ten foundations are installed", 0.8) if bbox[0] < 50 else ("blah", 0.3)

    def ground(self, image: Image.Image, phrase: str) -> list[tuple[list[float], float]]:
        return [([1.0, 2.0, 3.0, 4.0], 0.7)]

    def info(self) -> dict[str, str]:
        return {"engine": "stub"}


def test_api_logs_successes_and_rejections_but_not_when_disabled(tmp_path: Path) -> None:
    path = tmp_path / "req.jsonl"
    c = TestClient(create_app(Stub(), Limits(max_boxes=2), RequestLog(path)))
    files = {"image": ("a.png", png(texture(7)), "image/png")}
    assert c.post(
        "/v1/describe", files=files, data={"boxes": "[[0,0,20,20],[60,0,20,20]]"}
    ).is_success
    assert c.post("/v1/describe", files=files, data={"boxes": "[]"}).status_code == 422
    junk = {"image": ("a.png", b"junk", "image/png")}
    assert c.post("/v1/describe", files=junk, data={"boxes": "[[0,0,5,5]]"}).status_code == 415
    assert c.post("/v1/ground", files=files, data={"phrase": "rebar"}).is_success
    assert c.post("/v1/ground", files=files, data={"phrase": " "}).status_code == 422
    records, bad = read_log(path)
    assert bad == 0 and [r["status"] for r in records] == [200, 422, 415, 200, 422]
    assert [r["endpoint"] for r in records] == ["describe"] * 3 + ["ground"] * 2
    first = records[0]
    assert first["n_boxes"] == 2 and first["parsed_rate"] == 0.5 and first["review_rate"] == 0.5
    assert first["mean_confidence"] == pytest.approx(0.55)
    assert first["image"]["width"] == 96.0
    assert records[1]["error"].startswith("boxes must be") and records[1]["image"] is not None
    assert records[2]["image"] is None and records[3]["n_found"] == 1
    # no log object: the endpoints still work and nothing is written
    quiet = TestClient(create_app(Stub(), Limits()))
    assert quiet.post("/v1/ground", files=files, data={"phrase": "rebar"}).is_success


# ---------------------------------------------------------------- report and CLI


def test_report_states_the_verdict_and_escapes_untrusted_text() -> None:
    ref = {"a": np.arange(40.0)}
    ref["<script>alert(1)</script>"] = np.arange(40.0)
    bad = {"<script>alert(1)</script>": np.arange(40.0) + 1000, "a": np.arange(40.0) + 1000}
    res = detect_drift(ref, bad)
    page = render_report(res, ref, bad, [["blurry"], [], ["<b>x</b>"]], title="T<1>")
    assert "DRIFT ALERT" in page and "2 of 3 flagged" in page
    assert "<script>" not in page and "&lt;b&gt;x&lt;/b&gt;" in page and "T&lt;1&gt;" in page
    assert "ALERT</td>" in page
    small = detect_drift(ref, {"a": np.arange(3.0)})
    assert "Not enough data to test" in render_report(small, ref, {"a": np.arange(3.0)}, [])


def write_images(folder: Path, seeds: range, fn: object = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        im = texture(s)
        if fn is not None:
            im = fn(im)  # type: ignore[operator]
        im.save(folder / f"{s}.png")


def test_cli_fit_and_check_images_exit_codes_and_report(tmp_path: Path) -> None:
    write_images(tmp_path / "ref", range(0, 150))
    write_images(tmp_path / "clean", range(1000, 1040))
    write_images(tmp_path / "blur", range(1000, 1040), lambda im: perturb.blur(im, 3.0))
    write_images(tmp_path / "dark", range(1000, 1040), lambda im: perturb.scale_brightness(im, 0.4))
    write_images(tmp_path / "few", range(1000, 1005))
    ref = tmp_path / "ref.json"
    assert monitor_main(["fit", "--images", str(tmp_path / "ref"), "--out", str(ref)]) == 0

    def check(name: str) -> int:
        argv = ["check", "--reference", str(ref), "--images", str(tmp_path / name)]
        return monitor_main([*argv, "--report", str(tmp_path / f"{name}.html")])

    assert check("clean") == 0
    assert check("blur") == 1 and check("dark") == 1
    assert check("few") == 2
    assert "DRIFT ALERT" in (tmp_path / "blur.html").read_text(encoding="utf-8")
    assert "No drift detected" in (tmp_path / "clean.html").read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        monitor_main(["check", "--reference", str(ref), "--report", "x.html"])  # no batch source
    with pytest.raises(SystemExit):
        argv = ["fit", "--images", str(tmp_path), "--annotations", "a.json", "--split", "test"]
        monitor_main([*argv, "--out", str(ref)])


def test_cli_check_a_request_log_covers_model_outputs(tmp_path: Path) -> None:
    def make_log(path: Path, seeds: range, conf: float, fn: object = None) -> None:
        log = RequestLog(path)
        for s in seeds:
            im = texture(s) if fn is None else fn(texture(s))  # type: ignore[operator]
            log.describe_ok(
                latency_ms=1.0,
                image=image_stats(im),
                n_boxes=1,
                confidences=[conf + 0.01 * (s % 7)],
                n_parsed=1,
                n_review=0,
                summary={},
            )

    make_log(tmp_path / "good.jsonl", range(0, 150), 0.80)
    make_log(tmp_path / "same.jsonl", range(1000, 1040), 0.80)
    make_log(tmp_path / "lowconf.jsonl", range(1000, 1040), 0.40)  # only the outputs change
    ref = tmp_path / "ref.json"
    assert monitor_main(["fit", "--log", str(tmp_path / "good.jsonl"), "--out", str(ref)]) == 0
    base = ["check", "--reference", str(ref), "--json", str(tmp_path / "o.json")]
    same = [*base, "--log", str(tmp_path / "same.jsonl"), "--report", str(tmp_path / "a.html")]
    assert monitor_main(same) == 0
    code = monitor_main(
        [*base, "--log", str(tmp_path / "lowconf.jsonl"), "--report", str(tmp_path / "b.html")]
    )
    assert code == 1
    result = json.loads((tmp_path / "o.json").read_text())
    assert any(t["feature"] == "mean_confidence" and t["alerted"] for t in result["tests"])


def test_stack_orders_columns_by_feature() -> None:
    rows = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
    cols = stack(rows, ("b", "a"))
    assert list(cols) == ["b", "a"] and cols["a"].tolist() == [1.0, 3.0]
