"""Parser tests. Expected values are derived by hand in tests/conftest.py's docstring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from sitesight.data.schema import FLAG_CLIPPED, FLAG_DUPLICATE, FLAG_EMPTY_CAPTION, Dataset
from sitesight.data.via_parse import (
    DatasetError,
    jpeg_size,
    main,
    normalise_caption,
    parse_via,
)
from tests.conftest import WRONG_SIZE_FOR_10


@pytest.mark.parametrize("size", [(100, 60), (960, 540), (7, 3), (1, 1), (3000, 2)])
@pytest.mark.parametrize("progressive", [False, True])
def test_jpeg_size_matches_pillow(tmp_path: Path, size: tuple[int, int], progressive: bool) -> None:
    p = tmp_path / "x.jpg"
    Image.new("RGB", size).save(p, "JPEG", progressive=progressive)
    assert jpeg_size(p) == size


def test_jpeg_size_rejects_non_jpeg(tmp_path: Path) -> None:
    p = tmp_path / "x.jpg"
    p.write_bytes(b"not a jpeg at all")
    with pytest.raises(ValueError, match="not a JPEG"):
        jpeg_size(p)


def test_normalise_caption() -> None:
    assert normalise_caption("  A  worker\tis here \n") == "A worker is here"
    assert normalise_caption("   ") == ""


def test_counts_and_order(via_fixture: tuple[Path, Path]) -> None:
    ds, report = parse_via(*via_fixture)
    # 2.jpg sorts before 10.jpg (numeric, not lexicographic).
    assert [i.file_name for i in ds.images] == ["1.jpg", "2.jpg", "10.jpg"]
    assert [i.id for i in ds.images] == [1, 2, 3]
    assert (ds.images[0].width, ds.images[0].height) == (100, 60)
    assert (ds.images[2].width, ds.images[2].height) == (80, 40)
    assert report.n_images == 3
    assert report.n_regions_in_source == 9  # 8 + 0 + 1
    assert report.n_dropped == 4  # r4, r5, r6, r7 of 1.jpg
    assert report.n_annotations_kept == 5
    assert len(ds.annotations) == 5
    assert report.counts == {
        "clipped": 1,
        "dropped_degenerate_box": 2,
        "dropped_malformed_region": 1,
        "dropped_unsupported_shape": 1,
        "duplicate": 1,
        "empty_caption": 1,
        "size_mismatch": 1,
        "zero_regions": 1,
    }
    assert [a.id for a in ds.annotations] == [1, 2, 3, 4, 5]


def test_each_region_outcome(via_fixture: tuple[Path, Path]) -> None:
    ds, report = parse_via(*via_fixture)
    by_region = {
        (a.source["via_key"].split("jpg")[0], a.source["region_id"]): a for a in ds.annotations
    }
    ok = by_region[("1.", "0")]
    assert ok.bbox == [10, 10, 20, 20] and ok.area == 400 and ok.flags == []
    assert ok.bbox_raw is None and ok.category_id is None and ok.fields is None

    clipped = by_region[("1.", "1")]
    assert clipped.bbox == [0, 50, 25, 10]  # x: -5..25 -> 0..25 ; y: 50..70 -> 50..60
    assert clipped.bbox_raw == [-5, 50, 30, 20]
    assert clipped.flags == [FLAG_CLIPPED]
    assert clipped.caption == "Fences are installed in a line"
    assert clipped.caption_raw == "Fences are installed in a line "  # raw is preserved verbatim

    dup = by_region[("1.", "2")]
    assert dup.flags == [FLAG_DUPLICATE]
    assert by_region[("1.", "0")].flags == []  # the first occurrence is not the duplicate

    empty = by_region[("1.", "3")]
    assert empty.caption == "" and empty.flags == [FLAG_EMPTY_CAPTION]

    full = by_region[("10.", "0")]
    assert full.bbox == [0, 0, 80, 40] and full.flags == []  # touching the border is not clipping

    dropped = {(i.image, i.region): i.code for i in report.issues if i.code.startswith("dropped")}
    assert dropped == {
        ("1.jpg", "4"): "dropped_degenerate_box",
        ("1.jpg", "5"): "dropped_degenerate_box",
        ("1.jpg", "6"): "dropped_unsupported_shape",
        ("1.jpg", "7"): "dropped_malformed_region",
    }
    size_issue = next(i for i in report.issues if i.code == "size_mismatch")
    assert size_issue.image == "10.jpg" and str(WRONG_SIZE_FOR_10) in size_issue.detail


def test_missing_image_is_an_error(via_fixture: tuple[Path, Path]) -> None:
    via, images = via_fixture
    (images / "2.jpg").unlink()
    with pytest.raises(DatasetError, match=r"2\.jpg"):
        parse_via(via, images)


def test_via2_list_regions_are_accepted(tmp_path: Path) -> None:
    images = tmp_path / "img"
    images.mkdir()
    Image.new("RGB", (50, 50)).save(images / "1.jpg", "JPEG")
    size = (images / "1.jpg").stat().st_size
    via = {
        f"1.jpg{size}": {
            "filename": "1.jpg",
            "size": size,
            "regions": [
                {
                    "shape_attributes": {
                        "name": "rect",
                        "x": 1.4,
                        "y": 2,
                        "width": 10,
                        "height": 9.6,
                    },
                    "region_attributes": {"phrase": "A worker"},
                }
            ],
        }
    }
    p = tmp_path / "via.json"
    p.write_text(json.dumps(via), encoding="utf-8")
    ds, _ = parse_via(p, images)
    assert ds.annotations[0].bbox == [1, 2, 10, 10]  # floats rounded to nearest pixel


def test_save_load_roundtrip_and_determinism(
    via_fixture: tuple[Path, Path], tmp_path: Path
) -> None:
    ds, _ = parse_via(*via_fixture)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    ds.save(a)
    parse_via(*via_fixture)[0].save(b)
    assert a.read_bytes() == b.read_bytes()  # same input -> byte-identical output
    assert Dataset.load(a).to_dict() == ds.to_dict()
    assert ds.info["source_doi"] == "10.17632/4h68fmktwh.1"
    assert len(ds.info["source_sha256"]) == 64


def test_cli_writes_outputs(
    via_fixture: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    via, images = via_fixture
    out, rep = tmp_path / "out" / "annotations.json", tmp_path / "out" / "report.json"
    assert (
        main(["--via", str(via), "--images", str(images), "--out", str(out), "--report", str(rep)])
        == 0
    )
    assert json.loads(rep.read_text())["counts"]["clipped"] == 1
    assert len(json.loads(out.read_text())["annotations"]) == 5
    assert "annotations kept:       5" in capsys.readouterr().out
