"""Audit tests. Every expected number is computed by hand in a comment."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from sitesight.data.audit import (
    AUDIT_UNPARSED,
    BOX_SHIFTED,
    BOX_TOO_LARGE,
    BOX_TOO_SMALL,
    COLOUR_MISMATCH,
    COUNT_MISMATCH,
    MISSED_BY_ORIGINAL,
    NOT_CONFIRMED_BY_AUDIT,
    STATE_MISMATCH,
    TARGET_MISMATCH,
    AuditFile,
    AuditImage,
    Region,
    audit_image,
    audit_images,
    build_report,
    classify_pair,
    cohen_kappa,
    format_markdown,
    iou,
    main,
    match_regions,
    sample_audit_images,
)
from sitesight.data.schema import AnnotationRecord, Dataset, ImageRecord


def test_iou_hand_examples() -> None:
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 5, 5]) == 0.0
    assert iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(1 / 3)  # inter 50, union 150
    assert iou([0, 0, 10, 10], [0, 0, 5, 5]) == pytest.approx(0.25)  # contained: 25 / 100
    assert iou([0, 0, 10, 10], [10, 0, 10, 10]) == 0.0  # touching edges


def test_match_regions_greedy_and_threshold() -> None:
    orig = [Region([0, 0, 10, 10], "a"), Region([20, 0, 10, 10], "b")]
    audit = [Region([1, 0, 10, 10], "a"), Region([20, 0, 10, 10], "b"), Region([50, 50, 5, 5], "c")]
    m = match_regions(orig, audit)
    assert [(i, j) for i, j, _ in m.pairs] == [(0, 0), (1, 1)]
    assert m.pairs[0][2] == pytest.approx(90 / 110) and m.pairs[1][2] == 1.0
    assert m.unmatched_audit == [2] and m.unmatched_orig == []
    # Greedy picks the HIGHEST-IoU pair: Q (0.818) beats P (0.667) for the single original box.
    one = [Region([0, 0, 10, 10], "x")]
    pq = [Region([2, 0, 10, 10], "p"), Region([1, 0, 10, 10], "q")]
    m2 = match_regions(one, pq)
    assert [(i, j) for i, j, _ in m2.pairs] == [(0, 1)] and m2.unmatched_audit == [0]
    # IoU exactly 0.5 is a match (>=): inter 50, union 100.
    assert len(match_regions(one, [Region([0, 0, 10, 5], "h")]).pairs) == 1
    assert len(match_regions(one, [Region([0, 0, 10, 4], "h")]).pairs) == 0  # 0.4 < 0.5


def test_cohen_kappa_textbook_examples() -> None:
    # 2x2 table [[20,5],[10,15]]: p_o = 35/50 = 0.7; p_e = (25*30 + 25*20)/2500 = 0.5; kappa = 0.4.
    a = ["x"] * 25 + ["y"] * 25
    b = ["x"] * 20 + ["y"] * 5 + ["x"] * 10 + ["y"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa(["x", "y"], ["x", "y"]) == 1.0
    assert cohen_kappa(["x", "x", "y", "y"], ["x", "y", "x", "y"]) == pytest.approx(0.0)
    assert cohen_kappa(["x", "x"], ["x", "x"]) is None  # p_e = 1: kappa undefined
    with pytest.raises(ValueError):
        cohen_kappa([], [])


W = "A worker is located on soil"


def _r(box: Sequence[float], cap: str) -> Region:
    return Region(list(box), cap)


def test_classify_pair_agreement_and_each_field() -> None:
    b = [10, 10, 20, 20]
    assert classify_pair(_r(b, W), _r(b, W), 1.0) == []
    assert classify_pair(_r(b, W), _r(b, "Two workers are located on soil"), 1.0) == [
        COUNT_MISMATCH
    ]
    assert classify_pair(
        _r(b, "A yellow excavator is located on soil"),
        _r(b, "An orange excavator is located on soil"),
        1.0,
    ) == [COLOUR_MISMATCH]
    # digging has no preposition: state differs, and (preposition, target) differs from "on soil".
    codes = classify_pair(
        _r(b, "An orange excavator is located on soil"),
        _r(b, "An orange excavator is digging soil"),
        1.0,
    )
    assert codes == [STATE_MISMATCH, TARGET_MISMATCH]
    assert classify_pair(_r(b, W), _r(b, "garbage"), 1.0) == [AUDIT_UNPARSED]
    assert "resource_mismatch" in classify_pair(
        _r(b, W), _r(b, "A blue tarpaulin is located on soil"), 1.0
    )


def test_classify_pair_box_quality() -> None:
    # orig 20x20 (area 400) vs audit 15x15 (225): IoU = 225/400 = 0.5625 -> original too large.
    big, small = _r([0, 0, 20, 20], W), _r([0, 0, 15, 15], W)
    assert classify_pair(big, small, iou(big.bbox, small.bbox)) == [BOX_TOO_LARGE]
    assert classify_pair(small, big, iou(big.bbox, small.bbox)) == [BOX_TOO_SMALL]
    # same size, shifted by (3,3): inter 17*17 = 289, union 511, IoU = 0.5656 -> shifted.
    a, c = _r([0, 0, 20, 20], W), _r([3, 3, 20, 20], W)
    assert iou(a.bbox, c.bbox) == pytest.approx(289 / 511)
    assert classify_pair(a, c, iou(a.bbox, c.bbox)) == [BOX_SHIFTED]


def _example_image() -> tuple[list[Region], list[Region]]:
    orig = [
        _r([10, 10, 20, 20], W),
        _r([50, 10, 20, 20], "An orange excavator is located on soil"),
        _r([0, 40, 30, 10], "Fences are installed in a line"),
    ]
    audit = [
        _r([10, 10, 20, 20], "Two workers are located on soil"),
        _r([50, 10, 20, 20], "An orange excavator is located on soil"),
        _r([70, 40, 10, 10], "A blue tarpaulin is located on soil"),
    ]
    return orig, audit


def test_audit_image_hand_example() -> None:
    orig, audit = _example_image()
    r = audit_image("1.jpg", orig, audit)
    assert (r.n_orig, r.n_audit, r.n_matched) == (3, 3, 2)
    assert r.taxonomy == {COUNT_MISMATCH: 1, NOT_CONFIRMED_BY_AUDIT: 1, MISSED_BY_ORIGINAL: 1}
    assert r.field_checks["resource"] == (2, 2)  # worker/worker, excavator/excavator
    assert r.field_checks["count"] == (1, 2)  # 1 vs 2 differs; 1 vs 1 agrees
    assert r.exact_caption_matches == 1  # only the excavator caption is identical
    assert r.ious == [1.0, 1.0]


def test_audit_images_summary_values_and_ci() -> None:
    orig, audit = _example_image()
    perfect = [_r([5, 5, 10, 10], "A worker is located on soil")]
    rep = audit_images(
        {"1.jpg": (orig, audit), "2.jpg": (perfect, perfect)}, mode="blind", n_boot=200
    )
    s = rep.summary
    # totals: n_orig 3+1 = 4, n_audit 3+1 = 4, matched 2+1 = 3.
    assert (s["n_original_regions"], s["n_audit_regions"], s["n_matched"]) == (4, 4, 3)
    assert s["precision"] == pytest.approx(0.75) and s["recall"] == pytest.approx(0.75)
    assert s["f1"] == pytest.approx(0.75)
    assert s["resource_agreement"] == 1.0 and s["resource_n_compared"] == 3
    assert s["count_agreement"] == pytest.approx(2 / 3)  # 1 of 2 + 1 of 1
    assert rep.taxonomy[COUNT_MISMATCH] == 1 and rep.taxonomy[MISSED_BY_ORIGINAL] == 1
    lo, hi = rep.ci95["precision"]
    assert lo is not None and hi is not None and lo <= 0.75 <= hi
    again = audit_images(
        {"1.jpg": (orig, audit), "2.jpg": (perfect, perfect)}, mode="blind", n_boot=200
    )
    assert again.ci95 == rep.ci95
    with pytest.raises(ValueError):
        audit_images({}, mode="blind")


def test_precision_and_recall_use_different_denominators() -> None:
    # Original has 3 regions, the audit has 2; both audit regions match -> matched = 2.
    # precision = matched / n_orig = 2/3; recall = matched / n_audit = 2/2 = 1;
    # F1 = 2PR / (P+R) = 2 * (2/3) / (5/3) = 0.8.
    orig = [_r([0, 0, 10, 10], W), _r([20, 0, 10, 10], W), _r([40, 0, 10, 10], W)]
    audit = [_r([0, 0, 10, 10], W), _r([20, 0, 10, 10], W)]
    s = audit_images({"1.jpg": (orig, audit)}, "blind", n_boot=10).summary
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == 1.0
    assert s["f1"] == pytest.approx(0.8)


def test_bootstrap_ci_collapses_when_all_images_identical() -> None:
    perfect = [_r([5, 5, 10, 10], W)]
    rep = audit_images({f"{i}.jpg": (perfect, perfect) for i in range(1, 4)}, "blind", n_boot=50)
    assert rep.ci95["precision"] == [1.0, 1.0] and rep.ci95["resource_agreement"] == [1.0, 1.0]


def _dataset(n: int = 20) -> Dataset:
    ds = Dataset(
        images=[
            ImageRecord(
                id=i, file_name=f"{i}.jpg", width=100, height=60, split="train" if i % 2 else "test"
            )
            for i in range(1, n + 1)
        ]
    )
    for i in range(1, n + 1):
        if i % 5:  # every 5th image has no regions
            ds.annotations.append(
                AnnotationRecord(
                    id=i, image_id=i, bbox=[10, 10, 20, 20], area=400, caption=W, caption_raw=W
                )
            )
    return ds


def test_sampling_is_deterministic_sorted_unique_and_validated() -> None:
    ds = _dataset(30)
    s1 = sample_audit_images(ds, 8, seed=0)
    assert s1 == sample_audit_images(ds, 8, seed=0)
    assert s1 != sample_audit_images(ds, 8, seed=1)
    names = [r["file_name"] for r in s1]
    assert len(set(names)) == 8
    nums = [int(n.split(".")[0]) for n in names]
    assert nums == sorted(nums)  # numeric order: 2.jpg before 10.jpg
    assert all(r["split"] in ("train", "test") and r["n_original_regions"] in (0, 1) for r in s1)
    with pytest.raises(ValueError):
        sample_audit_images(ds, 31)


def test_audit_file_roundtrip_and_build_report(tmp_path: Path) -> None:
    ds = _dataset(10)
    af = AuditFile(mode="blind", sample=["1.jpg", "2.jpg", "5.jpg", "10.jpg"])
    af.images["1.jpg"] = AuditImage("done", [{"bbox": [10, 10, 20, 20], "caption": W}])
    af.images["2.jpg"] = AuditImage(
        "done", [{"bbox": [10, 10, 20, 20], "caption": "Two workers are located on soil"}]
    )
    af.images["5.jpg"] = AuditImage("done", [], "image has nothing")  # original also has none
    af.images["10.jpg"] = AuditImage("todo")
    p = tmp_path / "audit.json"
    af.save(p)
    back = AuditFile.load(p)
    assert back.to_dict() == af.to_dict() and back.done() == ["1.jpg", "2.jpg", "5.jpg"]
    rep = build_report(ds, back, n_boot=50)
    assert (
        rep.summary["n_images"] == 3
        and rep.summary["n_sampled"] == 4
        and rep.summary["n_not_done"] == 1
    )
    assert rep.summary["n_original_regions"] == 2 and rep.summary["n_audit_regions"] == 2
    assert rep.summary["precision"] == 1.0 and rep.taxonomy == {COUNT_MISMATCH: 1}
    bad = AuditFile(mode="blind", sample=["nope.jpg"], images={"nope.jpg": AuditImage("done")})
    with pytest.raises(ValueError, match="not in dataset"):
        build_report(ds, bad)


def test_markdown_mentions_mode_caveat() -> None:
    orig, audit = _example_image()
    rep = audit_images({"1.jpg": (orig, audit)}, mode="review", n_boot=20)
    md = format_markdown(rep)
    assert "review mode" in md.lower() and "upper bounds" in md and "| count_mismatch | 1 |" in md
    assert "upper bounds" not in format_markdown(
        audit_images({"1.jpg": (orig, audit)}, "blind", n_boot=20)
    )


def test_cli_sample_and_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ds = _dataset(12)
    ann = tmp_path / "ann.json"
    ds.save(ann)
    sample = tmp_path / "sample.json"
    assert (
        main(["sample", "--annotations", str(ann), "--n", "4", "--seed", "3", "--out", str(sample)])
        == 0
    )
    rows = json.loads(sample.read_text())["images"]
    assert len(rows) == 4
    af = AuditFile(mode="review", sample=[r["file_name"] for r in rows])
    for r in rows:
        af.images[r["file_name"]] = AuditImage(
            "done", [{"bbox": [10, 10, 20, 20], "caption": W}] if r["n_original_regions"] else []
        )
    audit = tmp_path / "audit.json"
    af.save(audit)
    out = tmp_path / "m3" / "audit_report.json"
    assert (
        main(["report", "--annotations", str(ann), "--audit", str(audit), "--out", str(out)]) == 0
    )
    rep = json.loads(out.read_text())
    assert rep["summary"]["n_images"] == 4 and rep["mode"] == "review"
    assert out.with_suffix(".md").exists() and "Label audit results" in capsys.readouterr().out
