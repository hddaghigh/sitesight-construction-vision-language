"""Failure classification: hand-built cases where the expected type is obvious."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from sitesight.data.caption_fields import CATEGORY_IDS, apply_caption_fields
from sitesight.data.predictions import Prediction, PredictionSet
from sitesight.data.via_parse import parse_via
from sitesight.eval.data import RegionGT, build_ground_truth
from sitesight.eval.gallery import (
    Failure,
    classify_caption_failure,
    classify_caption_failures,
    classify_grounding_failures,
    main,
    pick_examples,
    render,
)
from sitesight.eval.grounding import GT, Det


def _g(box: list[float], area: float, kind: str = "single") -> GT:
    return GT("a.jpg", "worker", box, area, kind, 3)


def _d(box: list[float], s: float = 0.9) -> Det:
    return Det("a.jpg", "worker", box, s)


def kinds(fails: list[Failure]) -> list[str]:
    return [f.kind for f in fails]


def test_each_grounding_failure_type() -> None:
    # group box with two individual predictions inside (IoU of each with the group is 0.09 -> unmatched)
    group = _g([0, 0, 100, 100], 0.2, "group")
    split = classify_grounding_failures([group], [_d([5, 5, 30, 30]), _d([50, 50, 30, 30])])
    missed = [f for f in split if f.gt_bbox]  # ground-truth boxes that were not found
    assert kinds(missed) == ["group_split"] and "2 predicted boxes" in missed[0].detail
    assert kinds([f for f in split if not f.gt_bbox]) == ["spurious", "spurious"]  # unmatched preds
    # a small ground-truth box with nothing predicted
    assert kinds(classify_grounding_failures([_g([0, 0, 10, 10], 0.005)], [])) == ["small_missed"]
    # a normal-size box with only a far-away prediction: no overlap at all (+ the prediction is spurious)
    far = classify_grounding_failures([_g([0, 0, 40, 40], 0.1)], [_d([200, 200, 40, 40])])
    assert kinds(far) == ["no_overlap", "spurious"] and far[0].best_iou == 0.0
    # overlap but below 0.5: inter 800 / union 2400 = 0.333
    poor = classify_grounding_failures([_g([0, 0, 40, 40], 0.1)], [_d([20, 0, 40, 40])])
    assert poor[0].kind == "poor_localisation" and poor[0].best_iou == pytest.approx(1 / 3)


def test_matched_boxes_are_not_failures_and_duplicates_are_named() -> None:
    gt = _g([0, 0, 40, 40], 0.1)
    ok = classify_grounding_failures([gt], [_d([0, 0, 40, 40], 0.9), _d([1, 1, 40, 40], 0.8)])
    # the first prediction matches; the second overlaps it (IoU > 0.5) and is a duplicate
    assert kinds(ok) == ["duplicate"]
    assert classify_grounding_failures([gt], [_d([0, 0, 40, 40])]) == []


def test_top_k_limits_what_counts_as_found() -> None:
    gt = _g([0, 0, 40, 40], 0.1)
    far = [_d([200 + i, 200, 10, 10], 0.9 - 0.01 * i) for i in range(3)]
    exact = [_d([0, 0, 40, 40], 0.1)]  # correct but ranked last
    assert kinds(classify_grounding_failures([gt], far + exact, top_k=3))[0] == "no_overlap"
    assert [f for f in classify_grounding_failures([gt], far + exact, top_k=10) if f.gt_bbox] == []


def _kind(reference: str, prediction: str) -> str | None:
    res = classify_caption_failure(reference, prediction)
    return None if res is None else res[0]


def test_caption_failure_types() -> None:
    ref = "An orange excavator is digging soil"
    assert _kind(ref, ref) is None
    assert _kind(ref, "AN orange excavator is digging soil.") is None  # case and punctuation
    assert _kind(ref, "Two orange excavators are digging soil") == "count_mismatch"
    assert _kind(ref, "A yellow excavator is digging soil") == "colour_mismatch"
    assert _kind(ref, "An orange dump truck is digging soil") == "resource_mismatch"
    assert _kind(ref, "An orange excavator is located on soil") == "state_mismatch"
    assert _kind(ref, "An orange excavator is digging rocks") == "target_mismatch"
    assert _kind(ref, "a big machine") == "unparsed"
    # several differences: the first in the fixed order (resource, count, colour, ...) wins
    assert _kind(ref, "Two yellow dump trucks are digging soil") == "resource_mismatch"
    assert _kind("garbage reference", "A worker is located on soil") is None


def test_classify_caption_failures_skips_missing_predictions() -> None:
    regions = [
        RegionGT(1, "a.jpg", [0, 0, 5, 5], "A worker is located on soil", {"resource": "worker"}),
        RegionGT(2, "a.jpg", [9, 9, 5, 5], "Fences are installed in a line", {"resource": "fence"}),
    ]
    f = classify_caption_failures(regions, {1: "Two workers are located on soil"})
    assert kinds(f) == ["count_mismatch"] and f[0].category == "worker"


def test_pick_examples_is_seeded_and_capped() -> None:
    fails = [Failure("no_overlap", f"{i}.jpg", "worker", [0, 0, 5, 5]) for i in range(10)]
    a, b = pick_examples(fails, 3, seed=1), pick_examples(fails, 3, seed=1)
    assert a == b and len(a["no_overlap"]) == 3
    assert pick_examples(fails, 3, seed=2) != a
    assert len(pick_examples(fails[:2], 3, 0)["no_overlap"]) == 2


def test_render_writes_an_image(tmp_path: Path) -> None:
    src = tmp_path / "x.jpg"
    Image.new("RGB", (100, 60)).save(src)
    out = tmp_path / "o" / "r.png"
    render(src, Failure("no_overlap", "x.jpg", "worker", [10, 10, 20, 20], [[50, 20, 10, 10]]), out)
    assert Image.open(out).size == (100, 60)


def test_cli_end_to_end(via_fixture: tuple[Path, Path], tmp_path: Path) -> None:
    ds, _ = parse_via(*via_fixture)
    apply_caption_fields(ds)
    for img in ds.images:
        img.split = "val"
    ann = tmp_path / "ann.json"
    ds.save(ann)
    gts, regions, _ = build_ground_truth(ds, "val")
    preds = [  # everything found except the excavator group box in 10.jpg
        Prediction(g.file_name, g.bbox, 0.9, None, CATEGORY_IDS[g.category])
        for g in gts
        if g.category != "excavator"
    ]
    gpath = tmp_path / "g.json"
    PredictionSet("m", preds, {"task": "grounding", "images": ["1.jpg", "2.jpg", "10.jpg"]}).save(
        gpath
    )
    rpreds = [
        Prediction(r.file_name, r.bbox, 0.5, "A worker is located on soil", None, r.ann_id)
        for r in regions
    ]
    rpath = tmp_path / "r.json"
    PredictionSet("m", rpreds, {"task": "region_captioning"}).save(rpath)
    out = tmp_path / "gallery"
    assert main(["--annotations", str(ann), "--images", str(via_fixture[1]), "--grounding", str(gpath),
                 "--regions", str(rpath), "--out-dir", str(out)]) == 0  # fmt: skip
    rep = json.loads((out / "failure_summary.json").read_text())
    assert rep["grounding"]["n_failures"] == 1 and rep["grounding"]["by_kind"] == {"no_overlap": 1}
    # 3 regions predicted as "A worker ..."; the worker region agrees, fence and excavators differ
    assert rep["captioning"]["n_failures"] == 2
    assert (out / "grounding" / "no_overlap_1.png").exists() and (out / "captioning").is_dir()
