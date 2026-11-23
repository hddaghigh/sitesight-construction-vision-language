"""Ground-truth building, summary metric and the evaluation runner, end to end on real files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sitesight.data.caption_fields import CATEGORY_IDS, apply_caption_fields
from sitesight.data.predictions import Prediction, PredictionSet
from sitesight.data.schema import Dataset
from sitesight.data.via_parse import parse_via
from sitesight.eval.data import build_ground_truth, detections_from_predictions
from sitesight.eval.grounding import GT, Det
from sitesight.eval.run import evaluate_file, main, markdown_table
from sitesight.eval.summary import presence_summary

ALL_IMAGES = ["1.jpg", "2.jpg", "10.jpg"]


@pytest.fixture
def ds(via_fixture: tuple[Path, Path]) -> Dataset:
    d, _ = parse_via(*via_fixture)
    apply_caption_fields(d)
    for img in d.images:
        img.split = "val"
    return d


def _perfect_grounding(ds: Dataset) -> PredictionSet:
    gts, _, _ = build_ground_truth(ds, "val")
    preds = [Prediction(g.file_name, g.bbox, 1.0, None, CATEGORY_IDS[g.category]) for g in gts]
    return PredictionSet(
        "perfect", preds, {"task": "grounding", "queried_classes": "present", "images": ALL_IMAGES}
    )


def test_build_ground_truth_and_exclusions(ds: Dataset) -> None:
    gts, regions, stats = build_ground_truth(ds, "val")
    # Fixture: 5 annotations; one exact duplicate and one empty caption are excluded.
    assert stats == {
        "annotations_in_split": 5,
        "excluded_duplicate": 1,
        "excluded_empty_caption": 1,
        "evaluated": 3,
        "images": 3,
    }
    by = {(g.file_name, g.category): g for g in gts}
    worker = by[("1.jpg", "worker")]
    assert (
        worker.kind == "single"
        and worker.crowd == 2
        and worker.area_frac == pytest.approx(400 / 6000)
    )
    assert by[("1.jpg", "fence")].kind == "group"  # no count: an uncounted plural
    assert by[("10.jpg", "excavator")].kind == "group"  # "Two orange excavators"
    assert by[("10.jpg", "excavator")].crowd == 1 and by[
        ("10.jpg", "excavator")
    ].area_frac == pytest.approx(1.0)
    assert len(regions) == 3 and regions[0].fields["resource"] == "worker"
    # restricting to a subset of images and to a split with no images
    assert build_ground_truth(ds, "val", {"10.jpg"})[2]["evaluated"] == 1
    assert build_ground_truth(ds, "test")[2] == {"evaluated": 0, "images": 0}


def test_perfect_grounding_scores_one(ds: Dataset) -> None:
    r = evaluate_file(_perfect_grounding(ds), ds, "val")
    m = r["protocols"]["given_phrases"]
    assert set(r["protocols"]) == {"given_phrases"}  # queried only where present
    assert m["map50"] == 1.0 and m["map75"] == 1.0 and m["recall50"] == 1.0 and m["recall75"] == 1.0
    assert m["calibration"]["ece"] == 0.0  # all score-1.0 predictions are right
    assert (
        m["breakdown"]["kind"]["single"]["n_gt"] == 1
        and m["breakdown"]["kind"]["group"]["n_gt"] == 2
    )
    assert r["images_evaluated"] == 3 and r["ground_truth"]["evaluated"] == 3
    # queried only where present -> image-level presence F1 would be trivially perfect: not computed
    assert r["presence_summary"] is None


def test_shifted_boxes_score_zero_and_all_queried_gets_both_protocols(ds: Dataset) -> None:
    gts, _, _ = build_ground_truth(ds, "val")
    preds = [
        Prediction(
            g.file_name,
            [g.bbox[0] + g.bbox[2] * 0.9, g.bbox[1], g.bbox[2], g.bbox[3]],
            0.7,
            None,
            CATEGORY_IDS[g.category],
        )
        for g in gts
    ]
    # an extra class predicted in an image that does not contain it (false alarm under 'detection')
    preds.append(Prediction("2.jpg", [0, 0, 20, 20], 0.9, None, CATEGORY_IDS["dump_truck"]))
    r = evaluate_file(
        PredictionSet(
            "bad", preds, {"task": "grounding", "queried_classes": "all", "images": ALL_IMAGES}
        ),
        ds,
        "val",
    )
    assert set(r["protocols"]) == {"detection", "given_phrases"}
    for proto in r["protocols"].values():
        assert proto["map50"] == 0.0 and proto["recall50"] == 0.0
    assert r["protocols"]["detection"]["n_predictions_scored"] == 4  # includes the false alarm
    assert r["protocols"]["given_phrases"]["n_predictions_scored"] == 3  # false alarm dropped
    assert r["protocols"]["given_phrases"]["calibration"]["ece"] == pytest.approx(
        0.7
    )  # confident and wrong


def test_region_captioning_perfect_and_zero_shot_style(ds: Dataset) -> None:
    _, regions, _ = build_ground_truth(ds, "val")
    perfect = PredictionSet(
        "copy",
        [Prediction(r.file_name, r.bbox, 1.0, r.caption, None, r.ann_id) for r in regions],
        {"task": "region_captioning", "images": ALL_IMAGES},
    )
    r = evaluate_file(perfect, ds, "val")
    assert r["n_evaluated"] == 3 and r["coverage"] == 1.0
    assert r["bleu4"] == pytest.approx(1.0) and r["meteor_exact"] > 0.99
    assert r["fields"]["strict_parse_rate"] == 1.0 and r["fields"]["resource_acc_loose"] == 1.0
    free = ["a worker standing", "fence", "yellow machine"]
    zero = PredictionSet(
        "free-text",
        [
            Prediction(rg.file_name, rg.bbox, 0.5, c, None, rg.ann_id)
            for rg, c in zip(regions, free, strict=True)
        ],
        {"task": "region_captioning"},
    )
    z = evaluate_file(zero, ds, "val")
    assert z["fields"]["resource_acc_loose"] == pytest.approx(
        2 / 3
    )  # "yellow machine" names no known resource
    assert z["fields"]["strict_parse_rate"] == 0.0  # free text never fits the template
    assert z["bleu4"] == 0.0 and z["fields"]["colour_acc_loose"] is not None


def test_partial_run_is_scored_on_the_images_it_covers(ds: Dataset) -> None:
    pset = _perfect_grounding(ds)
    pset.model_info["images"] = ["10.jpg"]
    r = evaluate_file(pset, ds, "val")
    assert r["images_evaluated"] == 1 and r["ground_truth"]["evaluated"] == 1
    assert r["protocols"]["given_phrases"]["n_gt"] == 1


def test_presence_summary_only_for_models_queried_on_all_classes(ds: Dataset) -> None:
    """Regression: a model asked only about present classes scored 99% presence F1 (meaningless)."""
    gts, _, _ = build_ground_truth(ds, "val")
    preds = [Prediction(g.file_name, g.bbox, 0.9, None, CATEGORY_IDS[g.category]) for g in gts]
    present = PredictionSet("p", preds, {"task": "grounding", "queried_classes": "present"})
    everything = PredictionSet("a", preds, {"task": "grounding", "queried_classes": "all"})
    assert evaluate_file(present, ds, "val")["presence_summary"] is None
    assert (
        evaluate_file(everything, ds, "val")["presence_summary"]["thresholds"]["0.5"]["f1"] == 1.0
    )
    md = markdown_table([evaluate_file(present, ds, "val"), evaluate_file(everything, ds, "val")])
    assert "Not computed for p" in md and "| a | 100.0 |" in md
    only_present = markdown_table([evaluate_file(present, ds, "val")])
    assert "resource presence" not in only_present  # no section when there is nothing valid to show


def test_sanitize_and_from_metrics_cli(ds: Dataset, tmp_path: Path) -> None:
    """Old metrics files (made before the rule existed) are repaired when re-rendered."""
    from sitesight.eval.run import sanitize_results

    res = evaluate_file(_perfect_grounding(ds), ds, "val")
    res["presence_summary"] = {"thresholds": {"0.5": {"f1": 0.99}}}  # as the old code wrote it
    res["queried_classes"] = "present"
    assert sanitize_results([res])[0]["presence_summary"] is None
    assert sanitize_results([res])[0]["presence_summary"] is None  # idempotent
    old = tmp_path / "old.json"
    res["presence_summary"] = {"thresholds": {"0.5": {"f1": 0.99}}}
    old.write_text(json.dumps([res]))
    out = tmp_path / "fixed.json"
    assert main(["--from-metrics", str(old), "--out", str(out)]) == 0
    assert json.loads(out.read_text())[0]["presence_summary"] is None
    assert "Grounding on `val`" in out.with_suffix(".md").read_text()
    with pytest.raises(SystemExit):  # neither mode given
        main(["--out", str(out)])
    # several metrics files are merged in order (zero-shot + fine-tuned side by side)
    second = tmp_path / "second.json"
    other = dict(res, model="fine-tuned")
    second.write_text(json.dumps([other]))
    both = tmp_path / "both.json"
    assert main(["--from-metrics", str(old), str(second), "--out", str(both)]) == 0
    assert [r["model"] for r in json.loads(both.read_text())] == [res["model"], "fine-tuned"]


def test_restrict_intersects_with_the_images_a_file_covers(ds: Dataset) -> None:
    pset = _perfect_grounding(ds)  # covers all three images
    r = evaluate_file(pset, ds, "val", restrict={"1.jpg", "10.jpg", "not-in-split.jpg"})
    assert (
        r["images_evaluated"] == 2 and r["ground_truth"]["evaluated"] == 3
    )  # 1.jpg has 2, 10.jpg has 1
    narrow = _perfect_grounding(ds)
    narrow.model_info["images"] = ["10.jpg"]
    assert evaluate_file(narrow, ds, "val", restrict={"1.jpg", "10.jpg"})["images_evaluated"] == 1


def test_cli_common_images(ds: Dataset, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ann = tmp_path / "ann.json"
    ds.save(ann)
    a, b = _perfect_grounding(ds), _perfect_grounding(ds)
    a.model_info["images"] = ["1.jpg", "10.jpg"]
    b.model_info["images"] = ["10.jpg", "2.jpg"]
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    a.save(pa)
    b.save(pb)
    out = tmp_path / "m.json"
    argv = ["--annotations", str(ann), "--predictions", str(pa), str(pb), "--out", str(out)]
    assert main([*argv, "--common-images"]) == 0
    assert [r["images_evaluated"] for r in json.loads(out.read_text())] == [1, 1]
    assert "1 images are covered by every file" in capsys.readouterr().out


def test_unknown_task_is_rejected(ds: Dataset) -> None:
    with pytest.raises(ValueError, match=r"model\.task"):
        evaluate_file(PredictionSet("x", [], {"task": "poetry"}), ds, "val")


def test_presence_summary_hand_example() -> None:
    gts = [
        GT("1.jpg", "worker", [0, 0, 1, 1]),
        GT("1.jpg", "fence", [0, 0, 1, 1]),
        GT("10.jpg", "excavator", [0, 0, 1, 1]),
    ]
    dets = [
        Det("1.jpg", "worker", [0, 0, 1, 1], 0.9),
        Det("1.jpg", "fence", [0, 0, 1, 1], 0.2),
        Det("2.jpg", "dump_truck", [0, 0, 1, 1], 0.6),
    ]
    s = presence_summary(dets, gts, ["1.jpg", "2.jpg", "10.jpg"])["thresholds"]
    # t=0.1: pred {worker, fence, dump(2.jpg)} tp=2 -> P=2/3, R=2/3, F1=2/3.
    assert s["0.1"]["precision"] == pytest.approx(2 / 3) and s["0.1"]["f1"] == pytest.approx(2 / 3)
    # t=0.3: pred {worker, dump} tp=1 -> P=1/2, R=1/3, F1 = 2*(1/2)(1/3)/(5/6) = 0.4.
    assert s["0.3"]["precision"] == 0.5 and s["0.3"]["recall"] == pytest.approx(1 / 3)
    assert s["0.3"]["f1"] == pytest.approx(0.4)
    empty = presence_summary([], gts, ["1.jpg"])["thresholds"]["0.5"]
    assert empty["precision"] is None and empty["f1"] == 0.0


def test_detections_from_predictions_skips_classless(ds: Dataset) -> None:
    pset = PredictionSet(
        "m",
        [
            Prediction("1.jpg", [0, 0, 5, 5], 0.5, "x", None),
            Prediction("1.jpg", [0, 0, 5, 5], 0.5, None, 999),
        ],
    )
    assert detections_from_predictions(pset, ds) == []


def test_cli_end_to_end(ds: Dataset, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ann = tmp_path / "ann.json"
    ds.save(ann)
    g = tmp_path / "g.json"
    _perfect_grounding(ds).save(g)
    out = tmp_path / "reports" / "m.json"
    assert (
        main(
            [
                "--annotations",
                str(ann),
                "--split",
                "val",
                "--predictions",
                str(g),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    data = json.loads(out.read_text())
    assert data[0]["protocols"]["given_phrases"]["map50"] == 1.0
    md = out.with_suffix(".md").read_text()
    assert "| perfect | given_phrases | 100.0 |" in md and "Grounding on `val`" in md
    assert md == markdown_table(data) and "val" in capsys.readouterr().out
    with pytest.raises(SystemExit):  # a split with no images asks for the curation step
        main(
            [
                "--annotations",
                str(ann),
                "--split",
                "test",
                "--predictions",
                str(g),
                "--out",
                str(out),
            ]
        )
