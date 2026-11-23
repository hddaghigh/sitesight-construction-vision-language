"""Grounding / calibration metric tests. Every expected value is derived by hand."""

from __future__ import annotations

import numpy as np
import pytest

from sitesight.eval.boxes import iou_matrix, nms
from sitesight.eval.calibration import expected_calibration_error
from sitesight.eval.grounding import (
    GT,
    Det,
    average_precision,
    crowd_bucket,
    evaluate_grounding,
    match_pair,
    size_bucket,
)


def test_iou_matrix() -> None:
    m = iou_matrix([[0, 0, 10, 10], [20, 20, 5, 5]], [[0, 0, 10, 10], [5, 0, 10, 10], [0, 0, 5, 5]])
    assert m.shape == (2, 3)
    assert m[0, 0] == 1.0
    assert m[0, 1] == pytest.approx(1 / 3)  # inter 50, union 150
    assert m[0, 2] == pytest.approx(0.25)  # contained: 25 / 100
    assert np.all(m[1] == 0.0)
    assert iou_matrix([], [[0, 0, 1, 1]]).shape == (0, 1)


def test_nms_hand_example() -> None:
    boxes = [[0, 0, 10, 10], [1, 0, 10, 10], [50, 50, 10, 10], [0, 0, 10, 10]]
    scores = [0.9, 0.8, 0.7, 0.95]
    # D (idx 3, 0.95) wins; A (IoU 1.0) and B (IoU 0.818) are suppressed; C is far away.
    assert nms(boxes, scores, 0.5) == [3, 2]
    # IoU exactly 0.5 is NOT suppressed (strictly greater is required).
    assert nms([[0, 0, 10, 10], [0, 0, 10, 5]], [0.9, 0.8], 0.5) == [0, 1]
    assert nms([[0, 0, 10, 10], [0, 0, 10, 5]], [0.9, 0.8], 0.4) == [0]


def test_average_precision_hand_examples() -> None:
    # n_gt=2; ranks: TP, FP, TP. precision = [1, .5, 2/3]; recall = [.5, .5, 1].
    # envelope = [1, 2/3, 2/3]; AP = .5*1 + 0*(2/3) + .5*(2/3) = 5/6.
    assert average_precision([True, False, True], 2) == pytest.approx(5 / 6)
    assert average_precision([True, True], 2) == 1.0
    assert average_precision([], 3) == 0.0
    # A wrong first guess hurts: [FP, TP], n_gt=1 -> precision [0, .5], envelope [.5, .5], AP = .5.
    assert average_precision([False, True], 1) == pytest.approx(0.5)
    # Missed ground truth lowers recall: one TP of n_gt=2 -> AP = .5 * 1.
    assert average_precision([True], 2) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        average_precision([True], 0)


def _g(c: str, box: list[float]) -> GT:
    return GT("x.jpg", c, box)


def _d(box: list[float], s: float) -> Det:
    return Det("x.jpg", "c", box, s)


def test_match_pair_consumes_each_gt_once() -> None:
    gts = [_g("c", [0, 0, 10, 10]), _g("c", [20, 0, 10, 10])]
    dets = [_d([1, 0, 10, 10], 0.9), _d([0, 0, 10, 10], 0.8), _d([20, 0, 10, 10], 0.7)]
    flags, used = match_pair(dets, gts, 0.5)
    # d1 (IoU .818) takes g1; d2 (IoU 1.0) finds g1 already used -> FP; d3 takes g2.
    assert flags == [True, False, True] and used == [True, True]
    # IoU exactly at the threshold counts (>=): inter 50 / union 100.
    assert match_pair([_d([0, 0, 10, 5], 1.0)], [_g("c", [0, 0, 10, 10])], 0.5)[0] == [True]
    assert match_pair([_d([0, 0, 10, 4], 1.0)], [_g("c", [0, 0, 10, 10])], 0.5)[0] == [False]
    # A prediction overlapping two ground truths takes the one with the higher IoU.
    two = [_g("c", [0, 0, 10, 10]), _g("c", [4, 0, 10, 10])]
    flags2, used2 = match_pair([_d([3, 0, 10, 10], 1.0)], two, 0.5)
    assert flags2 == [True] and used2 == [False, True]  # IoU with g2 is 0.82 > 0.54 with g1


def test_expected_calibration_error_hand_examples() -> None:
    # bin 9: two preds at 0.9, one correct -> |0.5 - 0.9| = .4 weight .5 -> .2
    # bin 1: two preds at 0.1, none correct -> |0 - 0.1| = .1 weight .5 -> .05 ; ECE = .25
    r = expected_calibration_error([0.9, 0.9, 0.1, 0.1], [True, False, False, False])
    assert r["ece"] == pytest.approx(0.25) and r["n"] == 4
    assert r["bins"][9]["n"] == 2 and r["bins"][9]["accuracy"] == 0.5
    assert expected_calibration_error([1.0, 1.0], [True, True])["ece"] == 0.0  # 1.0 -> last bin
    assert expected_calibration_error([0.55] * 10, [True] * 5 + [False] * 5)[
        "ece"
    ] == pytest.approx(0.05)
    assert expected_calibration_error([], [])["ece"] is None
    with pytest.raises(ValueError):
        expected_calibration_error([1.2], [True])
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [True, False])


def test_bucket_boundaries() -> None:
    assert [size_bucket(a) for a in (0.0099, 0.01, 0.0499, 0.05)] == [
        "small",
        "medium",
        "medium",
        "large",
    ]
    assert [crowd_bucket(n) for n in (1, 4, 5, 8, 9, 14)] == [
        "sparse",
        "sparse",
        "medium",
        "medium",
        "crowded",
        "crowded",
    ]


def _scenario() -> tuple[list[Det], list[GT]]:
    gts = [
        GT("A.jpg", "worker", [0, 0, 10, 10], 0.005, "single", 3),
        GT("A.jpg", "worker", [20, 0, 10, 10], 0.02, "single", 3),
        GT("A.jpg", "dump_truck", [50, 50, 20, 20], 0.06, "group", 3),
        GT("B.jpg", "worker", [0, 0, 10, 10], 0.005, "single", 1),
    ]
    dets = [
        Det("A.jpg", "worker", [0, 0, 10, 10], 0.9),  # TP (g1)
        Det("A.jpg", "worker", [60, 60, 5, 5], 0.8),  # FP
        Det("A.jpg", "worker", [20, 0, 10, 10], 0.4),  # TP (g2)
        Det("B.jpg", "worker", [0, 0, 10, 10], 0.7),  # TP
        Det("B.jpg", "dump_truck", [0, 0, 10, 10], 0.95),  # FP, and B has no dump_truck GT
        Det("A.jpg", "dump_truck", [50, 50, 20, 20], 0.6),  # TP
    ]
    return dets, gts


def test_detection_protocol_hand_example() -> None:
    dets, gts = _scenario()
    r = evaluate_grounding(dets, gts, protocol="detection")
    # worker (n_gt=3): ranks T,F,T,T -> precision [1,.5,2/3,.75], recall [1/3,1/3,2/3,1],
    # envelope [1,.75,.75,.75]; AP = 1/3 + (1/3)(.75) + (1/3)(.75) = 5/6.
    assert r["per_class"]["worker"]["ap50"] == pytest.approx(5 / 6)
    # dump_truck (n_gt=1): ranks F(0.95, image B), T(0.6) -> envelope [.5,.5]; AP = .5.
    assert r["per_class"]["dump_truck"]["ap50"] == pytest.approx(0.5)
    assert r["map50"] == pytest.approx((5 / 6 + 0.5) / 2)
    assert r["n_gt"] == 4 and r["n_predictions_scored"] == 6


def test_given_phrases_protocol_ignores_absent_classes() -> None:
    dets, gts = _scenario()
    r = evaluate_grounding(dets, gts, protocol="given_phrases")
    assert r["per_class"]["dump_truck"]["ap50"] == pytest.approx(
        1.0
    )  # the image-B false alarm is dropped
    assert r["per_class"]["worker"]["ap50"] == pytest.approx(5 / 6)
    assert r["map50"] == pytest.approx((5 / 6 + 1.0) / 2)
    assert r["n_predictions_scored"] == 5


def test_recall_top_k_and_breakdowns() -> None:
    dets, gts = _scenario()
    full = evaluate_grounding(dets, gts, top_k=10)
    assert full["recall50"] == 1.0 and full["recall75"] == 1.0
    # top_k=1: image A keeps only its 0.9 worker box -> worker 2/3 (A:g1, B:g1), dump_truck 1/1 -> 3/4.
    one = evaluate_grounding(dets, gts, top_k=1)
    assert one["recall50"] == pytest.approx(0.75)
    assert one["per_class"]["worker"]["recall50"] == pytest.approx(2 / 3)
    assert one["per_class"]["dump_truck"]["recall50"] == 1.0
    b = one["breakdown"]
    # kind: single = the 3 workers -> 2/3 ; group = the dump truck -> 1.
    assert b["kind"]["single"] == {"n_gt": 3, "recall": pytest.approx(2 / 3)}
    assert b["kind"]["group"] == {"n_gt": 1, "recall": 1.0}
    # size: area 0.005 (x2) small -> A.g1 hit, B.g1 hit = 2/2 ; 0.02 medium (A.g2 missed at k=1) = 0/1 ; 0.06 large = 1/1.
    assert b["size"]["small"] == {"n_gt": 2, "recall": 1.0}
    assert b["size"]["medium"] == {"n_gt": 1, "recall": 0.0}
    assert b["size"]["large"] == {"n_gt": 1, "recall": 1.0}
    # crowding: A's three regions (crowd=3 -> sparse) ; B (crowd=1 -> sparse): all 4 in "sparse".
    assert b["crowding"]["sparse"]["n_gt"] == 4


def test_calibration_inside_grounding() -> None:
    dets, gts = _scenario()
    cal = evaluate_grounding(dets, gts, protocol="given_phrases")["calibration"]
    # scored predictions: (.9 T) (.8 F) (.7 T) (.4 T) (.6 T), each in its own bin ->
    # |1-.9| + |0-.8| + |1-.7| + |1-.4| + |1-.6| = .1+.8+.3+.6+.4 = 2.2 ; ECE = 2.2 / 5.
    assert cal["ece"] == pytest.approx(0.44) and cal["n"] == 5


def test_unknown_protocol_and_predicted_only_classes() -> None:
    dets, gts = _scenario()
    with pytest.raises(ValueError):
        evaluate_grounding(dets, gts, protocol="nope")
    extra = [*dets, Det("A.jpg", "tarpaulin", [0, 0, 5, 5], 0.5)]
    r = evaluate_grounding(extra, gts, protocol="detection")
    assert r["classes_without_gt_in_predictions_only"] == ["tarpaulin"]
    assert "tarpaulin" not in r["per_class"] and r["map50"] == pytest.approx((5 / 6 + 0.5) / 2)
