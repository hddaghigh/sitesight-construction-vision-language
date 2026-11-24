"""Grounding metrics: average precision, recall@IoU, calibration and breakdowns.

Definitions (all boxes ``[x, y, w, h]``; a prediction has a class, a box and a score):

**Matching.** Within one (image, class) pair, predictions are visited from highest to lowest
score. Each takes the not-yet-matched ground-truth box of the same class with the highest IoU,
provided IoU >= the threshold; that prediction is a true positive (TP) and the ground-truth box
is consumed. Otherwise the prediction is a false positive (FP). (COCO-style.)

**Average precision (AP)** for one class and IoU threshold: sort all predictions of the class
(over all images) by score, accumulate TP and FP, giving at rank ``k``
``precision_k = TP_k / k`` and ``recall_k = TP_k / n_gt``. Replace precision by its *monotone
envelope* ``p'_k = max_{j >= k} precision_j`` (so a lucky early dip does not hide a later
rise) and integrate the precision-recall curve exactly (all-point interpolation, as in Pascal
VOC 2010+):  ``AP = sum_k (recall_k - recall_{k-1}) * p'_k``. **mAP** is the mean of AP over the
classes that have at least one ground-truth box in the evaluated set (the others are listed,
not averaged as zeros).

**Two protocols.**
* ``detection``: every prediction for every queried class in every image counts, so a class
  predicted in an image that does not contain it is a false positive. This is real open-set
  detection.
* ``given_phrases``: for each image only the classes whose ground truth is present in that
  image are scored (predictions for other classes there are discarded). This is *phrase
  grounding with the phrase given*, the setting a grounding VLM is asked in (it is told what
  to find), and it is easier because it removes false alarms on absent classes. Never compare a
  number from one protocol with a number from the other.

**Recall@IoU** is the fraction of ground-truth boxes matched at that IoU when at most ``top_k``
highest-scoring predictions per (image, class) are kept (``top_k`` stops a model that returns
hundreds of boxes from buying recall for free). Breakdowns report recall@0.5 by object size
(box area as a fraction of the image), crowding (annotated regions in the image) and region
kind (``single`` object vs ``group``/pile box).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from sitesight.eval.boxes import iou_matrix
from sitesight.eval.calibration import expected_calibration_error

# Fixed BEFORE looking at any model result (documented in PLAN.md / DATASHEET.md):
SMALL_AREA = 0.01  # area_frac below this = "small" (the dataset median is 0.0275)
LARGE_AREA = 0.05
CROWD_SPARSE_MAX = 4  # regions per image <= 4: sparse (dataset median is 5)
CROWD_CROWDED_MIN = 9

DEFAULT_IOU_THRESHOLDS = (0.5, 0.75)
DEFAULT_TOP_K = 10


@dataclass
class GT:
    file_name: str
    category: str
    bbox: list[float]
    area_frac: float = 0.0
    kind: str = "single"  # "single" | "group"
    crowd: int = 1  # number of evaluated regions in the image


@dataclass
class Det:
    file_name: str
    category: str
    bbox: list[float]
    score: float


def size_bucket(area_frac: float) -> str:
    return "small" if area_frac < SMALL_AREA else "large" if area_frac >= LARGE_AREA else "medium"


def crowd_bucket(n: int) -> str:
    return "sparse" if n <= CROWD_SPARSE_MAX else "crowded" if n >= CROWD_CROWDED_MIN else "medium"


def average_precision(tp_flags: Sequence[bool], n_gt: int) -> float:
    """AP from TP/FP flags already sorted by descending score (see module docstring)."""
    if n_gt == 0:
        raise ValueError("AP is undefined without ground truth")
    if not tp_flags:
        return 0.0
    tp = np.cumsum(np.asarray(tp_flags, dtype=np.float64))
    k = np.arange(1, len(tp_flags) + 1, dtype=np.float64)
    precision = tp / k
    recall = tp / n_gt
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev) * envelope))


def match_pair(
    dets: Sequence[Det], gts: Sequence[GT], iou_threshold: float
) -> tuple[list[bool], list[bool]]:
    """Match the predictions of ONE (image, class) pair. ``dets`` must be sorted by descending
    score. Returns (TP flag per prediction, matched flag per ground-truth box)."""
    ious = iou_matrix([d.bbox for d in dets], [g.bbox for g in gts])
    gt_used = [False] * len(gts)
    flags: list[bool] = []
    for i in range(len(dets)):
        best, best_iou = -1, iou_threshold
        for j in range(len(gts)):
            if not gt_used[j] and ious[i, j] >= best_iou and (best < 0 or ious[i, j] > best_iou):
                best, best_iou = j, float(ious[i, j])
        if best >= 0:
            gt_used[best] = True
        flags.append(best >= 0)
    return flags, gt_used


def _by_pair(items: Sequence[Any]) -> dict[tuple[str, str], list[Any]]:
    out: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for it in items:
        out[(it.file_name, it.category)].append(it)
    return out


def _sorted_desc(dets: Sequence[Det]) -> list[Det]:
    return sorted(dets, key=lambda d: -d.score)  # stable: equal scores keep input order


def evaluate_grounding(
    dets: Sequence[Det],
    gts: Sequence[GT],
    protocol: str = "given_phrases",
    iou_thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    if protocol not in ("detection", "given_phrases"):
        raise ValueError(f"unknown protocol {protocol!r}")
    gt_pairs = _by_pair(gts)
    det_pairs = _by_pair(dets)
    classes = sorted({g.category for g in gts})
    n_gt = {c: sum(1 for g in gts if g.category == c) for c in classes}
    scored_pairs = set(det_pairs) if protocol == "detection" else set(det_pairs) & set(gt_pairs)

    result: dict[str, Any] = {
        "protocol": protocol,
        "top_k": top_k,
        "n_gt": len(gts),
        "n_predictions_scored": sum(len(det_pairs[p]) for p in scored_pairs),
        "iou_thresholds": list(iou_thresholds),
    }
    per_class: dict[str, dict[str, Any]] = {c: {"n_gt": n_gt[c]} for c in classes}
    flags_at: dict[float, dict[str, list[tuple[float, bool]]]] = {}

    for thr in iou_thresholds:
        records: dict[str, list[tuple[float, bool]]] = defaultdict(list)
        for pair in scored_pairs:
            d_sorted = _sorted_desc(det_pairs[pair])
            flags, _ = match_pair(d_sorted, gt_pairs.get(pair, []), thr)
            for d, f in zip(d_sorted, flags, strict=True):
                records[pair[1]].append((d.score, f))
        flags_at[thr] = records
        for c in classes:
            recs = sorted(records.get(c, []), key=lambda r: -r[0])
            per_class[c][f"ap{round(thr * 100)}"] = average_precision([f for _, f in recs], n_gt[c])
        result[f"map{round(thr * 100)}"] = (
            float(np.mean([per_class[c][f"ap{round(thr * 100)}"] for c in classes]))
            if classes
            else None
        )

    # Recall: only (image, class) pairs that have ground truth matter; keep the top_k boxes.
    matched_at: dict[float, dict[int, bool]] = {}
    gt_index = {id(g): i for i, g in enumerate(gts)}
    for thr in iou_thresholds:
        matched: dict[int, bool] = {}
        for pair, g_list in gt_pairs.items():
            d_sorted = _sorted_desc(det_pairs.get(pair, []))[:top_k]
            _, used = match_pair(d_sorted, g_list, thr)
            for g, u in zip(g_list, used, strict=True):
                matched[gt_index[id(g)]] = u
        matched_at[thr] = matched
        for c in classes:
            idx = [i for i, g in enumerate(gts) if g.category == c]
            per_class[c][f"recall{round(thr * 100)}"] = sum(matched[i] for i in idx) / len(idx)
        result[f"recall{round(thr * 100)}"] = sum(matched.values()) / len(gts) if gts else None
    result["per_class"] = per_class
    result["classes_without_gt_in_predictions_only"] = sorted(
        {p[1] for p in det_pairs} - set(classes)
    )

    thr0 = iou_thresholds[0]
    m0 = matched_at[thr0]
    groups: dict[str, dict[str, list[int]]] = {
        "size": defaultdict(list),
        "crowding": defaultdict(list),
        "kind": defaultdict(list),
    }
    for i, g in enumerate(gts):
        groups["size"][size_bucket(g.area_frac)].append(i)
        groups["crowding"][crowd_bucket(g.crowd)].append(i)
        groups["kind"][g.kind].append(i)
    result["breakdown"] = {
        "iou": thr0,
        **{
            axis: {
                k: {"n_gt": len(v), "recall": sum(m0[i] for i in v) / len(v)}
                for k, v in sorted(d.items())
            }
            for axis, d in groups.items()
        },
    }

    # Calibration at the first IoU threshold: is a score-0.8 box right 80% of the time?
    recs0 = flags_at[thr0]
    conf = [s for rs in recs0.values() for s, _ in rs]
    ok = [f for rs in recs0.values() for _, f in rs]
    result["calibration"] = expected_calibration_error(conf, ok) if conf else {"ece": None, "n": 0}
    return result
