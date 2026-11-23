"""Image-level structured summary: *which kinds of resource are on this site image?*

For each image the ground-truth summary is the set of resource classes among its evaluated
regions, and the predicted summary is the set of classes the model returned at least one box
for with score >= t. Precision/recall/F1 are micro-averaged over (image, class) pairs, at each
threshold in ``thresholds`` (fixed in advance, so no threshold is tuned on the data being scored).

Counts are deliberately NOT scored here: a ground-truth box may stand for a whole group ("Ten
foundations are installed"), so the number of boxes is not the number of objects. Counts are
evaluated on generated captions instead (``count_acc`` in :mod:`sitesight.eval.captioning`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sitesight.eval.grounding import GT, Det

DEFAULT_THRESHOLDS = (0.1, 0.3, 0.5)


def presence_summary(
    dets: Sequence[Det],
    gts: Sequence[GT],
    images: Sequence[str],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    keep = set(images)
    truth = {(g.file_name, g.category) for g in gts if g.file_name in keep}
    out: dict[str, Any] = {"n_images": len(images), "n_true_pairs": len(truth), "thresholds": {}}
    for t in thresholds:
        pred = {(d.file_name, d.category) for d in dets if d.score >= t and d.file_name in keep}
        tp = len(pred & truth)
        p = tp / len(pred) if pred else None
        r = tp / len(truth) if truth else None
        f1 = 2 * p * r / (p + r) if p and r else (0.0 if truth else None)
        out["thresholds"][str(t)] = {
            "n_predicted_pairs": len(pred),
            "precision": p,
            "recall": r,
            "f1": f1,
        }
    return out
