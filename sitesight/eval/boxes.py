"""Box utilities (all boxes are ``[x, y, w, h]`` in pixels)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def iou_matrix(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> NDArray[np.float64]:
    """Pairwise intersection-over-union, shape ``(len(a), len(b))``.

    ``IoU = |A ∩ B| / (|A| + |B| - |A ∩ B|)``; 0 when the union is empty.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    A = np.asarray(a, dtype=np.float64)
    B = np.asarray(b, dtype=np.float64)
    ax0, ay0, ax1, ay1 = A[:, 0], A[:, 1], A[:, 0] + A[:, 2], A[:, 1] + A[:, 3]
    bx0, by0, bx1, by1 = B[:, 0], B[:, 1], B[:, 0] + B[:, 2], B[:, 1] + B[:, 3]
    iw = np.clip(
        np.minimum(ax1[:, None], bx1[None, :]) - np.maximum(ax0[:, None], bx0[None, :]), 0, None
    )
    ih = np.clip(
        np.minimum(ay1[:, None], by1[None, :]) - np.maximum(ay0[:, None], by0[None, :]), 0, None
    )
    inter = iw * ih
    union = (A[:, 2] * A[:, 3])[:, None] + (B[:, 2] * B[:, 3])[None, :] - inter
    out: NDArray[np.float64] = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
    return out


def nms(
    boxes: Sequence[Sequence[float]], scores: Sequence[float], iou_threshold: float
) -> list[int]:
    """Greedy non-maximum suppression. Returns kept indices, highest score first.

    Repeatedly keep the best-scoring box and drop every remaining box whose IoU with it is
    **greater than** ``iou_threshold`` (so IoU exactly equal to the threshold survives).
    """
    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], i))  # ties: lower index first
    keep: list[int] = []
    ious = iou_matrix(boxes, boxes)
    suppressed = [False] * len(boxes)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if j != i and not suppressed[j] and ious[i, j] > iou_threshold:
                suppressed[j] = True
    return keep
