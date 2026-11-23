"""Calibration: do confidence scores mean what they say?

A model is *calibrated* if, among predictions given confidence ``c``, a fraction ``c`` are
correct. Expected Calibration Error (ECE) measures the gap:

    ECE = sum_b (n_b / N) * | acc_b - conf_b |

where predictions are split into ``B`` equal-width confidence bins, ``n_b`` is the number in
bin ``b``, ``acc_b`` the fraction correct there and ``conf_b`` the mean confidence there.
A confidence of exactly 1.0 goes in the last bin. ECE is 0 for a perfectly calibrated model;
it says nothing about ranking quality (AP does), and it depends on the number of bins.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool | int], n_bins: int = 10
) -> dict[str, Any]:
    """ECE plus the per-bin reliability table."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    n = len(confidences)
    bins: list[dict[str, Any]] = []
    totals = [[0, 0.0, 0.0] for _ in range(n_bins)]  # count, sum conf, sum correct
    for c, ok in zip(confidences, correct, strict=True):
        if not 0.0 <= c <= 1.0:
            raise ValueError(f"confidence {c} outside [0, 1]")
        b = min(int(c * n_bins), n_bins - 1)
        totals[b][0] += 1
        totals[b][1] += c
        totals[b][2] += float(ok)
    ece = 0.0
    for b, (cnt, sc, sk) in enumerate(totals):
        entry: dict[str, Any] = {"bin": b, "lo": b / n_bins, "hi": (b + 1) / n_bins, "n": cnt}
        if cnt:
            entry["mean_confidence"] = sc / cnt
            entry["accuracy"] = sk / cnt
            ece += (cnt / n) * abs(sk / cnt - sc / cnt)
        bins.append(entry)
    return {"ece": ece if n else None, "n": n, "n_bins": n_bins, "bins": bins}
