"""Benchmark helpers: percentile maths, agreement, deterministic sampling (hand-derived)."""

from __future__ import annotations

import pytest

from sitesight.eval.data import RegionGT
from sitesight.serve.bench import agreement, latency_summary, percentile, sample_regions


def test_percentile_interpolates_between_order_statistics() -> None:
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(xs, 50) == 3.0
    assert percentile(xs, 95) == pytest.approx(4.8)  # position 0.95 * 4 = 3.8 -> 4 + 0.8 * (5 - 4)
    assert percentile([2.0], 95) == 2.0


def test_latency_summary_fields() -> None:
    s = latency_summary([0.1, 0.2, 0.3])
    assert s["n"] == 3 and s["p50_s"] == pytest.approx(0.2) and s["mean_s"] == pytest.approx(0.2)
    assert s["p95_s"] == pytest.approx(0.29)  # position 0.95 * 2 = 1.9 -> 0.2 + 0.9 * 0.1


def test_agreement_counts_identical_captions_and_confidence_gap() -> None:
    base = [("a", 0.9), ("b", 0.5), ("c", 0.7)]
    other = [("a", 0.8), ("x", 0.5), ("c", 0.4)]
    out = agreement(base, other)
    assert out["identical_caption_rate"] == pytest.approx(2 / 3)
    assert out["mean_abs_confidence_diff"] == pytest.approx((0.1 + 0.0 + 0.3) / 3)
    with pytest.raises(ValueError, match="aligned"):
        agreement(base, other[:2])
    with pytest.raises(ValueError, match="non-empty"):
        agreement([], [])


def test_sampling_is_seeded_and_independent_of_input_order() -> None:
    regs = [RegionGT(i, f"{i}.jpg", [0, 0, 1, 1], "c", {}) for i in range(20)]
    a = sample_regions(regs, 5, seed=1)
    assert [r.ann_id for r in a] == [r.ann_id for r in sample_regions(regs[::-1], 5, seed=1)]
    assert [r.ann_id for r in a] != [r.ann_id for r in sample_regions(regs, 5, seed=2)]
    assert len(sample_regions(regs, 50, seed=0)) == 20  # asking for more returns everything
