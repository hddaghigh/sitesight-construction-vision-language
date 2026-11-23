"""Acquisition strategies and learning-curve summaries for active learning (numpy only).

Setting. A pool of unlabeled images; each round the labeller annotates the captions of ``k`` more
images (all regions of an image together: the unit of labelling cost is the image, and the number
of regions is reported next to it). A strategy chooses which ``k`` images.

Strategies.
* ``random``: uniform without replacement (the baseline everything must beat).
* ``uncertainty``: the ``k`` images whose generated captions the current model is least sure of.
  Confidence of a caption = geometric-mean token probability ``exp(mean_t log p(y_t))`` (the same
  score as ``baselines.prompts.sequence_score``); image uncertainty = ``1 - mean`` of that over the
  image's regions.
* ``uncertainty_diversity``: take the ``m * k`` most uncertain images (``m`` = multiplier),
  then pick ``k`` of them by greedy k-center (farthest-point) selection on L2-normalised image
  embeddings, seeded by the labelled set. Greedy k-center repeatedly adds the candidate whose
  cosine distance to its nearest already-chosen image is largest, so near-duplicate frames (this
  dataset is full of them) are not all bought at once. It is a 2-approximation of the optimal
  k-center cover (Sener and Savarese, 2018).

Learning curves are summarised by the number of labelled images needed to reach a fraction of the
full-data score, found by linear interpolation between measured points (never extrapolated: ``None``
if the curve never gets there).
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import numpy as np

STRATEGIES = ("random", "uncertainty", "uncertainty_diversity")


def image_uncertainty(confidences: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """``1 - mean caption confidence`` per image (images with no regions are not scored)."""
    return {img: 1.0 - float(np.mean(c)) for img, c in confidences.items() if len(c) > 0}


def kcenter_greedy(
    embeddings: np.ndarray, candidates: Sequence[int], k: int, labelled: Sequence[int]
) -> list[int]:
    """Greedy farthest-point selection of ``k`` of ``candidates`` (row indices into ``embeddings``).

    Distance is cosine distance ``1 - u.v`` on L2-normalised rows. Each candidate starts with the
    distance to its nearest labelled row (``1.0`` if none yet); the farthest is taken,
    distances are updated with it, and the process repeats. Ties break by lowest position in
    ``candidates`` so runs are reproducible.
    """
    if k <= 0 or not candidates:
        return []
    x = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    cand = list(candidates)
    dist = np.ones(len(cand))
    if len(labelled):
        dist = 1.0 - (x[cand] @ x[list(labelled)].T).max(axis=1)
    chosen: list[int] = []
    available = np.ones(len(cand), dtype=bool)
    for _ in range(min(k, len(cand))):
        j = int(np.argmax(np.where(available, dist, -np.inf)))  # first maximum on ties
        chosen.append(cand[j])
        available[j] = False
        dist = np.minimum(dist, 1.0 - x[cand] @ x[cand[j]])
    return chosen


def select(
    strategy: str,
    pool: Sequence[str],
    labelled: Sequence[str],
    k: int,
    *,
    seed: int,
    uncertainty: Mapping[str, float] | None = None,
    embeddings: Mapping[str, np.ndarray] | None = None,
    candidate_multiplier: int = 3,
) -> list[str]:
    """Pick ``k`` pool images not in ``labelled``; deterministic given ``seed``."""
    done = set(labelled)
    unlabelled = sorted(n for n in pool if n not in done)  # sorted: must not depend on input order
    k = min(k, len(unlabelled))
    if strategy == "random":
        return random.Random(seed).sample(unlabelled, k)
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    if uncertainty is None:
        raise ValueError(f"strategy {strategy!r} needs uncertainty scores")
    # most uncertain first; the name breaks ties so equal scores are deterministic
    ranked = sorted(unlabelled, key=lambda n: (-uncertainty.get(n, -math.inf), n))
    if strategy == "uncertainty":
        return ranked[:k]
    if embeddings is None:
        raise ValueError("uncertainty_diversity needs embeddings")
    cands = ranked[: candidate_multiplier * k]
    names = sorted(set(cands) | set(labelled))
    row = {n: i for i, n in enumerate(names)}
    mat = np.stack([embeddings[n] for n in names]).astype(np.float64)
    picked = kcenter_greedy(mat, [row[n] for n in cands], k, [row[n] for n in labelled])
    return [names[i] for i in picked]


def labels_to_reach(curve: Sequence[tuple[float, float]], target: float) -> float | None:
    """Smallest x at which the (x-sorted) curve reaches ``target``, by linear interpolation."""
    pts = sorted(curve)
    if not pts:
        return None
    if pts[0][1] >= target:
        return float(pts[0][0])
    for (x0, y0), (x1, y1) in pairwise(pts):
        if y0 < target <= y1:
            return float(x0 + (x1 - x0) * (target - y0) / (y1 - y0))
    return None


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1; 0.0 for a single value)."""
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def summarise_curves(
    rows: Sequence[Mapping[str, Any]], full_scores: Mapping[int, float], fraction: float
) -> dict[str, Any]:
    """Aggregate per-run rows into per-strategy curves across seeds.

    ``rows``: dicts with ``strategy``, ``seed``, ``n_images``, ``score``. ``full_scores``: the
    full-data score per seed. For every strategy and seed the labels needed to reach ``fraction`` of
    that seed's full score are interpolated; ``None`` (never reached) is reported, not dropped.
    """
    out: dict[str, Any] = {}
    for strat in sorted({str(r["strategy"]) for r in rows}):
        by_n: dict[int, list[float]] = {}
        per_seed: dict[int, list[tuple[float, float]]] = {}
        for r in rows:
            if r["strategy"] != strat:
                continue
            by_n.setdefault(int(r["n_images"]), []).append(float(r["score"]))
            per_seed.setdefault(int(r["seed"]), []).append(
                (float(r["n_images"]), float(r["score"]))
            )
        curve = [
            {"n_images": n, "mean": mean_std(v)[0], "std": mean_std(v)[1], "seeds": len(v)}
            for n, v in sorted(by_n.items())
        ]
        reach = {
            seed: labels_to_reach(pts, fraction * full_scores[seed])
            for seed, pts in sorted(per_seed.items())
            if seed in full_scores
        }
        out[strat] = {"curve": curve, "labels_to_reach": reach}
    return out
