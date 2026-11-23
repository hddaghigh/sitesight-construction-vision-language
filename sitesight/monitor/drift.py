"""Two-sample drift tests between a reference sample and a new batch (numpy only).

Two kinds of test, combined so the overall false-alarm budget is ``alpha``:

1. **Per-feature Kolmogorov-Smirnov.** ``D = max_x |F_ref(x) - F_batch(x)|``, the largest gap
   between the two empirical CDFs. With ``n`` reference and ``m`` batch values the effective size is
   ``n_e = n m / (n + m)`` and the asymptotic p-value is the Kolmogorov survival function
   ``Q(lambda) = 2 sum_{k>=1} (-1)^(k-1) exp(-2 k^2 lambda^2)`` at ``lambda = (sqrt(n_e) + 0.12 +
   0.11 / sqrt(n_e)) D`` (Stephens' small-sample correction). Sees one feature at a time and is
   rank-based, so it does not care about units. Bonferroni: with ``k`` features each is tested at
   ``(alpha / 2) / k``.
2. **Multivariate MMD.** ``MMD^2 = E k(x, x') + E k(y, y') - 2 E k(x, y)`` with the Gaussian kernel
   ``k(a, b) = exp(-|a - b|^2 / (2 sigma^2))`` on features standardised by the reference mean and
   standard deviation; ``sigma`` is the median pairwise distance of the reference (median
   heuristic). The unbiased estimator drops the diagonal terms. Its null distribution is obtained
   by a permutation test (shuffle which points are "batch"), ``p = (1 + #{perm >= observed}) /
   (1 + B)``; it catches a joint shift that no single feature shows. Tested at ``alpha / 2``.

The overall chance that a batch drawn from the reference distribution raises an alert is therefore
at most about ``alpha`` (union bound); the demo measures it on real held-out batches instead of
trusting that bound. A batch smaller than ``min_batch`` is not tested: the result says
``too_small`` rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def kolmogorov_sf(lam: float) -> float:
    """``Q(lambda)`` above; 1 for lambda < 0.2 where the series needs no terms."""
    if lam < 0.2:
        return 1.0
    k = np.arange(1, 101, dtype=np.float64)
    q = 2.0 * float(np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * k * k * lam * lam)))
    return min(1.0, max(0.0, q))


def ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample KS statistic and asymptotic p-value."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        raise ValueError("both samples must be non-empty")
    pooled = np.concatenate([a, b])
    d = float(
        np.max(
            np.abs(
                np.searchsorted(a, pooled, side="right") / n
                - np.searchsorted(b, pooled, side="right") / m
            )
        )
    )
    ne = n * m / (n + m)
    return d, kolmogorov_sf((np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * d)


def _sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    d = (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2.0 * x @ y.T
    return np.asarray(np.maximum(d, 0.0), dtype=np.float64)


def mmd2_unbiased(kxx: np.ndarray, kyy: np.ndarray, kxy: np.ndarray) -> float:
    """Unbiased MMD^2 from the three kernel blocks (diagonals of kxx, kyy excluded)."""
    n, m = len(kxx), len(kyy)
    return float(
        (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
        + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
        - 2.0 * kxy.mean()
    )


def mmd_permutation_test(
    x: np.ndarray, y: np.ndarray, n_perm: int, seed: int, max_ref: int = 300
) -> tuple[float, float]:
    """(MMD^2, permutation p-value); ``x`` is the (sub-sampled) reference, ``y`` the batch."""
    rng = np.random.default_rng(seed)
    if len(x) > max_ref:
        x = x[rng.choice(len(x), max_ref, replace=False)]
    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    x, y = (x - mu) / sd, (y - mu) / sd
    dxx = _sq_dists(x, x)
    iu = np.triu_indices(len(x), 1)
    sigma2 = float(np.median(dxx[iu])) or 1.0  # median of squared distances = (median dist)^2
    z = np.concatenate([x, y])
    k = np.exp(-_sq_dists(z, z) / (2.0 * sigma2))
    n, m = len(x), len(y)

    def stat(idx_x: np.ndarray, idx_y: np.ndarray) -> float:
        return mmd2_unbiased(
            k[np.ix_(idx_x, idx_x)], k[np.ix_(idx_y, idx_y)], k[np.ix_(idx_x, idx_y)]
        )

    observed = stat(np.arange(n), np.arange(n, n + m))
    ge = 0
    for _ in range(n_perm):
        p = rng.permutation(n + m)
        ge += stat(p[:n], p[n:]) >= observed
    return observed, (1 + ge) / (1 + n_perm)


@dataclass
class TestResult:
    __test__ = False  # not a pytest class
    name: str
    feature: str | None
    statistic: float
    p_value: float
    threshold: float
    alerted: bool


@dataclass
class DriftResult:
    status: str  # "ok" | "alert" | "too_small"
    n_reference: int
    n_batch: int
    alpha: float
    tests: list[TestResult] = field(default_factory=list)

    @property
    def alerts(self) -> list[TestResult]:
        return [t for t in self.tests if t.alerted]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "n_reference": self.n_reference,
            "n_batch": self.n_batch,
            "alpha": self.alpha,
            "tests": [t.__dict__ for t in self.tests],
        }


def detect_drift(
    reference: dict[str, np.ndarray],
    batch: dict[str, np.ndarray],
    alpha: float = 0.01,
    n_perm: int = 300,
    seed: int = 0,
    min_batch: int = 20,
) -> DriftResult:
    """KS per feature (Bonferroni) plus multivariate MMD; see the module docstring."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if 1 / (1 + n_perm) > alpha / 2:
        raise ValueError("n_perm is too small to reach the MMD threshold alpha / 2")
    feats = [f for f in reference if f in batch]
    if not feats:
        raise ValueError("reference and batch share no features")
    n_ref, n_batch = len(reference[feats[0]]), len(batch[feats[0]])
    if n_batch < min_batch:
        return DriftResult("too_small", n_ref, n_batch, alpha)
    tests: list[TestResult] = []
    thr_ks = (alpha / 2) / len(feats)
    for f in feats:
        d, p = ks_2samp(reference[f], batch[f])
        tests.append(TestResult("ks", f, d, p, thr_ks, p < thr_ks))
    xr = np.column_stack([np.asarray(reference[f], dtype=np.float64) for f in feats])
    yb = np.column_stack([np.asarray(batch[f], dtype=np.float64) for f in feats])
    stat, p = mmd_permutation_test(xr, yb, n_perm, seed)
    tests.append(TestResult("mmd", None, stat, p, alpha / 2, p < alpha / 2))
    status = "alert" if any(t.alerted for t in tests) else "ok"
    return DriftResult(status, n_ref, n_batch, alpha, tests)
