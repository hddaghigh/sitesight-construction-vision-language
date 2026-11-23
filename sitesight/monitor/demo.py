"""Does the monitor alert on degraded batches and stay quiet on normal ones? (real images)

Protocol (fixed before the run; alpha = 0.01 is not tuned):

1. **Same-site control.** The training images are split at random into two halves, reference ``A``
   and clean pool ``B``. Batches of ``n`` images are drawn from ``B`` and compared with ``A``;
   the share of batches that alert is the empirical false-alarm rate (nominal budget: alpha).
2. **Degraded batches.** Every setting in :data:`perturb.PERTURBATIONS` is applied to all of ``B``;
   batches are drawn from the degraded copy and compared with ``A``. The share that alert is the
   detection rate. A second variant degrades only 25% of each batch (a realistic partial fault).
3. **New sites.** The reference is all training images; batches are drawn from the validation
   split (different sites, by construction of the grouped split). Alerts here are real
   differences between sites, not false alarms: reported separately and never tuned away.

``A`` and ``B`` are random halves of the training split, so they share sites: protocol 1 measures
the false-alarm rate for traffic from already-seen sites only. The test split is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sitesight.monitor.cli import list_images, stats_for
from sitesight.monitor.drift import detect_drift
from sitesight.monitor.perturb import PERTURBATIONS
from sitesight.monitor.quality import FEATURES, QualityReference, image_stats, stack

BATCH_SIZES = (20, 40, 80)


def draw(
    cols: dict[str, np.ndarray], n: int, rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    idx = rng.choice(len(cols["width"]), n, replace=False)
    return {k: v[idx] for k, v in cols.items()}, idx


def alert_rate(
    ref: dict[str, np.ndarray],
    pool: dict[str, np.ndarray],
    n: int,
    draws: int,
    seed: int,
    mixed_with: dict[str, np.ndarray] | None = None,
    bad_fraction: float = 0.25,
    n_perm: int = 300,
) -> float:
    """Share of ``draws`` random batches that raise an alert.

    With ``mixed_with`` each batch takes ``bad_fraction`` of its images from that degraded copy
    (same image indices as ``pool``) and the rest from ``pool``.
    """
    rng = np.random.default_rng(seed)
    hits = 0
    for d in range(draws):
        batch, idx = draw(pool, n, rng)
        if mixed_with is not None:
            k = round(bad_fraction * n)
            batch = {f: np.concatenate([batch[f][k:], mixed_with[f][idx[:k]]]) for f in batch}
        hits += detect_drift(ref, batch, n_perm=n_perm, seed=seed * 1000 + d).status == "alert"
    return hits / draws


def image_flag_rate(qref: QualityReference, cols: dict[str, np.ndarray]) -> float:
    n = len(cols["width"])
    return sum(bool(qref.flag({f: float(cols[f][i]) for f in FEATURES})) for i in range(n)) / n


def degraded_stats(paths: list[Path], fn: Any) -> dict[str, np.ndarray]:
    rows = []
    for p in paths:
        with Image.open(p) as im:
            rows.append(image_stats(fn(im.convert("RGB"))))
    return stack(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    train_paths = list_images(args.images, args.annotations, "train")
    perm = np.random.default_rng(args.seed).permutation(len(train_paths))
    half = len(train_paths) // 2
    a_paths = [train_paths[i] for i in perm[:half]]
    b_paths = [train_paths[i] for i in perm[half:]]
    train = stack(stats_for(train_paths))
    val = stack(stats_for(list_images(args.images, args.annotations, "val")))
    pos = {p: i for i, p in enumerate(train_paths)}
    ref_a = {k: v[[pos[p] for p in a_paths]] for k, v in train.items()}
    pool_b = {k: v[[pos[p] for p in b_paths]] for k, v in train.items()}
    qref = QualityReference.fit([{k: float(ref_a[k][i]) for k in FEATURES} for i in range(half)])

    out: dict[str, Any] = {
        "alpha": 0.01,
        "n_reference_A": half,
        "n_pool_B": len(b_paths),
        "draws_per_cell": args.draws,
        "batch_sizes": list(BATCH_SIZES),
        "image_flag_rate_clean_B": image_flag_rate(qref, pool_b),
        "same_site_false_alarm_rate": {
            str(n): alert_rate(ref_a, pool_b, n, args.draws, args.seed) for n in BATCH_SIZES
        },
        "new_sites_alert_rate_val_vs_train": {
            str(n): alert_rate(train, val, n, args.draws, args.seed) for n in BATCH_SIZES
        },
        "perturbations": {},
    }
    for name, fn in PERTURBATIONS.items():
        bad = degraded_stats(b_paths, fn)
        out["perturbations"][name] = {
            "image_flag_rate": image_flag_rate(qref, bad),
            "detection_rate_full_batch": {
                str(n): alert_rate(ref_a, bad, n, args.draws, args.seed) for n in BATCH_SIZES
            },
            "detection_rate_25pct_degraded": {
                str(n): alert_rate(ref_a, pool_b, n, args.draws, args.seed, mixed_with=bad)
                for n in BATCH_SIZES
            },
        }
        print(name, json.dumps(out["perturbations"][name]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
