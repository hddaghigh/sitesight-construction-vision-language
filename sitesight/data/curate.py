"""Curation pipeline: dedup -> groups -> leakage-free split -> report.

Inputs: ``annotations.json`` (with caption fields from M2), the images, and an embedding
matrix (one row per image, same order as ``images`` in the JSON; produced by
``scripts/embed_images.py``). All thresholds and seeds come from a TOML config so every
number in the report is reproducible from (config, data, code).

Outputs (in ``--out-dir``):

* ``annotations_curated.json``: the dataset with ``group_id`` and ``split`` on every image;
* ``curation_report.json``: thresholds, group statistics, split sizes, per-class counts,
  the leakage a naive random split would have had, and a threshold-sensitivity table.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sitesight.data.dedup import (
    connected_groups,
    cosine_matrix,
    group_stats,
    hamming_matrix,
    near_duplicate_adjacency,
    phash_file,
)
from sitesight.data.schema import Dataset
from sitesight.data.splits import (
    assign_splits,
    grouped_split,
    leakage,
    naive_leakage_distribution,
    summarise_splits,
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def compute_phashes(ds: Dataset, images_dir: Path, cache: Path | None) -> list[int]:
    """pHash per image (cached as uint64 .npy keyed by file-name order)."""
    if cache is not None and cache.exists():
        arr = np.load(cache)
        if len(arr) == len(ds.images):
            return [int(x) for x in arr]
    hashes = [phash_file(images_dir / i.file_name) for i in ds.images]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, np.array(hashes, dtype=np.uint64))
    return hashes


def run_curation(
    ds: Dataset,
    hashes: list[int],
    embeddings: NDArray[np.float32],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Assign groups/splits onto ``ds`` (in place) and return the report dictionary."""
    n = len(ds.images)
    if embeddings.shape[0] != n or len(hashes) != n:
        raise ValueError(
            f"{n} images but {len(hashes)} hashes and {embeddings.shape[0]} embeddings"
        )
    hd = hamming_matrix(hashes)
    cos = cosine_matrix(embeddings)

    dd, sp, lk = cfg["dedup"], cfg["split"], cfg["leakage"]
    adj = near_duplicate_adjacency(hd, cos, dd["max_hamming"], dd["min_cosine"])
    labels = connected_groups(adj)
    fractions = tuple(sp["fractions"])
    split = grouped_split(labels, fractions, sp["seed"])
    assign_splits(ds, labels, split)

    seeds = list(range(lk["naive_seeds"]))
    sensitivity = []
    for h in cfg["sensitivity"]["max_hamming"]:
        for c in cfg["sensitivity"]["min_cosine"]:
            a = near_duplicate_adjacency(hd, cos, h, c)
            lab = connected_groups(a)
            gs = group_stats(lab)
            s_g = grouped_split(lab, fractions, sp["seed"])
            sensitivity.append(
                {
                    "max_hamming": h,
                    "min_cosine": c,
                    "n_edges": int(np.triu(a, 1).sum()),
                    "n_groups": gs["n_groups"],
                    "largest_group": gs["largest_group"],
                    "naive_train_test_pairs_mean": naive_leakage_distribution(a, fractions, seeds)[
                        "train_test_pairs_mean"
                    ],
                    "grouped_train_test_pairs": leakage(s_g, a).crossing_train_test,
                }
            )

    # Residual leakage of the CHOSEN grouped split when "similar" is defined more loosely.
    residual = []
    for h, c in lk["looser_rules"]:
        a = near_duplicate_adjacency(hd, cos, h, c)
        r = leakage(split, a)
        residual.append(
            {
                "max_hamming": h,
                "min_cosine": c,
                "train_test_pairs": r.crossing_train_test,
                "test_fraction_with_train_neighbour": r.test_fraction_with_train_neighbour,
            }
        )

    summary = summarise_splits(ds)
    return {
        "config": cfg,
        "n_images": n,
        "chosen_rule": {"max_hamming": dd["max_hamming"], "min_cosine": dd["min_cosine"]},
        "groups": group_stats(labels),
        "n_near_duplicate_pairs": int(np.triu(adj, 1).sum()),
        "grouped_split_leakage": leakage(split, adj).to_dict(),
        "naive_split_leakage": naive_leakage_distribution(adj, fractions, seeds),
        "grouped_split_residual_under_looser_rules": residual,
        "sensitivity": sensitivity,
        "splits": summary.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Group near-duplicates and make a leakage-free split.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    ds = Dataset.load(args.annotations)
    hashes = compute_phashes(ds, args.images, args.out_dir / "phash.npy")
    emb = np.load(args.embeddings).astype(np.float32)
    report = run_curation(ds, hashes, emb, cfg)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ds.save(args.out_dir / "annotations_curated.json")
    # Compact, embedding-free record of the split so it can be reused without re-running dedup.
    splits = {i.file_name: {"split": i.split, "group_id": i.group_id} for i in ds.images}
    (args.out_dir / "splits.json").write_text(
        json.dumps(splits, indent=0, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "curation_report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    g, nl, gl = report["groups"], report["naive_split_leakage"], report["grouped_split_leakage"]
    print(f"groups: {g['n_groups']} (largest {g['largest_group']}, singletons {g['n_singletons']})")
    print(f"splits: {report['splits']['images']}")
    print(
        "train-test near-duplicate pairs: naive random split mean "
        f"{nl['train_test_pairs_mean']:.1f} "
        f"(min {nl['train_test_pairs_min']}, max {nl['train_test_pairs_max']}); "
        f"grouped split {gl['crossing_train_test']}"
    )
    print(
        "test images with a train near-duplicate: naive "
        f"{nl['test_fraction_with_train_neighbour_mean']:.1%} vs grouped "
        f"{gl['test_fraction_with_train_neighbour']:.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
