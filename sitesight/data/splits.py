"""Leakage-free splits: split by *group*, and measure what a naive split would have leaked.

The problem in one sentence: frames from the same site look alike, so if some of them are in
train and some in test, the test score partly measures recall of training pictures. Splitting
by connected near-duplicate group (see :mod:`sitesight.data.dedup`) removes the leak by
construction at the chosen similarity rule; this module also *quantifies* the leak a naive
random split would have had, on the same data, with the same rule.

Group assignment (deterministic for a seed): groups are sorted by size, largest first (ties
broken by a seeded random key), and each group goes to the split with the largest remaining
deficit ``target_images - assigned_images``. Placing large groups first while every deficit is
big is what keeps the realised proportions close to the targets even when a group has 40
images.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sitesight.data.schema import Dataset

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_FRACTIONS = (0.70, 0.15, 0.15)


def grouped_split(
    labels: NDArray[np.int64],
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    seed: int = 0,
) -> list[str]:
    """Assign every image to train/val/test without splitting any group (one name per image)."""
    if len(fractions) != len(SPLIT_NAMES) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must be 3 numbers summing to 1")
    n = len(labels)
    targets = [f * n for f in fractions]
    sizes = Counter(labels.tolist())
    rng = random.Random(seed)
    tiebreak = {g: rng.random() for g in sizes}
    order = sorted(sizes, key=lambda g: (-sizes[g], tiebreak[g]))
    assigned = [0.0] * len(SPLIT_NAMES)
    group_split: dict[int, int] = {}
    for g in order:
        deficits = [t - a for t, a in zip(targets, assigned, strict=True)]
        k = max(range(len(deficits)), key=lambda i: (deficits[i], -i))
        group_split[g] = k
        assigned[k] += sizes[g]
    return [SPLIT_NAMES[group_split[int(g)]] for g in labels]


def random_split(
    n: int, fractions: Sequence[float] = DEFAULT_FRACTIONS, seed: int = 0
) -> list[str]:
    """The naive image-level random split (the thing we are comparing against)."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = round(fractions[0] * n)
    n_val = round(fractions[1] * n)
    out = [""] * n
    for rank, i in enumerate(idx):
        out[i] = SPLIT_NAMES[0 if rank < n_train else 1 if rank < n_train + n_val else 2]
    return out


@dataclass
class LeakageReport:
    """Near-duplicate pairs that straddle split boundaries under a given split."""

    n_pairs_total: int
    crossing_train_test: int
    crossing_train_val: int
    crossing_val_test: int
    test_images_with_train_neighbour: int
    n_test: int

    @property
    def test_fraction_with_train_neighbour(self) -> float:
        return self.test_images_with_train_neighbour / self.n_test if self.n_test else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["test_fraction_with_train_neighbour"] = self.test_fraction_with_train_neighbour
        return d


def leakage(split: Sequence[str], adjacent: NDArray[np.bool_]) -> LeakageReport:
    """Count adjacent (near-duplicate) pairs whose two images are in different splits."""
    s = np.array(split)
    upper = np.triu(adjacent, k=1)
    rows, cols = np.nonzero(upper)
    a, b = s[rows], s[cols]

    def cross(x: str, y: str) -> int:
        return int((((a == x) & (b == y)) | ((a == y) & (b == x))).sum())

    is_test = s == "test"
    is_train = s == "train"
    has_train_neighbour = (adjacent[:, is_train].any(axis=1)) & is_test
    return LeakageReport(
        n_pairs_total=len(rows),
        crossing_train_test=cross("train", "test"),
        crossing_train_val=cross("train", "val"),
        crossing_val_test=cross("val", "test"),
        test_images_with_train_neighbour=int(has_train_neighbour.sum()),
        n_test=int(is_test.sum()),
    )


def naive_leakage_distribution(
    adjacent: NDArray[np.bool_],
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    seeds: Sequence[int] = tuple(range(20)),
) -> dict[str, Any]:
    """Leakage of a naive random split over many seeds (mean, std, min, max, per-seed values)."""
    reports = [leakage(random_split(adjacent.shape[0], fractions, s), adjacent) for s in seeds]
    ct = np.array([r.crossing_train_test for r in reports], dtype=float)
    fr = np.array([r.test_fraction_with_train_neighbour for r in reports], dtype=float)
    return {
        "seeds": list(seeds),
        "train_test_pairs_mean": float(ct.mean()),
        "train_test_pairs_std": float(ct.std(ddof=1)) if len(ct) > 1 else 0.0,
        "train_test_pairs_min": int(ct.min()),
        "train_test_pairs_max": int(ct.max()),
        "test_fraction_with_train_neighbour_mean": float(fr.mean()),
        "test_fraction_with_train_neighbour_min": float(fr.min()),
        "test_fraction_with_train_neighbour_max": float(fr.max()),
    }


@dataclass
class SplitSummary:
    images: dict[str, int] = field(default_factory=dict)
    annotations: dict[str, int] = field(default_factory=dict)
    class_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    classes_missing_in_test: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assign_splits(ds: Dataset, labels: NDArray[np.int64], split: Sequence[str]) -> None:
    """Write ``group_id`` and ``split`` onto the images (index-aligned with ``ds.images``)."""
    if not (len(ds.images) == len(labels) == len(split)):
        raise ValueError("labels/split must align with ds.images")
    for img, g, s in zip(ds.images, labels.tolist(), split, strict=True):
        img.group_id = int(g)
        img.split = s


def apply_saved_splits(ds: Dataset, saved: dict[str, dict[str, Any]]) -> None:
    """Copy ``group_id`` / ``split`` from a saved ``splits.json`` (file name -> {split, group_id}).

    Used to reproduce the exact split elsewhere (e.g. Colab) without recomputing embeddings,
    whose last digits can differ between CPU and GPU. Raises if the file names do not match the
    dataset exactly, so a stale or foreign split file can never be applied silently.
    """
    names = {i.file_name for i in ds.images}
    if names != set(saved):
        missing, extra = sorted(names - set(saved))[:3], sorted(set(saved) - names)[:3]
        raise ValueError(
            f"split file does not match the dataset (missing {missing}, extra {extra})"
        )
    for img in ds.images:
        entry = saved[img.file_name]
        img.split, img.group_id = entry["split"], entry["group_id"]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    p = argparse.ArgumentParser(description="Apply a saved splits.json to an annotations file.")
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    ds = Dataset.load(args.annotations)
    apply_saved_splits(ds, json.loads(args.splits.read_text(encoding="utf-8")))
    ds.save(args.out)
    print(f"applied splits to {len(ds.images)} images: {summarise_splits(ds).images}")
    return 0


def summarise_splits(ds: Dataset) -> SplitSummary:
    """Images, annotations and per-class annotation counts in each split."""
    split_of = {i.id: i.split for i in ds.images}
    names = {c.id: c.name for c in ds.categories}
    out = SplitSummary()
    img_count: Counter[str] = Counter(i.split or "none" for i in ds.images)
    ann_count: Counter[str] = Counter()
    per_class: dict[str, Counter[str]] = defaultdict(Counter)
    for a in ds.annotations:
        s = split_of[a.image_id] or "none"
        ann_count[s] += 1
        if a.category_id is not None:
            per_class[s][names[a.category_id]] += 1
    out.images = dict(img_count)
    out.annotations = dict(ann_count)
    out.class_counts = {s: dict(sorted(c.items())) for s, c in sorted(per_class.items())}
    all_classes = {n for c in per_class.values() for n in c}
    out.classes_missing_in_test = sorted(all_classes - set(per_class.get("test", {})))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
