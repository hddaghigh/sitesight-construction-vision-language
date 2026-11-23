"""Split / leakage tests with hand-computed expectations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sitesight.data.dedup import connected_groups
from sitesight.data.schema import AnnotationRecord, CategoryRecord, Dataset, ImageRecord
from sitesight.data.splits import (
    apply_saved_splits,
    assign_splits,
    grouped_split,
    leakage,
    naive_leakage_distribution,
    random_split,
    summarise_splits,
)
from sitesight.data.splits import main as split_main


def _counts(split: list[str]) -> tuple[int, int, int]:
    return split.count("train"), split.count("val"), split.count("test")


def test_grouped_split_hits_exact_targets_on_equal_groups() -> None:
    # 20 groups of 5 images = 100 images; targets 70/15/15 are exact multiples of 5.
    # Greedy largest-deficit assignment (hand-simulated) ends at exactly 14/3/3 groups.
    labels = np.repeat(np.arange(20), 5).astype(np.int64)
    assert _counts(grouped_split(labels, seed=0)) == (70, 15, 15)


def test_giant_group_goes_to_train_and_rest_fills_exactly() -> None:
    # One 40-image group + 60 singletons: the giant group takes the biggest deficit (train: 70),
    # then train needs 30 more, val 15, test 15 -> exactly 70/15/15.
    labels = np.concatenate([np.zeros(40, dtype=np.int64), np.arange(1, 61, dtype=np.int64)])
    split = grouped_split(labels, seed=3)
    assert _counts(split) == (70, 15, 15)
    assert set(split[:40]) == {"train"}


def test_no_group_is_ever_split_and_seed_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 40, size=300).astype(np.int64)
    s1 = grouped_split(labels, seed=5)
    assert s1 == grouped_split(labels, seed=5)
    assert s1 != grouped_split(labels, seed=6)
    for g in np.unique(labels):
        assert len(set(np.array(s1)[labels == g])) == 1


def test_grouped_split_validates_fractions() -> None:
    with pytest.raises(ValueError):
        grouped_split(np.arange(10, dtype=np.int64), fractions=(0.5, 0.5, 0.5))


def test_random_split_sizes_and_seed() -> None:
    s = random_split(100, seed=1)
    assert _counts(s) == (70, 15, 15)
    assert s != random_split(100, seed=2)


def test_leakage_hand_example() -> None:
    split = ["train", "train", "val", "test", "test", "train"]
    adj = np.zeros((6, 6), dtype=bool)
    for a, b in [(0, 3), (1, 2), (3, 4), (2, 3), (4, 5)]:
        adj[a, b] = adj[b, a] = True
    r = leakage(split, adj)
    # train-test: (0,3) and (4,5) -> 2; train-val: (1,2) -> 1; val-test: (2,3) -> 1;
    # (3,4) is test-test and does not count.
    assert r.n_pairs_total == 5
    assert (r.crossing_train_test, r.crossing_train_val, r.crossing_val_test) == (2, 1, 1)
    # Test images 3 (neighbour 0, train) and 4 (neighbour 5, train) both have a train neighbour.
    assert r.test_images_with_train_neighbour == 2 and r.n_test == 2
    assert r.test_fraction_with_train_neighbour == 1.0


def test_grouped_split_has_zero_leakage_but_random_split_leaks() -> None:
    rng = np.random.default_rng(1)
    n = 200
    # 40 tight clusters of 5 near-identical images.
    cluster = np.repeat(np.arange(40), 5)
    adj = cluster[:, None] == cluster[None, :]
    np.fill_diagonal(adj, False)
    labels = connected_groups(adj)
    grouped = leakage(grouped_split(labels, seed=0), adj)
    assert (
        grouped.crossing_train_test == grouped.crossing_train_val == grouped.crossing_val_test == 0
    )
    assert grouped.test_images_with_train_neighbour == 0
    naive = leakage(random_split(n, seed=int(rng.integers(0, 100))), adj)
    assert (
        naive.crossing_train_test > 50
    )  # 5-image clusters scattered over 70/15/15 almost always leak
    dist = naive_leakage_distribution(adj, seeds=range(5))
    assert dist["train_test_pairs_min"] > 0 and dist["test_fraction_with_train_neighbour_min"] > 0.5


def _tiny_dataset() -> Dataset:
    ds = Dataset(
        images=[ImageRecord(id=i, file_name=f"{i}.jpg", width=10, height=10) for i in (1, 2, 3, 4)],
        categories=[CategoryRecord(id=1, name="worker"), CategoryRecord(id=2, name="fence")],
    )
    cats = {1: 1, 2: 1, 3: 2, 4: 1}  # annotation i -> category
    for k, (img, cat) in enumerate([(1, 1), (2, 1), (3, 2), (4, 1), (4, 1)], start=1):
        ds.annotations.append(
            AnnotationRecord(
                id=k,
                image_id=img,
                bbox=[0, 0, 5, 5],
                area=25,
                caption="x",
                caption_raw="x",
                category_id=cat,
                fields={"resource": ds.categories[cat - 1].name},
            )
        )
    del cats
    return ds


def test_apply_saved_splits_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ds = _tiny_dataset()
    saved = {
        "1.jpg": {"split": "train", "group_id": 0},
        "2.jpg": {"split": "train", "group_id": 0},
        "3.jpg": {"split": "val", "group_id": 1},
        "4.jpg": {"split": "test", "group_id": 2},
    }
    apply_saved_splits(ds, saved)
    assert [i.split for i in ds.images] == ["train", "train", "val", "test"]
    assert [i.group_id for i in ds.images] == [0, 0, 1, 2]
    with pytest.raises(ValueError, match="does not match"):
        apply_saved_splits(_tiny_dataset(), {k: v for k, v in saved.items() if k != "4.jpg"})
    with pytest.raises(ValueError, match="does not match"):
        apply_saved_splits(_tiny_dataset(), {**saved, "5.jpg": saved["1.jpg"]})
    ann, spl, out = tmp_path / "a.json", tmp_path / "s.json", tmp_path / "o.json"
    _tiny_dataset().save(ann)
    spl.write_text(json.dumps(saved))
    assert split_main(["--annotations", str(ann), "--splits", str(spl), "--out", str(out)]) == 0
    assert (
        Dataset.load(out).images[3].split == "test" and "applied splits" in capsys.readouterr().out
    )


def test_assign_and_summarise_splits() -> None:
    ds = _tiny_dataset()
    labels = np.array([0, 0, 1, 2], dtype=np.int64)
    assign_splits(ds, labels, ["train", "train", "val", "test"])
    assert [i.group_id for i in ds.images] == [0, 0, 1, 2]
    s = summarise_splits(ds)
    assert s.images == {"train": 2, "val": 1, "test": 1}
    assert s.annotations == {"train": 2, "val": 1, "test": 2}
    assert s.class_counts["train"] == {"worker": 2}
    assert s.class_counts["val"] == {"fence": 1}
    assert s.classes_missing_in_test == ["fence"]  # fence appears only in val
    with pytest.raises(ValueError):
        assign_splits(ds, labels[:3], ["train", "val", "test"])
