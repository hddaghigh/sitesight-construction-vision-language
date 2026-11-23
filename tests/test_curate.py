"""Curation pipeline tests: synthetic clusters with known structure, and a small real-file run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from sitesight.data.curate import compute_phashes, load_config, main, run_curation
from sitesight.data.dedup import ThumbnailEmbedder, l2_normalise
from sitesight.data.schema import AnnotationRecord, CategoryRecord, Dataset, ImageRecord

CFG: dict[str, Any] = {
    "split": {"seed": 0, "fractions": [0.7, 0.15, 0.15]},
    "dedup": {"max_hamming": 4, "min_cosine": 0.95},
    "leakage": {"naive_seeds": 5, "looser_rules": [[8, 0.9]]},
    "sensitivity": {"max_hamming": [0, 4], "min_cosine": [0.95, 0.99]},
}


def _dataset(n: int) -> Dataset:
    ds = Dataset(
        images=[
            ImageRecord(id=i + 1, file_name=f"{i + 1}.jpg", width=10, height=10) for i in range(n)
        ],
        categories=[CategoryRecord(id=1, name="worker")],
    )
    for i in range(n):
        ds.annotations.append(
            AnnotationRecord(
                id=i + 1,
                image_id=i + 1,
                bbox=[0, 0, 5, 5],
                area=25,
                caption="x",
                caption_raw="x",
                category_id=1,
                fields={"resource": "worker"},
            )
        )
    return ds


def test_run_curation_on_known_clusters() -> None:
    n_clusters, per = 20, 5
    n = n_clusters * per
    rng = np.random.default_rng(0)
    centres = l2_normalise(rng.normal(size=(n_clusters, 64)).astype(np.float32))
    cluster = np.repeat(np.arange(n_clusters), per)
    emb = l2_normalise(centres[cluster] + 0.01 * rng.normal(size=(n, 64)).astype(np.float32))
    # Cluster-specific random 64-bit hashes, each member differing from its cluster hash by 1 bit.
    base = [int(x) for x in rng.integers(0, 2**63, size=n_clusters, dtype=np.int64)]
    hashes = [base[c] ^ (1 << (i % 60)) for i, c in enumerate(cluster)]
    ds = _dataset(n)
    rep = run_curation(ds, hashes, emb, CFG)
    assert rep["groups"]["n_groups"] == n_clusters and rep["groups"]["largest_group"] == per
    assert rep["grouped_split_leakage"]["crossing_train_test"] == 0
    assert rep["naive_split_leakage"]["train_test_pairs_mean"] > 20
    assert rep["splits"]["images"] == {"train": 70, "val": 15, "test": 15}
    assert all(i.split in ("train", "val", "test") and i.group_id is not None for i in ds.images)
    assert (
        len(rep["sensitivity"]) == 4
        and rep["grouped_split_residual_under_looser_rules"][0]["max_hamming"] == 8
    )
    # Images of one cluster share a split.
    by_group: dict[int, set[str]] = {}
    for img in ds.images:
        by_group.setdefault(int(img.group_id or 0), set()).add(img.split or "")
    assert all(len(s) == 1 for s in by_group.values())


def test_run_curation_rejects_misaligned_inputs() -> None:
    ds = _dataset(4)
    with pytest.raises(ValueError, match="4 images"):
        run_curation(ds, [0, 1, 2], np.zeros((4, 3), dtype=np.float32), CFG)


def _write_images(root: Path, n_clusters: int, per: int) -> list[Path]:
    paths = []
    for c in range(n_clusters):
        rng = np.random.default_rng(c)
        yy, xx = np.mgrid[0:90, 0:160]
        canvas = np.zeros((90, 160))
        for _ in range(5):
            cx, cy = rng.uniform(0, 160), rng.uniform(0, 90)
            canvas += rng.uniform(40, 120) * np.exp(
                -(((xx - cx) / 30) ** 2 + ((yy - cy) / 20) ** 2)
            )
        for k in range(per):
            arr = np.clip(
                canvas + np.random.default_rng(100 * c + k).normal(0, 1.0, canvas.shape), 0, 255
            )
            p = root / f"{c * per + k + 1}.jpg"
            Image.fromarray(np.stack([arr.astype(np.uint8)] * 3, axis=-1)).save(p, quality=95)
            paths.append(p)
    return paths


def test_cli_end_to_end_with_real_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    images = tmp_path / "img"
    images.mkdir()
    paths = _write_images(images, n_clusters=10, per=3)
    ds = _dataset(len(paths))
    ann = tmp_path / "ann.json"
    ds.save(ann)
    emb = tmp_path / "emb.npy"
    np.save(emb, ThumbnailEmbedder().embed(paths))
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[split]\nseed = 1\nfractions = [0.7, 0.15, 0.15]\n"
        "[dedup]\nmax_hamming = 6\nmin_cosine = 0.98\n"
        "[leakage]\nnaive_seeds = 4\nlooser_rules = [[10, 0.9]]\n"
        "[sensitivity]\nmax_hamming = [4, 6]\nmin_cosine = [0.98]\n",
        encoding="utf-8",
    )
    assert load_config(cfg)["split"]["seed"] == 1
    out = tmp_path / "out"
    assert (
        main(
            [
                "--config",
                str(cfg),
                "--annotations",
                str(ann),
                "--images",
                str(images),
                "--embeddings",
                str(emb),
                "--out-dir",
                str(out),
            ]
        )
        == 0
    )
    report = json.loads((out / "curation_report.json").read_text())
    assert report["groups"]["n_groups"] == 10  # the 3 noisy copies of each picture group together
    assert report["grouped_split_leakage"]["crossing_train_test"] == 0
    curated = Dataset.load(out / "annotations_curated.json")
    assert {i.split for i in curated.images} <= {"train", "val", "test"}
    assert "groups:" in capsys.readouterr().out
    splits = json.loads((out / "splits.json").read_text())
    assert len(splits) == 30 and splits["1.jpg"] == {
        "split": curated.images[0].split,
        "group_id": curated.images[0].group_id,
    }
    # pHash cache is written and reused.
    assert (out / "phash.npy").exists()
    first = compute_phashes(curated, images, out / "phash.npy")
    assert first == compute_phashes(curated, images, out / "phash.npy")
