"""Dedup tests. Expected values are derived by hand in each test's comment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sitesight.data.dedup import (
    ThumbnailEmbedder,
    _dct_matrix,
    connected_groups,
    cosine_matrix,
    group_stats,
    hamming_matrix,
    l2_normalise,
    near_duplicate_adjacency,
    phash,
    phash_file,
)


def _noise_image(seed: int, size: tuple[int, int] = (160, 90)) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _smooth_image(seed: int) -> Image.Image:
    """Low-frequency picture (what a hash is meant to describe): a few random gaussians."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:90, 0:160]
    canvas = np.zeros((90, 160), dtype=np.float64)
    for _ in range(6):
        cx, cy = rng.uniform(0, 160), rng.uniform(0, 90)
        canvas += rng.uniform(40, 120) * np.exp(-(((xx - cx) / 30) ** 2 + ((yy - cy) / 20) ** 2))
    arr = np.clip(canvas, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([arr] * 3, axis=-1))


def test_dct_matrix_is_orthonormal() -> None:
    c = _dct_matrix(32)
    assert np.allclose(c @ c.T, np.eye(32), atol=1e-12)  # C C^T = I defines an orthonormal DCT


def test_hamming_matrix_hand_example() -> None:
    # 0b1011 ^ 0b0110 = 0b1101 -> 3 bits differ; 0 vs 1<<63 differ in exactly the top bit.
    h = [0b1011, 0b0110, 0, 1 << 63]
    m = hamming_matrix(h)
    assert m[0, 1] == 3 and m[1, 0] == 3
    assert m[2, 3] == 1
    assert m[0, 2] == 3  # popcount(0b1011) = 3
    assert m[1, 2] == 2  # popcount(0b0110) = 2
    assert m[0, 3] == 4  # 3 bits + the top bit
    assert np.all(np.diag(m) == 0)


def test_phash_identical_and_robust_to_small_changes() -> None:
    im = _smooth_image(1)
    assert phash(im) == phash(im.copy())
    # A mild brightness change and mild noise leave the low-frequency layout almost intact.
    arr = np.asarray(im, dtype=np.int16)
    noisy = Image.fromarray(
        np.clip(arr + np.random.default_rng(0).integers(-3, 4, arr.shape) + 10, 0, 255).astype(
            np.uint8
        )
    )
    d_same = hamming_matrix([phash(im), phash(noisy)])[0, 1]
    other = hamming_matrix([phash(im), phash(_smooth_image(2))])[0, 1]
    assert d_same <= 6
    assert other > d_same


def test_phash_has_31_of_63_ac_bits_set() -> None:
    # 63 AC coefficients, bit = (coef > median): with no ties exactly 31 are strictly above
    # the median. This is why all pairwise distances are even.
    for seed in range(5):
        h = phash(_noise_image(seed))
        ac_bits = bin(h >> 1).count("1")
        assert ac_bits == 31
    d = hamming_matrix([phash(_noise_image(s)) for s in range(5)])
    assert np.all(d % 2 == 0)


def test_phash_flip_changes_hash() -> None:
    im = _smooth_image(3)
    flipped = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    assert hamming_matrix([phash(im), phash(flipped)])[0, 1] > 8


def test_phash_file_matches_phash(tmp_path: Path) -> None:
    p = tmp_path / "a.png"
    _smooth_image(4).save(p)
    assert phash_file(p) == phash(Image.open(p))


def test_connected_groups_hand_examples() -> None:
    adj = np.zeros((6, 6), dtype=bool)
    for a, b in [(0, 1), (1, 2), (3, 4)]:
        adj[a, b] = adj[b, a] = True
    assert connected_groups(adj).tolist() == [0, 0, 0, 1, 1, 2]
    adj2 = np.zeros((5, 5), dtype=bool)
    for a, b in [(4, 0), (2, 3)]:
        adj2[a, b] = adj2[b, a] = True
    # Labels are numbered by first appearance: {0,4} -> 0, {1} -> 1, {2,3} -> 2.
    assert connected_groups(adj2).tolist() == [0, 1, 2, 2, 0]
    # Chaining: 0-1, 1-2 puts 0 and 2 together even though 0-2 is not an edge.
    assert len(set(connected_groups(adj)[:3].tolist())) == 1


def test_group_stats_hand_example() -> None:
    s = group_stats(np.array([0, 0, 0, 1, 1, 2], dtype=np.int64))
    assert s["n_images"] == 6 and s["n_groups"] == 3 and s["n_singletons"] == 1
    assert s["largest_group"] == 3 and s["largest_group_fraction"] == pytest.approx(0.5)
    assert s["median_group_size"] == 2.0


def test_near_duplicate_adjacency_or_rule_and_no_self_edges() -> None:
    hd = np.array([[0, 2, 20], [2, 0, 20], [20, 20, 0]], dtype=np.int16)
    cos = np.array([[1, 0.1, 0.96], [0.1, 1, 0.1], [0.96, 0.1, 1]], dtype=np.float32)
    adj = near_duplicate_adjacency(hd, cos, max_hamming=4, min_cosine=0.95)
    expected = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool)  # 0-1 by hash, 0-2 by cosine
    assert np.array_equal(adj, expected)
    only_hash = near_duplicate_adjacency(hd, None, 4, None)
    assert only_hash.tolist() == [[False, True, False], [True, False, False], [False, False, False]]
    with pytest.raises(ValueError):
        near_duplicate_adjacency(None, None, None, None)


def test_l2_normalise_and_cosine() -> None:
    e = l2_normalise(np.array([[3.0, 4.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float32))
    assert np.allclose(e[0], [0.6, 0.8]) and np.allclose(e[1], [0.0, 1.0])
    assert np.allclose(e[2], [0.0, 0.0])  # zero vector stays zero, no NaN
    c = cosine_matrix(e[:2])
    assert c[0, 1] == pytest.approx(0.8)  # 0.6*0 + 0.8*1
    assert c[0, 0] == pytest.approx(1.0)


def test_thumbnail_embedder_on_real_files(tmp_path: Path) -> None:
    a, b, c = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"
    _smooth_image(1).save(a)
    _smooth_image(1).save(b)
    _smooth_image(7).save(c)
    emb = ThumbnailEmbedder().embed([a, b, c])
    assert emb.shape == (3, 24 * 24 * 3)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    cos = cosine_matrix(emb)
    assert cos[0, 1] == pytest.approx(1.0, abs=1e-5)  # identical pictures
    assert cos[0, 2] < 0.95  # different layout
