"""Near-duplicate detection: perceptual hash + embedding similarity + grouping.

Why this matters: the 1,431 images come from drone flights over a few construction sites,
so consecutive frames are almost identical. A random image-level train/test split then lets
the model see a near-copy of every test image during training and the test score measures
memorisation, not generalisation. We (1) find near-duplicates, (2) group them, and (3) split
by group (see :mod:`sitesight.data.splits`).

Two independent similarity signals are used, because each has a blind spot:

* **pHash** (perceptual hash) catches near-identical pixels (same frame, re-encoded, tiny
  shifts) but is fooled by a few metres of camera motion.
* **Embedding cosine similarity** (a visual feature vector per image) catches "same scene
  from a slightly different position" but depends on the embedding model.

pHash maths. Resize to 32x32 grey, take the 2-D DCT-II ``D = C X C^T`` where ``C[k, n] =
sqrt(2/N) * s_k * cos(pi (2n+1) k / 2N)`` (``s_0 = 1/sqrt(2)``, else 1). The top-left 8x8
block holds the lowest spatial frequencies (overall layout, not fine detail). Each of the 64
coefficients becomes one bit: ``1`` if it is above the median of the 63 non-DC coefficients.
The Hamming distance between two hashes (number of differing bits, 0..64) is small for images
with the same coarse layout.

Grouping. Images are nodes; an edge joins two images that are near-duplicates under the
chosen rule. A *group* is a connected component, i.e. single-linkage clustering at that
threshold. Single linkage is deliberately conservative for leakage control: if A~B and B~C,
all three stay together even if A and C differ. Its known downside (chaining into one giant
group) is measured and reported, not hidden.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

HASH_SIZE = 8  # 8x8 low-frequency block -> 64 bits
_DCT_SIZE = 32
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _dct_matrix(n: int) -> NDArray[np.float64]:
    k = np.arange(n, dtype=np.float64)[:, None]
    i = np.arange(n, dtype=np.float64)[None, :]
    m: NDArray[np.float64] = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2.0)
    return m


_DCT = _dct_matrix(_DCT_SIZE)


def phash(image: Image.Image) -> int:
    """64-bit DCT perceptual hash of a PIL image (see module docstring for the maths)."""
    grey = image.convert("L").resize((_DCT_SIZE, _DCT_SIZE), Image.Resampling.LANCZOS)
    x = np.asarray(grey, dtype=np.float64)
    d = _DCT @ x @ _DCT.T
    low = d[:HASH_SIZE, :HASH_SIZE].reshape(-1)
    median = np.median(low[1:])  # exclude the DC term: it only encodes brightness
    bits = low > median
    value = 0
    for k, b in enumerate(bits):
        if b:
            value |= 1 << k
    return value


def phash_file(path: Path) -> int:
    with Image.open(path) as im:
        return phash(im)


def hamming_matrix(hashes: Sequence[int]) -> NDArray[np.int16]:
    """Pairwise Hamming distances between 64-bit hashes (N x N)."""
    h = np.array(hashes, dtype=np.uint64)
    x = np.bitwise_xor(h[:, None], h[None, :])
    as_bytes = x.view(np.uint8).reshape(len(h), len(h), 8)
    dist: NDArray[np.int16] = _POPCOUNT[as_bytes].sum(axis=-1).astype(np.int16)
    return dist


class Embedder(Protocol):
    """Anything that maps images to L2-normalised feature vectors."""

    name: str

    def embed(self, paths: Sequence[Path]) -> NDArray[np.float32]: ...


class ThumbnailEmbedder:
    """Classical embedding: 24x24 colour thumbnail, mean-centred, L2-normalised.

    Cheap, deterministic, no model download. Cosine similarity of these vectors is the
    correlation of the two thumbnails, so it measures coarse colour-layout similarity. It is
    the laptop-side baseline; the DINOv2 embedder (``sitesight.data.embed``) is used in Colab.
    """

    name = "thumbnail24"

    def embed(self, paths: Sequence[Path]) -> NDArray[np.float32]:
        rows = []
        for p in paths:
            with Image.open(p) as im:
                t = np.asarray(
                    im.convert("RGB").resize((24, 24), Image.Resampling.BOX), dtype=np.float32
                )
            v = t.reshape(-1)
            rows.append(v - v.mean())
        e = np.stack(rows)
        return l2_normalise(e)


def l2_normalise(e: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    out: NDArray[np.float32] = (e / np.maximum(norms, 1e-12)).astype(np.float32)
    return out


def cosine_matrix(e: NDArray[np.float32]) -> NDArray[np.float32]:
    """Pairwise cosine similarity of L2-normalised rows (N x N), clipped to [-1, 1]."""
    sim: NDArray[np.float32] = np.clip(e @ e.T, -1.0, 1.0)
    return sim


def connected_groups(adjacent: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Label connected components of a symmetric boolean adjacency matrix (union-find).

    Labels are 0..G-1 numbered by the smallest member index, so the output is deterministic.
    """
    n = adjacent.shape[0]
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    rows, cols = np.nonzero(np.triu(adjacent, k=1))
    for a, b in zip(rows.tolist(), cols.tolist(), strict=True):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = [find(i) for i in range(n)]
    relabel: dict[int, int] = {}
    labels = np.empty(n, dtype=np.int64)
    for i, r in enumerate(roots):
        labels[i] = relabel.setdefault(r, len(relabel))
    return labels


def group_stats(labels: NDArray[np.int64]) -> dict[str, float | int]:
    """Summary of a grouping: how many groups, how large, how chained."""
    sizes = np.bincount(labels)
    return {
        "n_images": len(labels),
        "n_groups": len(sizes),
        "n_singletons": int((sizes == 1).sum()),
        "largest_group": int(sizes.max()),
        "largest_group_fraction": float(sizes.max() / len(labels)),
        "median_group_size": float(np.median(sizes)),
    }


def near_duplicate_adjacency(
    hash_dist: NDArray[np.int16] | None,
    cosine: NDArray[np.float32] | None,
    max_hamming: int | None,
    min_cosine: float | None,
) -> NDArray[np.bool_]:
    """Edge rule: pHash distance <= ``max_hamming`` OR cosine >= ``min_cosine`` (self-edges off)."""
    if hash_dist is None and cosine is None:
        raise ValueError("need at least one similarity matrix")
    n = (hash_dist if hash_dist is not None else cosine).shape[0]  # type: ignore[union-attr]
    adj = np.zeros((n, n), dtype=np.bool_)
    if hash_dist is not None and max_hamming is not None:
        adj |= hash_dist <= max_hamming
    if cosine is not None and min_cosine is not None:
        adj |= cosine >= min_cosine
    np.fill_diagonal(adj, False)
    return adj
