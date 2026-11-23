"""DINOv2 embedder test. Opt-in (needs torch + transformers + the cached model):

HF_HOME=... .venv-embed/Scripts/python -m pytest -m embed
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.embed


def _picture(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:540, 0:960]
    canvas = np.zeros((540, 960, 3))
    for _ in range(8):
        cx, cy = rng.uniform(0, 960), rng.uniform(0, 540)
        colour = rng.uniform(30, 120, size=3)
        canvas += np.exp(-(((xx - cx) / 120) ** 2 + ((yy - cy) / 80) ** 2))[..., None] * colour
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).save(path, quality=95)


def test_dinov2_embedder_properties(tmp_path: Path) -> None:
    from sitesight.data.embed import Dinov2Embedder

    a, b, c = tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "c.jpg"
    _picture(a, 1)
    _picture(b, 1)  # identical content
    _picture(c, 2)
    emb = Dinov2Embedder(batch_size=2)
    e = emb.embed([a, b, c])
    assert e.shape == (3, 384) and e.dtype == np.float32
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)
    cos = e @ e.T
    assert cos[0, 1] == pytest.approx(1.0, abs=1e-4)  # same picture -> same descriptor
    assert cos[0, 2] < cos[0, 1] - 0.01  # a different picture is measurably less similar
    # Batch composition must not change the result (batch of 1 vs batch of 2).
    assert np.allclose(Dinov2Embedder(batch_size=1).embed([c]), e[2:3], atol=1e-4)
