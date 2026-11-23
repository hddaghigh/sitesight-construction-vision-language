"""Controlled degradations used to check that the monitor reacts (and to find what it misses).

Each function maps an RGB image to a degraded copy. The settings in :data:`PERTURBATIONS` were fixed
before the demo was run; the demo reports every one of them, including the mild ones the monitor is
expected to miss.
"""

from __future__ import annotations

import io
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageFilter


def blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius))


def scale_brightness(img: Image.Image, factor: float) -> Image.Image:
    """Multiply every channel by ``factor`` (values above 255 are clipped)."""
    a = np.asarray(img.convert("RGB"), dtype=np.float64) * factor
    return Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8))


def downscale(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    return img.resize((max(1, round(w * factor)), max(1, round(h * factor))), Image.Resampling.BOX)


def downscale_then_restore(img: Image.Image, factor: float) -> Image.Image:
    """Shrink then enlarge back to the original size: same resolution, lost detail."""
    return downscale(img, factor).resize(img.size, Image.Resampling.BILINEAR)


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    with Image.open(buf) as im:
        return im.convert("RGB")


def add_noise(img: Image.Image, sigma: float, seed: int = 0) -> Image.Image:
    """Add Gaussian noise with standard deviation ``sigma`` (in 0..255 units)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    a = a + rng.normal(0.0, sigma, a.shape)
    return Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8))


PERTURBATIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "blur_mild_r0.5": lambda im: blur(im, 0.5),
    "blur_r2": lambda im: blur(im, 2.0),
    "blur_strong_r5": lambda im: blur(im, 5.0),
    "darker_x0.9": lambda im: scale_brightness(im, 0.9),
    "darker_x0.5": lambda im: scale_brightness(im, 0.5),
    "brighter_x1.4": lambda im: scale_brightness(im, 1.4),
    "rescaled_x0.5": lambda im: downscale(im, 0.5),
    "rescaled_x0.5_restored": lambda im: downscale_then_restore(im, 0.5),
    "jpeg_q10": lambda im: jpeg(im, 10),
    "noise_sigma10": lambda im: add_noise(im, 10.0),
}
