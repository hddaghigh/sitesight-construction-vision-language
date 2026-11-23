"""Cheap per-image quality statistics and a reference that says what "normal" looks like.

Every statistic is computed on the image reduced to at most ``ANALYSIS_SIDE`` pixels on its long
side (area average), so a 4000-pixel and a 960-pixel photo of the same scene give similar numbers
and the cost per image is small.

Statistics (all from the luma channel ``Y = 0.299 R + 0.587 G + 0.114 B``, scaled to 0..1):

* ``brightness``: mean of ``Y``.
* ``contrast``: standard deviation of ``Y``.
* ``dark_frac`` / ``bright_frac``: share of pixels with ``Y < 16/255`` / ``Y > 240/255`` (clipped
  shadows and highlights).
* ``sharpness``: variance of the 4-neighbour discrete Laplacian
  ``L(y, x) = 4 Y(y, x) - Y(y-1, x) - Y(y+1, x) - Y(y, x-1) - Y(y, x+1)`` over the interior.
  Blur removes high frequencies, so the variance of the second derivative falls; a flat image gives
  exactly 0. The value depends on scene texture, which is why it is judged against a reference
  distribution and never against a fixed number.
* ``width`` / ``height``: of the original upload (a rescaled batch shifts these).

``QualityReference`` stores the reference feature columns and flags an image when a feature lies
outside the reference's central band. A 1st-99th percentile band means about 1% of normal images
are flagged on each feature by construction; that is the price of the rule and is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

ANALYSIS_SIDE = 512
DARK_LEVEL = 16 / 255
BRIGHT_LEVEL = 240 / 255
FEATURES = ("width", "height", "brightness", "contrast", "sharpness", "dark_frac", "bright_frac")


def laplacian_variance(y: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian over the interior of a 2-D array."""
    if y.ndim != 2 or min(y.shape) < 3:
        return 0.0
    c = y[1:-1, 1:-1]
    lap = 4.0 * c - y[:-2, 1:-1] - y[2:, 1:-1] - y[1:-1, :-2] - y[1:-1, 2:]
    return float(lap.var())


def image_stats(img: Image.Image) -> dict[str, float]:
    """The quality features of one image (any mode; converted to RGB first)."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    scale = ANALYSIS_SIDE / max(w, h)
    if scale < 1:
        rgb = rgb.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BOX)
    y = np.asarray(rgb.convert("L"), dtype=np.float64) / 255.0
    return {
        "width": float(w),
        "height": float(h),
        "brightness": float(y.mean()),
        "contrast": float(y.std()),
        "sharpness": laplacian_variance(y),
        "dark_frac": float((y < DARK_LEVEL).mean()),
        "bright_frac": float((y > BRIGHT_LEVEL).mean()),
    }


def stack(rows: list[dict[str, float]], features: tuple[str, ...] = FEATURES) -> dict[str, Any]:
    """List of per-image dicts -> ``{feature: np.ndarray}``."""
    return {f: np.asarray([r[f] for r in rows], dtype=np.float64) for f in features}


@dataclass
class QualityReference:
    """Reference feature columns plus the per-feature acceptance band."""

    columns: dict[str, list[float]]
    lower_q: float = 1.0
    upper_q: float = 99.0

    def band(self, feature: str) -> tuple[float, float]:
        v = np.asarray(self.columns[feature], dtype=np.float64)
        return float(np.percentile(v, self.lower_q)), float(np.percentile(v, self.upper_q))

    def flag(self, stats: dict[str, float]) -> list[str]:
        """Reasons this image looks unlike the reference (empty list = nothing unusual)."""
        reasons: list[str] = []
        for feature, low_name, high_name in (
            ("sharpness", "blurry", None),
            ("brightness", "too_dark", "too_bright"),
            ("contrast", "low_contrast", None),
            ("dark_frac", None, "many_clipped_shadows"),
            ("bright_frac", None, "many_clipped_highlights"),
        ):
            lo, hi = self.band(feature)
            if low_name and stats[feature] < lo:
                reasons.append(low_name)
            if high_name and stats[feature] > hi:
                reasons.append(high_name)
        for feature in ("width", "height"):
            lo, hi = self.band(feature)
            if not lo <= stats[feature] <= hi:
                reasons.append(f"{feature}_outside_reference")
        return reasons

    def to_dict(self) -> dict[str, Any]:
        return {"lower_q": self.lower_q, "upper_q": self.upper_q, "columns": self.columns}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QualityReference:
        return cls(columns=d["columns"], lower_q=d["lower_q"], upper_q=d["upper_q"])

    @classmethod
    def fit(cls, rows: list[dict[str, float]]) -> QualityReference:
        if len(rows) < 30:
            raise ValueError("a reference needs at least 30 images")
        return cls(columns={f: [float(v) for v in stack(rows)[f]] for f in FEATURES})
