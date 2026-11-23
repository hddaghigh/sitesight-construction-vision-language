"""Pre- and post-processing for serving, with no torch and no transformers.

Everything here reproduces what ``Florence2Processor`` does, so the ONNX path can run in a slim
image. Each piece is checked against the real processor in the Colab parity run
(``sitesight.serve.bench``), not assumed equal.

* Image: RGB, bicubic resize to ``size x size`` (no aspect-ratio preservation, no crop), rescale
  by 1/255, normalise with the ImageNet mean/std, channel-first.
* Prompt ids: ``[<image> x N] <s> prompt </s>``.
* Boxes: the model writes a coordinate as one of 1000 bins per axis. A bin ``n`` maps back to the
  bin centre ``(n + 0.5) * size / 1000`` pixels (the processor then truncates to int).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from sitesight.baselines.prompts import box_to_loc_tokens

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GROUNDING_PROMPT = "Locate the phrases in the caption: {}"
REGION_PROMPT = "What does the region {} describe?"
_LOC = re.compile(r"^<loc_(\d+)>$")


def preprocess(image: Image.Image, size: int = 768) -> np.ndarray:
    """``(1, 3, size, size)`` float32 tensor, as the Florence-2 image processor produces."""
    rgb = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(rgb, dtype=np.float32) / np.float32(255.0)
    arr = (arr - MEAN) / STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None])


def region_prompt(bbox: Sequence[float], width: int, height: int) -> str:
    """Prompt text asking what is inside ``bbox`` (``[x, y, w, h]`` pixels)."""
    return REGION_PROMPT.format(box_to_loc_tokens(bbox, width, height))


def grounding_prompt(phrase: str) -> str:
    return GROUNDING_PROMPT.format(phrase)


def prompt_ids(tokenizer: Any, text: str, image_token_id: int, n_image_tokens: int) -> list[int]:
    """``[image placeholders] <s> text </s>`` ids (``tokenizer``: a ``tokenizers.Tokenizer``)."""
    body = tokenizer.encode("<s>" + text + "</s>", add_special_tokens=False).ids
    return [image_token_id] * n_image_tokens + list(body)


def loc_value(token: str) -> int | None:
    m = _LOC.match(token)
    return int(m.group(1)) if m else None


def dequantize_box(bins: Sequence[int], width: int, height: int) -> list[int]:
    """Four location bins ``(x0, y0, x1, y1)`` -> pixel box ``[x0, y0, x1, y1]`` (centre, int)."""
    pw, ph = width / 1000, height / 1000
    x0, y0, x1, y1 = bins
    return [int((x0 + 0.5) * pw), int((y0 + 0.5) * ph), int((x1 + 0.5) * pw), int((y1 + 0.5) * ph)]


def boxes_from_tokens(tokens: Sequence[str], width: int, height: int) -> list[list[int]]:
    """Pixel boxes from every complete run of four ``<loc_N>`` tokens (leftovers are dropped)."""
    boxes: list[list[int]] = []
    run: list[int] = []

    def flush() -> None:
        for k in range(0, len(run) - len(run) % 4, 4):
            boxes.append(dequantize_box(run[k : k + 4], width, height))
        run.clear()

    for tok in tokens:
        v = loc_value(tok)
        if v is not None:
            run.append(v)
        else:
            flush()
    flush()
    return boxes
