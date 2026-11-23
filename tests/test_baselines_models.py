"""Tests that run the REAL models (slow on CPU). Opt-in: needs torch, transformers, scipy, the
model weights in the Hugging Face cache and the extracted dataset:

    SITESIGHT_DATA_DIR=sitesight_data HF_HOME=... .venv-embed/Scripts/python -m pytest -m embed
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest
from PIL import Image

from sitesight.baselines.prompts import box_to_loc_tokens, owl_queries
from sitesight.data.caption_fields import CATEGORY_NAMES
from sitesight.eval.boxes import iou_matrix

pytestmark = pytest.mark.embed


@pytest.fixture(scope="module")
def image() -> Image.Image:
    root = os.environ.get("SITESIGHT_DATA_DIR")
    if not root:
        pytest.fail("set SITESIGHT_DATA_DIR to the folder containing 'extracted/'")
    return Image.open(Path(root) / "extracted" / "UAVdata" / "3.jpg").convert("RGB")


def test_loc_quantisation_matches_the_library_exactly() -> None:
    import torch
    from transformers import AutoProcessor

    from sitesight.baselines.florence import MODEL_ID

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    rng = random.Random(0)
    w, h = 960, 540
    boxes = [
        [rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0, w), rng.uniform(0, h)]
        for _ in range(2000)
    ]
    # exact bin edges and the image border are where a float rounding difference would show up
    boxes += [[0.96 * k, 0.54 * k, 0.96 * (k + 7), 0.54 * (k + 3)] for k in range(0, 990, 11)]
    boxes += [[0, 0, w, h], [w, h, w, h]]
    q = proc.post_processor.quantize(torch.tensor(boxes, dtype=torch.float32), (w, h)).tolist()
    for (x0, y0, x1, y1), qq in zip(boxes, q, strict=True):
        ours = box_to_loc_tokens([x0, y0, x1 - x0, y1 - y0], w, h)
        assert ours == "".join(f"<loc_{v}>" for v in qq), (x0, y0, x1, y1)


def test_florence_grounding_cache_is_exact_and_used(image: Image.Image) -> None:
    from sitesight.baselines.florence import FlorenceRunner

    runner = FlorenceRunner()
    first = runner.ground(image, "excavator")  # computes the vision features
    hits = runner.n_feature_cache_hits
    second = runner.ground(image, "excavator")  # must reuse them and give the identical answer
    assert runner.n_feature_cache_hits > hits and first == second
    for (x, y, w, h), score in first:
        assert (
            x >= 0
            and y >= 0
            and w > 0
            and h > 0
            and x + w <= image.width + 1
            and y + h <= image.height + 1
        )
        assert 0.0 < score <= 1.0
    # a different image must NOT be served stale features
    other = Image.new("RGB", image.size, (10, 10, 10))
    before = runner.n_feature_cache_hits
    runner.ground(other, "excavator")
    assert runner.n_feature_cache_hits == before  # no hit: new pixels, new features
    # region description: no leaked location tokens, sensible confidence
    caption, score = runner.describe_region(image, [550, 480, 125, 190])
    assert "<loc_" not in caption and caption and 0.0 < score <= 1.0
    assert runner.n_score_fallbacks == 0  # every box got a token-derived confidence


def test_owlv2_detections_are_well_formed(image: Image.Image) -> None:
    from sitesight.baselines.owlv2 import Owlv2Runner

    runner = Owlv2Runner()
    dets = runner.detect(image, owl_queries())
    assert dets, "OWLv2 returned nothing at min_score 0.01"
    by_class: dict[int, list[tuple[list[float], float]]] = {}
    for qi, (x, y, w, h), score in [(d[0], d[1], d[2]) for d in dets]:
        assert 0 <= qi < len(CATEGORY_NAMES) and 0.01 <= score <= 1.0
        assert x >= 0 and y >= 0 and x + w <= image.width + 1e-3 and y + h <= image.height + 1e-3
        by_class.setdefault(qi, []).append(([x, y, w, h], score))
    for items in by_class.values():
        assert len(items) <= 100  # max_per_class
        boxes = [b for b, _ in items]
        m = iou_matrix(boxes, boxes)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert m[i, j] <= 0.5 + 1e-9  # per-class NMS left no near-duplicates
