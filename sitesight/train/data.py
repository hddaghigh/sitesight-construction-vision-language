"""Turn the curated dataset into (prompt, target) text pairs for Florence-2 fine-tuning.

Two tasks, matching exactly what is evaluated (``sitesight.eval``):

* **region** (region captioning): prompt ``<REGION_TO_DESCRIPTION>`` + the ground-truth box as four
  location tokens; target = the region's normalised caption. (The zero-shot model also echoes the
  box after its description; we train on the caption alone so the fine-tuned model stops there.)
* **grounding** (phrase grounding): prompt ``<CAPTION_TO_PHRASE_GROUNDING>`` + the class phrase;
  target = the phrase followed by one ``<loc_x0><loc_y0><loc_x1><loc_y1>`` group per ground-truth
  box of that class in the image, in reading order (top to bottom, then left to right), which is
  the format the model itself writes. One sample per (image, class present).

The same exclusions as evaluation apply (:func:`sitesight.eval.data.build_ground_truth`): empty
captions, exact duplicates and captions the parser could not read are not trained on, and the
numbers excluded are returned, not hidden. Only the requested split is used: training uses
``train``; ``val`` provides the loss used to pick the epoch; ``test`` is never read here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from sitesight.baselines.prompts import (
    FLORENCE_GROUNDING_TASK,
    FLORENCE_REGION_TASK,
    QUERY_PHRASES,
    box_to_loc_tokens,
)
from sitesight.data.schema import Dataset
from sitesight.eval.data import build_ground_truth


@dataclass(frozen=True)
class Sample:
    file_name: str
    task: str  # "region" | "grounding"
    prompt: str  # task token + input, exactly what is given to the processor
    target: str  # text the decoder must produce
    ref: str  # region annotation id (region) or class name (grounding): for traceability


def reading_order(boxes: Sequence[Sequence[float]]) -> list[Sequence[float]]:
    """Sort ``[x, y, w, h]`` boxes top-to-bottom, then left-to-right, then by size."""
    return sorted(boxes, key=lambda b: (b[1], b[0], b[3], b[2]))


def build_samples(ds: Dataset, split: str) -> tuple[list[Sample], dict[str, int]]:
    """All training pairs for one split, plus counts (including what was excluded)."""
    gts, regions, stats = build_ground_truth(ds, split)
    size = {i.file_name: (i.width, i.height) for i in ds.images}
    samples: list[Sample] = []

    for r in regions:
        w, h = size[r.file_name]
        prompt = FLORENCE_REGION_TASK + box_to_loc_tokens(r.bbox, w, h)
        samples.append(Sample(r.file_name, "region", prompt, r.caption, str(r.ann_id)))

    by_pair: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for g in gts:
        by_pair[(g.file_name, g.category)].append(g.bbox)
    for (name, category), boxes in sorted(by_pair.items(), key=lambda kv: (len(kv[0][0]), kv[0])):
        w, h = size[name]
        phrase = QUERY_PHRASES[category]
        locs = "".join(box_to_loc_tokens(b, w, h) for b in reading_order(boxes))
        samples.append(
            Sample(name, "grounding", FLORENCE_GROUNDING_TASK + phrase, phrase + locs, category)
        )

    out = dict(stats)
    out["n_region_samples"] = sum(s.task == "region" for s in samples)
    out["n_grounding_samples"] = sum(s.task == "grounding" for s in samples)
    out["n_images_with_samples"] = len({s.file_name for s in samples})
    return samples, out
