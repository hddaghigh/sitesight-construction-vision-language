"""Turn the curated dataset into evaluation inputs, and say exactly what was excluded.

Ground truth used for evaluation = the annotations of one split that (a) parsed into a class
(``category_id`` set), (b) are not flagged ``duplicate`` (an exact repeat would be counted twice),
and (c) have a non-empty caption. Everything else is counted and reported, never silently dropped.

A region is a ``single`` object when its caption says exactly one item (``count == 1`` and
``count_unit == "item"``), otherwise a ``group`` (several objects, a pile, or an uncounted plural
such as "Fences are installed in a line"). One box covering a group is a labelling convention a
per-object detector cannot be expected to follow; the ``kind`` breakdown makes that visible.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from sitesight.data.predictions import PredictionSet
from sitesight.data.schema import FLAG_DUPLICATE, Dataset
from sitesight.eval.grounding import GT, Det


@dataclass
class RegionGT:
    ann_id: int
    file_name: str
    bbox: list[float]
    caption: str
    fields: dict[str, Any]


def split_images(ds: Dataset, split: str) -> list[str]:
    return sorted((i.file_name for i in ds.images if i.split == split), key=lambda n: (len(n), n))


def build_ground_truth(
    ds: Dataset, split: str, images: set[str] | None = None
) -> tuple[list[GT], list[RegionGT], dict[str, int]]:
    """Ground truth for grounding and captioning plus an exclusion report."""
    names = {c.id: c.name for c in ds.categories}
    img = {
        i.id: i for i in ds.images if i.split == split and (images is None or i.file_name in images)
    }
    kept: list[tuple[Any, Any]] = []
    stats: Counter[str] = Counter()
    for a in ds.annotations:
        im = img.get(a.image_id)
        if im is None:
            continue
        stats["annotations_in_split"] += 1
        if not a.caption:
            stats["excluded_empty_caption"] += 1
        elif FLAG_DUPLICATE in a.flags:
            stats["excluded_duplicate"] += 1
        elif a.category_id is None or a.fields is None:
            stats["excluded_unparsed"] += 1
        else:
            kept.append((im, a))
    per_image = Counter(im.file_name for im, _ in kept)
    gts: list[GT] = []
    regions: list[RegionGT] = []
    for im, a in kept:
        f = a.fields
        kind = "single" if f["count"] == 1 and f["count_unit"] == "item" else "group"
        gts.append(
            GT(
                file_name=im.file_name,
                category=names[a.category_id],
                bbox=[float(v) for v in a.bbox],
                area_frac=a.area / (im.width * im.height),
                kind=kind,
                crowd=per_image[im.file_name],
            )
        )
        regions.append(RegionGT(a.id, im.file_name, [float(v) for v in a.bbox], a.caption, f))
    stats["evaluated"] = len(gts)
    stats["images"] = len(img)
    return gts, regions, dict(stats)


def detections_from_predictions(pset: PredictionSet, ds: Dataset) -> list[Det]:
    """Grounding predictions -> :class:`Det` (class from ``category_id``); skips class-less ones."""
    names = {c.id: c.name for c in ds.categories}
    out = []
    for p in pset.predictions:
        if p.category_id is None or p.category_id not in names:
            continue
        out.append(Det(p.file_name, names[p.category_id], [float(v) for v in p.bbox], p.score))
    return out
