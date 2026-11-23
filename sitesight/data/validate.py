"""Invariant checker for a SiteSight :class:`~sitesight.data.schema.Dataset`.

:func:`parse_via` *repairs and flags* problems in the raw export; this module *verifies*
that the resulting dataset (or one edited later in the viewer) satisfies every invariant of
the schema. It is deliberately independent of the parser so a bug in one cannot hide a bug
in the other.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from sitesight.data.schema import (
    FLAG_CLIPPED,
    FLAG_DUPLICATE,
    FLAG_EMPTY_CAPTION,
    FLAG_VALUES,
    Dataset,
)

VALID_SPLITS = frozenset({"train", "val", "test"})


@dataclass(frozen=True)
class Violation:
    code: str
    annotation_id: int | None
    detail: str


def _norm(text: str) -> str:
    return " ".join(text.split())


def validate_dataset(ds: Dataset) -> list[Violation]:
    """Return every invariant violation (empty list means the dataset is valid)."""
    out: list[Violation] = []

    def bad(code: str, ann_id: int | None, detail: str) -> None:
        out.append(Violation(code, ann_id, detail))

    image_ids = Counter(i.id for i in ds.images)
    for iid, n in image_ids.items():
        if n > 1:
            bad("duplicate_image_id", None, f"image id {iid} appears {n} times")
    for name, n in Counter(i.file_name for i in ds.images).items():
        if n > 1:
            bad("duplicate_file_name", None, f"{name} appears {n} times")
    images = {i.id: i for i in ds.images}
    for img in ds.images:
        if img.width <= 0 or img.height <= 0:
            bad("bad_image_size", None, f"{img.file_name}: {img.width}x{img.height}")
        if img.split is not None and img.split not in VALID_SPLITS:
            bad("bad_split", None, f"{img.file_name}: split={img.split!r}")

    category_ids = {c.id for c in ds.categories}
    for aid, n in Counter(a.id for a in ds.annotations).items():
        if n > 1:
            bad("duplicate_annotation_id", aid, f"annotation id appears {n} times")

    seen: set[tuple[int, tuple[int, ...], str]] = set()
    for a in ds.annotations:
        owner = images.get(a.image_id)
        if owner is None:
            bad("unknown_image", a.id, f"image_id={a.image_id}")
            continue
        if len(a.bbox) != 4:
            bad("bad_bbox_length", a.id, f"bbox={a.bbox}")
            continue
        x, y, w, h = a.bbox
        if w <= 0 or h <= 0:
            bad("non_positive_box", a.id, f"bbox={a.bbox}")
        if x < 0 or y < 0 or x + w > owner.width or y + h > owner.height:
            bad("box_out_of_bounds", a.id, f"bbox={a.bbox} image={owner.width}x{owner.height}")
        if a.area != w * h:
            bad("area_mismatch", a.id, f"area={a.area} but w*h={w * h}")
        if a.category_id is not None and a.category_id not in category_ids:
            bad("unknown_category", a.id, f"category_id={a.category_id}")
        for flag in a.flags:
            if flag not in FLAG_VALUES:
                bad("unknown_flag", a.id, flag)
        if a.caption != _norm(a.caption_raw):
            bad("caption_not_normalised", a.id, f"{a.caption!r} vs raw {a.caption_raw!r}")
        if (not a.caption) != (FLAG_EMPTY_CAPTION in a.flags):
            bad("empty_caption_flag_mismatch", a.id, f"caption={a.caption!r} flags={a.flags}")
        if (a.bbox_raw is not None) != (FLAG_CLIPPED in a.flags):
            bad("clipped_flag_mismatch", a.id, f"bbox_raw={a.bbox_raw} flags={a.flags}")
        key = (a.image_id, tuple(a.bbox), a.caption)
        is_repeat = key in seen
        if is_repeat != (FLAG_DUPLICATE in a.flags):
            bad("duplicate_flag_mismatch", a.id, f"repeat={is_repeat} flags={a.flags}")
        seen.add(key)
    return out
