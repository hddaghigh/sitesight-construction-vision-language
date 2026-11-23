"""The one documented SiteSight dataset schema (COCO-style JSON).

Why COCO-style: every detection/grounding tool (pycocotools, Florence-2 fine-tuning
recipes, viewers) already speaks it, so we add only what the captions need.

Layout of the JSON file written by :meth:`Dataset.to_dict`::

    {
      "info":        {...provenance...},
      "licenses":    [{"id": 1, "name": "CC BY 4.0", "url": "..."}],
      "images":      [{"id", "file_name", "width", "height", "group_id", "split"}],
      "annotations": [{"id", "image_id", "bbox", "area", "iscrowd", "category_id",
                       "caption", "caption_raw", "fields", "flags", "bbox_raw", "source"}],
      "categories":  [{"id", "name"}]
    }

Conventions (all enforced by :mod:`sitesight.data.validate`):

* ``bbox`` is ``[x, y, width, height]`` in absolute pixels, origin top-left, and lies
  fully inside the image. If the source box stuck out of the image it is clipped and the
  original is kept in ``bbox_raw`` (otherwise ``bbox_raw`` is ``null``).
* ``caption`` is whitespace-normalised; ``caption_raw`` is exactly what the annotator typed.
* ``category_id`` and ``fields`` are ``null`` until the caption-field parser (milestone M2)
  fills them. M1 must not invent classes.
* ``flags`` records data-quality findings instead of silently dropping rows
  (see :data:`FLAG_VALUES`).
* ``group_id`` / ``split`` are ``null`` until the grouped split (M2).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Annotation flags. Rows with a flag are kept in the file; evaluation code decides
#: whether to exclude them, and says so.
FLAG_CLIPPED = "clipped"  # box extended past the image and was clipped
FLAG_EMPTY_CAPTION = "empty_caption"  # caption is empty after stripping whitespace
FLAG_DUPLICATE = "duplicate"  # same box and caption as an earlier row in the same image
FLAG_UNPARSED = "unparsed"  # set by the caption-field parser (M2) when a non-empty caption fails
FLAG_VALUES = frozenset({FLAG_CLIPPED, FLAG_EMPTY_CAPTION, FLAG_DUPLICATE, FLAG_UNPARSED})

LICENSE_CC_BY_4 = {
    "id": 1,
    "name": "CC BY 4.0",
    "url": "https://creativecommons.org/licenses/by/4.0/",
}


@dataclass
class ImageRecord:
    id: int
    file_name: str
    width: int
    height: int
    group_id: int | None = None
    split: str | None = None


@dataclass
class AnnotationRecord:
    id: int
    image_id: int
    bbox: list[int]
    area: int
    caption: str
    caption_raw: str
    iscrowd: int = 0
    category_id: int | None = None
    fields: dict[str, Any] | None = None
    flags: list[str] = field(default_factory=list)
    bbox_raw: list[int] | None = None
    source: dict[str, str] = field(default_factory=dict)


@dataclass
class CategoryRecord:
    id: int
    name: str


@dataclass
class Dataset:
    """In-memory dataset; round-trips losslessly through :meth:`to_dict` / :meth:`from_dict`."""

    images: list[ImageRecord] = field(default_factory=list)
    annotations: list[AnnotationRecord] = field(default_factory=list)
    categories: list[CategoryRecord] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    licenses: list[dict[str, Any]] = field(default_factory=lambda: [dict(LICENSE_CC_BY_4)])

    def to_dict(self) -> dict[str, Any]:
        return {
            "info": self.info,
            "licenses": self.licenses,
            "images": [asdict(i) for i in self.images],
            "annotations": [asdict(a) for a in self.annotations],
            "categories": [asdict(c) for c in self.categories],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Dataset:
        return cls(
            images=[ImageRecord(**i) for i in d["images"]],
            annotations=[AnnotationRecord(**a) for a in d["annotations"]],
            categories=[CategoryRecord(**c) for c in d.get("categories", [])],
            info=d.get("info", {}),
            licenses=d.get("licenses", []),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # sort_keys + fixed indent => byte-identical output for identical input (reproducible).
        path.write_text(
            json.dumps(self.to_dict(), indent=1, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Dataset:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
