"""Parse the Bang & Kim VIA (VGG Image Annotator 1.x) export into the SiteSight schema.

Source format (verified on the real file, see PLAN.md section 2)::

    { "<filename><filesize>": {
          "filename": "1.jpg", "size": 440260, "regions": {
              "0": {"shape_attributes": {"name": "rect", "x": 543, "y": 407,
                                         "width": 31, "height": 63},
                    "region_attributes": {"phrase": "A worker is located on soil"}},
              ... } } }

The parser never guesses and never silently drops data: every clipped, flagged or
dropped region becomes an :class:`Issue` in the :class:`ValidationReport`.

Box clipping, in one line: a box ``(x, y, w, h)`` on an image ``W x H`` becomes the
intersection of ``[x, x+w] x [y, y+h]`` with ``[0, W] x [0, H]``. A box with nothing left
after clipping is dropped as degenerate. Clipping is the right repair for the real data
(boxes stick out by 1-41 px because annotators dragged past the border), and the original
box is preserved in ``bbox_raw`` so the repair is auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sitesight.data.schema import (
    FLAG_CLIPPED,
    FLAG_DUPLICATE,
    FLAG_EMPTY_CAPTION,
    AnnotationRecord,
    Dataset,
    ImageRecord,
)

PARSER_VERSION = "via_parse/1"
SOURCE_DOI = "10.17632/4h68fmktwh.1"

# Issue codes written to the report.
ISSUE_CLIPPED = "clipped"
ISSUE_EMPTY_CAPTION = "empty_caption"
ISSUE_DUPLICATE = "duplicate"
ISSUE_ZERO_REGIONS = "zero_regions"
ISSUE_DROPPED_DEGENERATE = "dropped_degenerate_box"
ISSUE_DROPPED_SHAPE = "dropped_unsupported_shape"
ISSUE_DROPPED_MALFORMED = "dropped_malformed_region"
ISSUE_SIZE_MISMATCH = "size_mismatch"
ISSUE_KEY_MISMATCH = "key_mismatch"


class DatasetError(Exception):
    """The dataset is unusable as given (e.g. an annotated image file is missing)."""


@dataclass
class Issue:
    code: str
    image: str
    region: str | None = None
    detail: str = ""


@dataclass
class ValidationReport:
    n_images: int = 0
    n_regions_in_source: int = 0
    n_annotations_kept: int = 0
    n_dropped: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)
        self.counts[issue.code] = self.counts.get(issue.code, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["counts"] = dict(sorted(self.counts.items()))
        return d


def jpeg_size(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` of a JPEG by reading its Start-Of-Frame marker.

    Standard library only (no Pillow) so the parser runs with zero dependencies.
    SOF markers are 0xC0-0xCF except 0xC4 (DHT), 0xC8 (JPG) and 0xCC (DAC).
    """
    with path.open("rb") as f:
        if f.read(2) != b"\xff\xd8":
            raise ValueError(f"{path} is not a JPEG (missing SOI marker)")
        while True:
            byte = f.read(1)
            if not byte:
                raise ValueError(f"{path}: reached end of file before a frame header")
            if byte != b"\xff":
                continue
            marker = f.read(1)
            while marker == b"\xff":  # fill bytes
                marker = f.read(1)
            if not marker:
                raise ValueError(f"{path}: truncated marker")
            m = marker[0]
            if m == 0x00 or m == 0x01 or 0xD0 <= m <= 0xD8:  # no length field
                continue
            length_bytes = f.read(2)
            if len(length_bytes) < 2:
                raise ValueError(f"{path}: truncated segment")
            (length,) = struct.unpack(">H", length_bytes)
            if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                body = f.read(5)  # precision (1) + height (2) + width (2)
                if len(body) < 5:
                    raise ValueError(f"{path}: truncated frame header")
                height, width = struct.unpack(">HH", body[1:5])
                return width, height
            f.seek(length - 2, 1)


def normalise_caption(raw: str) -> str:
    """Strip and collapse internal whitespace. Case and wording are left untouched."""
    return " ".join(raw.split())


def _natural_key(name: str) -> tuple[int, int, str]:
    """Sort ``2.jpg`` before ``10.jpg`` (numeric stems first, others after by name)."""
    stem = Path(name).stem
    return (0, int(stem), name) if stem.isdigit() else (1, 0, name)


def _region_items(regions: Any) -> list[tuple[str, dict[str, Any]]]:
    """VIA 1.x stores regions as a dict keyed '0','1',...; VIA 2.x as a list. Accept both."""
    if isinstance(regions, dict):
        keys = sorted(regions, key=lambda k: (0, int(k), k) if str(k).isdigit() else (1, 0, k))
        return [(str(k), regions[k]) for k in keys]
    return [(str(i), r) for i, r in enumerate(regions)]


def _to_int(value: Any) -> int:
    """VIA coordinates are numbers; accept floats by rounding to the nearest pixel."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"not a number: {value!r}")
    return round(value)


def parse_via(via_path: Path, images_dir: Path) -> tuple[Dataset, ValidationReport]:
    """Convert a VIA export to a :class:`Dataset` plus a :class:`ValidationReport`.

    ``images_dir`` must contain every annotated file; a missing image raises
    :class:`DatasetError` because without it the box cannot be bounds-checked.
    Output order is deterministic: images by numeric file name, annotations by
    (image, region id).
    """
    raw = json.loads(via_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise DatasetError("VIA file must be a JSON object keyed by file name + size")

    entries = sorted(raw.items(), key=lambda kv: _natural_key(str(kv[1].get("filename", kv[0]))))
    report = ValidationReport()
    ds = Dataset(
        info={
            "description": "Bang & Kim UAV construction-site region captions, parsed from VIA.",
            "source_doi": SOURCE_DOI,
            "source_file": via_path.name,
            "source_sha256": hashlib.sha256(via_path.read_bytes()).hexdigest(),
            "parser": PARSER_VERSION,
        }
    )
    next_ann_id = 1

    for image_id, (via_key, entry) in enumerate(entries, start=1):
        file_name = entry["filename"]
        image_path = images_dir / file_name
        if not image_path.is_file():
            raise DatasetError(f"annotated image not found: {image_path}")
        width, height = jpeg_size(image_path)
        ds.images.append(ImageRecord(id=image_id, file_name=file_name, width=width, height=height))
        report.n_images += 1

        actual_size = image_path.stat().st_size
        if entry.get("size") != actual_size:
            report.add(
                Issue(
                    ISSUE_SIZE_MISMATCH,
                    file_name,
                    detail=f"VIA size={entry.get('size')} file size={actual_size}",
                )
            )
        if via_key != f"{file_name}{entry.get('size')}":
            report.add(Issue(ISSUE_KEY_MISMATCH, file_name, detail=f"key={via_key!r}"))

        regions = _region_items(entry.get("regions", {}))
        if not regions:
            report.add(Issue(ISSUE_ZERO_REGIONS, file_name))

        seen: set[tuple[tuple[int, int, int, int], str]] = set()
        for region_id, region in regions:
            report.n_regions_in_source += 1
            try:
                shape = region["shape_attributes"]
                caption_raw = region["region_attributes"]["phrase"]
                if not isinstance(caption_raw, str):
                    raise TypeError("phrase is not a string")
            except (KeyError, TypeError) as exc:
                report.n_dropped += 1
                report.add(Issue(ISSUE_DROPPED_MALFORMED, file_name, region_id, repr(exc)))
                continue
            if shape.get("name") != "rect":
                report.n_dropped += 1
                report.add(Issue(ISSUE_DROPPED_SHAPE, file_name, region_id, str(shape.get("name"))))
                continue
            try:
                x, y, w, h = (_to_int(shape[k]) for k in ("x", "y", "width", "height"))
            except (KeyError, TypeError) as exc:
                report.n_dropped += 1
                report.add(Issue(ISSUE_DROPPED_MALFORMED, file_name, region_id, repr(exc)))
                continue

            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + w), min(height, y + h)
            if w <= 0 or h <= 0 or x1 <= x0 or y1 <= y0:
                report.n_dropped += 1
                report.add(
                    Issue(ISSUE_DROPPED_DEGENERATE, file_name, region_id, f"raw box {[x, y, w, h]}")
                )
                continue

            bbox = [x0, y0, x1 - x0, y1 - y0]
            flags: list[str] = []
            bbox_raw: list[int] | None = None
            if bbox != [x, y, w, h]:
                flags.append(FLAG_CLIPPED)
                bbox_raw = [x, y, w, h]
                report.add(Issue(ISSUE_CLIPPED, file_name, region_id, f"{[x, y, w, h]} -> {bbox}"))

            caption = normalise_caption(caption_raw)
            if not caption:
                flags.append(FLAG_EMPTY_CAPTION)
                report.add(Issue(ISSUE_EMPTY_CAPTION, file_name, region_id))

            key = ((x0, y0, x1 - x0, y1 - y0), caption)
            if key in seen:
                flags.append(FLAG_DUPLICATE)
                report.add(Issue(ISSUE_DUPLICATE, file_name, region_id, caption))
            seen.add(key)

            ds.annotations.append(
                AnnotationRecord(
                    id=next_ann_id,
                    image_id=image_id,
                    bbox=bbox,
                    area=bbox[2] * bbox[3],
                    caption=caption,
                    caption_raw=caption_raw,
                    flags=flags,
                    bbox_raw=bbox_raw,
                    source={"via_key": via_key, "region_id": region_id},
                )
            )
            next_ann_id += 1
            report.n_annotations_kept += 1

    return ds, report


def summarise(report: ValidationReport) -> str:
    c = Counter(report.counts)
    lines = [
        f"images:                 {report.n_images}",
        f"regions in source:      {report.n_regions_in_source}",
        f"annotations kept:       {report.n_annotations_kept}",
        f"regions dropped:        {report.n_dropped}",
    ]
    lines += [f"{code + ':':<23} {n}" for code, n in sorted(c.items())]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Convert a VIA export to the SiteSight COCO-style JSON."
    )
    p.add_argument("--via", type=Path, required=True, help="via_region_data_final.json")
    p.add_argument("--images", type=Path, required=True, help="directory with the JPEGs")
    p.add_argument("--out", type=Path, required=True, help="output annotations.json")
    p.add_argument("--report", type=Path, help="optional validation_report.json")
    args = p.parse_args(argv)

    ds, report = parse_via(args.via, args.images)
    ds.save(args.out)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report.to_dict(), indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(summarise(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
