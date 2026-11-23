"""Batch CLI: caption the boxes of many images and write one JSON line per image.

``python -m sitesight.serve.cli describe --model-dir DIR --images IMG_DIR --boxes boxes.json
--out out.jsonl`` where ``boxes.json`` maps file name -> ``[[x, y, w, h], ...]``. It uses the same
validation and the same ``describe_regions`` as the HTTP API, so the two cannot disagree. An image
that cannot be processed is written as an ``error`` line and counted; the run continues and exits
with status 1 if any image failed (nothing is dropped silently).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from PIL import Image

from sitesight.serve.api import (
    Backend,
    Limits,
    decode_image,
    describe_regions,
    parse_boxes,
    summarise,
)


def describe_batch(
    backend: Backend, images_dir: Path, boxes: dict[str, Any], out: Path, limits: Limits
) -> tuple[int, int]:
    """Process every image in ``boxes``; returns (n_ok, n_failed)."""
    ok = failed = 0
    with out.open("w", encoding="utf-8") as f:
        for name in sorted(boxes):
            try:
                path = images_dir / name
                if not path.is_file():
                    raise HTTPException(404, f"{name} not found in {images_dir}")
                img: Image.Image = decode_image(path.read_bytes(), limits)
                valid, warnings = parse_boxes(json.dumps(boxes[name]), *img.size, limits)
                regions = describe_regions(backend, img, valid, limits)
                row: dict[str, Any] = {
                    "file_name": name,
                    "regions": [r.model_dump() for r in regions],
                    "summary": dict(sorted(summarise(regions).items())),
                    "warnings": warnings,
                }
                ok += 1
            except HTTPException as e:
                row = {"file_name": name, "error": e.detail}
                failed += 1
            f.write(json.dumps(row) + "\n")
    return ok, failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe", help="caption the boxes of every image in a boxes file")
    d.add_argument("--model-dir", type=Path, required=True)
    d.add_argument("--images", type=Path, required=True)
    d.add_argument("--boxes", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)

    from sitesight.serve.app import load_backend

    backend = load_backend(args.model_dir, args.threads)
    boxes = json.loads(args.boxes.read_text(encoding="utf-8"))
    ok, failed = describe_batch(backend, args.images, boxes, args.out, Limits.from_env())
    print(f"described {ok} images, {failed} failed -> {args.out}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
