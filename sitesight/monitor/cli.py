"""Monitoring CLI.

    python -m sitesight.monitor.cli fit --images DIR --annotations A.json --split train
        --out ref.json
    python -m sitesight.monitor.cli fit --log good_period.jsonl --out ref.json
    python -m sitesight.monitor.cli check --reference ref.json --images DIR
        [--annotations A --split val] --report out.html [--json out.json]
    python -m sitesight.monitor.cli check --reference ref.json --log requests.jsonl
        --report out.html

(each command is one line; they are wrapped here)

Exit status of ``check``: 0 no drift, 1 drift alert, 2 batch too small to test (so a scheduler can
tell "alert" from "could not decide"). ``--log`` batches also test the model-output features
(mean confidence, parse rate, review rate) when the reference holds them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sitesight.monitor.drift import detect_drift
from sitesight.monitor.quality import FEATURES, QualityReference, image_stats, stack
from sitesight.monitor.report import render_report
from sitesight.monitor.request_log import log_features, read_log

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def list_images(images: Path, annotations: Path | None, split: str | None) -> list[Path]:
    if annotations is not None:
        from sitesight.data.schema import Dataset

        ds = Dataset.load(annotations)
        return [images / i.file_name for i in ds.images if split is None or i.split == split]
    return sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def stats_for(paths: list[Path]) -> list[dict[str, float]]:
    rows = []
    for p in paths:
        with Image.open(p) as im:
            rows.append(image_stats(im))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="build a reference from images or from a request log")
    c = sub.add_parser("check", help="compare a batch with a reference")
    for p in (f, c):
        p.add_argument("--images", type=Path)
        p.add_argument("--annotations", type=Path)
        p.add_argument("--split", default=None)
        p.add_argument("--log", type=Path)
    f.add_argument("--out", type=Path, required=True)
    c.add_argument("--reference", type=Path, required=True)
    c.add_argument("--report", type=Path, required=True)
    c.add_argument("--json", type=Path, default=None)
    c.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args(argv)
    if args.split == "test":
        raise SystemExit("monitoring never touches the test split")
    if (args.images is None) == (args.log is None):
        raise SystemExit("give exactly one of --images or --log")

    cols: dict[str, np.ndarray]
    if args.images is not None:
        rows = stats_for(list_images(args.images, args.annotations, args.split))
        cols = stack(rows)
        flags_rows = rows
        source = f"images in {args.images}" + (f" (split {args.split})" if args.split else "")
    else:
        records, bad = read_log(args.log)
        cols = log_features(records)
        flags_rows = [
            r["image"] for r in records if r.get("status") == 200 and r.get("image") is not None
        ]
        source = f"request log {args.log} ({len(records)} lines, {bad} malformed skipped)"

    if args.cmd == "fit":
        n = len(next(iter(cols.values())))
        QualityReference.fit([{k: float(cols[k][i]) for k in FEATURES} for i in range(n)])
        ref = {"source": source, "columns": {k: [float(x) for x in v] for k, v in cols.items()}}
        args.out.write_text(json.dumps(ref), encoding="utf-8")
        print(f"reference of {n} samples from {source} -> {args.out}", file=sys.stderr)
        return 0

    saved: dict[str, Any] = json.loads(args.reference.read_text(encoding="utf-8"))
    ref_cols = {k: np.asarray(v, dtype=np.float64) for k, v in saved["columns"].items()}
    qref = QualityReference(columns={k: list(map(float, ref_cols[k])) for k in FEATURES})
    flags = [qref.flag({k: float(r[k]) for k in FEATURES}) for r in flags_rows]
    result = detect_drift(ref_cols, cols, alpha=args.alpha)
    args.report.write_text(
        render_report(result, ref_cols, cols, flags, source_note=f"Batch: {source}."),
        encoding="utf-8",
    )
    if args.json:
        args.json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"{result.status}: {len(result.alerts)} alerting tests -> {args.report}", file=sys.stderr)
    return {"ok": 0, "alert": 1, "too_small": 2}[result.status]


if __name__ == "__main__":
    raise SystemExit(main())
