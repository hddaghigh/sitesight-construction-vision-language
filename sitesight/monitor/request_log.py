"""Append-only JSON-lines log of API requests: enough to monitor, nothing that identifies a site.

One line per request. Stored: UTC time, endpoint, HTTP status, latency, the image quality
statistics of :mod:`sitesight.monitor.quality`, the number of boxes, mean confidence, the share of
captions that parsed, the share flagged for review, and the per-resource tally. **Not stored**:
the image, its file name, the captions or the box coordinates (the privacy rule of the project:
log what is needed to detect drift, no more). A rejected request is logged with its status and
reason so a rise in 4xx errors is visible too.

Writes take a lock and open the file per record, so a crash cannot leave a half-written line
behind another request's line; ``read_log`` still skips (and counts) any malformed line.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


class RequestLog:
    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self._clock = clock
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        record = {"ts": datetime.fromtimestamp(self._clock(), UTC).isoformat(), **record}
        line = json.dumps(record, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def describe_ok(
        self,
        *,
        latency_ms: float,
        image: dict[str, float],
        n_boxes: int,
        confidences: list[float],
        n_parsed: int,
        n_review: int,
        summary: dict[str, int],
    ) -> None:
        n = len(confidences)
        self.write(
            {
                "endpoint": "describe",
                "status": 200,
                "latency_ms": round(latency_ms, 3),
                "image": image,
                "n_boxes": n_boxes,
                "mean_confidence": float(np.mean(confidences)) if n else None,
                "parsed_rate": n_parsed / n if n else None,
                "review_rate": n_review / n if n else None,
                "summary": summary,
            }
        )

    def failure(
        self,
        endpoint: str,
        status: int,
        reason: str,
        latency_ms: float,
        image: dict[str, float] | None = None,
    ) -> None:
        self.write(
            {
                "endpoint": endpoint,
                "status": status,
                "latency_ms": round(latency_ms, 3),
                "image": image,
                "error": reason,
            }
        )

    def ground_ok(self, *, latency_ms: float, image: dict[str, float], n_found: int) -> None:
        self.write(
            {
                "endpoint": "ground",
                "status": 200,
                "latency_ms": round(latency_ms, 3),
                "image": image,
                "n_found": n_found,
            }
        )


def read_log(path: Path) -> tuple[list[dict[str, Any]], int]:
    """(records, number of malformed lines skipped)."""
    records: list[dict[str, Any]] = []
    bad = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(rec, dict):
            records.append(rec)
        else:
            bad += 1
    return records, bad


INPUT_FEATURES = (
    "width",
    "height",
    "brightness",
    "contrast",
    "sharpness",
    "dark_frac",
    "bright_frac",
)
OUTPUT_FEATURES = ("mean_confidence", "parsed_rate", "review_rate")


def log_features(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Feature columns from the successful describe requests (one value per request)."""
    ok = [
        r
        for r in records
        if r.get("endpoint") == "describe"
        and r.get("status") == 200
        and r.get("image")
        and r.get("mean_confidence") is not None
    ]
    cols: dict[str, list[float]] = {f: [] for f in (*INPUT_FEATURES, *OUTPUT_FEATURES)}
    for r in ok:
        for f in INPUT_FEATURES:
            cols[f].append(float(r["image"][f]))
        for f in OUTPUT_FEATURES:
            cols[f].append(float(r[f]))
    return {f: np.asarray(v, dtype=np.float64) for f, v in cols.items()}


def error_rate(records: list[dict[str, Any]]) -> float | None:
    """Share of requests that were rejected or failed (status >= 400)."""
    if not records:
        return None
    return sum(int(r.get("status", 0)) >= 400 for r in records) / len(records)
