"""Model predictions in a file format shared by the viewer and the M4 evaluation harness.

One JSON file per model (so two files can be compared side by side)::

    {
      "model": {"name": "florence2-base-zeroshot", "notes": "free text"},
      "predictions": [
        {"file_name": "1.jpg", "bbox": [x, y, w, h], "score": 0.83,
         "caption": "A worker is located on soil", "category_id": null}
      ]
    }

``bbox`` is absolute pixels ``[x, y, width, height]`` like the ground truth (floats allowed:
models do not predict integers). ``score`` is the model's confidence in ``[0, 1]``; how it is
derived (sequence log-probability for generative models, objectness for detectors) is stated
in ``model.notes``, because calibration (milestone M4) depends on it. ``caption`` and
``category_id`` are optional: a pure detector has boxes and scores but no text.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sitesight.data.schema import Dataset


@dataclass
class Prediction:
    file_name: str
    bbox: list[float]
    score: float
    caption: str | None = None
    category_id: int | None = None


@dataclass
class PredictionSet:
    model_name: str
    predictions: list[Prediction] = field(default_factory=list)
    model_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": {"name": self.model_name, **self.model_info},
            "predictions": [asdict(p) for p in self.predictions],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PredictionSet:
        model = dict(d["model"])
        name = model.pop("name")
        return cls(
            model_name=name,
            model_info=model,
            predictions=[Prediction(**p) for p in d["predictions"]],
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=1, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> PredictionSet:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def by_image(self, min_score: float = 0.0) -> dict[str, list[Prediction]]:
        """Group by file name, highest score first, keeping scores ``>= min_score``."""
        out: dict[str, list[Prediction]] = defaultdict(list)
        for p in self.predictions:
            if p.score >= min_score:
                out[p.file_name].append(p)
        for preds in out.values():
            preds.sort(key=lambda p: -p.score)
        return dict(out)


def validate_predictions(pset: PredictionSet, ds: Dataset) -> list[str]:
    """Return human-readable problems (empty list = valid) when checked against ``ds``."""
    sizes = {i.file_name: (i.width, i.height) for i in ds.images}
    problems: list[str] = []
    for k, p in enumerate(pset.predictions):
        where = f"prediction {k} ({p.file_name})"
        if p.file_name not in sizes:
            problems.append(f"{where}: unknown image")
            continue
        if len(p.bbox) != 4:
            problems.append(f"{where}: bbox must have 4 numbers")
            continue
        x, y, w, h = p.bbox
        width, height = sizes[p.file_name]
        if w <= 0 or h <= 0:
            problems.append(f"{where}: non-positive box {p.bbox}")
        elif x < 0 or y < 0 or x + w > width + 1e-6 or y + h > height + 1e-6:
            problems.append(f"{where}: box {p.bbox} outside {width}x{height}")
        if not 0.0 <= p.score <= 1.0:
            problems.append(f"{where}: score {p.score} not in [0, 1]")
    return problems
