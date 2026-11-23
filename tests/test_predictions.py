from __future__ import annotations

from pathlib import Path

from sitesight.data.predictions import Prediction, PredictionSet, validate_predictions
from sitesight.data.schema import Dataset, ImageRecord


def _ds() -> Dataset:
    return Dataset(
        images=[
            ImageRecord(id=1, file_name="1.jpg", width=100, height=60),
            ImageRecord(id=2, file_name="2.jpg", width=80, height=40),
        ]
    )


def _set() -> PredictionSet:
    return PredictionSet(
        model_name="m1",
        model_info={"notes": "score = mean token prob"},
        predictions=[
            Prediction("1.jpg", [10.5, 10, 20, 20], 0.4, "A worker is located on soil"),
            Prediction("1.jpg", [0, 0, 50, 30], 0.9),
            Prediction("2.jpg", [5, 5, 10, 10], 0.5, None, 3),
        ],
    )


def test_roundtrip(tmp_path: Path) -> None:
    s = _set()
    p = tmp_path / "pred.json"
    s.save(p)
    back = PredictionSet.load(p)
    assert back.to_dict() == s.to_dict()
    assert back.model_name == "m1" and back.model_info == {"notes": "score = mean token prob"}
    assert back.predictions[2].category_id == 3 and back.predictions[1].caption is None


def test_by_image_sorts_by_score_and_filters_inclusively() -> None:
    s = _set()
    grouped = s.by_image()
    assert [p.score for p in grouped["1.jpg"]] == [0.9, 0.4]  # highest first
    assert [p.score for p in s.by_image(min_score=0.4)["1.jpg"]] == [0.9, 0.4]  # >= is inclusive
    assert [p.score for p in s.by_image(min_score=0.41)["1.jpg"]] == [0.9]
    assert "2.jpg" not in s.by_image(min_score=0.6)


def test_validate_accepts_good_and_names_each_problem() -> None:
    assert validate_predictions(_set(), _ds()) == []
    bad = PredictionSet(
        model_name="bad",
        predictions=[
            Prediction("nope.jpg", [0, 0, 1, 1], 0.5),
            Prediction("1.jpg", [0, 0, 0, 5], 0.5),
            Prediction("1.jpg", [90, 0, 20, 5], 0.5),
            Prediction("1.jpg", [0, 0, 5, 5], 1.5),
            Prediction("1.jpg", [0, 0, 5], 0.5),
        ],
    )
    problems = validate_predictions(bad, _ds())
    assert len(problems) == 5
    joined = " | ".join(problems)
    for needle in ("unknown image", "non-positive", "outside 100x60", "not in [0, 1]", "4 numbers"):
        assert needle in joined
