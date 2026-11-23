"""Validator tests: a parsed dataset is clean, and each invariant breach is caught by name."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

import pytest

from sitesight.data.schema import CategoryRecord, Dataset
from sitesight.data.validate import validate_dataset
from sitesight.data.via_parse import parse_via


@pytest.fixture
def parsed(via_fixture: tuple[Path, Path]) -> Dataset:
    return parse_via(*via_fixture)[0]


def codes(ds: Dataset) -> set[str]:
    return {v.code for v in validate_dataset(ds)}


def test_parsed_fixture_is_valid(parsed: Dataset) -> None:
    assert validate_dataset(parsed) == []


def _oob(ds: Dataset) -> None:
    ds.annotations[0].bbox = [90, 50, 20, 20]  # 110 > width 100


def _zero(ds: Dataset) -> None:
    ds.annotations[0].bbox = [10, 10, 0, 5]
    ds.annotations[0].area = 0


def _area(ds: Dataset) -> None:
    ds.annotations[0].area += 1


def _image(ds: Dataset) -> None:
    ds.annotations[0].image_id = 999


def _caption(ds: Dataset) -> None:
    ds.annotations[0].caption = "A  worker"  # not normalised


def _category(ds: Dataset) -> None:
    ds.annotations[0].category_id = 7


def _flag(ds: Dataset) -> None:
    ds.annotations[0].flags.append("made_up")


def _split(ds: Dataset) -> None:
    ds.images[0].split = "validation"


def _dup_id(ds: Dataset) -> None:
    ds.annotations[1].id = ds.annotations[0].id


def _clip_flag(ds: Dataset) -> None:
    ds.annotations[0].bbox_raw = [0, 0, 1, 1]  # raw box present but no 'clipped' flag


def _dup_flag(ds: Dataset) -> None:
    ds.annotations[2].flags.clear()  # the repeated row loses its 'duplicate' flag


def _empty_flag(ds: Dataset) -> None:
    ds.annotations[3].flags.clear()  # empty caption loses its flag


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (_oob, "box_out_of_bounds"),
        (_zero, "non_positive_box"),
        (_area, "area_mismatch"),
        (_image, "unknown_image"),
        (_caption, "caption_not_normalised"),
        (_category, "unknown_category"),
        (_flag, "unknown_flag"),
        (_split, "bad_split"),
        (_dup_id, "duplicate_annotation_id"),
        (_clip_flag, "clipped_flag_mismatch"),
        (_dup_flag, "duplicate_flag_mismatch"),
        (_empty_flag, "empty_caption_flag_mismatch"),
    ],
)
def test_each_breach_is_detected(
    parsed: Dataset, mutate: Callable[[Dataset], None], expected: str
) -> None:
    broken = copy.deepcopy(parsed)
    mutate(broken)
    assert expected in codes(broken)


def test_known_category_is_accepted(parsed: Dataset) -> None:
    parsed.categories.append(CategoryRecord(id=1, name="worker"))
    parsed.annotations[0].category_id = 1
    parsed.annotations[0].fields = {"resource": "worker"}
    assert validate_dataset(parsed) == []
    parsed.annotations[0].fields = {"resource": "excavator"}  # category and fields disagree
    assert "category_resource_mismatch" in codes(parsed)
    parsed.annotations[0].fields = None  # category without fields
    assert "fields_category_mismatch" in codes(parsed)


def test_duplicate_image_rows_detected(parsed: Dataset) -> None:
    parsed.images.append(copy.deepcopy(parsed.images[0]))
    assert {"duplicate_image_id", "duplicate_file_name"} <= codes(parsed)
