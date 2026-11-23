"""Training-sample construction: hand-checked on the fixture dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from sitesight.baselines.prompts import box_to_loc_tokens
from sitesight.data.caption_fields import apply_caption_fields
from sitesight.data.schema import Dataset
from sitesight.data.via_parse import parse_via
from sitesight.train.data import build_samples, reading_order


@pytest.fixture
def ds(via_fixture: tuple[Path, Path]) -> Dataset:
    d, _ = parse_via(*via_fixture)
    apply_caption_fields(d)
    for img in d.images:
        img.split = "train"
    return d


def test_reading_order_is_top_to_bottom_then_left_to_right() -> None:
    boxes = [[50, 10, 5, 5], [10, 30, 5, 5], [10, 10, 5, 5], [30, 10, 9, 9]]
    assert reading_order(boxes) == [[10, 10, 5, 5], [30, 10, 9, 9], [50, 10, 5, 5], [10, 30, 5, 5]]
    assert reading_order([]) == []


def test_region_and_grounding_samples_on_the_fixture(ds: Dataset) -> None:
    samples, stats = build_samples(ds, "train")
    # Fixture: 3 evaluated regions (worker, fence in 1.jpg; excavators in 10.jpg); the duplicate and
    # the empty caption are excluded exactly as in evaluation.
    assert stats["n_region_samples"] == 3 and stats["n_grounding_samples"] == 3
    assert stats["excluded_duplicate"] == 1 and stats["excluded_empty_caption"] == 1
    regions = {s.target: s for s in samples if s.task == "region"}
    assert set(regions) == {
        "A worker is located on soil",
        "Fences are installed in a line",
        "Two orange excavators are digging soil",
    }
    worker = regions["A worker is located on soil"]
    assert worker.prompt == "<REGION_TO_DESCRIPTION>" + box_to_loc_tokens([10, 10, 20, 20], 100, 60)
    assert worker.file_name == "1.jpg" and worker.ref.isdigit()

    ground = {s.ref: s for s in samples if s.task == "grounding"}
    assert set(ground) == {"worker", "fence", "excavator"}
    g = ground["worker"]
    assert g.prompt == "<CAPTION_TO_PHRASE_GROUNDING>worker"
    assert g.target == "worker" + box_to_loc_tokens([10, 10, 20, 20], 100, 60)
    # the excavator box is the whole 80x40 image of 10.jpg
    assert ground["excavator"].target == "excavator" + box_to_loc_tokens([0, 0, 80, 40], 80, 40)
    assert ground["excavator"].target.count("<loc_") == 4


def test_several_boxes_of_one_class_are_concatenated_in_reading_order(ds: Dataset) -> None:
    from sitesight.data.schema import AnnotationRecord

    # Add two more workers to 1.jpg: one below, one to the right of the first.
    for k, box in enumerate([[10, 40, 10, 10], [50, 10, 10, 10]], start=100):
        ds.annotations.append(
            AnnotationRecord(
                id=k,
                image_id=1,
                bbox=box,
                area=100,
                caption="A worker is located on soil",
                caption_raw="A worker is located on soil",
                category_id=ds.annotations[0].category_id,
                fields=ds.annotations[0].fields,
            )
        )
    samples, _ = build_samples(ds, "train")
    g = next(s for s in samples if s.task == "grounding" and s.ref == "worker")
    expect = "worker" + "".join(
        box_to_loc_tokens(b, 100, 60)
        for b in ([10, 10, 20, 20], [50, 10, 10, 10], [10, 40, 10, 10])
    )
    assert g.target == expect and g.target.count("<loc_") == 12


def test_only_the_requested_split_is_used(ds: Dataset) -> None:
    ds.images[0].split = "test"  # 1.jpg leaves the training split
    samples, stats = build_samples(ds, "train")
    assert {s.file_name for s in samples} == {"10.jpg"} and stats["n_images_with_samples"] == 1
    assert build_samples(ds, "val")[0] == []
