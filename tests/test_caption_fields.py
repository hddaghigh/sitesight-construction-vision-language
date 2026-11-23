"""Caption-field parser tests.

The expected values below are labelled BY HAND from reading the caption (not produced by
running the parser). Row format:
(caption, count, unit, colour, resource, state, preposition, target, destination).
"""

from __future__ import annotations

import pytest

from sitesight.data.caption_fields import (
    CATEGORY_IDS,
    CATEGORY_NAMES,
    apply_caption_fields,
    parse_caption,
)
from sitesight.data.schema import (
    FLAG_EMPTY_CAPTION,
    FLAG_UNPARSED,
    AnnotationRecord,
    Dataset,
    ImageRecord,
)
from sitesight.data.validate import validate_dataset

Row = tuple[str, int | None, str | None, str | None, str, str, str | None, str | None, str | None]

GOOD: list[Row] = [
    ("A worker is located on soil", 1, "item", None, "worker", "located", "on", "soil", None),
    ("Two workers are located on soil", 2, "item", None, "worker", "located", "on", "soil", None),
    (
        "Twenty six H-beams are installed on soil",
        26,
        "item",
        None,
        "h_beam",
        "installed",
        "on",
        "soil",
        None,
    ),
    (
        "Twenty-eight foundations are installed on soil",
        28,
        "item",
        None,
        "foundation",
        "installed",
        "on",
        "soil",
        None,
    ),
    (
        "Eighty-two concrete pipes are located on soil",
        82,
        "item",
        None,
        "concrete_pipe",
        "located",
        "on",
        "soil",
        None,
    ),
    (
        "Thirty two H-beams are installed on soil",
        32,
        "item",
        None,
        "h_beam",
        "installed",
        "on",
        "soil",
        None,
    ),
    (
        "Thirty foundations are installed on soil",
        30,
        "item",
        None,
        "foundation",
        "installed",
        "on",
        "soil",
        None,
    ),
    (
        "An orange excavator is located on soil",
        1,
        "item",
        "orange",
        "excavator",
        "located",
        "on",
        "soil",
        None,
    ),
    (
        "A orange excavator is located on soil",
        1,
        "item",
        "orange",
        "excavator",
        "located",
        "on",
        "soil",
        None,
    ),
    (
        "An orange excavator is digging soil",
        1,
        "item",
        "orange",
        "excavator",
        "digging",
        None,
        "soil",
        None,
    ),
    (
        "Two orange excavators are digging soil",
        2,
        "item",
        "orange",
        "excavator",
        "digging",
        None,
        "soil",
        None,
    ),
    (
        "A yellow excavator is digging a rock,",
        1,
        "item",
        "yellow",
        "excavator",
        "digging",
        None,
        "rock",
        None,
    ),
    ("A white dump truck is empty", 1, "item", "white", "dump_truck", "empty", None, None, None),
    (
        "A white dump truck is loaded with soil",
        1,
        "item",
        "white",
        "dump_truck",
        "loaded",
        "with",
        "soil",
        None,
    ),
    (
        "A dump truck is loaded with gray rocks and soil",
        1,
        "item",
        None,
        "dump_truck",
        "loaded",
        "with",
        "gray rocks and soil",
        None,
    ),
    (
        "A white dump truck is loaded with nine concrete pillars",
        1,
        "item",
        "white",
        "dump_truck",
        "loaded",
        "with",
        "nine concrete pillars",
        None,
    ),
    (
        "A concrete pipe is covered with soil",
        1,
        "item",
        None,
        "concrete_pipe",
        "covered",
        "with",
        "soil",
        None,
    ),
    (
        "A black dust protection is covering rocks",
        1,
        "item",
        "black",
        "dust_protection",
        "covering",
        None,
        "rocks",
        None,
    ),
    (
        "A black dust protection is covering soil",
        1,
        "item",
        "black",
        "dust_protection",
        "covering",
        None,
        "soil",
        None,
    ),
    (
        "A temporary building is covered with a blue tarpaulin",
        1,
        "item",
        None,
        "temporary_building",
        "covered",
        "with",
        "blue tarpaulin",
        None,
    ),
    (
        "An orange excavator is putting soil in a white dump truck",
        1,
        "item",
        "orange",
        "excavator",
        "putting",
        "in",
        "soil",
        "white dump truck",
    ),
    (
        "An excavator is putting soil on a truck",
        1,
        "item",
        None,
        "excavator",
        "putting",
        "on",
        "soil",
        "truck",
    ),
    (
        "A yellow excavator is lifting two H-beams",
        1,
        "item",
        "yellow",
        "excavator",
        "lifting",
        None,
        "two h-beams",
        None,
    ),
    (
        "A white crane truck is pouring concrete",
        1,
        "item",
        "white",
        "crane_truck",
        "pouring",
        None,
        "concrete",
        None,
    ),
    (
        "A yellow excavator is drilling on rock",
        1,
        "item",
        "yellow",
        "excavator",
        "drilling",
        "on",
        "rock",
        None,
    ),
    (
        "A gray drilling is drilling on soil",
        1,
        "item",
        "gray",
        "drilling_rig",
        "drilling",
        "on",
        "soil",
        None,
    ),
    (
        "A green drilling is located on soil",
        1,
        "item",
        "green",
        "drilling_rig",
        "located",
        "on",
        "soil",
        None,
    ),
    ("A yellow excavator is on soil", 1, "item", "yellow", "excavator", "on", "on", "soil", None),
    ("Fences are installed in a line", None, None, None, "fence", "installed", "in", "line", None),
    (
        "Wood fences are installed in a line",
        None,
        None,
        None,
        "wood_fence",
        "installed",
        "in",
        "line",
        None,
    ),
    (
        "Safety fences are installed in a line",
        None,
        None,
        None,
        "safety_fence",
        "installed",
        "in",
        "line",
        None,
    ),
    (
        "Concrete is installed on soil",
        None,
        None,
        None,
        "concrete",
        "installed",
        "on",
        "soil",
        None,
    ),
    (
        "Three concrete pipes are installed in a line",
        3,
        "item",
        None,
        "concrete_pipe",
        "installed",
        "in",
        "line",
        None,
    ),
    (
        "A pile of H-beam is located on soil",
        1,
        "pile",
        None,
        "h_beam",
        "located",
        "on",
        "soil",
        None,
    ),
    (
        "A pile of black H-beam are located on soil",
        1,
        "pile",
        "black",
        "h_beam",
        "located",
        "on",
        "soil",
        None,
    ),
    (
        "Four piles of molds are located on soil",
        4,
        "pile",
        None,
        "mold",
        "located",
        "on",
        "soil",
        None,
    ),
    (
        "Eight pile of molds are located on soil",
        8,
        "pile",
        None,
        "mold",
        "located",
        "on",
        "soil",
        None,
    ),
    (
        "Piles of rebars are located on soil",
        None,
        "pile",
        None,
        "rebar",
        "located",
        "on",
        "soil",
        None,
    ),
    (
        "Eighteen piles of white materials are located on soil",
        18,
        "pile",
        "white",
        "material",
        "located",
        "on",
        "soil",
        None,
    ),
    (
        "Two piles wood bars are located on soil",
        2,
        "pile",
        None,
        "wood_bar",
        "located",
        "on",
        "soil",
        None,
    ),
    (
        "Three Piles of wood bars are located on soil",
        3,
        "pile",
        None,
        "wood_bar",
        "located",
        "on",
        "soil",
        None,
    ),
    (
        "A blue tarpaulin Is located on concrete",
        1,
        "item",
        "blue",
        "tarpaulin",
        "located",
        "on",
        "concrete",
        None,
    ),
    (
        "A pink temporary building is installed on soil",
        1,
        "item",
        "pink",
        "temporary_building",
        "installed",
        "on",
        "soil",
        None,
    ),
    (
        "An excavator bucket is located on soil",
        1,
        "item",
        None,
        "excavator_bucket",
        "located",
        "on",
        "soil",
        None,
    ),
    (
        "Three excavator buckets are located on soil",
        3,
        "item",
        None,
        "excavator_bucket",
        "located",
        "on",
        "soil",
        None,
    ),
    (
        "A white concrete truck is located on road",
        1,
        "item",
        "white",
        "concrete_truck",
        "located",
        "on",
        "road",
        None,
    ),
    (
        "A yellow tower crane is installed on concrete",
        1,
        "item",
        "yellow",
        "tower_crane",
        "installed",
        "on",
        "concrete",
        None,
    ),
    (
        "A worker is located on a pile of H-beam",
        1,
        "item",
        None,
        "worker",
        "located",
        "on",
        "pile of h-beam",
        None,
    ),
    (
        "A worker is located on a temporary building",
        1,
        "item",
        None,
        "worker",
        "located",
        "on",
        "temporary building",
        None,
    ),
    (
        "Two orange safety cones are located on road",
        2,
        "item",
        "orange",
        "safety_cone",
        "located",
        "on",
        "road",
        None,
    ),
    (
        "A blue water tank is located on soil",
        1,
        "item",
        "blue",
        "water_tank",
        "located",
        "on",
        "soil",
        None,
    ),
    (
        "A yellow forklift is lifting a H-beam",
        1,
        "item",
        "yellow",
        "forklift",
        "lifting",
        None,
        "h-beam",
        None,
    ),
    (
        "Two containers are installed on road",
        2,
        "item",
        None,
        "container",
        "installed",
        "on",
        "road",
        None,
    ),
    (
        "A red structure is installed on soil",
        1,
        "item",
        "red",
        "structure",
        "installed",
        "on",
        "soil",
        None,
    ),
    (
        "A concrete pillar is installed in concrete",
        1,
        "item",
        None,
        "concrete_pillar",
        "installed",
        "in",
        "concrete",
        None,
    ),
    (
        "Fourteen H-beams are embedded on soil",
        14,
        "item",
        None,
        "h_beam",
        "embedded",
        "on",
        "soil",
        None,
    ),
    (
        "A green crane truck is located on soil",
        1,
        "item",
        "green",
        "crane_truck",
        "located",
        "on",
        "soil",
        None,
    ),
]


@pytest.mark.parametrize("row", GOOD, ids=[r[0] for r in GOOD])
def test_hand_labelled_captions(row: Row) -> None:
    caption, count, unit, colour, resource, state, prep, target, dest = row
    res = parse_caption(caption)
    assert res.fields is not None, res.reason
    f = res.fields
    assert (f.count, f.count_unit, f.colour, f.resource) == (count, unit, colour, resource)
    assert (f.state, f.preposition, f.target, f.destination) == (state, prep, target, dest)
    assert f.repairs == []  # none of these needs a repair


def test_at_least_fifty_hand_labelled_rows() -> None:
    assert len(GOOD) >= 50
    assert len({r[0] for r in GOOD}) == len(GOOD)  # no accidental duplicates


REPAIRED: list[tuple[str, int | None, str, str, str | None, str | None, str]] = [
    # caption, count, resource, state, preposition, target, expected repair substring
    (
        "A blue tarpaulin is covering a red sturcture",
        1,
        "tarpaulin",
        "covering",
        None,
        "red structure",
        "typo:sturcture",
    ),
    (
        "A pile of rebara is installed on concrete",
        1,
        "rebar",
        "installed",
        "on",
        "concrete",
        "typo:rebara",
    ),
    ("A red truck is loaced", 1, "truck", "loaded", None, None, "typo:loaced"),
    ("wo workers are located on soil", 2, "worker", "located", "on", "soil", "typo:wo"),
    ("Fivr piles of rebars are located on soil", 5, "rebar", "located", "on", "soil", "typo:fivr"),
    ("A n orange excavator is located on soil", 1, "excavator", "located", "on", "soil", "phrase:"),
    ("An excavator is locate d on soil", 1, "excavator", "located", "on", "soil", "phrase:"),
    (
        "Ten concrete pipes are located on on concrete",
        10,
        "concrete_pipe",
        "located",
        "on",
        "concrete",
        "phrase:",
    ),
    ("A worker is located soil", 1, "worker", "located", None, "soil", "missing_preposition"),
    ("Fences are installed", None, "fence", "installed", None, None, "missing_target"),
    ("A white dump is located on soil", 1, "dump_truck", "located", "on", "soil", "truncated:dump"),
    (
        "A temporary is installed on soil",
        1,
        "temporary_building",
        "installed",
        "on",
        "soil",
        "truncated:temporary",
    ),
]


@pytest.mark.parametrize("row", REPAIRED, ids=[r[0] for r in REPAIRED])
def test_repairs_are_applied_and_recorded(
    row: tuple[str, int | None, str, str, str | None, str | None, str],
) -> None:
    caption, count, resource, state, prep, target, repair = row
    res = parse_caption(caption)
    assert res.fields is not None, res.reason
    f = res.fields
    assert (f.count, f.resource, f.state, f.preposition, f.target) == (
        count,
        resource,
        state,
        prep,
        target,
    )
    assert any(repair in r for r in f.repairs), f.repairs


FAILURES = [
    ("", "empty_caption"),
    ("   ", "empty_caption"),
    (
        "A pile of rebars is located on soilA black dust protection is located on soil",
        "glued_captions",
    ),
    ("Fences are installedTwenty six H-beams are installed on soil in a line", "glued_captions"),
    ("Fences are installed in a line H-beams are installed on soil", "multiple_clauses"),
    ("A white truck is located is located on soil", "multiple_clauses"),
    ("An orange is located on soil", "no_resource"),
    ("An orange exca", "no_resource"),
    ("Nine concrete piles are located on soil", "unknown_predicate:piles"),
    ("A white concrete truck Rebars are installed on soil", "unknown_predicate:rebars"),
    ("Two workers located are located on soil", "bad_target:are"),
    ("A worker is located on a banana", "bad_target:banana"),
    ("A worker", "no_predicate"),
    ("A worker is dancing on soil", "unknown_predicate:dancing"),
]


@pytest.mark.parametrize(("caption", "reason"), FAILURES)
def test_failures_report_a_reason_and_never_raise(caption: str, reason: str) -> None:
    res = parse_caption(caption)
    assert res.fields is None
    assert res.reason == reason


def test_categories_are_stable_and_complete() -> None:
    assert list(CATEGORY_NAMES) == sorted(CATEGORY_NAMES)  # deterministic ids
    assert [CATEGORY_IDS[n] for n in CATEGORY_NAMES] == list(range(1, len(CATEGORY_NAMES) + 1))
    assert {"worker", "excavator", "dump_truck", "h_beam"} <= set(CATEGORY_NAMES)


def _ds(captions: list[str]) -> Dataset:
    ds = Dataset(images=[ImageRecord(id=1, file_name="1.jpg", width=100, height=100)])
    for i, c in enumerate(captions, start=1):
        flags = [] if c.strip() else [FLAG_EMPTY_CAPTION]
        ds.annotations.append(
            AnnotationRecord(
                id=i,
                image_id=1,
                bbox=[i, i, 5, 5],
                area=25,
                caption=" ".join(c.split()),
                caption_raw=c,
                flags=flags,
            )
        )
    return ds


def test_apply_caption_fields_and_coverage() -> None:
    ds = _ds(
        [
            "A worker is located on soil",
            "Two workers are located on soil",
            "Fences are installed in a line",
            "",
            "Fences are installed in a line A concrete pipe is covered with",
            "wo workers are located on soil",
        ]
    )
    rep = apply_caption_fields(ds)
    assert (rep.n_annotations, rep.n_parsed, rep.n_unparsed, rep.n_empty) == (6, 4, 1, 1)
    assert rep.coverage == pytest.approx(4 / 5)  # empty captions are not counted as failures
    assert rep.failure_reasons == {"multiple_clauses": 1}
    assert rep.n_with_repairs == 1 and rep.n_with_count == 3
    assert rep.resource_counts == {"worker": 3, "fence": 1}
    a = ds.annotations
    assert a[0].category_id == CATEGORY_IDS["worker"] and a[0].fields is not None
    assert a[0].fields["count"] == 1
    assert FLAG_UNPARSED in a[4].flags and a[4].category_id is None and a[4].fields is None
    assert a[3].flags == [FLAG_EMPTY_CAPTION] and a[3].category_id is None
    assert validate_dataset(ds) == []


def test_apply_is_idempotent_and_clears_stale_results() -> None:
    ds = _ds(["A worker is located on soil", "garbage text here"])
    first = apply_caption_fields(ds).to_dict()
    snapshot = ds.to_dict()
    assert apply_caption_fields(ds).to_dict() == first
    assert ds.to_dict() == snapshot
    ds.annotations[1].caption = "A worker is located on soil"  # now parseable
    apply_caption_fields(ds)
    assert (
        FLAG_UNPARSED not in ds.annotations[1].flags and ds.annotations[1].category_id is not None
    )
