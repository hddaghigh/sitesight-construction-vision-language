"""Pure helper tests for the baselines (no torch needed)."""

from __future__ import annotations

import math

import pytest

from sitesight.baselines.prompts import (
    QUERY_PHRASES,
    box_scores_from_tokens,
    box_to_loc_tokens,
    owl_queries,
    sequence_score,
)
from sitesight.data.caption_fields import CATEGORY_NAMES


def test_prompts_cover_every_class_and_are_fixed() -> None:
    assert list(QUERY_PHRASES) == list(CATEGORY_NAMES) and len(owl_queries()) == len(CATEGORY_NAMES)
    assert QUERY_PHRASES["h_beam"] == "H-beam" and QUERY_PHRASES["dump_truck"] == "dump truck"
    assert QUERY_PHRASES["drilling_rig"] == "drilling rig"
    assert owl_queries()[CATEGORY_NAMES.index("worker")] == "a photo of a worker"


def test_box_to_loc_tokens_hand_examples() -> None:
    # 960x540 image: one bin is 0.96 px wide and 0.54 px tall.
    # [100.5, 54.3, 192, 108.4]: x0 = 100.5/0.96 = 104.69 -> 104 ; y0 = 54.3/0.54 = 100.56 -> 100 ;
    # x1 = 292.5/0.96 = 304.69 -> 304 ; y1 = 162.7/0.54 = 301.3 -> 301.
    assert (
        box_to_loc_tokens([100.5, 54.3, 192.0, 108.4], 960, 540)
        == "<loc_104><loc_100><loc_304><loc_301>"
    )
    # The far edge (960 / 0.96 = 1000) is clamped into the last bin, 999.
    assert box_to_loc_tokens([0, 0, 960, 540], 960, 540) == "<loc_0><loc_0><loc_999><loc_999>"
    assert (
        box_to_loc_tokens([-5, -5, 10, 10], 960, 540) == "<loc_0><loc_0><loc_5><loc_9>"
    )  # clamp below 0


def test_box_scores_geometric_mean_of_four_loc_tokens() -> None:
    toks = ["bird", "<loc_1>", "<loc_2>", "<loc_3>", "<loc_4>", "</s>"]
    lps = [-1.0, -0.1, -0.2, -0.3, -0.4, -0.01]
    scores, left = box_scores_from_tokens(toks, lps)
    assert scores == [pytest.approx(math.exp(-0.25))] and left == 0  # mean(-.1,-.2,-.3,-.4) = -.25
    # two boxes back to back, then a text token, then a second phrase with one box
    toks2 = ["a"] + [f"<loc_{i}>" for i in range(8)] + ["b"] + [f"<loc_{i}>" for i in range(4)]
    lps2 = [-9.0] + [-0.1] * 4 + [-0.5] * 4 + [-9.0] + [-0.2] * 4
    s2, l2 = box_scores_from_tokens(toks2, lps2)
    assert (
        s2
        == [
            pytest.approx(math.exp(-0.1)),
            pytest.approx(math.exp(-0.5)),
            pytest.approx(math.exp(-0.2)),
        ]
        and l2 == 0
    )
    # a run of 5 loc tokens is one box plus one stray token
    s3, l3 = box_scores_from_tokens(["<loc_1>"] * 5, [-0.1] * 5)
    assert len(s3) == 1 and l3 == 1
    assert box_scores_from_tokens([], []) == ([], 0)
    with pytest.raises(ValueError):
        box_scores_from_tokens(["a"], [])


def test_sequence_score() -> None:
    assert sequence_score([-0.5, -1.5]) == pytest.approx(math.exp(-1.0))
    assert sequence_score([0.0, 0.0]) == 1.0 and sequence_score([]) == 0.0
