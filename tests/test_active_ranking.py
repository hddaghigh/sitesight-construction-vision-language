"""Acquisition strategies and curve summaries: hand-derived expectations."""

from __future__ import annotations

import numpy as np
import pytest

from sitesight.active.ranking import (
    image_uncertainty,
    kcenter_greedy,
    labels_to_reach,
    mean_std,
    select,
    summarise_curves,
)

POOL = [f"{i}.jpg" for i in range(10)]


def test_image_uncertainty_is_one_minus_mean_confidence() -> None:
    u = image_uncertainty({"a": [0.9, 0.7], "b": [0.5], "c": []})
    assert u == {"a": pytest.approx(0.2), "b": pytest.approx(0.5)}  # empty images are not scored


def test_kcenter_picks_the_farthest_point_first_and_spreads() -> None:
    # unit vectors at 0, 5, 90, 95 degrees; row 0 is already labelled.
    # distance to it: row1 1-cos5 = 0.0038, row2 1.0, row3 1-cos95 = 1.087
    ang = np.radians([0, 5, 90, 95])
    emb = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    assert kcenter_greedy(emb, [1, 2, 3], 1, [0]) == [3]
    # after taking row 3: row2 is 1-cos5 = 0.0038 from it, row1 keeps 0.0038 from the labelled point;
    # the two are equal, so the lower position (row 1) wins
    assert kcenter_greedy(emb, [1, 2, 3], 2, [0]) == [3, 1]
    assert kcenter_greedy(emb, [1, 2, 3], 0, [0]) == []
    assert kcenter_greedy(emb, [], 3, [0]) == []


def test_kcenter_without_labelled_points_starts_from_unit_distance() -> None:
    emb = np.eye(3)  # mutually orthogonal: all distances 1, so ties go to the first candidate
    assert kcenter_greedy(emb, [2, 0, 1], 3, []) == [2, 0, 1]


def test_random_is_seeded_excludes_labelled_and_caps_k() -> None:
    a = select("random", POOL, ["0.jpg"], 4, seed=3)
    assert a == select("random", list(reversed(POOL)), ["0.jpg"], 4, seed=3)  # order-independent
    assert len(set(a)) == 4 and "0.jpg" not in a
    assert select("random", POOL, ["0.jpg"], 4, seed=4) != a
    assert len(select("random", POOL, POOL[:8], 5, seed=0)) == 2  # only two images left


def test_uncertainty_takes_top_k_with_name_tiebreak() -> None:
    u = {n: 0.1 for n in POOL}
    u["7.jpg"], u["3.jpg"], u["5.jpg"] = 0.9, 0.8, 0.8
    assert select("uncertainty", POOL, [], 3, seed=0, uncertainty=u) == ["7.jpg", "3.jpg", "5.jpg"]
    # a labelled image is never re-selected even if it is the most uncertain
    assert select("uncertainty", POOL, ["7.jpg"], 1, seed=0, uncertainty=u) == ["3.jpg"]


def test_uncertainty_diversity_avoids_near_duplicates() -> None:
    # 3.jpg and 4.jpg are the two most uncertain but identical; 5.jpg is third and different
    emb = {n: np.array([1.0, 0.0]) for n in POOL}
    emb["5.jpg"] = np.array([0.0, 1.0])
    u = {n: 0.0 for n in POOL}
    u["3.jpg"], u["4.jpg"], u["5.jpg"] = 0.9, 0.89, 0.5
    chosen = select("uncertainty_diversity", POOL, [], 2, seed=0, uncertainty=u,
                    embeddings=emb, candidate_multiplier=2)  # fmt: skip
    assert chosen == ["3.jpg", "5.jpg"]  # plain uncertainty would take 3 and 4
    assert select("uncertainty", POOL, [], 2, seed=0, uncertainty=u) == ["3.jpg", "4.jpg"]


def test_select_validates_inputs() -> None:
    with pytest.raises(ValueError, match="needs uncertainty"):
        select("uncertainty", POOL, [], 2, seed=0)
    with pytest.raises(ValueError, match="needs embeddings"):
        select("uncertainty_diversity", POOL, [], 2, seed=0, uncertainty={})
    with pytest.raises(ValueError, match="unknown strategy"):
        select("magic", POOL, [], 2, seed=0)


def test_labels_to_reach_interpolates_and_never_extrapolates() -> None:
    curve = [(100.0, 50.0), (50.0, 30.0), (200.0, 70.0)]  # unsorted on purpose
    assert labels_to_reach(curve, 40.0) == pytest.approx(75.0)  # halfway between 50 and 100
    assert labels_to_reach(curve, 30.0) == 50.0  # already reached at the first point
    assert labels_to_reach(curve, 50.0) == pytest.approx(100.0)
    assert labels_to_reach(curve, 70.1) is None and labels_to_reach([], 1.0) is None


def test_mean_std_uses_sample_std() -> None:
    m, sd = mean_std([2.0, 4.0])
    assert m == 3.0 and sd == pytest.approx(2**0.5)
    assert mean_std([5.0]) == (5.0, 0.0)


def test_summarise_curves_per_strategy_and_seed() -> None:
    rows = []
    for seed, shift in ((0, 0.0), (1, 10.0)):
        for strat, gain in (("random", 1.0), ("uncertainty", 2.0)):
            for n in (10, 20, 40):
                rows.append({"strategy": strat, "seed": seed, "n_images": n,
                             "score": gain * n / 2 + shift})  # fmt: skip
    s = summarise_curves(rows, {0: 40.0, 1: 50.0}, fraction=0.5)
    # random at n=10: scores 5 (seed 0) and 15 (seed 1) -> mean 10, sample std sqrt(50)
    assert s["random"]["curve"][0] == {
        "n_images": 10, "mean": 10.0, "std": pytest.approx(50**0.5), "seeds": 2,
    }  # fmt: skip
    # seed 0, target 20: random scores 5, 10, 20 -> reached at n=40; uncertainty 10, 20, 40 -> n=20
    assert s["random"]["labels_to_reach"][0] == pytest.approx(40.0)
    assert s["uncertainty"]["labels_to_reach"][0] == pytest.approx(20.0)
    # seed 1, target 25: random scores 15, 20, 30 -> 20 + 20 * (5 / 10) = 30
    assert s["random"]["labels_to_reach"][1] == pytest.approx(30.0)
    # a curve that never reaches the target reports None
    assert (
        summarise_curves(rows, {0: 1000.0, 1: 1000.0}, 1.0)["random"]["labels_to_reach"][0] is None
    )


def test_budgets_round_and_must_increase() -> None:
    from sitesight.active.simulate import budgets

    assert budgets(1000, [0.05, 0.1, 0.2, 0.4]) == [50, 100, 200, 400]
    assert budgets(10, [0.04, 0.5, 1.0]) == [1, 5, 10]  # at least one image; capped at the pool
    with pytest.raises(ValueError, match="non-increasing"):
        budgets(10, [0.04, 0.05])  # both round to 1 image


def test_field_score_is_the_mean_of_three_strict_accuracies() -> None:
    from sitesight.active.simulate import field_score

    acc = {"resource_acc_strict": 0.9, "colour_acc_strict": 0.6, "count_acc_strict": 0.3,
           "state_acc_strict": 1.0}  # fmt: skip
    assert field_score(acc) == pytest.approx(0.6)  # state is not part of the headline


def test_done_keys_reads_back_written_rows(tmp_path: object) -> None:
    from pathlib import Path

    from sitesight.active.simulate import done_keys

    p = Path(str(tmp_path)) / "r.jsonl"
    assert done_keys(p) == {}
    p.write_text('{"seed":1,"strategy":"random","round":2,"score":0.5}\n\n', encoding="utf-8")
    assert done_keys(p)[(1, "random", 2)]["score"] == 0.5


def test_uncertainty_diversity_only_considers_the_top_candidates() -> None:
    emb = {n: np.array([1.0, 0.0]) for n in POOL}
    emb["9.jpg"] = np.array([0.0, 1.0])  # very different, but the model is sure about it
    u = {n: 0.0 for n in POOL}
    u["3.jpg"], u["4.jpg"] = 0.9, 0.8
    got = select("uncertainty_diversity", POOL, [], 2, seed=0, uncertainty=u,
                 embeddings=emb, candidate_multiplier=1)  # fmt: skip
    assert got == ["3.jpg", "4.jpg"]  # 9.jpg is outside the 1 * k = 2 candidates
