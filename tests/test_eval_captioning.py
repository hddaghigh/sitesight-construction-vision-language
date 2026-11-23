"""Caption metric tests. Every expected value is derived by hand (formulas in the comments)."""

from __future__ import annotations

import math

import pytest

from sitesight.eval.captioning import (
    bleu,
    cider_d,
    field_accuracy,
    loose_fields,
    meteor_exact,
    meteor_exact_sentence,
    tokenize,
)


def test_tokenize() -> None:
    assert tokenize("A pile of H-beams, are located on soil.") == [
        "a",
        "pile",
        "of",
        "h-beams",
        "are",
        "located",
        "on",
        "soil",
    ]


def test_bleu_perfect_and_hand_example() -> None:
    same = bleu(["the cat sat on the mat"], [["the cat sat on the mat"]])
    assert all(v == pytest.approx(1.0) for v in same.values())
    # cand "the cat sat on the mat" vs ref "the cat is on the mat" (equal length -> BP = 1):
    # p1 = 5/6 ; p2 = 3/5 (the cat, on the, the mat) ; p3 = 1/4 (on the mat) ; p4 = 0/3.
    b = bleu(["the cat sat on the mat"], [["the cat is on the mat"]])
    assert b["bleu1"] == pytest.approx(5 / 6)
    assert b["bleu2"] == pytest.approx(math.sqrt(5 / 6 * 3 / 5))  # = sqrt(0.5)
    assert b["bleu3"] == pytest.approx((5 / 6 * 3 / 5 * 1 / 4) ** (1 / 3))  # = (1/8)^(1/3) = 0.5
    assert (
        b["bleu4"] == 0.0
    )  # an unmatched 4-gram order makes cumulative BLEU-4 zero (no smoothing)


def test_bleu_brevity_penalty_clipping_pooling_and_multi_reference() -> None:
    # c=2, r=6 -> BP = exp(1 - 6/2) = exp(-2); p1 = p2 = 1.
    short = bleu(["the cat"], [["the cat sat on the mat"]])
    assert short["bleu1"] == pytest.approx(math.exp(-2)) and short["bleu2"] == pytest.approx(
        math.exp(-2)
    )
    assert short["bleu3"] == 0.0  # no candidate trigram exists
    # clipping: "the" x3 against a reference with one "the": matches min(3, 1) = 1 of 3 -> p1 = 1/3, BP = 1.
    assert bleu(["the the the"], [["the cat"]])["bleu1"] == pytest.approx(1 / 3)
    # corpus pooling (NOT a mean of sentence scores): matches 2+0 over candidates 2+1 -> 2/3; c=3=r -> BP=1.
    pooled = bleu(["a b", "c"], [["a b"], ["d"]])
    assert pooled["bleu1"] == pytest.approx(2 / 3)
    # multi-reference: the closest-length reference sets r; clipping uses the max count over references.
    multi = bleu(
        ["a b c"], [["a b c d e f", "a b"]]
    )  # |ref| 6 and 2, candidate 3 -> closest is 2 (|3-2|=1 < 3)
    assert multi["bleu1"] == pytest.approx(
        1.0
    )  # c=3 > r=2 -> BP=1, all 3 unigrams matched (a,b from ref1/2, c from ref1)
    with pytest.raises(ValueError):
        bleu(["a"], [])


def test_cider_d_hand_examples() -> None:
    # One-word captions: only unigram vectors exist; identical -> cosine 1 for n=1 and 0 for
    # n=2..4 -> score = 10 * (1 + 0 + 0 + 0) / 4 = 2.5.
    assert cider_d(["cat", "dog"], [["cat"], ["dog"]]) == pytest.approx(2.5)
    # Two-word identical captions: n=1 and n=2 both give 1 -> 10 * (1 + 1) / 4 = 5.0.
    assert cider_d(["red cat", "blue dog"], [["red cat"], ["blue dog"]]) == pytest.approx(5.0)
    # Nothing in common -> 0.
    assert cider_d(["green frog", "pink pig"], [["red cat"], ["blue dog"]]) == 0.0
    # Partial: image 1 "red dog" vs "red cat": unigram cosine = L^2 / (sqrt(2) L * sqrt(2) L) = 0.5,
    # bigrams differ -> 10 * 0.5 / 4 = 1.25 ; image 2 identical = 5.0 ; corpus mean = 3.125.
    assert cider_d(["red dog", "blue dog"], [["red cat"], ["blue dog"]]) == pytest.approx(3.125)
    # Duplicated references change nothing.
    assert cider_d(
        ["red cat", "blue dog"], [["red cat", "red cat"], ["blue dog"]]
    ) == pytest.approx(5.0)


def test_cider_d_length_penalty_hand_formula() -> None:
    # cand "red cat" vs ref "red cat today": unigram cosine = 2 / (sqrt2 * sqrt3) = 2/sqrt(6);
    # bigram cosine = 1 / (1 * sqrt2) = 1/sqrt(2); length (bigram count) 1 vs 2 -> d = -1,
    # penalty exp(-1 / 72); score = 10 * (2/sqrt6 + 1/sqrt2) * exp(-1/72) / 4.
    got = cider_d(["red cat", "blue dog"], [["red cat today"], ["blue dog"]])
    first = 10 * (2 / math.sqrt(6) + 1 / math.sqrt(2)) * math.exp(-1 / 72) / 4
    assert got == pytest.approx((first + 5.0) / 2)


def test_cider_d_idf_discounts_words_that_appear_everywhere() -> None:
    # "the" is in the references of both images, so its weight is log(2) - log(2) = 0 and
    # sharing only "the" earns nothing: image 1 ("the dog" vs "the cat") = 0, image 2 = 5.0.
    assert cider_d(["the dog", "the dog"], [["the cat"], ["the dog"]]) == pytest.approx(2.5)


def test_meteor_exact_hand_examples() -> None:
    # Identical, 6 words: P=R=1 -> Fmean=1; 1 chunk -> penalty 0.5 * (1/6)^3.
    assert meteor_exact_sentence(
        "the cat sat on the mat", ["the cat sat on the mat"]
    ) == pytest.approx(1 - 0.5 * (1 / 6) ** 3)
    # "the cat sat" vs "the cat is on the mat": m=2, P=2/3, R=2/6=1/3 -> Fmean = PR/(0.9P+0.1R)
    # = (2/9) / (0.6 + 1/30) ; one chunk -> penalty 0.5 * (1/2)^3 = 0.0625.
    p, r = 2 / 3, 1 / 3
    assert meteor_exact_sentence("the cat sat", ["the cat is on the mat"]) == pytest.approx(
        p * r / (0.9 * p + 0.1 * r) * (1 - 0.0625)
    )
    # Reordering fragments the match: "cat the" vs "the cat": m=2, P=R=1, chunks=2 -> penalty 0.5 -> 0.5.
    assert meteor_exact_sentence("cat the", ["the cat"]) == pytest.approx(0.5)
    # Repeated token: "the the" vs "the cat": one match (a reference word is used once):
    # P=R=1/2 -> Fmean=.5 ; chunks=1,m=1 -> penalty .5 -> 0.25.
    assert meteor_exact_sentence("the the", ["the cat"]) == pytest.approx(0.25)
    assert meteor_exact_sentence("zzz", ["the cat"]) == 0.0
    # Best reference wins; corpus value is the mean over sentences.
    assert meteor_exact_sentence("cat the", ["the cat", "cat the"]) == pytest.approx(
        1 - 0.5 * (1 / 2) ** 3
    )
    assert meteor_exact(["cat the", "zzz"], [["the cat"], ["x"]]) == pytest.approx(0.25)


def test_loose_fields_on_free_text() -> None:
    f = loose_fields("a yellow excavator on the ground")
    assert (f.resource, f.colour, f.count) == ("excavator", "yellow", 1)
    f = loose_fields("Two orange dump trucks parked near a fence")
    assert (f.resource, f.colour, f.count) == (
        "dump_truck",
        "orange",
        2,
    )  # leftmost resource phrase
    f = loose_fields("construction site with some machinery")
    assert (f.resource, f.colour, f.count) == (None, None, None)
    f = loose_fields("3 workers")
    assert (f.resource, f.colour, f.count) == ("worker", None, 3)
    assert loose_fields("twenty six workers").count == 26
    assert loose_fields("A pile of H-beams").resource == "h_beam"
    assert loose_fields("a grey truck").colour == "gray"


def test_field_accuracy_hand_example() -> None:
    preds = ["A worker is located on soil", "a yellow excavator digging", "some stuff"]
    gt = [
        {"resource": "worker", "count": 1, "colour": None, "state": "located"},
        {"resource": "excavator", "count": 1, "colour": "yellow", "state": "digging"},
        {"resource": "worker", "count": 2, "colour": None, "state": "located"},
    ]
    r = field_accuracy(preds, gt)
    assert r["resource_acc_loose"] == pytest.approx(2 / 3)  # third prediction names nothing
    assert r["colour_acc_loose"] == 1.0 and r["colour_n_with_ground_truth"] == 1
    assert r["count_acc_loose"] == pytest.approx(2 / 3) and r["count_n_with_ground_truth"] == 3
    assert r["strict_parse_rate"] == pytest.approx(2 / 3)  # "some stuff" does not fit the template
    assert r["resource_acc_strict"] == pytest.approx(2 / 3) and r[
        "state_acc_strict"
    ] == pytest.approx(2 / 3)
    assert r["colour_acc_strict"] == 1.0 and r["count_acc_strict"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        field_accuracy(["a"], [])
