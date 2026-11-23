"""Codec and decoding-loop logic with hand-derived expectations (no model, no onnxruntime run)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from sitesight.serve import codec
from sitesight.serve.onnx_backend import OnnxFlorence, banned_by_ngram


def test_preprocess_matches_the_normalisation_formula() -> None:
    px = codec.preprocess(Image.new("RGB", (2, 2), (255, 0, 128)), size=4)
    assert px.shape == (1, 3, 4, 4) and px.dtype == np.float32
    # (value / 255 - mean) / std per channel
    assert px[0, 0, 0, 0] == pytest.approx((1.0 - 0.485) / 0.229, rel=1e-6)
    assert px[0, 1, 3, 3] == pytest.approx((0.0 - 0.456) / 0.224, rel=1e-6)
    assert px[0, 2, 1, 2] == pytest.approx((128 / 255 - 0.406) / 0.225, rel=1e-6)
    # a grayscale or RGBA input is converted, not passed through
    assert codec.preprocess(Image.new("L", (3, 5), 7), size=2).shape == (1, 3, 2, 2)
    assert codec.preprocess(Image.new("RGBA", (3, 5)), size=2).shape == (1, 3, 2, 2)


def test_prompts_use_the_florence_templates() -> None:
    assert codec.grounding_prompt("rebar") == "Locate the phrases in the caption: rebar"
    # x1 = 10 + 30 = 40, y1 = 20 + 40 = 60 on a 1000 x 1000 image (1 pixel per bin)
    assert (
        codec.region_prompt([10, 20, 30, 40], 1000, 1000)
        == "What does the region <loc_10><loc_20><loc_40><loc_60> describe?"
    )


def test_prompt_ids_put_the_image_placeholders_first() -> None:
    tok = SimpleNamespace(
        encode=lambda text, add_special_tokens: SimpleNamespace(ids=[0, len(text), 2])
    )
    assert codec.prompt_ids(tok, "ab", image_token_id=9, n_image_tokens=3) == [9, 9, 9, 0, 9, 2]
    # "<s>ab</s>" is 9 characters: the text is wrapped in bos/eos before tokenising
    assert codec.prompt_ids(tok, "", 9, 0) == [0, 7, 2]


def test_dequantize_uses_the_bin_centre_and_truncates() -> None:
    assert codec.dequantize_box([0, 0, 999, 999], 1000, 500) == [0, 0, 999, 499]
    # 2000 x 1000 image: 2 px per column bin, 1 px per row bin; (n + 0.5) * px, then int()
    assert codec.dequantize_box([100, 200, 300, 400], 2000, 1000) == [201, 200, 601, 400]


def test_boxes_come_from_complete_runs_of_four_location_tokens() -> None:
    def locs(*v: int) -> list[str]:
        return [f"<loc_{i}>" for i in v]

    tokens = ["rebar", *locs(100, 200, 300, 400), "</s>"]
    assert codec.boxes_from_tokens(tokens, 1000, 1000) == [[100, 200, 300, 400]]
    two = ["a", *locs(1, 2, 3, 4), "b", *locs(5, 6, 7, 8)]
    assert codec.boxes_from_tokens(two, 1000, 1000) == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert codec.boxes_from_tokens(["a", *locs(1, 2, 3), "</s>"], 1000, 1000) == []  # leftover
    assert codec.boxes_from_tokens(["a", *locs(1, 2, 3, 4, 5, 6)], 1000, 1000) == [[1, 2, 3, 4]]
    assert codec.loc_value("<loc_17>") == 17 and codec.loc_value("loc_17") is None


def test_ngram_ban_blocks_a_repeated_trigram_continuation() -> None:
    # history 1 2 3 1 2: with n = 3 the next token would repeat "1 2 3" if it were 3
    assert banned_by_ngram([1, 2, 3, 1, 2], 3) == {3}
    assert banned_by_ngram([1, 2, 3, 1, 2], 2) == {3}  # bigram (2, 3) seen after 2
    assert banned_by_ngram([1, 2, 1], 2) == {2}
    assert banned_by_ngram([1, 2, 3], 3) == set() and banned_by_ngram([1], 2) == set()
    assert banned_by_ngram([1, 2, 1, 2], 0) == set()


VOCAB = ["<s>", "<pad>", "</s>", "rebar", "<loc_100>", "<loc_200>", "<loc_300>", "<loc_400>"]


class FakeTokenizer:
    def id_to_token(self, i: int) -> str:
        return VOCAB[i]

    def token_to_id(self, t: str) -> int:
        return VOCAB.index(t)

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(VOCAB[i] for i in ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> Any:
        return SimpleNamespace(ids=[0, 3, 2])


class Scripted:
    """Stand-in for an InferenceSession: returns ``fn(feeds)`` and records the feeds."""

    def __init__(self, fn: Any) -> None:
        self.fn = fn
        self.feeds: list[dict[str, np.ndarray]] = []

    def run(self, _out: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds.append(feeds)
        return [self.fn(feeds)]


def make_backend(
    tmp_path: Path, steps: list[list[float]], **meta: Any
) -> tuple[OnnxFlorence, Scripted]:
    base = {
        "image_size": 4,
        "n_image_tokens": 2,
        "image_token_id": 7,
        "decoder_start_token_id": 2,
        "eos_token_id": 2,
        "forced_bos_token_id": None,
        "no_repeat_ngram_size": 0,
        **meta,
    }
    (tmp_path / "meta.json").write_text(json.dumps(base))
    dec = Scripted(lambda f: np.asarray([steps[f["decoder_input_ids"].shape[1] - 1]]))
    sessions = {
        "vision": Scripted(lambda f: np.zeros((1, 2, 3), dtype=np.float32)),
        "encoder": Scripted(lambda f: np.zeros((1, f["input_ids"].shape[1], 3), dtype=np.float32)),
        "decoder": dec,
    }
    return OnnxFlorence(tmp_path, FakeTokenizer(), sessions=sessions), dec


def lse(v: list[float]) -> float:
    return math.log(sum(math.exp(x) for x in v))


def test_greedy_loop_takes_argmax_stops_at_eos_and_scores_with_log_softmax(tmp_path: Path) -> None:
    steps = [
        [0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0],  # step 1 -> token 3
        [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # step 2 -> eos
        [9.0] * 8,  # never reached
    ]
    backend, dec = make_backend(tmp_path, steps)
    ids, lps = backend.generate_ids(np.zeros((1, 3, 4, 4), dtype=np.float32), [7, 7, 0, 3, 2], 10)
    assert ids == [3, 2]
    assert lps == pytest.approx([3.0 - lse(steps[0]), 5.0 - lse(steps[1])])
    # the decoder is fed the start token, then each chosen token (no cache: whole prefix each time)
    assert [f["decoder_input_ids"].tolist() for f in dec.feeds] == [[[2]], [[2, 3]]]


def test_greedy_loop_respects_the_length_cap(tmp_path: Path) -> None:
    steps = [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * 5
    backend, _ = make_backend(tmp_path, steps)
    ids, _ = backend.generate_ids(np.zeros((1, 3, 4, 4), dtype=np.float32), [7, 7, 0], 3)
    assert ids == [3, 3, 3]


def test_forced_bos_and_ngram_constraints_change_the_chosen_token(tmp_path: Path) -> None:
    steps = [[0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0]] * 6
    pv = np.zeros((1, 3, 4, 4), dtype=np.float32)
    forced, _ = make_backend(tmp_path, steps, forced_bos_token_id=0)
    ids, lps = forced.generate_ids(pv, [7, 7, 0], 2)
    assert ids[0] == 0 and lps[0] == pytest.approx(0.0)  # all other tokens masked: probability 1
    assert ids[1] == 3  # only the first step is forced
    # n = 2: history 2,3,3 already holds the bigram (3, 3), so a third 3 is banned and 4 is taken
    banned, _ = make_backend(
        tmp_path, [[0.0, 0.0, 0.0, 5.0, 4.0, 0.0, 0.0, 0.0]] * 6, no_repeat_ngram_size=2
    )
    ids, lps = banned.generate_ids(pv, [7, 7, 0], 4)
    assert ids == [3, 3, 4, 3]  # without the ban this would be 3, 3, 3, 3


def test_feature_cache_is_keyed_on_the_pixels(tmp_path: Path) -> None:
    backend, _ = make_backend(tmp_path, [[0.0, 0.0, 9.0] + [0.0] * 5])
    a = np.zeros((1, 3, 4, 4), dtype=np.float32)
    backend.generate_ids(a, [7, 7, 0], 1)
    backend.generate_ids(a.copy(), [7, 7, 0], 1)
    assert backend.n_feature_cache_hits == 1 and len(backend.vision.feeds) == 1
    backend.generate_ids(a + 1, [7, 7, 0], 1)
    assert backend.n_feature_cache_hits == 1 and len(backend.vision.feeds) == 2


def test_ground_and_describe_use_the_decoded_tokens(tmp_path: Path) -> None:
    def peak(token: int) -> list[float]:
        v = [0.0] * 8
        v[token] = 10.0
        return v

    # <loc_100> ... <loc_400> then eos
    steps = [peak(3), peak(4), peak(5), peak(6), peak(7), peak(2)]
    backend, _ = make_backend(tmp_path, steps)
    img = Image.new("RGB", (2000, 1000))
    boxes = backend.ground(img, "rebar")
    # bins 100,200,300,400 on 2000 x 1000: 2 px / 1 px per bin -> [201, 200, 601, 400] -> xywh
    assert [b for b, _ in boxes] == [[201.0, 200.0, 400.0, 200.0]]
    assert 0.0 < boxes[0][1] <= 1.0
    text, conf = backend.describe_region(img, [0, 0, 10, 10])
    assert text == "rebar" and 0.0 < conf <= 1.0  # location tokens are not part of the caption
