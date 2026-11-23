"""Florence-2 on ONNX Runtime: the two tasks of ``baselines.florence.FlorenceRunner``.

``generate_ids`` is the greedy decoding loop that ``transformers.generate`` normally provides:

    y_1 = argmax softmax(f(image, prompt, <start>)),  y_t = argmax softmax(f(image, prompt, y_<t))

stopping at ``eos`` or ``max_new_tokens``. Each token's log-probability is
``log softmax(logits)[y_t]`` AFTER the same constraints ``generate`` applies (``forced_bos``,
``no_repeat_ngram_size``), so confidences are the ones the PyTorch path would report. The vision
features are cached per exact pixel tensor, because one image is queried for many regions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sitesight.baselines.prompts import (
    box_scores_from_tokens,
    sequence_score,
    strip_loc_tokens,
)
from sitesight.serve import codec


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    return np.asarray(x - (m + np.log(np.exp(x - m).sum())))


def banned_by_ngram(generated: list[int], n: int) -> set[int]:
    """Tokens that would complete an n-gram already in ``generated`` (as ``generate`` does)."""
    if n <= 0 or len(generated) < n:
        return set()
    prefix = tuple(generated[len(generated) - n + 1 :])
    banned: set[int] = set()
    for i in range(len(generated) - n + 1):
        if tuple(generated[i : i + n - 1]) == prefix:
            banned.add(generated[i + n - 1])
    return banned


class OnnxFlorence:
    def __init__(
        self,
        model_dir: Path,
        tokenizer: Any,
        *,
        providers: list[str] | None = None,
        threads: int | None = None,
        sessions: dict[str, Any] | None = None,
    ) -> None:
        """``sessions`` lets tests inject stand-ins for the three ``InferenceSession`` objects."""
        self.model_dir = Path(model_dir)
        self.meta = json.loads((self.model_dir / "meta.json").read_text(encoding="utf-8"))
        self.tokenizer = tokenizer
        if sessions is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            if threads:
                opts.intra_op_num_threads = threads
            sessions = {
                name: ort.InferenceSession(
                    str(self.model_dir / f"{name}.onnx"),
                    opts,
                    providers=providers or ["CPUExecutionProvider"],
                )
                for name in ("vision", "encoder", "decoder")
            }
        self.vision, self.encoder, self.decoder = (
            sessions["vision"],
            sessions["encoder"],
            sessions["decoder"],
        )
        self._feat_key: str | None = None
        self._feat_val: np.ndarray | None = None
        self.n_feature_cache_hits = 0

    def info(self) -> dict[str, Any]:
        import onnxruntime as ort

        return {
            "engine": "onnxruntime",
            "onnxruntime": ort.__version__,
            "precision": self.meta.get("precision", "fp32"),
            "model_dir": self.model_dir.name,
        }

    def image_features(self, pixel_values: np.ndarray) -> np.ndarray:
        key = hashlib.md5(pixel_values.tobytes()).hexdigest()
        if key == self._feat_key and self._feat_val is not None:
            self.n_feature_cache_hits += 1
            return self._feat_val
        feats = self.vision.run(None, {"pixel_values": pixel_values})[0]
        self._feat_key, self._feat_val = key, feats
        return np.asarray(feats)

    def generate_ids(
        self, pixel_values: np.ndarray, input_ids: list[int], max_new_tokens: int
    ) -> tuple[list[int], list[float]]:
        """Greedy decoding; returns (generated ids incl. the final eos, their log-probabilities)."""
        meta = self.meta
        ids = np.asarray([input_ids], dtype=np.int64)
        mask = np.ones_like(ids)
        enc = self.encoder.run(
            None,
            {
                "input_ids": ids,
                "image_features": self.image_features(pixel_values),
                "attention_mask": mask,
            },
        )[0]
        eos = int(meta["eos_token_id"])
        forced = meta.get("forced_bos_token_id")
        ngram = int(meta.get("no_repeat_ngram_size") or 0)
        prev = [int(meta["decoder_start_token_id"])]
        out: list[int] = []
        lps: list[float] = []
        for step in range(max_new_tokens):
            logits = self.decoder.run(
                None,
                {
                    "decoder_input_ids": np.asarray([prev], dtype=np.int64),
                    "encoder_hidden_states": enc,
                    "encoder_attention_mask": mask,
                },
            )[0][0].astype(np.float64)
            if step == 0 and forced is not None:
                keep = logits[forced]
                logits = np.full_like(logits, -np.inf)
                logits[forced] = keep
            for t in banned_by_ngram(prev, ngram):
                logits[t] = -np.inf
            logp = _log_softmax(logits)
            tok = int(np.argmax(logp))
            out.append(tok)
            lps.append(float(logp[tok]))
            prev.append(tok)
            if tok == eos:
                break
        return out, lps

    def _run(
        self, image: Image.Image, text: str, max_new_tokens: int
    ) -> tuple[list[str], list[float]]:
        pv = codec.preprocess(image, int(self.meta["image_size"]))
        ids = codec.prompt_ids(
            self.tokenizer, text, int(self.meta["image_token_id"]), int(self.meta["n_image_tokens"])
        )
        gen, lps = self.generate_ids(pv, ids, max_new_tokens)
        tokens = [self.tokenizer.id_to_token(i) for i in gen]
        return tokens, lps

    def ground(self, image: Image.Image, phrase: str) -> list[tuple[list[float], float]]:
        """Boxes ``[x, y, w, h]`` with confidences for one phrase (same contract as the runner)."""
        w, h = image.size
        tokens, lps = self._run(image, codec.grounding_prompt(phrase), 128)
        boxes = codec.boxes_from_tokens(tokens, w, h)
        scores, _ = box_scores_from_tokens(tokens, lps)
        if len(scores) != len(boxes):
            loc = [lp for t, lp in zip(tokens, lps, strict=True) if codec.loc_value(t) is not None]
            scores = [sequence_score(loc)] * len(boxes)
        return [
            ([float(x0), float(y0), float(x1 - x0), float(y1 - y0)], s)
            for (x0, y0, x1, y1), s in zip(boxes, scores, strict=True)
            if x1 > x0 and y1 > y0
        ]

    def describe_region(self, image: Image.Image, bbox: list[float]) -> tuple[str, float]:
        """Description of what is inside ``bbox`` (``[x, y, w, h]``) and its confidence."""
        w, h = image.size
        tokens, lps = self._run(image, codec.region_prompt(bbox, w, h), 48)
        words = [
            (t, lp)
            for t, lp in zip(tokens, lps, strict=True)
            if codec.loc_value(t) is None and t not in ("</s>", "<s>", "<pad>")
        ]
        text = self.tokenizer.decode(
            [self.tokenizer.token_to_id(t) for t, _ in words], skip_special_tokens=True
        )
        return strip_loc_tokens(text), sequence_score([lp for _, lp in words])
