"""Florence-2 (Microsoft, MIT) zero-shot phrase grounding and region description.

Uses the native ``transformers`` implementation published as ``florence-community/Florence-2-base``
(a conversion of Microsoft's weights): no ``trust_remote_code``, so no third-party code is
executed. Decoding is **greedy** (``num_beams=1``) so every generated token has an exact
conditional probability ``p(token | previous tokens, image, prompt)``, which we turn into the
confidence scores that calibration (ECE) needs. (Beam search would score whole beams, not tokens.)

Two tasks, both with prompts fixed in advance:

* ``<CAPTION_TO_PHRASE_GROUNDING>`` + the class phrase: the model writes the phrase followed by
  ``<loc_x0><loc_y0><loc_x1><loc_y1>`` per box it finds. Box confidence = geometric-mean token
  probability of its four location tokens.
* ``<REGION_TO_DESCRIPTION>`` + the ground-truth box as four location tokens: the model writes a
  description of what is inside. Confidence = geometric-mean probability of the text tokens.
"""

from __future__ import annotations

import hashlib
from typing import Any

from PIL import Image

from sitesight.baselines.prompts import (
    FLORENCE_GROUNDING_TASK,
    FLORENCE_REGION_TASK,
    box_scores_from_tokens,
    box_to_loc_tokens,
    sequence_score,
    strip_loc_tokens,
)

MODEL_ID = "florence-community/Florence-2-base"


class FlorenceRunner:
    def __init__(
        self, model_id: str = MODEL_ID, device: str = "cpu", adapter: str | None = None
    ) -> None:
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        self._torch = torch
        self.device = device
        self.model_id = model_id
        self.adapter = adapter
        self.processor = AutoProcessor.from_pretrained(model_id)
        model = Florence2ForConditionalGeneration.from_pretrained(model_id)
        if adapter:  # LoRA weights from sitesight.train.lora, folded into the base weights
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.eval().to(device)
        # grounding outputs whose box count did not match the token groups (reported, never hidden)
        self.n_score_fallbacks = 0
        self._install_image_feature_cache()

    def _install_image_feature_cache(self) -> None:
        """Compute the (slow) vision-tower features once per image and reuse them for every query.

        On CPU the image encoder is ~80% of the cost of one query (about 45 of 56 seconds on the
        development laptop) while the text side is cheap, and we ask several questions about the
        same image. The cache key is a hash of the exact pixel tensor, so a different image or a
        different preprocessing can never receive stale features; results are identical to the
        uncached model (checked in tests/test_baselines_models.py).
        """
        inner = self.model.model
        original = inner.get_image_features
        self._feat_key: str | None = None
        self._feat_val: Any = None
        self.n_feature_cache_hits = 0

        def cached(pixel_values: Any, *args: Any, **kwargs: Any) -> Any:
            key = hashlib.md5(pixel_values.detach().cpu().numpy().tobytes()).hexdigest()
            if key == self._feat_key:
                self.n_feature_cache_hits += 1
            else:
                self._feat_val = original(pixel_values, *args, **kwargs)
                self._feat_key = key
            return self._feat_val

        inner.get_image_features = cached

    def _generate(
        self, image: Image.Image, text: str, max_new_tokens: int
    ) -> tuple[str, list[str], list[float]]:
        """Greedy generation. Returns (decoded text, generated tokens, log-prob of each token)."""
        torch = self._torch
        inputs: Any = self.processor(text=text, images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        seqs = out.sequences
        trans = self.model.compute_transition_scores(seqs, out.scores, normalize_logits=True)
        n = trans.shape[1]
        ids = seqs[0, -n:].tolist()
        tokens = self.processor.tokenizer.convert_ids_to_tokens(ids)
        decoded = self.processor.batch_decode(seqs, skip_special_tokens=False)[0]
        return decoded, tokens, [float(v) for v in trans[0].tolist()]

    def ground(self, image: Image.Image, phrase: str) -> list[tuple[list[float], float]]:
        """Boxes ``[x, y, w, h]`` with confidences for one phrase."""
        w, h = image.size
        text, tokens, lps = self._generate(image, FLORENCE_GROUNDING_TASK + phrase, 128)
        parsed = self.processor.post_process_generation(
            text, task=FLORENCE_GROUNDING_TASK, image_size=(w, h)
        )[FLORENCE_GROUNDING_TASK]
        boxes = parsed["bboxes"]
        scores, _ = box_scores_from_tokens(tokens, lps)
        if len(scores) != len(boxes):  # should not happen; stay honest if it does
            self.n_score_fallbacks += 1
            loc = [lp for t, lp in zip(tokens, lps, strict=True) if t.startswith("<loc_")]
            scores = [sequence_score(loc)] * len(boxes)
        return [
            ([float(x0), float(y0), float(x1 - x0), float(y1 - y0)], s)
            for (x0, y0, x1, y1), s in zip(boxes, scores, strict=True)
            if x1 > x0 and y1 > y0
        ]

    def describe_region(self, image: Image.Image, bbox: list[float]) -> tuple[str, float]:
        """Description of what is inside ``bbox`` (``[x, y, w, h]``) and its confidence."""
        w, h = image.size
        text, tokens, lps = self._generate(
            image, FLORENCE_REGION_TASK + box_to_loc_tokens(bbox, w, h), 48
        )
        caption = self.processor.post_process_generation(
            text, task=FLORENCE_REGION_TASK, image_size=(w, h)
        )[FLORENCE_REGION_TASK]
        # The model echoes the region as trailing <loc_N> tokens after the description; they are
        # not part of the caption and must not count towards its confidence either.
        special = set(self.processor.tokenizer.all_special_tokens)
        word_lps = [
            lp
            for t, lp in zip(tokens, lps, strict=True)
            if t not in special and not t.startswith("<loc_")
        ]
        return strip_loc_tokens(str(caption)), sequence_score(word_lps)
