"""OWLv2 open-vocabulary detector as a zero-shot grounding baseline.

OWLv2 (Google, Apache-2.0) embeds the image once into ~3,600 candidate boxes and scores each
against every text query, so querying all 31 classes costs about the same as querying one.
For every (box, query) pair the model outputs a logit; ``score = sigmoid(logit)`` in [0, 1].

Choices (fixed in advance, see :mod:`sitesight.baselines.prompts`):

* **Per-class scoring, no argmax.** The library's post-processor gives each box only its single
  best query; we keep every (box, class) pair above ``min_score`` so a box that looks like both
  "truck" and "dump truck" can be right for either (each class is evaluated independently).
* **Per-class non-maximum suppression** at IoU 0.5 and at most ``max_per_class`` boxes per class
  and image: OWL emits many overlapping boxes for one object, which would otherwise count as
  false positives.
* The processor pads the image to a square (at the bottom/right) and resizes to 960 px; box
  coordinates come back normalised to that square, so multiplying by the longer image side
  gives original pixel coordinates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from sitesight.eval.boxes import nms

MODEL_ID = "google/owlv2-base-patch16-ensemble"


class Owlv2Runner:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu") -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self._torch = torch
        self.device = device
        self.model_id = model_id
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).eval().to(device)

    def detect(
        self,
        image: Image.Image,
        queries: list[str],
        min_score: float = 0.01,
        nms_iou: float = 0.5,
        max_per_class: int = 100,
    ) -> list[tuple[int, list[float], float]]:
        """Return ``(query index, [x, y, w, h], score)`` for every kept detection."""
        torch = self._torch
        w, h = image.size
        inputs: Any = self.processor(text=[queries], images=image, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        scores = torch.sigmoid(out.logits[0]).cpu().numpy()  # (num_boxes, num_queries)
        cxcywh = out.pred_boxes[0].cpu().numpy()  # normalised to the padded square
        side = float(max(w, h))
        x0 = np.clip((cxcywh[:, 0] - cxcywh[:, 2] / 2) * side, 0, w)
        y0 = np.clip((cxcywh[:, 1] - cxcywh[:, 3] / 2) * side, 0, h)
        x1 = np.clip((cxcywh[:, 0] + cxcywh[:, 2] / 2) * side, 0, w)
        y1 = np.clip((cxcywh[:, 1] + cxcywh[:, 3] / 2) * side, 0, h)
        boxes = np.stack([x0, y0, x1 - x0, y1 - y0], axis=1)
        valid = (boxes[:, 2] > 1) & (boxes[:, 3] > 1)  # drop boxes that clipped away
        results: list[tuple[int, list[float], float]] = []
        for qi in range(scores.shape[1]):
            idx = np.nonzero((scores[:, qi] >= min_score) & valid)[0]
            if idx.size == 0:
                continue
            keep = nms(boxes[idx].tolist(), scores[idx, qi].tolist(), nms_iou)[:max_per_class]
            for k in keep:
                j = int(idx[k])
                results.append((qi, [float(v) for v in boxes[j]], float(scores[j, qi])))
        return results
