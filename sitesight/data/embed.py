"""DINOv2 image embeddings for near-duplicate / same-site grouping.

Needs ``torch`` and ``transformers`` (dependency group ``embed``); imported lazily so the rest
of the package works without them.

Choices worth being able to defend:

* **Model:** ``facebook/dinov2-small`` (22 M params, Apache-2.0). DINOv2 is trained with
  self-supervision to make features that are the same for the same scene and object, with no
  captions involved, which is exactly "do these two photos show the same place".
* **Whole frame, not a centre crop:** the default image processor centre-crops to a square
  and would throw away the left/right thirds of a 16:9 drone frame. We resize to 224 x 392
  (both multiples of the 14-px patch size) and keep the entire image. DINOv2 interpolates its
  position embeddings, so a non-square input is valid.
* **Descriptor:** the final-layer CLS token (a global summary of the image), L2-normalised, so
  the dot product of two descriptors is their cosine similarity.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from sitesight.data.dedup import l2_normalise

MODEL_ID = "facebook/dinov2-small"
INPUT_HEIGHT, INPUT_WIDTH = 224, 392  # multiples of the 14-px patch size, 16:9
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet statistics used by DINOv2
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Dinov2Embedder:
    name = "dinov2-small-cls"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        batch_size: int = 16,
        threads: int | None = None,
        device: str = "cpu",
    ):
        import torch
        from transformers import AutoModel

        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self._device = device
        self._model = AutoModel.from_pretrained(model_id).eval().to(device)
        self._batch = batch_size
        self.model_id = model_id

    def _prepare(self, path: Path) -> NDArray[np.float32]:
        with Image.open(path) as im:
            rgb = im.convert("RGB").resize((INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.BICUBIC)
        x = (np.asarray(rgb, dtype=np.float32) / 255.0 - _MEAN) / _STD
        return x.transpose(2, 0, 1)  # HWC -> CHW

    def embed(self, paths: Sequence[Path]) -> NDArray[np.float32]:
        out: list[NDArray[np.float32]] = []
        with self._torch.no_grad():
            for start in range(0, len(paths), self._batch):
                batch = np.stack([self._prepare(p) for p in paths[start : start + self._batch]])
                pixels = self._torch.from_numpy(batch).to(self._device)
                hidden = self._model(pixel_values=pixels).last_hidden_state
                out.append(hidden[:, 0].cpu().numpy())  # CLS token
        return l2_normalise(np.concatenate(out).astype(np.float32))
