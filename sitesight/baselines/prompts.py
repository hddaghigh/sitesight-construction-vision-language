"""Pre-registered prompts and the pure helpers shared by the baseline runners (no torch here).

**Prompts are fixed before any result is seen** and not tuned on the evaluation split: tuning
prompt wording on validation data would make the zero-shot number optimistic. Each class is
queried by its plain name (underscores to spaces, "H-beam" capitalised). OWLv2 additionally uses
the standard "a photo of a {phrase}" template, because that is how the model family was
trained to be queried. Changing any of this is a new experiment and must be recorded as such.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np

from sitesight.data.caption_fields import CATEGORY_NAMES

_SPECIAL = {"h_beam": "H-beam", "drilling_rig": "drilling rig"}
QUERY_PHRASES: dict[str, str] = {c: _SPECIAL.get(c, c.replace("_", " ")) for c in CATEGORY_NAMES}
OWL_TEMPLATE = "a photo of a {}"
FLORENCE_GROUNDING_TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
FLORENCE_REGION_TASK = "<REGION_TO_DESCRIPTION>"

_LOC = re.compile(r"^<loc_(\d+)>$")


def owl_queries() -> list[str]:
    """OWLv2 text queries, in ``CATEGORY_NAMES`` order (index i <-> class i)."""
    return [OWL_TEMPLATE.format(QUERY_PHRASES[c]) for c in CATEGORY_NAMES]


def box_to_loc_tokens(bbox: Sequence[float], width: int, height: int, bins: int = 1000) -> str:
    """Florence-2 region input: ``[x, y, w, h]`` pixels -> ``<loc_x0><loc_y0><loc_x1><loc_y1>``.

    Florence-2 writes coordinates as one of ``bins`` (1000) discrete tokens per axis: the image
    is cut into 1000 columns and 1000 rows, ``q = clamp(floor(coordinate / (size / 1000)), 0,
    999)``. The arithmetic is done in float32 to match the library's own quantiser bit for bit
    (a float64 division can land on the other side of a bin edge).
    """
    x, y, w, h = bbox
    per_w = np.float32(width / bins)
    per_h = np.float32(height / bins)

    def q(v: float, per: np.float32) -> int:
        return int(min(max(math.floor(np.float32(v) / per), 0), bins - 1))

    return "".join(
        f"<loc_{v}>" for v in (q(x, per_w), q(y, per_h), q(x + w, per_w), q(y + h, per_h))
    )


def box_scores_from_tokens(
    tokens: Sequence[str], logprobs: Sequence[float]
) -> tuple[list[float], int]:
    """Confidence for each box in a Florence-2 generation, from the model's own token probabilities.

    A box is four consecutive ``<loc_N>`` tokens (x0, y0, x1, y1). Its confidence is the
    **geometric mean of the probabilities of those four tokens**, ``exp(mean(log p))``: the average
    per-token certainty about where the box is. (The plain product would shrink with every extra
    token; the geometric mean does not.) Returns ``(scores in generation order, number of loc
    tokens left over that did not form a complete box)``.
    """
    if len(tokens) != len(logprobs):
        raise ValueError("tokens and logprobs must be aligned")
    scores: list[float] = []
    leftover = 0
    run: list[float] = []

    def flush() -> None:
        nonlocal leftover
        for k in range(0, len(run) - len(run) % 4, 4):
            scores.append(math.exp(sum(run[k : k + 4]) / 4))
        leftover += len(run) % 4
        run.clear()

    for tok, lp in zip(tokens, logprobs, strict=True):
        if _LOC.match(tok):
            run.append(lp)
        else:
            flush()
    flush()
    return scores, leftover


def strip_loc_tokens(text: str) -> str:
    """Remove ``<loc_N>`` tokens (Florence-2 echoes the region after a description)."""
    return re.sub(r"<loc_\d+>", "", text).strip()


def sequence_score(logprobs: Sequence[float]) -> float:
    """Geometric-mean token probability of a whole generated text (0.0 if nothing generated)."""
    return math.exp(sum(logprobs) / len(logprobs)) if logprobs else 0.0
