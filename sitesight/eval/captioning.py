"""Caption metrics: BLEU, CIDEr-D, METEOR (exact match) and field-level accuracy.

Why field accuracy matters more than the n-gram scores here: the ~735 distinct ground-truth
captions come from one template, so a model that learns the template scores high on BLEU/CIDEr
without being right about *what is in the box*. Field accuracy checks the facts (resource, count,
colour, state); n-gram metrics are reported alongside, with that caveat.

Tokenisation for all n-gram metrics: lower-case, ``[a-z0-9]+`` words, hyphenated words kept whole.

**BLEU-N** (corpus level): for n = 1..N, ``p_n = sum of clipped n-gram matches / sum of candidate
n-grams`` over the whole corpus (a candidate n-gram counts at most as often as it occurs in any
single reference). ``BLEU-N = BP * exp((1/N) sum_n log p_n)`` with brevity penalty
``BP = 1`` if ``c > r`` else ``exp(1 - r/c)``, where ``c`` is the total candidate length and ``r``
the sum over sentences of the reference length closest to the candidate's. No smoothing: any
``p_n = 0`` gives 0.

**CIDEr-D** (Vedantam et al.; same formulation as the widely used coco-caption code): each
n-gram (n = 1..4) of a caption becomes a TF-IDF weight ``tf * (log N_img - log max(1, df))``
where ``df`` is the number of images whose references contain it, so common n-grams count less.
For each n the score is the cosine similarity of candidate and reference vectors (with each
candidate weight clipped to the reference weight, and a Gaussian length penalty
``exp(-d^2 / (2 sigma^2))``, ``sigma = 6``, ``d`` = difference in bigram counts), averaged over
n = 1..4 and over references, times 10. Scores are *not* in [0, 1]: a caption identical to its
reference scores about 10 only when all four n-gram orders exist.

**METEOR (exact)**: unigram alignment between candidate and a reference using exact word matches
only (no stemming, WordNet synonyms or paraphrases, so it is lower than full METEOR and named
accordingly). With ``m`` matches, precision ``P = m / |cand|``, recall ``R = m / |ref|``,
``Fmean = P R / (0.9 P + 0.1 R)`` and a fragmentation penalty ``0.5 (chunks / m)^3``, where a
chunk is a maximal run of matches that are adjacent in both texts; score = ``Fmean * (1 -
penalty)``, best over references, averaged over sentences (not corpus-pooled).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from sitesight.data.caption_fields import (
    COLOURS,
    find_resource,
    parse_caption,
    parse_number_at,
    tokenize_caption,
)

_WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu(
    candidates: Sequence[str], references: Sequence[Sequence[str]], max_n: int = 4
) -> dict[str, float]:
    """Corpus BLEU-1..``max_n`` (cumulative). ``references[i]`` lists the references of image i."""
    if len(candidates) != len(references):
        raise ValueError("need one reference list per candidate")
    matches = [0] * max_n
    totals = [0] * max_n
    cand_len = ref_len = 0
    for cand, refs in zip(candidates, references, strict=True):
        c = tokenize(cand)
        rs = [tokenize(r) for r in refs]
        cand_len += len(c)
        ref_len += min((abs(len(r) - len(c)), len(r)) for r in rs)[
            1
        ]  # closest length, shorter on ties
        for n in range(1, max_n + 1):
            cn = _ngrams(c, n)
            max_ref: Counter[tuple[str, ...]] = Counter()
            for r in rs:
                for g, k in _ngrams(r, n).items():
                    max_ref[g] = max(max_ref[g], k)
            matches[n - 1] += sum(min(k, max_ref[g]) for g, k in cn.items())
            totals[n - 1] += sum(cn.values())
    bp = 1.0 if cand_len > ref_len else (math.exp(1 - ref_len / cand_len) if cand_len else 0.0)
    out: dict[str, float] = {}
    for n in range(1, max_n + 1):
        precisions = [matches[k] / totals[k] if totals[k] else 0.0 for k in range(n)]
        out[f"bleu{n}"] = (
            0.0
            if min(precisions) == 0.0
            else bp * math.exp(sum(math.log(p) for p in precisions) / n)
        )
    return out


def cider_d(
    candidates: Sequence[str], references: Sequence[Sequence[str]], n: int = 4, sigma: float = 6.0
) -> float:
    """Corpus CIDEr-D (mean over images; see module docstring)."""
    if len(candidates) != len(references):
        raise ValueError("need one reference list per candidate")
    ref_counts = [
        [[_ngrams(tokenize(r), k) for k in range(1, n + 1)] for r in refs] for refs in references
    ]
    df: Counter[tuple[str, ...]] = Counter()
    for per_image in ref_counts:
        seen: set[tuple[str, ...]] = set()
        for per_ref in per_image:
            for cnt in per_ref:
                seen.update(cnt)
        df.update(seen)
    log_n = math.log(float(len(references)))

    def vectorise(
        per_order: list[Counter[tuple[str, ...]]],
    ) -> tuple[list[dict[tuple[str, ...], float]], list[float], int]:
        vec: list[dict[tuple[str, ...], float]] = [defaultdict(float) for _ in range(n)]
        norm = [0.0] * n
        length = 0
        for k, cnt in enumerate(per_order):
            for g, tf in cnt.items():
                w = float(tf) * (log_n - math.log(max(1.0, df[g])))
                vec[k][g] = w
                norm[k] += w * w
                if k == 1:  # coco-caption uses the bigram count as the length proxy
                    length += tf
        return vec, [math.sqrt(x) for x in norm], length

    total = 0.0
    for cand, per_image in zip(candidates, ref_counts, strict=True):
        vh, nh, lh = vectorise([_ngrams(tokenize(cand), k) for k in range(1, n + 1)])
        score = [0.0] * n
        for per_ref in per_image:
            vr, nr, lr = vectorise(per_ref)
            delta = float(lh - lr)
            for k in range(n):
                val = sum(min(w, vr[k][g]) * vr[k][g] for g, w in vh[k].items())
                if nh[k] != 0 and nr[k] != 0:
                    val /= nh[k] * nr[k]
                score[k] += val * math.exp(-(delta**2) / (2 * sigma**2))
        total += sum(score) / n / len(per_image) * 10.0
    return total / len(candidates) if candidates else 0.0


def _align(hyp: list[str], ref: list[str]) -> list[tuple[int, int]]:
    """Greedy exact alignment that prefers continuing the previous match (fewer chunks)."""
    free: dict[str, list[int]] = defaultdict(list)
    for j, w in enumerate(ref):
        free[w].append(j)
    pairs: list[tuple[int, int]] = []
    prev = -2
    for i, w in enumerate(hyp):
        cands = free.get(w, [])
        if not cands:
            continue
        nxt = [j for j in cands if j >= prev + 1]
        j = min(nxt) if nxt else min(cands)
        cands.remove(j)
        pairs.append((i, j))
        prev = j
    return pairs


def meteor_exact_sentence(hyp: str, refs: Sequence[str]) -> float:
    h = tokenize(hyp)
    best = 0.0
    for ref in refs:
        r = tokenize(ref)
        pairs = _align(h, r)
        m = len(pairs)
        if m == 0:
            continue
        p, rec = m / len(h), m / len(r)
        fmean = p * rec / (0.9 * p + 0.1 * rec)
        # a new chunk starts wherever two consecutive matches are not adjacent in both texts
        chunks = 1 + sum(
            1 for (i0, j0), (i1, j1) in pairwise(pairs) if not (i1 == i0 + 1 and j1 == j0 + 1)
        )
        best = max(best, fmean * (1 - 0.5 * (chunks / m) ** 3))
    return best


def meteor_exact(candidates: Sequence[str], references: Sequence[Sequence[str]]) -> float:
    if len(candidates) != len(references):
        raise ValueError("need one reference list per candidate")
    scores = [meteor_exact_sentence(c, r) for c, r in zip(candidates, references, strict=True)]
    return sum(scores) / len(scores) if scores else 0.0


@dataclass
class LooseFields:
    resource: str | None
    count: int | None
    colour: str | None


def loose_fields(text: str) -> LooseFields:
    """Pull resource / count / colour out of FREE text (zero-shot models do not use our template).

    resource = the first resource phrase from the dataset's lexicon found in the text (longest
    phrase at each position); colour = first colour word; count = first number word or digit,
    else 1 for a leading "a"/"an". Anything not found is ``None`` (counted as wrong, not skipped).
    """
    tokens = tokenize_caption(text)
    resource = None
    for i in range(len(tokens)):
        found = find_resource(tokens, i)
        if found is not None:
            resource = found[0]
            break
    colour = next(("gray" if t == "grey" else t for t in tokens if t in COLOURS), None)
    count: int | None = None
    for i, t in enumerate(tokens):
        num = parse_number_at(tokens, i)
        if num is not None:
            count = num[0]
            break
        if t.isdigit():
            count = int(t)
            break
    if count is None and tokens and tokens[0] in ("a", "an"):
        count = 1
    return LooseFields(resource, count, colour)


def field_accuracy(
    predictions: Sequence[str], gt_fields: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Accuracy of facts in predicted captions against the parsed ground-truth fields.

    * ``resource_acc_loose`` / ``colour_acc_loose`` / ``count_acc_loose``: facts found in free
      text by :func:`loose_fields` (colour and count are scored only where the ground truth has
      one; a prediction that omits it is wrong).
    * ``strict_parse_rate``: share of predictions that fit the dataset template, and for those
      ``resource/count/colour/state_acc_strict`` (a model fine-tuned on the data should be near
      the parse rate; a zero-shot model is not expected to use the template at all).
    """
    if len(predictions) != len(gt_fields):
        raise ValueError("one ground-truth fields dict per prediction")
    n = len(predictions)
    loose = [loose_fields(p) for p in predictions]
    strict = [parse_caption(p).fields for p in predictions]
    out: dict[str, Any] = {"n": n}
    out["resource_acc_loose"] = (
        sum(loose[i].resource == gt_fields[i]["resource"] for i in range(n)) / n if n else None
    )
    for f in ("colour", "count"):
        idx = [i for i in range(n) if gt_fields[i].get(f) is not None]
        out[f"{f}_acc_loose"] = (
            sum(getattr(loose[i], f) == gt_fields[i][f] for i in idx) / len(idx) if idx else None
        )
        out[f"{f}_n_with_ground_truth"] = len(idx)
    parsed = [i for i in range(n) if strict[i] is not None]
    out["strict_parse_rate"] = len(parsed) / n if n else None
    for f in ("resource", "colour", "count", "state"):
        idx = [
            i
            for i in range(n)
            if f == "resource" or f == "state" or gt_fields[i].get(f) is not None
        ]
        ok = 0
        for i in idx:
            s = strict[i]
            ok += s is not None and getattr(s, f) == gt_fields[i][f]
        out[f"{f}_acc_strict"] = ok / len(idx) if idx else None
    return out
