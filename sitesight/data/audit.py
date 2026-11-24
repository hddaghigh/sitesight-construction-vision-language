"""Label-quality audit: sample images, compare original labels with re-annotated ones, report.

Protocol. A random sample of images is re-annotated by a person in the review tool, in one of
two modes:

* ``blind``: the original labels are hidden and the auditor draws boxes and writes captions
  from scratch. This is an *independent* second annotation, so agreement is not anchored on the
  original labels (preferred; costs more time).
* ``review``: the original labels are shown and the auditor corrects them. Much faster, but the
  auditor is anchored on what they see, so agreement is an *upper bound*. Reports say which mode
  produced them.

Either way the analysis is the same: compare the original region set with the audited region
set of each image.

Matching. Regions are matched greedily by box IoU (intersection over union of the two boxes):
all pairs with IoU >= ``iou_threshold`` (default 0.5, the usual detection convention) are sorted
by IoU, highest first, and a pair is accepted if neither box is used yet. Greedy matching is
optimal when matches are unambiguous, which is the typical case here (separate objects); it is
documented rather than hidden because crowded scenes can make it suboptimal.

Reading the numbers (``orig`` = original annotators, ``audit`` = the re-annotation):

* *precision* = matched / n_orig: the share of original labels the audit confirms;
* *recall* = matched / n_audit: the share of audited objects the original annotators had;
* field agreement is measured on matched pairs only, for fields both captions parse into;
* Cohen's kappa corrects resource-label agreement for chance:
  ``kappa = (p_o - p_e) / (1 - p_e)`` with ``p_o`` the observed agreement and ``p_e`` the
  agreement expected if both sides labelled independently with their own label frequencies.
* 95% confidence intervals are percentile bootstrap intervals over *images* (resampling whole
  images with replacement), because regions inside an image are not independent.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sitesight.data.caption_fields import parse_caption
from sitesight.data.schema import Dataset

# Taxonomy codes (see ``classify_pair`` / ``audit_images``).
BOX_TOO_LARGE = "box_too_large"
BOX_TOO_SMALL = "box_too_small"
BOX_SHIFTED = "box_shifted"
RESOURCE_MISMATCH = "resource_mismatch"
COUNT_MISMATCH = "count_mismatch"
COLOUR_MISMATCH = "colour_mismatch"
STATE_MISMATCH = "state_mismatch"
TARGET_MISMATCH = "target_mismatch"
ORIGINAL_UNPARSED = "original_caption_unparsed"
AUDIT_UNPARSED = "audit_caption_unparsed"
MISSED_BY_ORIGINAL = "missed_by_original"
NOT_CONFIRMED_BY_AUDIT = "not_confirmed_by_audit"

GOOD_IOU = 0.75  # matched boxes at or above this are "tight"; below get a box-quality code
SIZE_RATIO = 1.25  # area ratio beyond which a loose/tight box is called too large / too small


@dataclass
class Region:
    """A box + caption, comparable between annotators."""

    bbox: list[float]  # [x, y, w, h]
    caption: str


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``[x, y, w, h]`` boxes."""
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Matching:
    pairs: list[tuple[int, int, float]]  # (orig index, audit index, IoU)
    unmatched_orig: list[int]
    unmatched_audit: list[int]


def match_regions(
    orig: Sequence[Region], audit: Sequence[Region], iou_threshold: float = 0.5
) -> Matching:
    """Greedy one-to-one matching by descending IoU (ties broken by index, so deterministic)."""
    cands = []
    for i, o in enumerate(orig):
        for j, a in enumerate(audit):
            v = iou(o.bbox, a.bbox)
            if v >= iou_threshold:
                cands.append((-v, i, j))
    cands.sort()
    used_o: set[int] = set()
    used_a: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for neg, i, j in cands:
        if i in used_o or j in used_a:
            continue
        used_o.add(i)
        used_a.add(j)
        pairs.append((i, j, -neg))
    return Matching(
        pairs=sorted(pairs),
        unmatched_orig=[i for i in range(len(orig)) if i not in used_o],
        unmatched_audit=[j for j in range(len(audit)) if j not in used_a],
    )


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Cohen's kappa for two raters' categorical labels; ``None`` if undefined (p_e = 1)."""
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty label lists of equal length")
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    if p_e >= 1.0:
        return None
    return (p_o - p_e) / (1.0 - p_e)


def classify_pair(orig: Region, audit: Region, overlap: float) -> list[str]:
    """Taxonomy codes for one matched pair (empty list = fully agrees)."""
    codes: list[str] = []
    if overlap < GOOD_IOU:
        ratio = (orig.bbox[2] * orig.bbox[3]) / (audit.bbox[2] * audit.bbox[3])
        codes.append(
            BOX_TOO_LARGE
            if ratio > SIZE_RATIO
            else BOX_TOO_SMALL
            if ratio < 1 / SIZE_RATIO
            else BOX_SHIFTED
        )
    po, pa = parse_caption(orig.caption), parse_caption(audit.caption)
    if po.fields is None:
        codes.append(ORIGINAL_UNPARSED)
    if pa.fields is None:
        codes.append(AUDIT_UNPARSED)
    if po.fields is not None and pa.fields is not None:
        fo, fa = po.fields, pa.fields
        if fo.resource != fa.resource:
            codes.append(RESOURCE_MISMATCH)
        if (fo.count, fo.count_unit) != (fa.count, fa.count_unit):
            codes.append(COUNT_MISMATCH)
        if fo.colour != fa.colour:
            codes.append(COLOUR_MISMATCH)
        if fo.state != fa.state:
            codes.append(STATE_MISMATCH)
        if (fo.preposition, fo.target, fo.destination) != (
            fa.preposition,
            fa.target,
            fa.destination,
        ):
            codes.append(TARGET_MISMATCH)
    return codes


@dataclass
class ImageResult:
    file_name: str
    n_orig: int
    n_audit: int
    n_matched: int
    ious: list[float]
    taxonomy: dict[str, int]
    resource_pairs: list[tuple[str, str]]  # (orig, audit) resource for pairs both parse into
    field_checks: dict[str, tuple[int, int]]  # field -> (n agree, n compared)
    exact_caption_matches: int


_FIELDS = ("resource", "count", "colour", "state", "target")


def audit_image(
    file_name: str, orig: Sequence[Region], audit: Sequence[Region], iou_threshold: float = 0.5
) -> ImageResult:
    m = match_regions(orig, audit, iou_threshold)
    tax: Counter[str] = Counter()
    res_pairs: list[tuple[str, str]] = []
    checks = {f: [0, 0] for f in _FIELDS}
    exact = 0
    for i, j, overlap in m.pairs:
        for code in classify_pair(orig[i], audit[j], overlap):
            tax[code] += 1
        po, pa = parse_caption(orig[i].caption), parse_caption(audit[j].caption)
        if " ".join(orig[i].caption.lower().split()) == " ".join(audit[j].caption.lower().split()):
            exact += 1
        if po.fields is not None and pa.fields is not None:
            fo, fa = po.fields, pa.fields
            res_pairs.append((fo.resource, fa.resource))
            same = {
                "resource": fo.resource == fa.resource,
                "count": (fo.count, fo.count_unit) == (fa.count, fa.count_unit),
                "colour": fo.colour == fa.colour,
                "state": fo.state == fa.state,
                "target": (fo.preposition, fo.target, fo.destination)
                == (fa.preposition, fa.target, fa.destination),
            }
            for f in _FIELDS:
                checks[f][1] += 1
                checks[f][0] += same[f]
    tax[NOT_CONFIRMED_BY_AUDIT] += len(m.unmatched_orig)
    tax[MISSED_BY_ORIGINAL] += len(m.unmatched_audit)
    return ImageResult(
        file_name=file_name,
        n_orig=len(orig),
        n_audit=len(audit),
        n_matched=len(m.pairs),
        ious=[p[2] for p in m.pairs],
        taxonomy={k: v for k, v in tax.items() if v},
        resource_pairs=res_pairs,
        field_checks={f: (c[0], c[1]) for f, c in checks.items()},
        exact_caption_matches=exact,
    )


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def _summary(results: Sequence[ImageResult]) -> dict[str, Any]:
    n_o = sum(r.n_orig for r in results)
    n_a = sum(r.n_audit for r in results)
    n_m = sum(r.n_matched for r in results)
    ious = [v for r in results for v in r.ious]
    out: dict[str, Any] = {
        "n_images": len(results),
        "n_original_regions": n_o,
        "n_audit_regions": n_a,
        "n_matched": n_m,
        "precision": _ratio(n_m, n_o),
        "recall": _ratio(n_m, n_a),
        "mean_iou_of_matches": float(np.mean(ious)) if ious else None,
        "fraction_matches_iou_ge_0_75": _ratio(sum(v >= GOOD_IOU for v in ious), len(ious)),
        "exact_caption_agreement": _ratio(sum(r.exact_caption_matches for r in results), n_m),
    }
    p, r_ = out["precision"], out["recall"]
    out["f1"] = 2 * p * r_ / (p + r_) if p and r_ else None
    for f in _FIELDS:
        agree = sum(r.field_checks[f][0] for r in results)
        total = sum(r.field_checks[f][1] for r in results)
        out[f"{f}_agreement"] = _ratio(agree, total)
        out[f"{f}_n_compared"] = total
    pairs = [p for r in results for p in r.resource_pairs]
    out["resource_kappa"] = (
        cohen_kappa([a for a, _ in pairs], [b for _, b in pairs]) if pairs else None
    )
    return out


def bootstrap_ci(
    results: Sequence[ImageResult], keys: Sequence[str], n_boot: int = 1000, seed: int = 0
) -> dict[str, list[float | None]]:
    """95% percentile bootstrap CI for summary metrics, resampling images with replacement."""
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {k: [] for k in keys}
    n = len(results)
    for _ in range(n_boot):
        boot = [results[rng.randrange(n)] for _ in range(n)]
        s = _summary(boot)
        for k in keys:
            if s[k] is not None:
                samples[k].append(float(s[k]))
    return {
        k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [None, None]
        for k, v in samples.items()
    }


@dataclass
class AuditReport:
    mode: str
    iou_threshold: float
    summary: dict[str, Any]
    ci95: dict[str, list[float | None]]
    taxonomy: dict[str, int]
    per_image: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CI_KEYS = (
    "precision",
    "recall",
    "resource_agreement",
    "count_agreement",
    "colour_agreement",
    "state_agreement",
    "resource_kappa",
    "mean_iou_of_matches",
)


def audit_images(
    data: Mapping[str, tuple[Sequence[Region], Sequence[Region]]],
    mode: str,
    iou_threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 0,
) -> AuditReport:
    """Compare ``{file_name: (original regions, audited regions)}`` and build the report."""
    results = [audit_image(name, o, a, iou_threshold) for name, (o, a) in sorted(data.items())]
    if not results:
        raise ValueError("no audited images")
    tax: Counter[str] = Counter()
    for r in results:
        tax.update(r.taxonomy)
    return AuditReport(
        mode=mode,
        iou_threshold=iou_threshold,
        summary=_summary(results),
        ci95=bootstrap_ci(results, CI_KEYS, n_boot, seed),
        taxonomy=dict(sorted(tax.items(), key=lambda kv: -kv[1])),
        per_image=[
            {
                "file_name": r.file_name,
                "n_orig": r.n_orig,
                "n_audit": r.n_audit,
                "n_matched": r.n_matched,
                "taxonomy": r.taxonomy,
            }
            for r in results
        ],
    )


def _numeric_key(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    return (int(stem) if stem.isdigit() else 10**9, name)


def sample_audit_images(ds: Dataset, n: int = 100, seed: int = 0) -> list[dict[str, Any]]:
    """Uniform random sample (without replacement) of ``n`` images, recorded for reproducibility.

    Drawn from *all* images (including the 19 with no regions, which test missed labels), so
    the audit estimates the noise of the whole dataset, not just of one split.
    """
    if n > len(ds.images):
        raise ValueError("sample larger than the dataset")
    by_name = {i.file_name: i for i in ds.images}
    names = sorted(by_name, key=_numeric_key)
    chosen = sorted(random.Random(seed).sample(names, n), key=_numeric_key)
    counts = Counter(a.image_id for a in ds.annotations)
    return [
        {"file_name": nm, "split": by_name[nm].split, "n_original_regions": counts[by_name[nm].id]}
        for nm in chosen
    ]


@dataclass
class AuditImage:
    status: str = "todo"  # todo | done | skipped
    annotations: list[dict[str, Any]] = field(
        default_factory=list
    )  # {"bbox": [...], "caption": str}
    notes: str = ""


@dataclass
class AuditFile:
    mode: str  # "blind" | "review"
    sample: list[str]  # file names, in audit order
    images: dict[str, AuditImage] = field(default_factory=dict)
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "mode": self.mode,
            "sample": self.sample,
            "images": {
                k: asdict(v)
                for k, v in sorted(self.images.items(), key=lambda kv: _numeric_key(kv[0]))
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AuditFile:
        return cls(
            mode=d["mode"],
            sample=list(d["sample"]),
            images={k: AuditImage(**v) for k, v in d.get("images", {}).items()},
            version=d.get("version", 1),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> AuditFile:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def done(self) -> list[str]:
        return [n for n in self.sample if self.images.get(n, AuditImage()).status == "done"]


def build_report(
    ds: Dataset, audit: AuditFile, iou_threshold: float = 0.5, n_boot: int = 1000
) -> AuditReport:
    """Report for the images the auditor marked ``done`` (others are ignored, and counted)."""
    id_of = {i.file_name: i.id for i in ds.images}
    orig_by_image: dict[int, list[Region]] = {}
    for a in ds.annotations:
        orig_by_image.setdefault(a.image_id, []).append(
            Region([float(v) for v in a.bbox], a.caption)
        )
    data: dict[str, tuple[list[Region], list[Region]]] = {}
    for name in audit.done():
        if name not in id_of:
            raise ValueError(f"audited image not in dataset: {name}")
        aud = [
            Region([float(v) for v in x["bbox"]], x["caption"])
            for x in audit.images[name].annotations
        ]
        data[name] = (orig_by_image.get(id_of[name], []), aud)
    rep = audit_images(data, audit.mode, iou_threshold, n_boot)
    rep.summary["n_sampled"] = len(audit.sample)
    rep.summary["n_not_done"] = len(audit.sample) - len(data)
    return rep


def _kappa_text(k: float | None) -> str:
    return "n/a" if k is None else f"{k:.3f}"


def format_markdown(rep: AuditReport) -> str:
    s = rep.summary

    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{100 * v:.1f}%"

    def ci(k: str) -> str:
        lo, hi = rep.ci95.get(k, [None, None])
        return "" if lo is None or hi is None else f" (95% CI {100 * lo:.1f}-{100 * hi:.1f}%)"

    lines = [
        f"# Label audit results ({rep.mode} mode, IoU >= {rep.iou_threshold})",
        "",
        f"Images audited: {s['n_images']} of {s.get('n_sampled', s['n_images'])} sampled. "
        f"Original regions: {s['n_original_regions']}; audit regions: {s['n_audit_regions']}; "
        f"matched: {s['n_matched']}.",
        "",
        f"- Original labels confirmed (precision): {pct(s['precision'])}{ci('precision')}",
        f"- Audited objects present in original (recall): {pct(s['recall'])}{ci('recall')}",
        f"- Mean IoU of matched boxes: {pct(s['mean_iou_of_matches'])}",
        f"- Resource agreement: {pct(s['resource_agreement'])}{ci('resource_agreement')}; "
        f"Cohen's kappa: {_kappa_text(s['resource_kappa'])}",
        f"- Count agreement: {pct(s['count_agreement'])}{ci('count_agreement')}",
        f"- Colour agreement: {pct(s['colour_agreement'])}; state: {pct(s['state_agreement'])}; "
        f"target: {pct(s['target_agreement'])}",
        f"- Exact caption agreement (matched): {pct(s['exact_caption_agreement'])}",
        "",
        "## Error taxonomy (counts)",
        "",
        "| Code | Count |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in rep.taxonomy.items()]
    if rep.mode == "review":
        lines += [
            "",
            "*Review mode: the auditor saw the original labels, so these are upper bounds.*",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Label-audit tools.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="draw the random audit sample")
    s.add_argument("--annotations", type=Path, required=True)
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("report", help="agreement metrics + taxonomy from a finished audit")
    r.add_argument("--annotations", type=Path, required=True)
    r.add_argument("--audit", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True, help="audit_report.json (+ .md next to it)")
    r.add_argument("--iou", type=float, default=0.5)
    args = p.parse_args(argv)

    ds = Dataset.load(args.annotations)
    if args.cmd == "sample":
        rows = sample_audit_images(ds, args.n, args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"seed": args.seed, "n": args.n, "images": rows}, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"sampled {len(rows)} images (seed {args.seed}) -> {args.out}")
        return 0
    audit = AuditFile.load(args.audit)
    rep = build_report(ds, audit, args.iou)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(rep.to_dict(), indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.out.with_suffix(".md").write_text(format_markdown(rep), encoding="utf-8")
    print(format_markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
