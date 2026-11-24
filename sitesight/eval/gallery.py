"""Failure analysis: classify every miss into a type, count them, and draw real examples.

A gallery of hand-picked images proves little; a *classification of all failures with counts* plus
a few drawn examples per type tells you what actually goes wrong and how often. Types are assigned
by fixed rules (below), then examples are drawn with a seeded random choice (not the prettiest).

Grounding (one record per ground-truth box that was NOT matched at IoU 0.5 by its class's top-k
predictions, using the same matching as the metrics):

* ``group_split``: the ground truth is a group/pile box and two or more predicted boxes of that
  class lie (mostly) inside it: the model found the individual objects, the label is one box.
* ``small_missed``: the ground-truth box is small (< 1% of the image) and nothing matches it.
* ``no_overlap``: no predicted box of the class overlaps it (IoU < 0.1): missed or misplaced.
* ``poor_localisation``: the best predicted box overlaps it (0.1 <= IoU < 0.5), not well enough.
  Rules are applied in this order; a record gets the first type that fits.

Predicted boxes that match nothing are counted as ``false_positive`` (``duplicate`` if they overlap
an already matched box at IoU >= 0.5, else ``spurious``).

Captioning (one record per region whose caption differs from the reference): the first differing
field in the order resource, count, colour, state, target; ``unparsed`` if the predicted caption
does not fit the dataset template.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from sitesight.data.caption_fields import parse_caption
from sitesight.data.predictions import PredictionSet
from sitesight.data.schema import Dataset
from sitesight.eval.boxes import iou_matrix
from sitesight.eval.data import RegionGT, build_ground_truth, detections_from_predictions
from sitesight.eval.grounding import DEFAULT_TOP_K, GT, SMALL_AREA, Det, match_pair

GROUP_INSIDE = 0.8  # a predicted box counts as "inside" a group box if >= 80% of it lies in the GT


@dataclass
class Failure:
    kind: str
    file_name: str
    category: str
    gt_bbox: list[float]
    pred_bboxes: list[list[float]] = field(default_factory=list)
    best_iou: float = 0.0
    detail: str = ""


def _inside_fraction(inner: Sequence[float], outer: Sequence[float]) -> float:
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1 = min(inner[0] + inner[2], outer[0] + outer[2])
    iy1 = min(inner[1] + inner[3], outer[1] + outer[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    return inter / (inner[2] * inner[3]) if inner[2] * inner[3] > 0 else 0.0


def classify_grounding_failures(
    gts: Sequence[GT], dets: Sequence[Det], iou_threshold: float = 0.5, top_k: int = DEFAULT_TOP_K
) -> list[Failure]:
    gt_pairs: dict[tuple[str, str], list[GT]] = defaultdict(list)
    det_pairs: dict[tuple[str, str], list[Det]] = defaultdict(list)
    for g in gts:
        gt_pairs[(g.file_name, g.category)].append(g)
    for d in dets:
        det_pairs[(d.file_name, d.category)].append(d)
    out: list[Failure] = []
    for pair, g_list in sorted(gt_pairs.items()):
        d_sorted = sorted(det_pairs.get(pair, []), key=lambda d: -d.score)[:top_k]
        flags, used = match_pair(d_sorted, g_list, iou_threshold)
        ious = iou_matrix([g.bbox for g in g_list], [d.bbox for d in d_sorted])
        for j, g in enumerate(g_list):
            if used[j]:
                continue
            best = float(ious[j].max()) if d_sorted else 0.0
            inside = [d.bbox for d in d_sorted if _inside_fraction(d.bbox, g.bbox) >= GROUP_INSIDE]
            if g.kind == "group" and len(inside) >= 2:
                kind, detail = "group_split", f"{len(inside)} predicted boxes inside the group box"
            elif g.area_frac < SMALL_AREA:
                kind, detail = "small_missed", f"GT covers {100 * g.area_frac:.2f}% of the image"
            elif best < 0.1:
                kind, detail = "no_overlap", "no predicted box overlaps"
            else:
                kind, detail = "poor_localisation", f"best IoU {best:.2f}"
            out.append(
                Failure(kind, pair[0], pair[1], g.bbox, [d.bbox for d in d_sorted], best, detail)
            )
        matched_boxes = [d.bbox for d, f in zip(d_sorted, flags, strict=True) if f]
        for d, f in zip(d_sorted, flags, strict=True):
            if f:
                continue
            dup = bool(matched_boxes) and (
                float(iou_matrix([d.bbox], matched_boxes)[0].max()) >= iou_threshold
            )
            out.append(
                Failure(
                    "duplicate" if dup else "spurious",
                    pair[0],
                    pair[1],
                    [],
                    [d.bbox],
                    0.0,
                    f"score {d.score:.2f}",
                )
            )
    # false positives for classes that are present nowhere in the image are outside the
    # given_phrases protocol and are not listed here.
    return out


_FIELDS = ("resource", "count", "colour", "state", "target")


def classify_caption_failure(reference: str, prediction: str) -> tuple[str, str] | None:
    """``None`` if the captions agree on every field, else (failure type, detail)."""
    ref, pred = parse_caption(reference).fields, parse_caption(prediction).fields
    if pred is None:
        return "unparsed", "prediction does not fit the dataset template"
    if ref is None:
        return None  # cannot compare against an unparseable reference
    pairs = {
        "resource": (ref.resource, pred.resource),
        "count": ((ref.count, ref.count_unit), (pred.count, pred.count_unit)),
        "colour": (ref.colour, pred.colour),
        "state": (ref.state, pred.state),
        "target": (
            (ref.preposition, ref.target, ref.destination),
            (pred.preposition, pred.target, pred.destination),
        ),
    }
    for name in _FIELDS:
        if pairs[name][0] != pairs[name][1]:
            return f"{name}_mismatch", f"{pairs[name][0]} -> {pairs[name][1]}"
    return None


def classify_caption_failures(
    regions: Sequence[RegionGT], predictions: dict[int, str]
) -> list[Failure]:
    out: list[Failure] = []
    for r in regions:
        pred = predictions.get(r.ann_id)
        if pred is None:
            continue
        res = classify_caption_failure(r.caption, pred)
        if res is not None:
            out.append(
                Failure(
                    res[0],
                    r.file_name,
                    str(r.fields["resource"]),
                    r.bbox,
                    [],
                    0.0,
                    f"reference: {r.caption} | prediction: {pred} | {res[1]}",
                )
            )
    return out


def summarise(failures: Sequence[Failure], n_total: int) -> dict[str, Any]:
    c = Counter(f.kind for f in failures)
    return {"n_total": n_total, "n_failures": len(failures), "by_kind": dict(c.most_common())}


def pick_examples(
    failures: Sequence[Failure], per_kind: int, seed: int
) -> dict[str, list[Failure]]:
    """Seeded random examples per type (not the most striking ones)."""
    by_kind: dict[str, list[Failure]] = defaultdict(list)
    for f in failures:
        by_kind[f.kind].append(f)
    rng = random.Random(seed)
    return {k: rng.sample(v, min(per_kind, len(v))) for k, v in sorted(by_kind.items())}


def render(image_path: Path, failure: Failure, out_path: Path) -> None:
    """Green = ground truth, orange = predictions; the title says the failure type."""
    im = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(im)
    if failure.gt_bbox:
        x, y, w, h = failure.gt_bbox
        d.rectangle([x, y, x + w, y + h], outline=(46, 158, 91), width=4)
    for b in failure.pred_bboxes:
        d.rectangle([b[0], b[1], b[0] + b[2], b[1] + b[3]], outline=(217, 119, 6), width=2)
    d.rectangle([0, 0, im.width, 22], fill=(0, 0, 0))
    d.text(
        (6, 5),
        f"{failure.kind} | {failure.category} | {failure.file_name} | green=GT orange=pred",
        fill=(255, 255, 255),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--grounding", type=Path, required=True, help="grounding predictions JSON")
    p.add_argument("--regions", type=Path, help="region-caption predictions JSON")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--per-kind", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    ds = Dataset.load(args.annotations)
    gp = PredictionSet.load(args.grounding)
    wanted = set(gp.model_info["images"]) if gp.model_info.get("images") else None
    gts, regions, _ = build_ground_truth(ds, args.split, wanted)
    dets = detections_from_predictions(gp, ds)
    report: dict[str, Any] = {"split": args.split, "seed": args.seed}
    kinds: dict[str, list[Failure]] = {}

    gf = classify_grounding_failures(gts, dets)
    report["grounding"] = {
        **summarise([f for f in gf if f.gt_bbox], len(gts)),
        "false_positives": dict(Counter(f.kind for f in gf if not f.gt_bbox)),
    }
    kinds["grounding"] = gf
    if args.regions:
        rp = PredictionSet.load(args.regions)
        preds = {
            int(x.region_id): x.caption or "" for x in rp.predictions if x.region_id is not None
        }
        cf = classify_caption_failures(regions, preds)
        report["captioning"] = summarise(cf, len(preds))
        kinds["captioning"] = cf

    for task, fails in kinds.items():
        for kind, examples in pick_examples(fails, args.per_kind, args.seed).items():
            for i, f in enumerate(examples, start=1):
                render(args.images / f.file_name, f, args.out_dir / task / f"{kind}_{i}.png")
                report.setdefault("examples", []).append(
                    {"task": task, **asdict(f), "image": f"{task}/{kind}_{i}.png"}
                )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "failure_summary.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
