"""Evaluate prediction files against a split of the curated dataset.

    python -m sitesight.eval.run --annotations curated/annotations_curated.json --split val \\
        --predictions owlv2.json florence_grounding.json florence_regions.json \\
        --out reports/m4/zero_shot_val.json

Each prediction file declares in ``model.task`` what it is:

* ``grounding``: boxes with ``category_id`` and a score. ``model.queried_classes`` says how it was
  queried: ``"all"`` (every class asked in every image: scored under BOTH the ``detection`` and
  ``given_phrases`` protocols) or ``"present"`` (each class asked only where it is present in
  the ground truth, the phrase-grounding setting: scored under ``given_phrases`` only, because
  false alarms on absent classes were never possible).
* ``region_captioning``: a caption per ground-truth region (``region_id``) given its box; scored
  with BLEU / CIDEr-D / METEOR(exact) and field accuracy against the ground-truth captions.

``model.images`` (optional) lists the images the model was run on; ground truth is restricted to
them so a partial run is scored fairly and the coverage is reported.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sitesight.data.predictions import PredictionSet
from sitesight.data.schema import Dataset
from sitesight.eval.captioning import bleu, cider_d, field_accuracy, meteor_exact
from sitesight.eval.data import (
    RegionGT,
    build_ground_truth,
    detections_from_predictions,
    split_images,
)
from sitesight.eval.grounding import GT, evaluate_grounding
from sitesight.eval.summary import presence_summary


def evaluate_grounding_file(
    pset: PredictionSet, ds: Dataset, gts: Sequence[GT], images: Sequence[str]
) -> dict[str, Any]:
    dets = [d for d in detections_from_predictions(pset, ds) if d.file_name in set(images)]
    queried = pset.model_info.get("queried_classes", "present")
    protocols = ["detection", "given_phrases"] if queried == "all" else ["given_phrases"]
    out: dict[str, Any] = {
        "task": "grounding",
        "queried_classes": queried,
        "n_predictions": len(dets),
        "protocols": {p: evaluate_grounding(dets, gts, protocol=p) for p in protocols},
        "presence_summary": presence_summary(dets, gts, images),
    }
    return out


def evaluate_captioning_file(pset: PredictionSet, regions: Sequence[RegionGT]) -> dict[str, Any]:
    by_id = {r.ann_id: r for r in regions}
    pairs = [
        (p, by_id[p.region_id])
        for p in pset.predictions
        if p.region_id in by_id and p.caption is not None
    ]
    cands = [p.caption or "" for p, _ in pairs]
    refs = [[r.caption] for _, r in pairs]
    n = len(pairs)
    out: dict[str, Any] = {
        "task": "region_captioning",
        "n_evaluated": n,
        "n_ground_truth_regions": len(regions),
        "coverage": n / len(regions) if regions else None,
    }
    if n:
        out.update(bleu(cands, refs))
        out["cider_d"] = cider_d(cands, refs)
        out["meteor_exact"] = meteor_exact(cands, refs)
        out["fields"] = field_accuracy(cands, [r.fields for _, r in pairs])
        out["examples"] = [
            {"prediction": c, "reference": r[0]} for c, r in list(zip(cands, refs, strict=True))[:8]
        ]
    return out


def evaluate_file(
    pset: PredictionSet, ds: Dataset, split: str, restrict: set[str] | None = None
) -> dict[str, Any]:
    """Score one prediction file on its images (intersected with ``restrict`` if given)."""
    processed = pset.model_info.get("images")
    wanted = set(processed) if processed else None
    if restrict is not None:
        wanted = restrict if wanted is None else wanted & restrict
    gts, regions, stats = build_ground_truth(ds, split, wanted)
    images = sorted(
        {
            i.file_name
            for i in ds.images
            if i.split == split and (wanted is None or i.file_name in wanted)
        }
    )
    task = pset.model_info.get("task")
    if task == "grounding":
        body = evaluate_grounding_file(pset, ds, gts, images)
    elif task == "region_captioning":
        body = evaluate_captioning_file(pset, regions)
    else:
        raise ValueError(
            f"{pset.model_name}: model.task must be 'grounding' or 'region_captioning', "
            f"got {task!r}"
        )
    return {
        "model": pset.model_name,
        "model_info": {k: v for k, v in pset.model_info.items() if k != "images"},
        "split": split,
        "images_evaluated": len(images),
        "ground_truth": stats,
        **body,
    }


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}"


def _row(*cells: object) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    return [_row(*header), _row(*["---"] * len(header)), *[_row(*r) for r in rows], ""]


def markdown_table(results: Sequence[dict[str, Any]]) -> str:
    """Human-readable tables: grounding, recall breakdowns, per-class AP, summary, captioning."""
    split = results[0]["split"] if results else "?"
    grounding = [r for r in results if r["task"] == "grounding"]
    lines = [f"## Grounding on `{split}` (percent; IoU 0.5 unless stated)", ""]
    rows = []
    for r in grounding:
        for proto, m in r["protocols"].items():
            ece = m["calibration"].get("ece")
            rows.append(
                (r["model"], proto, _pct(m["map50"]), _pct(m["map75"]), _pct(m["recall50"]),
                 _pct(m["recall75"]), "n/a" if ece is None else f"{ece:.3f}", r["images_evaluated"])
            )  # fmt: skip
    head_g = (
        "Model",
        "Protocol",
        "mAP@0.5",
        "mAP@0.75",
        "Recall@0.5",
        "Recall@0.75",
        "ECE",
        "Images",
    )
    lines += _table(head_g, rows)

    lines += ["### Recall@0.5 by object size, crowding and region kind", ""]
    rows = []
    for r in grounding:
        b = next(iter(r["protocols"].values()))["breakdown"]
        cells = []
        for axis, keys in (("size", ("small", "medium", "large")),
                           ("crowding", ("sparse", "medium", "crowded")),
                           ("kind", ("single", "group"))):  # fmt: skip
            for k in keys:
                e = b[axis].get(k)
                cells.append("n/a" if e is None else f"{_pct(e['recall'])} (n={e['n_gt']})")
        rows.append((r["model"], *cells))
    head_b = (
        "Model",
        "small",
        "medium",
        "large",
        "sparse",
        "mid-crowd",
        "crowded",
        "single",
        "group",
    )
    lines += _table(head_b, rows)

    lines += ["### Image-level resource presence (F1 at fixed score thresholds)", ""]
    rows = []
    for r in grounding:
        t = r["presence_summary"]["thresholds"]
        rows.append((r["model"], *[_pct(t[k]["f1"]) for k in t]))
    thr = list(grounding[0]["presence_summary"]["thresholds"]) if grounding else []
    lines += _table(("Model", *[f"F1 @ score>={k}" for k in thr]), rows)

    if grounding:
        lines += ["### Per-class AP@0.5 (given_phrases protocol)", ""]
        classes = sorted(
            {c for r in grounding for c in r["protocols"]["given_phrases"]["per_class"]}
        )
        rows = []
        per = [r["protocols"]["given_phrases"]["per_class"] for r in grounding]
        for c in classes:
            n = max(p.get(c, {}).get("n_gt", 0) for p in per)
            rows.append((c, n, *[_pct(p.get(c, {}).get("ap50")) for p in per]))
        lines += _table(("Class", "n GT", *[r["model"] for r in grounding]), rows)

    lines += ["## Region captioning (given the ground-truth box)", ""]
    rows = []
    for r in results:
        if r["task"] != "region_captioning" or not r.get("n_evaluated"):
            continue
        f = r["fields"]
        n_gram = (_pct(r["bleu4"]), f"{r['cider_d']:.2f}", _pct(r["meteor_exact"]))
        facts = tuple(
            _pct(f[k]) for k in ("resource_acc_loose", "colour_acc_loose", "count_acc_loose")
        )
        rows.append((r["model"], r["n_evaluated"], *n_gram, *facts, _pct(f["strict_parse_rate"])))
    head_c = (
        "Model", "Regions", "BLEU-4", "CIDEr-D", "METEOR (exact)",
        "Resource acc", "Colour acc", "Count acc", "Template parse rate",
    )  # fmt: skip
    lines += _table(head_c, rows)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--predictions", type=Path, nargs="+", required=True)
    p.add_argument(
        "--out", type=Path, required=True, help="metrics JSON (a .md table is written next to it)"
    )
    p.add_argument(
        "--common-images",
        action="store_true",
        help="score every file on the images ALL files cover (apples-to-apples comparison)",
    )
    args = p.parse_args(argv)
    ds = Dataset.load(args.annotations)
    if not split_images(ds, args.split):
        p.error(f"no images in split {args.split!r}: run the curation step first")
    psets = [PredictionSet.load(path) for path in args.predictions]
    restrict: set[str] | None = None
    if args.common_images:
        covered = [set(ps.model_info["images"]) for ps in psets if ps.model_info.get("images")]
        restrict = set.intersection(*covered) if covered else None
        print(f"--common-images: {len(restrict or [])} images are covered by every file")
    results = [evaluate_file(ps, ds, args.split, restrict) for ps in psets]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    md = markdown_table(results)
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
