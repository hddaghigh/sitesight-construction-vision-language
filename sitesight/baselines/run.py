"""Run a zero-shot baseline over a split and write prediction file(s) (resumable).

    python -m sitesight.baselines.run --model owlv2 \\
        --annotations curated/annotations_curated.json --images extracted/UAVdata \\
        --split val --out predictions/owlv2_val.json

Models:

* ``owlv2``: all 31 classes queried in every image -> ``--out``.
* ``florence``: both Florence-2 tasks in ONE pass per image, so the slow vision encoder runs once:
  phrase grounding (each class queried only where it is present in the evaluated ground truth,
  the phrase-grounding setting) -> ``<out stem>_grounding.json``, and region description (the
  model describes each evaluated ground-truth box) -> ``<out stem>_regions.json``.

``--limit N`` takes a seeded random sample of N images (same seed, same sample, so repeated and
resumed runs cover the same images, and two models run with the same ``--limit/--seed`` are
scored on the same images). Progress is saved every ``--save-every`` images and existing output
files are resumed, so an interrupted run loses little. ``--adapter DIR`` runs a LoRA-fine-tuned
Florence-2 (from ``sitesight.train.lora``). The ``test`` split is refused unless ``--allow-test`` is
given (for the one final evaluation only): it is
reserved for the final evaluation.
"""

from __future__ import annotations

import argparse
import platform
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from sitesight.baselines.prompts import (
    FLORENCE_GROUNDING_TASK,
    FLORENCE_REGION_TASK,
    OWL_TEMPLATE,
    QUERY_PHRASES,
    owl_queries,
)
from sitesight.data.caption_fields import CATEGORY_IDS, CATEGORY_NAMES
from sitesight.data.predictions import Prediction, PredictionSet
from sitesight.data.schema import Dataset
from sitesight.eval.data import build_ground_truth, split_images


def select_images(ds: Dataset, split: str, limit: int | None, seed: int) -> list[str]:
    names = split_images(ds, split)
    if limit is not None and limit < len(names):
        names = sorted(random.Random(seed).sample(names, limit), key=lambda n: (len(n), n))
    return names


@dataclass
class Task:
    """One output file: its predictions, where it is saved and how to answer one image."""

    pset: PredictionSet
    path: Path
    info: dict[str, Any]
    answer: Callable[[str, Image.Image], list[Prediction]]


def _load_or_new(path: Path, model_name: str) -> PredictionSet:
    if path.exists():
        pset = PredictionSet.load(path)
        print(f"resuming {path}: {len(pset.model_info.get('images', []))} images already done")
        return pset
    return PredictionSet(model_name=model_name, model_info={"images": []})


def build_tasks(
    model: str,
    out: Path,
    ds: Dataset,
    split: str,
    names: list[str],
    device: str,
    model_id: str | None = None,
    adapter: str | None = None,
) -> tuple[list[Task], Any]:
    gts, regions, _ = build_ground_truth(ds, split, set(names))
    present: dict[str, set[str]] = {}
    for g in gts:
        present.setdefault(g.file_name, set()).add(g.category)
    regs: dict[str, list[Any]] = {}
    for r in regions:
        regs.setdefault(r.file_name, []).append(r)

    if model == "owlv2":
        from sitesight.baselines.owlv2 import MODEL_ID as DEFAULT_ID
        from sitesight.baselines.owlv2 import Owlv2Runner

        MODEL_ID = model_id or DEFAULT_ID
        owl = Owlv2Runner(MODEL_ID, device=device)
        queries = owl_queries()

        def owl_answer(name: str, image: Image.Image) -> list[Prediction]:
            out_preds = []
            for qi, bbox, score in owl.detect(image, queries):
                cls = CATEGORY_NAMES[qi]
                out_preds.append(
                    Prediction(name, bbox, score, QUERY_PHRASES[cls], CATEGORY_IDS[cls])
                )
            return out_preds

        info = {
            "task": "grounding",
            "queried_classes": "all",
            "model_id": MODEL_ID,
            "prompt_template": OWL_TEMPLATE,
            "min_score": 0.01,
            "nms_iou": 0.5,
            "max_per_class": 100,
            "score": "sigmoid(logit) of the (box, query) pair",
        }
        owl_name = MODEL_ID.split("/")[-1]
        return [Task(_load_or_new(out, owl_name), out, info, owl_answer)], owl

    if model == "florence":
        from sitesight.baselines.florence import MODEL_ID as DEFAULT_ID
        from sitesight.baselines.florence import FlorenceRunner

        MODEL_ID = model_id or DEFAULT_ID
        flo = FlorenceRunner(MODEL_ID, device=device, adapter=adapter)
        flo_name = MODEL_ID.split("/")[-1].lower()
        if (
            adapter
        ):  # fine-tuned: name it after the run and epoch, e.g. florence-2-base-lora-epoch_3
            flo_name += "-lora-" + Path(adapter).name

        def ground_answer(name: str, image: Image.Image) -> list[Prediction]:
            preds = []
            for cls in sorted(present.get(name, set())):
                for bbox, score in flo.ground(image, QUERY_PHRASES[cls]):
                    preds.append(
                        Prediction(name, bbox, score, QUERY_PHRASES[cls], CATEGORY_IDS[cls])
                    )
            return preds

        def region_answer(name: str, image: Image.Image) -> list[Prediction]:
            preds = []
            for r in regs.get(name, []):
                caption, score = flo.describe_region(image, r.bbox)
                preds.append(Prediction(name, r.bbox, score, caption, None, r.ann_id))
            return preds

        base = {"model_id": MODEL_ID, "decoding": "greedy", "adapter": adapter}
        g_path = out.with_name(out.stem + "_grounding.json")
        r_path = out.with_name(out.stem + "_regions.json")
        g_info = {
            **base,
            "task": "grounding",
            "queried_classes": "present",
            "florence_task": FLORENCE_GROUNDING_TASK,
            "score": "geometric-mean token probability of the 4 location tokens of each box",
        }
        r_info = {
            **base,
            "task": "region_captioning",
            "florence_task": FLORENCE_REGION_TASK,
            "score": "geometric-mean token probability of the description words",
        }
        return [
            Task(_load_or_new(g_path, f"{flo_name}-grounding"), g_path, g_info, ground_answer),
            Task(_load_or_new(r_path, f"{flo_name}-regions"), r_path, r_info, region_answer),
        ], flo
    raise ValueError(f"unknown model {model!r}")


def _stamp(
    task: Task, runner: Any, device: str, started: float, n_now: int, meta: dict[str, Any]
) -> None:
    import torch
    import transformers

    task.pset.model_info.update(task.info)
    task.pset.model_info.update(meta)
    task.pset.model_info.update(
        {
            "device": device,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "machine": platform.processor() or platform.machine(),
            "n_score_fallbacks": getattr(runner, "n_score_fallbacks", 0),
            "seconds_per_image_this_session": (time.time() - started) / n_now if n_now else None,
        }
    )


def run(
    model: str,
    ds: Dataset,
    images_dir: Path,
    split: str,
    out: Path,
    limit: int | None = None,
    seed: int = 0,
    device: str = "cpu",
    save_every: int = 5,
    model_id: str | None = None,
    adapter: str | None = None,
) -> list[PredictionSet]:
    names = select_images(ds, split, limit, seed)
    meta = {"split": split, "sample_seed": seed, "limit": limit}
    tasks, runner = build_tasks(model, out, ds, split, names, device, model_id, adapter)
    # An image counts as done only if EVERY output file has it; others are redone cleanly.
    done = set.intersection(*(set(t.pset.model_info.get("images", [])) for t in tasks))
    todo = [n for n in names if n not in done]
    for t in tasks:
        t.pset.predictions = [p for p in t.pset.predictions if p.file_name not in set(todo)]
        t.pset.model_info["images"] = [n for n in t.pset.model_info.get("images", []) if n in done]
    started = time.time()
    for k, name in enumerate(todo, start=1):
        image = Image.open(images_dir / name).convert("RGB")
        for t in tasks:
            t.pset.predictions.extend(t.answer(name, image))
            t.pset.model_info["images"].append(name)
        if k % save_every == 0 or k == len(todo):
            for t in tasks:
                _stamp(t, runner, device, started, k, meta)
                t.pset.save(t.path)
            n_preds = sum(len(t.pset.predictions) for t in tasks)
            print(
                f"{len(names) - len(todo) + k}/{len(names)} images, {n_preds} predictions, "
                f"{(time.time() - started) / k:.1f} s/image",
                flush=True,
            )
    if not todo:
        for t in tasks:
            _stamp(t, runner, device, started, 0, meta)
            t.pset.save(t.path)
    return [t.pset for t in tasks]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, choices=["owlv2", "florence"])
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--model-id", help="override the Hugging Face model (e.g. Florence-2-large)")
    p.add_argument("--adapter", help="LoRA adapter directory from sitesight.train.lora")
    p.add_argument(
        "--allow-test",
        action="store_true",
        help="permit --split test: ONLY for the single final evaluation, never to choose anything",
    )
    args = p.parse_args(argv)
    if args.split == "test" and not args.allow_test:
        p.error("the test split is reserved for the final evaluation (pass --allow-test once)")
    if args.adapter and args.model != "florence":
        p.error("--adapter applies to --model florence")
    ds = Dataset.load(args.annotations)
    run(args.model, ds, args.images, args.split, args.out, args.limit, args.seed,
        args.device, args.save_every, args.model_id, args.adapter)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
