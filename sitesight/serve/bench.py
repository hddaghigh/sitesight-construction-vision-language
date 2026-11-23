"""Parity, accuracy and latency of PyTorch vs ONNX fp32 vs ONNX int8 on the same validation regions.

Run in the Colab CPU runtime (hardware is logged; the number is that machine's, not a laptop's):

    python -m sitesight.serve.bench --annotations ... --images ... --adapter ... \
        --onnx-dir export/ --out-dir bench/ --n-regions 300

What is measured, on one fixed random sample of validation regions (seeded, never ``test``):

* **Preprocessing parity**: the pure-numpy image/prompt code vs ``Florence2Processor`` (max absolute
  pixel difference, prompt-id equality).
* **Accuracy**: strict resource / colour / count accuracy of each engine against the parsed ground
  truth, so the int8 cost is stated in the metric the project cares about, not only in agreement.
* **Agreement with PyTorch**: share of identical captions and mean absolute confidence difference.
* **Latency**: per request, cold (vision features recomputed: one request = one new image),
  ``p50`` and ``p95`` over the sample after warm-up, single process, thread count logged.
  ``p95`` is the 95th percentile by linear interpolation between order statistics.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sitesight.baselines.prompts import FLORENCE_REGION_TASK, box_to_loc_tokens
from sitesight.data.schema import Dataset
from sitesight.eval.captioning import field_accuracy
from sitesight.eval.data import RegionGT, build_ground_truth
from sitesight.serve import codec


def percentile(values: list[float], q: float) -> float:
    """``q``-th percentile (0..100) by linear interpolation between order statistics."""
    return float(np.percentile(np.asarray(values, dtype=float), q))


def latency_summary(seconds: list[float]) -> dict[str, float]:
    return {
        "n": float(len(seconds)),
        "p50_s": percentile(seconds, 50),
        "p95_s": percentile(seconds, 95),
        "mean_s": float(np.mean(seconds)),
    }


def agreement(base: list[tuple[str, float]], other: list[tuple[str, float]]) -> dict[str, float]:
    """Caption identity rate and mean |confidence difference| of ``other`` against ``base``."""
    if len(base) != len(other) or not base:
        raise ValueError("need two non-empty, aligned prediction lists")
    same = sum(a[0] == b[0] for a, b in zip(base, other, strict=True)) / len(base)
    dconf = float(np.mean([abs(a[1] - b[1]) for a, b in zip(base, other, strict=True)]))
    return {"identical_caption_rate": same, "mean_abs_confidence_diff": dconf}


def sample_regions(regions: list[RegionGT], n: int, seed: int) -> list[RegionGT]:
    ordered = sorted(regions, key=lambda r: r.ann_id)
    return random.Random(seed).sample(ordered, min(n, len(ordered)))


def hardware() -> dict[str, Any]:
    import onnxruntime as ort

    cpu = platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    import os

    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
    }


def run_engine(
    describe: Any, regions: list[RegionGT], images: Path, reset_cache: Any
) -> tuple[list[tuple[str, float]], list[float]]:
    """Caption every region once (cold: the vision cache is cleared first); time each call."""
    preds: list[tuple[str, float]] = []
    secs: list[float] = []
    for k, r in enumerate(regions):
        with Image.open(images / r.file_name) as im:
            img = im.convert("RGB")
        reset_cache()
        t0 = time.perf_counter()
        preds.append(describe(img, r.bbox))
        dt = time.perf_counter() - t0
        if k >= 3:  # the first calls pay one-off costs (allocation, lazy init)
            secs.append(dt)
    return preds, secs


def preprocessing_parity(
    processor: Any, tokenizer: Any, meta: dict[str, Any], images: Path, regions: list[RegionGT]
) -> dict[str, Any]:
    max_diff, id_equal = 0.0, 0
    for r in regions:
        with Image.open(images / r.file_name) as im:
            img = im.convert("RGB")
        text = codec.region_prompt(r.bbox, *img.size)
        locs = box_to_loc_tokens(r.bbox, *img.size)
        ref = processor(text=FLORENCE_REGION_TASK + locs, images=img, return_tensors="np")
        mine = codec.preprocess(img, int(meta["image_size"]))
        max_diff = max(max_diff, float(np.abs(ref["pixel_values"] - mine).max()))
        ids = codec.prompt_ids(
            tokenizer, text, int(meta["image_token_id"]), int(meta["n_image_tokens"])
        )
        id_equal += int(ids == ref["input_ids"][0].tolist())
    return {
        "n": len(regions),
        "max_abs_pixel_diff": max_diff,
        "prompt_ids_identical": id_equal,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--onnx-dir", type=Path, required=True, help="holds fp32/ and int8/")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-regions", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--split", default="val")
    args = ap.parse_args(argv)
    if args.split == "test":
        raise SystemExit("the benchmark never touches the test split")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from tokenizers import Tokenizer

    from sitesight.baselines.florence import FlorenceRunner
    from sitesight.serve.onnx_backend import OnnxFlorence

    ds = Dataset.load(args.annotations)
    regions = sample_regions(build_ground_truth(ds, args.split)[1], args.n_regions, args.seed)
    gt = [r.fields for r in regions]
    tok = Tokenizer.from_file(str(args.onnx_dir / "fp32" / "tokenizer.json"))

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    runner = FlorenceRunner(adapter=str(args.adapter))

    def reset_torch() -> None:
        runner._feat_key = None

    engines: dict[str, tuple[Any, Any]] = {"pytorch_fp32": (runner.describe_region, reset_torch)}
    onnx: dict[str, OnnxFlorence] = {}
    for prec in ("fp32", "int8"):
        b = OnnxFlorence(args.onnx_dir / prec, tok, threads=args.threads)
        onnx[prec] = b

        def reset(b: OnnxFlorence = b) -> None:
            b._feat_key = None

        engines[f"onnx_{prec}"] = (b.describe_region, reset)

    results: dict[str, Any] = {
        "hardware": hardware() | {"torch": torch.__version__, "threads": args.threads},
        "split": args.split,
        "seed": args.seed,
        "n_regions": len(regions),
        "preprocessing_parity": preprocessing_parity(
            runner.processor, tok, onnx["fp32"].meta, args.images, regions[:20]
        ),
        "engines": {},
    }
    preds: dict[str, list[tuple[str, float]]] = {}
    for name, (describe, reset_cache) in engines.items():
        print(f"running {name} on {len(regions)} regions", flush=True)
        preds[name], secs = run_engine(describe, regions, args.images, reset_cache)
        acc = field_accuracy([p[0] for p in preds[name]], gt)
        results["engines"][name] = {
            "latency_cold": latency_summary(secs),
            "accuracy": {k: v for k, v in acc.items() if "strict" in k or k == "n"},
        }
    for prec in ("fp32", "int8"):
        d = args.onnx_dir / prec
        results["engines"][f"onnx_{prec}"]["size_mb"] = round(
            sum(f.stat().st_size for f in d.glob("*.onnx")) / 1e6, 1
        )
        results["engines"][f"onnx_{prec}"]["agreement_with_pytorch"] = agreement(
            preds["pytorch_fp32"], preds[f"onnx_{prec}"]
        )
    out = args.out_dir / "bench.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.out_dir / "predictions.json").write_text(
        json.dumps(
            {
                k: [
                    {"ann_id": r.ann_id, "caption": c, "confidence": s}
                    for r, (c, s) in zip(regions, v, strict=True)
                ]
                for k, v in preds.items()
            }
        ),
        encoding="utf-8",
    )
    print("BENCH_JSON_BEGIN")
    print(json.dumps(results, separators=(",", ":"), sort_keys=True))
    print("BENCH_JSON_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
