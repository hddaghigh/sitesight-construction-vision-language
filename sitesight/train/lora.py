"""LoRA fine-tuning of Florence-2 with cached (frozen) vision features.

Why this is cheap enough for a free-tier GPU. Florence-2 = frozen-able vision encoder (DaViT) +
transformer encoder-decoder language model. We adapt only the language model with LoRA, so the
image features (the expensive part: ~80% of the cost of one forward pass) never change; we compute
them ONCE per image, keep them in CPU memory in half precision, and feed them to the model through
its own forward pass (by replacing ``get_image_features`` with a lookup). Every training step then
costs only the language-model side. Consequence: no image augmentation (stated limitation).

LoRA in two lines. A weight ``W`` (d_out x d_in) is kept frozen and a low-rank update is learned:
``W' = W + (alpha / r) * B @ A`` with ``A`` (r x d_in), ``B`` (d_out x r), ``r`` much smaller than
``d``. Only ``A`` and ``B`` are trained (about 1% of the parameters here), and ``B`` starts at
zero so training starts exactly at the pretrained model.

Loss. The decoder is trained with teacher forcing: it reads the target text shifted right and
predicts the next token; the loss is the mean token-level cross-entropy
``-(1/N) sum_t log p(y_t | y_<t, image, prompt)``, averaged over all target tokens in the batch
(padding ignored). Boxes are just location tokens in the target, so one loss covers both tasks.

Run (Colab GPU): ``python -m sitesight.train.lora --config configs/train.toml --annotations ...
--images ... --out-dir runs/lora_v1``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from sitesight.data.schema import Dataset
from sitesight.train.data import Sample, build_samples


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cache_image_features(
    model: Any, processor: Any, images_dir: Path, names: list[str], device: str
) -> dict[str, Any]:
    """Projected vision features per image (CPU, float16). Call BEFORE the model is wrapped."""
    import torch

    feats: dict[str, Any] = {}
    t0 = time.time()
    with torch.no_grad():
        for k, name in enumerate(names, start=1):
            with Image.open(images_dir / name) as im:
                px = processor.image_processor(im.convert("RGB"), return_tensors="pt")[
                    "pixel_values"
                ]
            out = model.model.get_image_features(px.to(device))
            feats[name] = out.pooler_output[0].to("cpu", torch.float16)
            if k % 100 == 0 or k == len(names):
                print(
                    f"  cached vision features {k}/{len(names)} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
    return feats


def encode_samples(
    processor: Any, samples: list[Sample], max_target_tokens: int
) -> list[dict[str, Any]]:
    """Token ids for every sample. The prompt goes through the processor (it inserts the image
    placeholder tokens exactly as at inference); the target is the plain tokenised text, which the
    tokenizer wraps as ``<s> ... </s>``, the format the model itself generates."""
    tok = processor.tokenizer
    dummy = Image.new("RGB", (32, 32))
    prompt_cache: dict[str, list[int]] = {}
    out = []
    for s in samples:
        ids = prompt_cache.get(s.prompt)
        if ids is None:
            ids = processor(text=s.prompt, images=dummy, return_tensors="pt")["input_ids"][
                0
            ].tolist()
            prompt_cache[s.prompt] = ids
        target = tok(s.target).input_ids
        if len(target) > max_target_tokens:
            raise ValueError(f"target of {len(target)} tokens exceeds max_target_tokens: {s}")
        out.append({"sample": s, "input_ids": ids, "labels": target})
    return out


def make_batch(
    items: list[dict[str, Any]], feats: dict[str, Any], pad_id: int, device: str
) -> dict[str, Any]:
    import torch

    n = len(items)
    max_in = max(len(i["input_ids"]) for i in items)
    max_lab = max(len(i["labels"]) for i in items)
    input_ids = torch.full((n, max_in), pad_id, dtype=torch.long)
    attn = torch.zeros((n, max_in), dtype=torch.long)
    labels = torch.full((n, max_lab), -100, dtype=torch.long)
    for k, it in enumerate(items):
        ids, lab = it["input_ids"], it["labels"]
        input_ids[k, : len(ids)] = torch.tensor(ids)
        attn[k, : len(ids)] = 1
        labels[k, : len(lab)] = torch.tensor(lab)
    f = torch.stack([feats[it["sample"].file_name] for it in items]).to(device, torch.float32)
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attn.to(device),
        "labels": labels.to(device),
        "feats": f,
    }


def _loss(peft_model: Any, inner: Any, batch: dict[str, Any]) -> Any:
    import torch

    feats = batch["feats"]
    # Replace the vision tower call by a lookup of the cached features for THIS batch.
    inner.get_image_features = lambda pixel_values, **kw: SimpleNamespace(pooler_output=feats)
    dummy_pixels = torch.zeros(
        (feats.shape[0], 3, 1, 1), device=feats.device
    )  # only its presence matters
    out = peft_model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=dummy_pixels,
        labels=batch["labels"],
    )
    return out.loss


def evaluate_loss(
    peft_model: Any,
    inner: Any,
    items: list[dict[str, Any]],
    feats: dict[str, Any],
    bs: int,
    pad_id: int,
    device: str,
) -> dict[str, float]:
    """Mean token loss on a set of samples, overall and per task."""
    import torch

    peft_model.eval()
    sums: dict[str, list[float]] = {
        "all": [0.0, 0.0],
        "region": [0.0, 0.0],
        "grounding": [0.0, 0.0],
    }
    with torch.no_grad():
        for task in ("region", "grounding"):
            sub = [i for i in items if i["sample"].task == task]
            for k in range(0, len(sub), bs):
                chunk = sub[k : k + bs]
                loss = float(_loss(peft_model, inner, make_batch(chunk, feats, pad_id, device)))
                n_tok = sum(len(c["labels"]) for c in chunk)  # weight batches by their token count
                for key in (task, "all"):
                    sums[key][0] += loss * n_tok
                    sums[key][1] += n_tok
    return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in sums.items()}


def run_training(
    cfg: dict[str, Any],
    ds: Dataset,
    images_dir: Path,
    out_dir: Path,
    device: str = "cuda",
    limit_images: int | None = None,
) -> dict[str, Any]:
    import peft
    import torch
    import transformers
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    t_start = time.time()
    tc, lc = cfg["train"], cfg["lora"]
    set_seed(tc["seed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples, train_stats = build_samples(ds, cfg["data"]["train_split"])
    val_samples, val_stats = build_samples(ds, cfg["data"]["val_split"])
    if limit_images:  # smoke tests only: first N images (sorted), and N // 3 + 1 validation images
        keep_t = sorted({s.file_name for s in train_samples}, key=lambda n: (len(n), n))[
            :limit_images
        ]
        keep_v = sorted({s.file_name for s in val_samples}, key=lambda n: (len(n), n))[
            : limit_images // 3 + 1
        ]
        train_samples = [s for s in train_samples if s.file_name in set(keep_t)]
        val_samples = [s for s in val_samples if s.file_name in set(keep_v)]
    print(f"train samples {len(train_samples)}, val samples {len(val_samples)}", flush=True)

    processor = AutoProcessor.from_pretrained(cfg["model"]["id"])
    model = Florence2ForConditionalGeneration.from_pretrained(cfg["model"]["id"]).to(device)
    model.eval()
    names = sorted({s.file_name for s in train_samples + val_samples}, key=lambda n: (len(n), n))
    print(f"caching vision features for {len(names)} images on {device}...", flush=True)
    feats = cache_image_features(model, processor, images_dir, names, device)

    for p in model.parameters():
        p.requires_grad_(False)
    pcfg = peft.LoraConfig(
        r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"],
        target_modules=cfg["lora"]["target_regex"],
    )
    peft_model = peft.get_peft_model(model, pcfg)
    inner = peft_model.get_base_model().model
    n_train = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in peft_model.parameters())
    print(
        f"trainable parameters: {n_train:,} of {n_total:,} ({100 * n_train / n_total:.2f}%)",
        flush=True,
    )

    pad_id = processor.tokenizer.pad_token_id
    train_items = encode_samples(processor, train_samples, tc["max_target_tokens"])
    val_items = encode_samples(processor, val_samples, tc["max_target_tokens"])
    params = [p for p in peft_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    bs = tc["batch_size"]
    steps_per_epoch = math.ceil(len(train_items) / bs)
    total_steps = steps_per_epoch * tc["epochs"]
    warm = max(1, int(tc["warmup_frac"] * total_steps))

    def lr_at(step: int) -> float:  # linear warm-up, then cosine decay to 0
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    log_rows: list[dict[str, Any]] = []
    best = {"epoch": 0, "val_loss": float("inf")}
    base_val = evaluate_loss(peft_model, inner, val_items, feats, bs, pad_id, device)
    print(f"epoch 0 (pretrained): val loss {base_val}", flush=True)
    log_rows.append({"epoch": 0, "step": 0, "train_loss": "", "lr": 0.0, "seconds": 0.0,
                     **{f"val_loss_{k}": round(v, 5) for k, v in base_val.items()}})  # fmt: skip
    step = 0
    for epoch in range(1, tc["epochs"] + 1):
        t_ep = time.time()
        peft_model.train()
        order = list(range(len(train_items)))
        random.Random(tc["seed"] * 1000 + epoch).shuffle(order)
        run_loss, run_tok = 0.0, 0
        for k in range(0, len(order), bs):
            chunk = [train_items[i] for i in order[k : k + bs]]
            batch = make_batch(chunk, feats, pad_id, device)
            loss = _loss(peft_model, inner, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tc["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            n_tok = sum(len(c["labels"]) for c in chunk)
            run_loss += float(loss) * n_tok
            run_tok += n_tok
            if step % 50 == 0:
                print(f"  step {step}/{total_steps} loss {float(loss):.4f}", flush=True)
        val = evaluate_loss(peft_model, inner, val_items, feats, bs, pad_id, device)
        row = {"epoch": epoch, "step": step, "train_loss": round(run_loss / run_tok, 5),
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t_ep, 1),
               **{f"val_loss_{k}": round(v, 5) for k, v in val.items()}}  # fmt: skip
        log_rows.append(row)
        print(f"epoch {epoch}: {row}", flush=True)
        peft_model.save_pretrained(out_dir / f"epoch_{epoch}")
        if val["all"] < best["val_loss"]:
            best = {"epoch": epoch, "val_loss": val["all"]}

    with (out_dir / "experiment_log.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        w.writeheader()
        w.writerows(log_rows)
    gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") else None
    summary = {
        "config": cfg,
        "seed": tc["seed"],
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "pretrained_val_loss": base_val["all"],
        "trainable_parameters": n_train,
        "total_parameters": n_total,
        "train_data": train_stats,
        "val_data": val_stats,
        "n_train_samples": len(train_items),
        "n_val_samples": len(val_items),
        "n_images_cached": len(names),
        "limit_images": limit_images,
        "total_seconds": round(time.time() - t_start, 1),
        "device": device,
        "gpu": gpu,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "python": platform.python_version(),
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8"
    )
    best_dir = out_dir / f"epoch_{best['epoch']}"
    (out_dir / "BEST_EPOCH.txt").write_text(f"{best['epoch']}\n{best_dir.name}\n", encoding="utf-8")
    print(f"done: best epoch {best['epoch']} (val loss {best['val_loss']:.4f})", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--limit-images", type=int, help="smoke test: use only the first N training images"
    )
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if cfg["data"]["train_split"] == "test" or cfg["data"]["val_split"] == "test":
        p.error("the test split must not be used for training or model selection")
    run_training(
        cfg,
        Dataset.load(args.annotations),
        args.images,
        args.out_dir,
        args.device,
        args.limit_images,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
