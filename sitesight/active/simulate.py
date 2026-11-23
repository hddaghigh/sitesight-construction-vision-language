"""Active-learning simulation: how many labelled images does each acquisition strategy need?

Protocol (fixed in ``configs/active.toml`` before any run):

1. The pool is the training images that have region captions. Vision features are cached once
   (``train.lora.cache_image_features``) and also give a per-image embedding (mean over the feature
   tokens) for the diversity strategy.
2. Per seed: draw the initial labelled set at random (identical for every strategy), then grow it to
   each cumulative budget. At every budget a FRESH LoRA (same recipe as M5, 3 epochs) is trained on
   the labelled images' region captions and scored on the validation split. The score is the mean of
   the strict resource, colour and count accuracies of the generated captions (field-level, because
   caption n-gram metrics reward the dataset's rigid template; see ``eval.captioning``).
3. After each round the strategy scores the still-unlabelled pool images with the current model:
   it generates a caption for every pool region (boxes are assumed known; only captions are bought)
   and records the geometric-mean token probability. See ``active.ranking`` for the strategies.
4. A full-data reference (all pool images, same recipe) is trained once per seed.

Results are appended to ``results.jsonl`` after every round, so an interrupted Colab session resumes
without redoing finished rounds. The test split is never read.

Run (Colab GPU): ``python -m sitesight.active.simulate --config configs/active.toml --annotations
... --images ... --out-dir runs/active``. ``--limit-images N`` is a smoke test (tiny pool).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from sitesight.active.ranking import (
    STRATEGIES,
    image_uncertainty,
    select,
    summarise_curves,
)
from sitesight.data.schema import Dataset
from sitesight.eval.captioning import field_accuracy
from sitesight.eval.data import build_ground_truth
from sitesight.train.data import Sample, build_samples
from sitesight.train.lora import (
    _loss,
    cache_image_features,
    encode_samples,
    load_config,
    make_batch,
    set_seed,
)

FIELD_KEYS = ("resource_acc_strict", "colour_acc_strict", "count_acc_strict")


def field_score(acc: dict[str, Any]) -> float:
    """Headline score: mean of the strict resource, colour and count accuracies (0-1)."""
    return float(np.mean([acc[k] for k in FIELD_KEYS]))


def budgets(pool_size: int, fractions: list[float]) -> list[int]:
    """Cumulative image budgets (rounded, at least 1, strictly increasing, at most the pool)."""
    out: list[int] = []
    for f in fractions:
        n = min(pool_size, max(1, round(f * pool_size)))
        if out and n <= out[-1]:
            raise ValueError(
                f"budget fractions {fractions} give non-increasing budgets at pool {pool_size}"
            )
        out.append(n)
    return out


def generate(
    peft_model: Any, inner: Any, items: list[dict[str, Any]], feats: dict[str, Any],
    processor: Any, device: str, max_new_tokens: int, bs: int,
) -> tuple[list[str], list[float]]:  # fmt: skip
    """Greedy captions and their confidences (geometric-mean token probability) for region items."""
    import torch

    from sitesight.baselines.prompts import strip_loc_tokens

    tok = processor.tokenizer
    pad, bos, eos = tok.pad_token_id, tok.bos_token_id, tok.eos_token_id
    peft_model.eval()
    texts: list[str] = []
    conf: list[float] = []
    with torch.no_grad():
        for k in range(0, len(items), bs):
            batch = make_batch(items[k : k + bs], feats, pad, device)
            f = batch["feats"]
            inner.get_image_features = lambda pixel_values, _f=f, **kw: type(
                "O", (), {"pooler_output": _f}
            )()
            out = peft_model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=torch.zeros((f.shape[0], 3, 1, 1), device=device),
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
            steps = len(out.scores)
            toks = out.sequences[:, -steps:]
            logp = torch.stack(
                [torch.log_softmax(s.float(), -1).gather(1, toks[:, t : t + 1])[:, 0]
                 for t, s in enumerate(out.scores)], dim=1,
            )  # fmt: skip
            is_eos = (toks == eos).long()
            before_or_at_first_eos = (is_eos.cumsum(1) - is_eos) == 0
            valid = before_or_at_first_eos & (toks != pad) & (toks != bos)
            for i in range(toks.shape[0]):
                n = int(valid[i].sum())
                conf.append(math.exp(float(logp[i][valid[i]].sum()) / n) if n else 0.0)
            texts += [
                strip_loc_tokens(t)
                for t in processor.batch_decode(out.sequences, skip_special_tokens=True)
            ]
    return texts, conf


def train_fresh_lora(
    model: Any, cfg: dict[str, Any], items: list[dict[str, Any]], feats: dict[str, Any],
    pad_id: int, device: str, seed: int,
) -> tuple[Any, Any]:  # fmt: skip
    """Train a new LoRA on the pretrained ``model``; returns (peft model, inner model)."""
    import peft
    import torch

    tc, lc = cfg["train"], cfg["lora"]
    assert not any("lora_" in n for n, _ in model.named_parameters()), (
        "stale LoRA on the base model"
    )
    set_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    pm = peft.get_peft_model(
        model,
        peft.LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                        target_modules=lc["target_regex"]),
    )  # fmt: skip
    inner = pm.get_base_model().model
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    bs = tc["batch_size"]
    total = math.ceil(len(items) / bs) * tc["epochs"]
    warm = max(1, int(tc["warmup_frac"] * total))

    def lr_at(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    pm.train()
    for epoch in range(1, tc["epochs"] + 1):
        order = list(range(len(items)))
        random.Random(seed * 1000 + epoch).shuffle(order)
        for k in range(0, len(order), bs):
            batch = make_batch([items[i] for i in order[k : k + bs]], feats, pad_id, device)
            loss = _loss(pm, inner, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, tc["grad_clip"])
            opt.step()
            sched.step()
    return pm, inner


def done_keys(path: Path) -> dict[tuple[int, str, int], dict[str, Any]]:
    """Rows already written, keyed by (seed, strategy, round); 'full' is the reference."""
    rows: dict[tuple[int, str, int], dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["seed"], r["strategy"], r["round"])] = r
    return rows


def run_simulation(
    cfg: dict[str, Any], ds: Dataset, images_dir: Path, out_dir: Path, device: str = "cuda",
    limit_images: int | None = None,
) -> dict[str, Any]:  # fmt: skip
    import torch
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    ac = cfg["active"]
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.jsonl"
    finished = done_keys(res_path)

    def emit(row: dict[str, Any]) -> None:
        finished[(row["seed"], row["strategy"], row["round"])] = row
        with res_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

    train_s = [s for s in build_samples(ds, cfg["data"]["train_split"])[0] if s.task == "region"]
    val_s = [s for s in build_samples(ds, cfg["data"]["val_split"])[0] if s.task == "region"]
    pool = sorted({s.file_name for s in train_s}, key=lambda n: (len(n), n))
    if limit_images:  # smoke test only
        pool = pool[:limit_images]
        train_s = [s for s in train_s if s.file_name in set(pool)]
        val_s = val_s[: 4 * limit_images]
    val_regions = {str(r.ann_id): r for r in build_ground_truth(ds, cfg["data"]["val_split"])[1]}
    val_gt = [val_regions[s.ref].fields for s in val_s]
    sizes = budgets(len(pool), ac["budget_fractions"])
    print(
        f"pool {len(pool)} images, {len(train_s)} regions; budgets {sizes}; val {len(val_s)}",
        flush=True,
    )

    processor = AutoProcessor.from_pretrained(cfg["model"]["id"])
    model = Florence2ForConditionalGeneration.from_pretrained(cfg["model"]["id"]).to(device)
    model.eval()
    names = sorted({s.file_name for s in train_s + val_s}, key=lambda n: (len(n), n))
    feats = cache_image_features(model, processor, images_dir, names, device)
    emb = {n: feats[n].float().mean(0).numpy() for n in pool}
    pad_id = processor.tokenizer.pad_token_id
    mt = cfg["train"]["max_target_tokens"]
    by_image: dict[str, list[Sample]] = {}
    for s in train_s:
        by_image.setdefault(s.file_name, []).append(s)
    train_items = {
        s.ref: it for s, it in zip(train_s, encode_samples(processor, train_s, mt), strict=True)
    }
    val_items = encode_samples(processor, val_s, mt)

    def items_for(images: list[str]) -> list[dict[str, Any]]:
        return [train_items[s.ref] for n in images for s in by_image[n]]

    def evaluate(pm: Any, inner: Any) -> dict[str, Any]:
        texts, _ = generate(pm, inner, val_items, feats, processor, device,
                            ac["max_new_tokens"], ac["gen_batch_size"])  # fmt: skip
        acc = field_accuracy(texts, val_gt)
        return {
            "score": field_score(acc),
            **{k: acc[k] for k in (*FIELD_KEYS, "strict_parse_rate")},
        }

    def pool_uncertainty(pm: Any, inner: Any, labelled: list[str]) -> dict[str, float]:
        rest = [n for n in pool if n not in set(labelled)]
        its = items_for(rest)
        _, conf = generate(pm, inner, its, feats, processor, device,
                           ac["max_new_tokens"], ac["gen_batch_size"])  # fmt: skip
        per: dict[str, list[float]] = {}
        for it, c in zip(its, conf, strict=True):
            per.setdefault(it["sample"].file_name, []).append(c)
        return image_uncertainty(per)

    def run_round(
        labelled: list[str], seed: int, need_unc: bool
    ) -> tuple[dict[str, Any], dict[str, float] | None]:
        t0 = time.time()
        nonlocal model
        pm, inner = train_fresh_lora(model, cfg, items_for(labelled), feats, pad_id, device, seed)
        metrics = evaluate(pm, inner)
        unc = pool_uncertainty(pm, inner, labelled) if need_unc else None
        model = (
            pm.unload()
        )  # drop the LoRA layers: the next round starts from the pretrained weights
        metrics.update(n_images=len(labelled), n_regions=len(items_for(labelled)),
                       seconds=round(time.time() - t0, 1))  # fmt: skip
        return metrics, unc

    def unc_path(seed: int, strat: str, rnd: int) -> Path:
        return out_dir / f"unc_s{seed}_{strat}_r{rnd}.json"

    def save_unc(path: Path, unc: dict[str, float] | None) -> None:
        if unc is not None:
            path.write_text(json.dumps(unc, sort_keys=True), encoding="utf-8")

    def load_unc(path: Path) -> dict[str, float] | None:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    strategies = ac["strategies"]
    mult = ac["candidate_multiplier"]
    for seed in ac["seeds"]:
        if (seed, "full", 0) not in finished:
            m, _ = run_round(pool, seed, False)
            emit({"seed": seed, "strategy": "full", "round": 0, **m})
        initial = select("random", pool, [], sizes[0], seed=seed)
        need = len(sizes) > 1 and any(st != "random" for st in strategies)
        shared_unc = load_unc(unc_path(seed, "shared", 0))
        if not all((seed, st, 0) in finished for st in strategies) or (need and shared_unc is None):
            r0, shared_unc = run_round(
                initial, seed, need
            )  # identical for every strategy: train once
            save_unc(unc_path(seed, "shared", 0), shared_unc)
            for st in strategies:
                if (seed, st, 0) not in finished:
                    emit({"seed": seed, "strategy": st, "round": 0, **r0})
        for strat in strategies:
            labelled = list(initial)
            unc = shared_unc
            for rnd in range(1, len(sizes)):
                done = finished.get((seed, strat, rnd))
                if done is not None:  # resume: reuse the recorded choice and the saved scores
                    labelled += done["selected"]
                    unc = load_unc(unc_path(seed, strat, rnd))
                    continue
                k = sizes[rnd] - len(labelled)
                new = select(strat, pool, labelled, k, seed=seed * 100 + rnd, uncertainty=unc,
                             embeddings=emb, candidate_multiplier=mult)  # fmt: skip
                labelled += new
                m, unc = run_round(labelled, seed, strat != "random" and rnd < len(sizes) - 1)
                save_unc(unc_path(seed, strat, rnd), unc)
                emit({"seed": seed, "strategy": strat, "round": rnd, "selected": sorted(new), **m})

    rows = [r for r in finished.values() if r["strategy"] != "full"]
    full = {r["seed"]: r["score"] for r in finished.values() if r["strategy"] == "full"}
    summary = {
        "config": cfg,
        "pool_images": len(pool),
        "pool_regions": len(train_s),
        "budgets": sizes,
        "full_scores": full,
        "curves": summarise_curves(rows, full, ac["target_fraction"]),
        "limit_images": limit_images,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True), encoding="utf-8"
    )
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
    p.add_argument("--limit-images", type=int, help="smoke test: tiny pool")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if "test" in (cfg["data"]["train_split"], cfg["data"]["val_split"]):
        p.error("the test split must not be used here")
    unknown = set(cfg["active"]["strategies"]) - set(STRATEGIES)
    if unknown:
        p.error(f"unknown strategies {sorted(unknown)}")
    run_simulation(
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
