"""Export Florence-2 (optionally with a merged LoRA adapter) to three ONNX graphs, then int8.

Why three graphs and not one. ``generate`` is a Python loop around the model, which ONNX cannot
express, so we export the pieces and write the greedy loop ourselves
(:mod:`sitesight.serve.onnx_backend`):

* ``vision.onnx``: ``pixel_values (1,3,H,W) -> image_features (1,N,D)``. The expensive part, run
  once per image.
* ``encoder.onnx``: ``input_ids (1,L), image_features (1,N,D), attention_mask (1,L) ->
  encoder_hidden_states (1,L,D)``. The prompt holds ``N`` image-placeholder ids; the graph swaps
  the k-th placeholder's embedding for the k-th image feature (``masked_scatter`` written with
  ``cumsum`` + ``gather`` + ``where`` so it exports).
* ``decoder.onnx``: ``decoder_input_ids (1,T), encoder_hidden_states, encoder_attention_mask ->
  logits (1,V)`` for the LAST position only. No key/value cache: every step re-runs the ``T``
  tokens so far. That costs ``O(T^2)`` instead of ``O(T)`` decoder work but is exactly the
  pretrained computation and cannot go wrong through cache plumbing; captions are short (<= 48
  tokens) and the decoder is small next to the vision tower. Batch size is fixed at 1 (serving
  handles one region at a time).

Int8: ``onnxruntime.quantization.quantize_dynamic`` stores ``MatMul``/``Gemm`` weights as int8 and
quantises activations on the fly. For a weight ``w`` with per-tensor scale ``s = max|w| / 127``
the stored value is ``round(w / s)`` clipped to [-127, 127] and the dequantised weight is
``s * q``, so the error per weight is at most ``s / 2`` (about 0.4% of the largest weight).
Convolutions (the vision tower's patch embeddings and depth-wise convs) are left in fp32:
dynamic quantisation of ``Conv`` needs constant weights, which the traced graph does not give.
Whether that matters for the task is measured, not assumed (``sitesight.serve.bench``).

Torch/onnx are imported lazily: the serving image never needs them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GRAPHS = ("vision", "encoder", "decoder")


def _wrappers(model: Any) -> tuple[Any, Any, Any]:
    import torch
    from torch import nn

    inner = model.model
    image_token_id = int(model.config.image_token_id)

    class Vision(nn.Module):
        def forward(self, pixel_values: Any) -> Any:
            return inner.get_image_features(pixel_values).pooler_output

    class Encoder(nn.Module):
        def forward(self, input_ids: Any, image_features: Any, attention_mask: Any) -> Any:
            embeds = inner.get_input_embeddings()(input_ids)
            is_img = input_ids == image_token_id  # (1, L)
            order = (torch.cumsum(is_img.long(), dim=1) - 1).clamp(min=0)  # k-th placeholder -> k
            picked = image_features[0][order[0]].unsqueeze(0)  # (1, L, D)
            embeds = torch.where(is_img.unsqueeze(-1), picked.to(embeds.dtype), embeds)
            out = inner.language_model.encoder(attention_mask=attention_mask, inputs_embeds=embeds)
            return out[0]

    class Decoder(nn.Module):
        def forward(self, decoder_input_ids: Any, encoder_hidden_states: Any, mask: Any) -> Any:
            out = inner.language_model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=mask,
                use_cache=False,
            )
            return model.lm_head(out.last_hidden_state[:, -1])

    return Vision().eval(), Encoder().eval(), Decoder().eval()


def export_onnx(
    model: Any, out_dir: Path, *, image_size: int = 768, opset: int = 17, prompt_len: int = 40
) -> dict[str, Any]:
    """Write the three graphs and ``meta.json`` into ``out_dir``; returns the metadata."""
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.eval().to("cpu").float().requires_grad_(False)  # weights export as constants
    vision, encoder, decoder = _wrappers(model)
    image_token_id = int(model.config.image_token_id)
    with torch.no_grad():
        px = torch.zeros(1, 3, image_size, image_size)
        feats = vision(px)
        n_img, dim = int(feats.shape[1]), int(feats.shape[2])
        ids = torch.cat(
            [
                torch.full((1, n_img), image_token_id, dtype=torch.long),
                torch.full((1, prompt_len), 5, dtype=torch.long),
            ],
            dim=1,
        )
        mask = torch.ones_like(ids)
        enc = encoder(ids, feats, mask)
        dec_ids = torch.tensor([[2, 0, 7]], dtype=torch.long)

        def dump(
            mod: Any, args: tuple[Any, ...], name: str, ins: list[str], axes: dict[str, Any]
        ) -> None:
            torch.onnx.export(
                mod,
                args,
                str(out_dir / f"{name}.onnx"),
                input_names=ins,
                output_names=["out"],
                dynamic_axes=axes,
                opset_version=opset,
                dynamo=False,
            )

        dump(vision, (px,), "vision", ["pixel_values"], {})
        dump(
            encoder,
            (ids, feats, mask),
            "encoder",
            ["input_ids", "image_features", "attention_mask"],
            {"input_ids": {1: "L"}, "attention_mask": {1: "L"}, "out": {1: "L"}},
        )
        dump(
            decoder,
            (dec_ids, enc, mask),
            "decoder",
            ["decoder_input_ids", "encoder_hidden_states", "encoder_attention_mask"],
            {
                "decoder_input_ids": {1: "T"},
                "encoder_hidden_states": {1: "L"},
                "encoder_attention_mask": {1: "L"},
            },
        )
    gen = model.generation_config
    meta = {
        "image_size": image_size,
        "n_image_tokens": n_img,
        "hidden_dim": dim,
        "image_token_id": image_token_id,
        "decoder_start_token_id": int(model.config.text_config.decoder_start_token_id),
        "eos_token_id": int(model.config.text_config.eos_token_id),
        "forced_bos_token_id": getattr(gen, "forced_bos_token_id", None),
        "no_repeat_ngram_size": int(getattr(gen, "no_repeat_ngram_size", 0) or 0),
        "opset": opset,
        "precision": "fp32",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def quantize_int8(src_dir: Path, dst_dir: Path) -> dict[str, int]:
    """Dynamic int8 weight quantisation of all three graphs; returns file sizes in bytes."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    dst_dir.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, int] = {}
    for name in GRAPHS:
        quantize_dynamic(
            src_dir / f"{name}.onnx",
            dst_dir / f"{name}.onnx",
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],  # convolutions stay fp32 (see docstring)
        )
        sizes[name] = (dst_dir / f"{name}.onnx").stat().st_size
    meta = json.loads((src_dir / "meta.json").read_text(encoding="utf-8"))
    meta["precision"] = "int8-dynamic"
    (dst_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return sizes


def main(argv: list[str] | None = None) -> int:
    """``python -m sitesight.serve.export --adapter DIR --out DIR``: fp32 + int8 + tokenizer."""
    import argparse

    ap = argparse.ArgumentParser(description="Export Florence-2 (+ LoRA) to ONNX fp32 and int8.")
    ap.add_argument("--model-id", default="florence-community/Florence-2-base")
    ap.add_argument("--adapter", type=Path, help="LoRA adapter directory to merge (optional)")
    ap.add_argument("--out", type=Path, required=True, help="writes OUT/fp32 and OUT/int8")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args(argv)

    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    model = Florence2ForConditionalGeneration.from_pretrained(args.model_id)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    meta = export_onnx(model, args.out / "fp32", opset=args.opset)
    sizes = quantize_int8(args.out / "fp32", args.out / "int8")
    tok_dir = args.out / "fp32"
    AutoProcessor.from_pretrained(args.model_id).tokenizer.save_pretrained(str(tok_dir))
    if not (tok_dir / "tokenizer.json").is_file():
        raise RuntimeError("tokenizer.json was not written; the ONNX backend needs it")
    (args.out / "int8" / "tokenizer.json").write_bytes((tok_dir / "tokenizer.json").read_bytes())
    print(json.dumps({"meta": meta, "int8_bytes": sizes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
