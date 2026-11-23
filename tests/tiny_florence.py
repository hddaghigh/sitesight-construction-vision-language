"""A tiny randomly initialised Florence-2 (real transformers classes, no download) for export tests."""

from __future__ import annotations

from typing import Any


def tiny_florence(seed: int = 0) -> Any:
    import torch
    from transformers import BartConfig, Florence2Config, Florence2ForConditionalGeneration
    from transformers.models.florence2.configuration_florence2 import Florence2VisionConfig

    torch.manual_seed(seed)
    vision = Florence2VisionConfig(
        depths=(1, 1, 1, 1),
        embed_dim=(8, 16, 16, 32),
        num_heads=(1, 2, 2, 4),
        num_groups=(1, 2, 2, 4),
        window_size=4,
        projection_dim=24,
    )
    text = BartConfig(
        vocab_size=100,
        d_model=24,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=128,
        decoder_start_token_id=2,
        eos_token_id=2,
        bos_token_id=0,
        pad_token_id=1,
    )
    cfg = Florence2Config(vision_config=vision, text_config=text, image_token_id=99)
    return Florence2ForConditionalGeneration(cfg).eval()
