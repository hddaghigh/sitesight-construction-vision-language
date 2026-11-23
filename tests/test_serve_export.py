"""ONNX export parity on a tiny random Florence-2 (real transformers classes, no download).

Opt-in (``pytest -m embed``): needs torch, transformers, onnx and onnxruntime. The real-weights
parity (processor, tokenizer, full model) is measured in Colab by ``sitesight.serve.bench``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sitesight.serve.export import _wrappers, export_onnx, quantize_int8
from sitesight.serve.onnx_backend import OnnxFlorence
from tests.tiny_florence import tiny_florence

pytestmark = pytest.mark.embed


SIZE, PROMPT = 64, 6


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Path, dict[str, Any]]:
    model = tiny_florence()
    out = tmp_path_factory.mktemp("onnx")
    meta = export_onnx(model, out, image_size=SIZE, prompt_len=PROMPT)
    return model, out, meta


def _sess(d: Path, name: str) -> Any:
    import onnxruntime as ort

    return ort.InferenceSession(str(d / f"{name}.onnx"), providers=["CPUExecutionProvider"])


def test_each_graph_matches_torch_at_other_lengths(
    exported: tuple[Any, Path, dict[str, Any]],
) -> None:
    model, d, meta = exported
    vision, encoder, decoder = _wrappers(model)
    import torch

    g = torch.Generator().manual_seed(1)
    px = torch.randn(1, 3, SIZE, SIZE, generator=g)
    with torch.no_grad():
        feats = vision(px)
        got = _sess(d, "vision").run(None, {"pixel_values": px.numpy()})[0]
        np.testing.assert_allclose(got, feats.numpy(), atol=1e-4)
        for text_len in (3, 11):  # different from the traced prompt length
            ids = torch.cat(
                [
                    torch.full((1, meta["n_image_tokens"]), meta["image_token_id"]),
                    torch.randint(3, 90, (1, text_len), generator=g),
                ],
                dim=1,
            )
            mask = torch.ones_like(ids)
            enc = encoder(ids, feats, mask)
            got_enc = _sess(d, "encoder").run(
                None,
                {
                    "input_ids": ids.numpy(),
                    "image_features": feats.numpy(),
                    "attention_mask": mask.numpy(),
                },
            )[0]
            np.testing.assert_allclose(got_enc, enc.numpy(), atol=1e-4)
            for t in (1, 4, 7):  # different decoder lengths
                dec = torch.randint(3, 90, (1, t), generator=g)
                want = decoder(dec, enc, mask).numpy()
                got_dec = _sess(d, "decoder").run(
                    None,
                    {
                        "decoder_input_ids": dec.numpy(),
                        "encoder_hidden_states": enc.numpy(),
                        "encoder_attention_mask": mask.numpy(),
                    },
                )[0]
                np.testing.assert_allclose(got_dec, want, atol=1e-4)


def test_greedy_loop_matches_a_torch_reference(exported: tuple[Any, Path, dict[str, Any]]) -> None:
    import torch

    model, d, meta = exported
    g = torch.Generator().manual_seed(2)
    px = torch.randn(1, 3, SIZE, SIZE, generator=g)
    text = torch.randint(3, 90, (1, 5), generator=g)
    ids = torch.cat([torch.full((1, meta["n_image_tokens"]), meta["image_token_id"]), text], 1)
    backend = OnnxFlorence(d, tokenizer=None)
    eos = (
        77  # a random model stops at once on its real eos; use a rarely-chosen id so the loop runs
    )
    backend.meta["eos_token_id"] = eos
    got_ids, got_lp = backend.generate_ids(px.numpy(), ids[0].tolist(), 8)

    prev = [meta["decoder_start_token_id"]]
    want_ids: list[int] = []
    want_lp: list[float] = []
    with torch.no_grad():
        for _ in range(8):  # the full model, no wrappers: independent of the export code
            out = model(input_ids=ids, pixel_values=px, decoder_input_ids=torch.tensor([prev]))
            logp = torch.log_softmax(out.logits[0, -1].double(), dim=-1)
            tok = int(logp.argmax())
            want_ids.append(tok)
            want_lp.append(float(logp[tok]))
            prev.append(tok)
            if tok == eos:
                break
    assert len(want_ids) >= 5  # vacuous if the loop stopped after a token or two
    assert got_ids == want_ids
    np.testing.assert_allclose(got_lp, want_lp, atol=1e-4)


def test_feature_cache_reuses_the_vision_pass(exported: tuple[Any, Path, dict[str, Any]]) -> None:
    _, d, meta = exported
    px = np.random.default_rng(0).standard_normal((1, 3, SIZE, SIZE)).astype(np.float32)
    ids = [meta["image_token_id"]] * meta["n_image_tokens"] + [5, 6]
    backend = OnnxFlorence(d, tokenizer=None)
    first = backend.generate_ids(px, ids, 3)
    second = backend.generate_ids(px, ids, 3)
    assert first == second and backend.n_feature_cache_hits == 1
    backend.generate_ids(px + 1.0, ids, 3)  # different pixels: no stale features
    assert backend.n_feature_cache_hits == 1


def test_int8_graphs_run_and_stay_close(
    exported: tuple[Any, Path, dict[str, Any]], tmp_path: Path
) -> None:
    _, d, meta = exported
    sizes = quantize_int8(d, tmp_path)
    assert set(sizes) == {"vision", "encoder", "decoder"}
    px = np.random.default_rng(1).standard_normal((1, 3, SIZE, SIZE)).astype(np.float32)
    ids = [meta["image_token_id"]] * meta["n_image_tokens"] + [5, 6, 7]
    fp = OnnxFlorence(d, tokenizer=None).generate_ids(px, ids, 4)
    q = OnnxFlorence(tmp_path, tokenizer=None).generate_ids(px, ids, 4)
    assert np.isfinite(q[1]).all() and len(q[0]) >= 1
    # int8 may change a token, but the first-step log-probability must stay in the same ballpark
    assert abs(q[1][0] - fp[1][0]) < 1.0


def test_weights_are_initializers_and_int8_really_stores_int8(
    exported: tuple[Any, Path, dict[str, Any]], tmp_path: Path
) -> None:
    """Dynamic quantisation only touches initializers: a graph that inlines its weights as Constant
    nodes quantises to nothing (this happened on the real model: int8 files were fp32-sized)."""
    import onnx

    _, d, _ = exported
    quantize_int8(d, tmp_path)
    for name in ("encoder", "decoder"):
        fp = onnx.load(str(d / f"{name}.onnx"))
        assert len(fp.graph.initializer) >= 10, f"{name}: weights were inlined as constants"
        q = onnx.load(str(tmp_path / f"{name}.onnx"))
        int8 = [t for t in q.graph.initializer if t.data_type == onnx.TensorProto.INT8]
        assert int8, f"{name}: no int8 weights after quantisation"
        assert any(n.op_type == "MatMulInteger" for n in q.graph.node)
