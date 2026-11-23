"""Process entry point: ``uvicorn --factory sitesight.serve.app:build_app`` (the Dockerfile).

Configuration comes from the environment so the same image serves fp32 or int8 weights:

* ``SITESIGHT_MODEL_DIR``: directory with ``vision/encoder/decoder.onnx``, ``meta.json`` and
  ``tokenizer.json`` (required).
* ``SITESIGHT_THREADS``: intra-op threads for ONNX Runtime (default: runtime's choice).
* the ``SITESIGHT_MAX_*`` / ``SITESIGHT_REVIEW_THRESHOLD`` limits of :class:`api.Limits`.

``--factory`` builds the app (and loads the model) at start-up, so importing this module is free.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from sitesight.serve.api import Limits, create_app


def load_backend(model_dir: Path, threads: int | None = None) -> Any:
    from tokenizers import Tokenizer

    from sitesight.serve.onnx_backend import OnnxFlorence

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    return OnnxFlorence(model_dir, tok, threads=threads)


def build_app() -> FastAPI:
    model_dir = os.environ.get("SITESIGHT_MODEL_DIR")
    if not model_dir:
        raise RuntimeError("set SITESIGHT_MODEL_DIR to the exported ONNX model directory")
    threads = int(os.environ["SITESIGHT_THREADS"]) if "SITESIGHT_THREADS" in os.environ else None
    return create_app(load_backend(Path(model_dir), threads), Limits.from_env())
