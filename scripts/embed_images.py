"""Compute image embeddings for every image in an annotations.json and save them as .npy.

Usage (laptop embed venv or Colab)::

    python scripts/embed_images.py --annotations derived/annotations.json \\
        --images extracted/UAVdata --out derived/dinov2_small.npy
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from sitesight.data.embed import MODEL_ID, Dinov2Embedder
from sitesight.data.schema import Dataset


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu", help="'cpu' or 'cuda' (Colab GPU)")
    p.add_argument("--limit", type=int, help="only the first N images (smoke test)")
    args = p.parse_args(argv)

    ds = Dataset.load(args.annotations)
    paths = [args.images / i.file_name for i in ds.images][: args.limit]
    embedder = Dinov2Embedder(args.model, batch_size=args.batch_size, device=args.device)
    t = time.time()
    emb = embedder.embed(paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, emb)
    print(
        f"embedded {len(paths)} images -> {emb.shape} in {time.time() - t:.1f}s ({embedder.name})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
