"""Build the self-contained, token-free Colab notebook for the M4 zero-shot baselines.

The project's code (package, scripts, config, the saved split file) is zipped, base64-encoded and
embedded in the notebook's second cell, so Colab needs neither a GitHub token nor repository
access: upload the notebook, pick a GPU runtime, Run all.

    python scripts/build_colab_notebook.py --out colab/sitesight_m4_colab.ipynb

The notebook runs OWLv2, Florence-2-base and Florence-2-large on the validation split only (the
runner refuses ``test``), then evaluates and prints the metrics. Rebuild after any code change so
the embedded copy matches the repository.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EMBED_GLOBS = ("sitesight/**/*.py", "scripts/*.py", "configs/*.toml")
EMBED_FILES = (
    "reports/m2/splits.json",
    "reports/m4/zero_shot_val.json",  # zero-shot metrics, for the side-by-side comparison in M5
    "pyproject.toml",
    "README.md",
)


def embedded_zip() -> bytes:
    files = [p for g in EMBED_GLOBS for p in ROOT.glob(g) if "__pycache__" not in p.parts]
    files += [ROOT / f for f in EMBED_FILES]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            z.write(p, p.relative_to(ROOT).as_posix())
    return buf.getvalue()


def _md(text: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def _code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    }


def build_notebook(zip_bytes: bytes) -> dict[str, object]:
    b64 = base64.b64encode(zip_bytes).decode()
    cells = [
        _md(
            """
# SiteSight M4: zero-shot baselines on the validation split (GPU)

Self-contained: the project's code is embedded in the next cell, so no GitHub token is needed.
Runs OWLv2, Florence-2-base and Florence-2-large on the **validation** split only and prints the
metrics. Set **Runtime > Change runtime type > T4 GPU**, then Run all.
"""
        ),
        _code(
            f"""
import base64, io, os, zipfile
B64 = "{b64}"
zipfile.ZipFile(io.BytesIO(base64.b64decode(B64))).extractall("/content/sitesight")
print(sorted(os.listdir("/content/sitesight")))
"""
        ),
        _code(
            """
%cd /content/sitesight
%pip -q install -e . transformers scipy
import torch
assert torch.cuda.is_available(), "switch the runtime to a GPU"
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
"""
        ),
        _code(
            """
DATA = "/content/sitesight_data"
!python scripts/download_data.py --data-dir {DATA} --min-free-gb 3
!python -m sitesight.data.via_parse --via {DATA}/extracted/via_region_data_final.json \\
    --images {DATA}/extracted/UAVdata --out {DATA}/derived/annotations.json
!python -m sitesight.data.caption_fields --annotations {DATA}/derived/annotations.json \\
    --out {DATA}/derived/annotations_fields.json
# Reuse the laptop's exact split (no embeddings needed, so CPU/GPU float differences cannot change it).
!python -m sitesight.data.splits --annotations {DATA}/derived/annotations_fields.json \\
    --splits reports/m2/splits.json --out {DATA}/derived/annotations_curated.json
"""
        ),
        _code(
            """
ANN, IMG = f"{DATA}/derived/annotations_curated.json", f"{DATA}/extracted/UAVdata"
PRED = f"{DATA}/predictions"
!mkdir -p {PRED}
!python -m sitesight.baselines.run --model owlv2 --annotations {ANN} --images {IMG} --split val \\
    --device cuda --save-every 20 --out {PRED}/owlv2_val.json
"""
        ),
        _code(
            """
!python -m sitesight.baselines.run --model florence --annotations {ANN} --images {IMG} --split val \\
    --device cuda --save-every 20 --out {PRED}/florence2_base_val.json
"""
        ),
        _code(
            """
!python -m sitesight.baselines.run --model florence --model-id florence-community/Florence-2-large \\
    --annotations {ANN} --images {IMG} --split val --device cuda --save-every 20 \\
    --out {PRED}/florence2_large_val.json
"""
        ),
        _code(
            """
!mkdir -p reports/colab
!python -m sitesight.eval.run --annotations {ANN} --split val --common-images \\
    --predictions {PRED}/owlv2_val.json {PRED}/florence2_base_val_grounding.json \\
        {PRED}/florence2_base_val_regions.json {PRED}/florence2_large_val_grounding.json \\
        {PRED}/florence2_large_val_regions.json --out reports/colab/zero_shot_val_gpu.json
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_m5_notebook(zip_bytes: bytes) -> dict[str, object]:
    """LoRA fine-tuning of Florence-2-base, validation evaluation and zero-shot comparison."""
    b64 = base64.b64encode(zip_bytes).decode()
    cells = [
        _md(
            """
# SiteSight M5: LoRA fine-tuning of Florence-2-base (GPU)

Self-contained (code embedded, no GitHub token). Trains on the **train** split, picks the epoch with
the lowest **validation** loss (a rule fixed in `configs/train.toml`), evaluates that epoch on
validation with the same harness as the zero-shot baselines, and prints the comparison.
The test split is not used here. Set **Runtime > Change runtime type > T4 GPU**, then Run all.
"""
        ),
        _code(
            f"""
import base64, io, os, zipfile
B64 = "{b64}"
zipfile.ZipFile(io.BytesIO(base64.b64decode(B64))).extractall("/content/sitesight")
print(sorted(os.listdir("/content/sitesight")))
"""
        ),
        _code(
            """
%cd /content/sitesight
%pip -q install -e . transformers scipy peft
# Colab preinstalls an old torchao that peft refuses to import alongside; we do not use it.
%pip -q uninstall -y torchao
import torch
assert torch.cuda.is_available(), "switch the runtime to a GPU"
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
"""
        ),
        _code(
            """
DATA = "/content/sitesight_data"
!python scripts/download_data.py --data-dir {DATA} --min-free-gb 3
!python -m sitesight.data.via_parse --via {DATA}/extracted/via_region_data_final.json \\
    --images {DATA}/extracted/UAVdata --out {DATA}/derived/annotations.json
!python -m sitesight.data.caption_fields --annotations {DATA}/derived/annotations.json \\
    --out {DATA}/derived/annotations_fields.json
!python -m sitesight.data.splits --annotations {DATA}/derived/annotations_fields.json \\
    --splits reports/m2/splits.json --out {DATA}/derived/annotations_curated.json
ANN, IMG = f"{DATA}/derived/annotations_curated.json", f"{DATA}/extracted/UAVdata"
"""
        ),
        _code(
            """
# Quick smoke test first (4 training images, about a minute): fails fast if anything is wrong.
!python -m sitesight.train.lora --config configs/train.toml --annotations {ANN} --images {IMG} \\
    --out-dir /content/smoke --device cuda --limit-images 4
"""
        ),
        _code(
            """
# Training: caches vision features once, then trains LoRA on the language model; logs every epoch.
!python -m sitesight.train.lora --config configs/train.toml --annotations {ANN} --images {IMG} \\
    --out-dir /content/run --device cuda
"""
        ),
        _code(
            """
import json
best = open("/content/run/BEST_EPOCH.txt").read().split()[0]
print("best epoch by validation loss:", best)
PRED = f"{DATA}/predictions"
!mkdir -p {PRED}
!python -m sitesight.baselines.run --model florence --adapter /content/run/epoch_{best} \\
    --annotations {ANN} --images {IMG} --split val --device cuda --save-every 20 --out {PRED}/ft_val.json
"""
        ),
        _code(
            """
!mkdir -p reports/colab
!python -m sitesight.eval.run --annotations {ANN} --split val \\
    --predictions {PRED}/ft_val_grounding.json {PRED}/ft_val_regions.json --out reports/colab/ft_val.json
!python -m sitesight.eval.run --from-metrics reports/m4/zero_shot_val.json reports/colab/ft_val.json \\
    --out reports/colab/compare_val.json
"""
        ),
        _code(
            """
import json
print(open("/content/run/experiment_log.csv").read())
s = json.load(open("/content/run/train_summary.json"))
print({k: s[k] for k in ("best_epoch", "best_val_loss", "pretrained_val_loss", "trainable_parameters",
                          "total_parameters", "n_train_samples", "n_val_samples", "total_seconds", "gpu")})
print("METRICS_JSON_BEGIN")
print(json.dumps(json.load(open("reports/colab/ft_val.json")), separators=(",", ":"), sort_keys=True))
print("METRICS_JSON_END")
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--kind", choices=["m4", "m5"], default="m4", help="which notebook to build")
    args = p.parse_args(argv)
    nb = (build_m5_notebook if args.kind == "m5" else build_notebook)(embedded_zip())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB, {len(nb['cells'])} cells)")  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
