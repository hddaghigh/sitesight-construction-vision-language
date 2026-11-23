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


def embedded_zip(extra: dict[str, Path] | None = None) -> bytes:
    """Project code as a zip; ``extra`` maps an archive name to a file outside the repository."""
    files = [p for g in EMBED_GLOBS for p in ROOT.glob(g) if "__pycache__" not in p.parts]
    files += [ROOT / f for f in EMBED_FILES]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            z.write(p, p.relative_to(ROOT).as_posix())
        for name, path in sorted((extra or {}).items()):
            z.write(path, name)
    return buf.getvalue()


def adapter_as_fp16(path: Path, dest: Path) -> Path:
    """Copy a LoRA ``.safetensors`` file with every float32 tensor rounded to float16.

    Only to get the notebook under the browser-upload size limit (13.0 MB -> 6.5 MB). Each weight
    changes by at most a relative 2**-11 (about 0.05%); the benchmark compares engines that all load
    this same file, and the report states that the adapter was rounded.
    """
    import numpy as np
    from safetensors.numpy import load_file, save_file

    tensors = {
        k: v.astype(np.float16) if v.dtype == np.float32 else v for k, v in load_file(path).items()
    }
    save_file(tensors, dest)
    return dest


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


def build_m6_notebook(zip_bytes: bytes) -> dict[str, object]:
    """Active-learning simulation (random vs uncertainty vs uncertainty + diversity)."""
    b64 = base64.b64encode(zip_bytes).decode()
    cells = [
        _md(
            """
# SiteSight M6: active-learning simulation (GPU)

Self-contained (code embedded, no GitHub token). Compares random, uncertainty and uncertainty +
diversity acquisition at 5/10/20/40% of the training images (3 seeds, a fresh LoRA each round) plus a
full-data reference. Uses only train and validation images; the test split is not read. Results are
appended to `results.jsonl` after every round, so if the session drops, reconnect and Run all again:
finished rounds are skipped. Set **Runtime > Change runtime type > T4 GPU**, then Run all.
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
!python -m sitesight.data.via_parse --via {DATA}/extracted/via_region_data_final.json \
    --images {DATA}/extracted/UAVdata --out {DATA}/derived/annotations.json
!python -m sitesight.data.caption_fields --annotations {DATA}/derived/annotations.json \
    --out {DATA}/derived/annotations_fields.json
!python -m sitesight.data.splits --annotations {DATA}/derived/annotations_fields.json \
    --splits reports/m2/splits.json --out {DATA}/derived/annotations_curated.json
ANN, IMG = f"{DATA}/derived/annotations_curated.json", f"{DATA}/extracted/UAVdata"
"""
        ),
        _code(
            """
# Smoke test (tiny pool, about 2 minutes): fails fast if anything is wrong.
!python -m sitesight.active.simulate --config configs/active.toml --annotations {ANN} --images {IMG} \
    --out-dir /content/active_smoke --device cuda --limit-images 24
"""
        ),
        _code(
            """
# The real run. Resumable: re-running skips finished rounds.
!python -m sitesight.active.simulate --config configs/active.toml --annotations {ANN} --images {IMG} \
    --out-dir /content/active_run --device cuda
"""
        ),
        _code(
            """
import json
s = json.load(open("/content/active_run/summary.json"))
print("METRICS_JSON_BEGIN")
print(json.dumps(s, separators=(",", ":"), sort_keys=True))
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


def build_m7_notebook(zip_bytes: bytes) -> dict[str, object]:
    """ONNX export (fp32 + int8) and the PyTorch / ONNX fp32 / ONNX int8 benchmark on CPU."""
    raw = base64.b64encode(zip_bytes).decode()
    # Wrapped in short lines: the Colab editor freezes on a single multi-megabyte line.
    b64 = chr(10).join(raw[i : i + 4000] for i in range(0, len(raw), 4000))
    cells = [
        _md(
            """
# SiteSight M7: ONNX export, int8 and CPU benchmark

Self-contained (code and the M5 LoRA adapter embedded, no GitHub token; the adapter is rounded to
float16 when built with `--adapter-fp16`, to fit the browser-upload limit). Use a **CPU** runtime (the
point is CPU latency; the hardware is logged in the result). Exports the fine-tuned Florence-2 to
ONNX fp32 and int8, then compares PyTorch, ONNX fp32 and ONNX int8 on one fixed sample of
**validation** regions: preprocessing parity, strict field accuracy, agreement with PyTorch and
cold-request latency (p50/p95). The test split is not read.
"""
        ),
        _code(
            f"""
import base64, io, os, zipfile
B64 = '''
{b64}
'''
zipfile.ZipFile(io.BytesIO(base64.b64decode("".join(B64.split())))).extractall("/content/sitesight")
print(sorted(os.listdir("/content/sitesight")))
"""
        ),
        _code(
            """
%cd /content/sitesight
%pip -q install -e . transformers scipy peft onnx onnxruntime tokenizers
%pip -q uninstall -y torchao
import torch, onnxruntime, os
print("torch", torch.__version__, "| onnxruntime", onnxruntime.__version__, "| cpus", os.cpu_count())
"""
        ),
        _code(
            """
DATA = "/content/sitesight_data"
!python scripts/download_data.py --data-dir {DATA} --min-free-gb 3
!python -m sitesight.data.via_parse --via {DATA}/extracted/via_region_data_final.json     --images {DATA}/extracted/UAVdata --out {DATA}/derived/annotations.json
!python -m sitesight.data.caption_fields --annotations {DATA}/derived/annotations.json     --out {DATA}/derived/annotations_fields.json
!python -m sitesight.data.splits --annotations {DATA}/derived/annotations_fields.json     --splits reports/m2/splits.json --out {DATA}/derived/annotations_curated.json
ANN, IMG = f"{DATA}/derived/annotations_curated.json", f"{DATA}/extracted/UAVdata"
"""
        ),
        _code(
            """
# Export: merges the adapter, writes export/fp32 and export/int8 (+ tokenizer.json).
!python -m sitesight.serve.export --adapter adapter/epoch_4 --out /content/export
!ls -la /content/export/fp32 /content/export/int8
"""
        ),
        _code(
            """
# Smoke (20 regions): fails fast if parity is broken before the long run.
!python -m sitesight.serve.bench --annotations {ANN} --images {IMG} --adapter adapter/epoch_4     --onnx-dir /content/export --out-dir /content/bench_smoke --n-regions 20
"""
        ),
        _code(
            """
# The real benchmark (300 validation regions, seed 0).
!python -m sitesight.serve.bench --annotations {ANN} --images {IMG} --adapter adapter/epoch_4     --onnx-dir /content/export --out-dir /content/bench --n-regions 300
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_m9_notebook(zip_bytes: bytes) -> dict[str, object]:
    """TUM RGB-D fr1/desk: RGB-D odometry, ATE/RPE, cross-check of our ATE against ``evo``."""
    raw = base64.b64encode(zip_bytes).decode()
    b64 = chr(10).join(raw[i : i + 4000] for i in range(0, len(raw), 4000))
    cells = [
        _md(
            """
# SiteSight M9: RGB-D visual odometry on TUM fr1/desk

Self-contained (code embedded, no GitHub token). CPU runtime is enough. Downloads the public TUM
RGB-D sequence `freiburg1_desk` (about 0.36 GB), runs frame-to-frame and keyframe RGB-D odometry
(ORB + depth + PnP/RANSAC, TUM's recommended ROS-default intrinsics, no undistortion) and scores both
against the motion-capture ground truth: ATE (own Umeyama alignment) and RPE, plus the "camera never
moved" baseline. Our ATE is cross-checked against `evo_ape` on the very same frames. The ground truth
is used only for scoring.
"""
        ),
        _code(
            f"""
import base64, io, os, zipfile
B64 = '''
{b64}
'''
zipfile.ZipFile(io.BytesIO(base64.b64decode("".join(B64.split())))).extractall("/content/sitesight")
print(sorted(os.listdir("/content/sitesight")))
"""
        ),
        _code(
            """
%cd /content/sitesight
%pip -q uninstall -y opencv-python opencv-contrib-python opencv-python-headless
%pip -q install -e . "opencv-python-headless>=4.9,<5" evo
import cv2, evo, os
print("cv2", cv2.__version__, "| evo", evo.__version__, "| cpus", os.cpu_count())
"""
        ),
        _code(
            """
!curl -sSL -o /content/desk.tgz https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
!ls -la /content/desk.tgz && sha256sum /content/desk.tgz
!tar xzf /content/desk.tgz -C /content
ROOT = "/content/rgbd_dataset_freiburg1_desk"
!ls {ROOT} && ls {ROOT}/rgb | wc -l && ls {ROOT}/depth | wc -l && wc -l {ROOT}/groundtruth.txt
"""
        ),
        _code(
            """
!python -m sitesight.geometry.run --root {ROOT} --out-dir /content/m9
"""
        ),
        _code(
            """
# Cross-check our ATE against evo (same frames, SE(3) alignment, no scale).
import json, re, subprocess
m = json.load(open("/content/m9/metrics.json"))
for mode in m["modes"]:
    out = subprocess.run(
        ["evo_ape", "tum", "/content/m9/gt.txt", f"/content/m9/est_{mode}.txt", "-a", "--t_max_diff", "0.01"],
        capture_output=True, text=True,
    ).stdout
    print(out)
    m["modes"][mode]["evo_ate_rmse_m"] = float(re.search(r"rmse\\s+([0-9.eE+-]+)", out).group(1))
print("M9_JSON_BEGIN")
print(json.dumps(m, separators=(",", ":"), sort_keys=True))
print("M9_JSON_END")
"""
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--adapter-dir", type=Path, help="m7: LoRA adapter directory to embed")
    p.add_argument(
        "--adapter-fp16", action="store_true", help="m7: round the adapter to float16 (smaller)"
    )
    p.add_argument(
        "--kind",
        choices=["m4", "m5", "m6", "m7", "m9"],
        default="m4",
        help="which notebook to build",
    )
    args = p.parse_args(argv)
    builders = {
        "m4": build_notebook,
        "m5": build_m5_notebook,
        "m6": build_m6_notebook,
        "m9": build_m9_notebook,
    }
    if args.kind == "m7":
        if not args.adapter_dir:
            p.error("--kind m7 needs --adapter-dir (the M5 LoRA adapter to export)")
        extra = {
            f"adapter/epoch_4/{n}": args.adapter_dir / n
            for n in ("adapter_config.json", "adapter_model.safetensors")
        }
        if args.adapter_fp16:
            tmp = args.out.parent / "adapter_model_fp16.safetensors"
            extra["adapter/epoch_4/adapter_model.safetensors"] = adapter_as_fp16(
                args.adapter_dir / "adapter_model.safetensors", tmp
            )
        nb = build_m7_notebook(embedded_zip(extra))
    else:
        nb = builders[args.kind](embedded_zip())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB, {len(nb['cells'])} cells)")  # type: ignore[arg-type]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
