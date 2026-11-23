"""The token-free Colab notebook builder: valid notebook, embedded code is complete and imports."""

from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts.build_colab_notebook import main


def test_m5_notebook_trains_on_train_never_touches_test_and_embeds_its_inputs(
    tmp_path: Path,
) -> None:
    out = tmp_path / "m5.ipynb"
    assert main(["--out", str(out), "--kind", "m5"]) == 0
    nb = json.loads(out.read_text(encoding="utf-8"))
    text = "".join("".join(c["source"]) for c in nb["cells"])
    assert "sitesight.train.lora" in text and "--adapter" in text and "peft" in text
    assert "--limit-images 4" in text  # a quick smoke stage precedes the real training
    assert "GITHUB_TOKEN" not in text and "--split test" not in text and "--allow-test" not in text
    m = re.search(r'B64 = "([^"]+)"', "".join(nb["cells"][1]["source"]))
    assert m is not None
    names = set(zipfile.ZipFile(io.BytesIO(base64.b64decode(m.group(1)))).namelist())
    for needed in (
        "configs/train.toml",
        "sitesight/train/lora.py",
        "reports/m4/zero_shot_val.json",
    ):
        assert needed in names, needed


def test_notebook_is_valid_and_embeds_the_code(tmp_path: Path) -> None:
    out = tmp_path / "nb.ipynb"
    assert main(["--out", str(out)]) == 0
    nb = json.loads(out.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and all(c["cell_type"] in ("markdown", "code") for c in nb["cells"])
    text = "".join("".join(c["source"]) for c in nb["cells"])
    assert "GITHUB_TOKEN" not in text and "userdata" not in text  # token-free by construction
    assert "--split val" in text and "--split test" not in text  # never touches the test split
    m = re.search(r'B64 = "([^"]+)"', "".join(nb["cells"][1]["source"]))
    assert m is not None
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(m.group(1))))
    names = set(z.namelist())
    for needed in (
        "sitesight/baselines/run.py",
        "sitesight/eval/run.py",
        "sitesight/data/splits.py",
        "reports/m2/splits.json",
        "pyproject.toml",
        "scripts/download_data.py",
    ):
        assert needed in names, needed
    assert not any("__pycache__" in n or n.startswith("tests/") for n in names)
    # The embedded copy must import and run on its own, exactly as it will in Colab.
    dest = tmp_path / "unpacked"
    z.extractall(dest)
    r = subprocess.run(
        [sys.executable, "-m", "sitesight.eval.run", "--help"],
        cwd=dest,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(dest), "PATH": ""},
        timeout=120,
    )
    assert r.returncode == 0 and "usage:" in r.stdout.lower(), r.stderr[-400:]


def test_m6_notebook_is_resumable_smoke_tested_and_never_touches_test(tmp_path: Path) -> None:
    out = tmp_path / "m6.ipynb"
    assert main(["--out", str(out), "--kind", "m6"]) == 0
    nb = json.loads(out.read_text(encoding="utf-8"))
    text = "".join("".join(c["source"]) for c in nb["cells"])
    assert "sitesight.active.simulate" in text and "--limit-images 24" in text
    assert "GITHUB_TOKEN" not in text and "--split test" not in text and "--allow-test" not in text
    m = re.search(r'B64 = "([^"]+)"', "".join(nb["cells"][1]["source"]))
    assert m is not None
    names = set(zipfile.ZipFile(io.BytesIO(base64.b64decode(m.group(1)))).namelist())
    assert {"configs/active.toml", "sitesight/active/simulate.py"} <= names


def test_m7_notebook_embeds_the_adapter_and_benchmarks_validation_only(tmp_path: Path) -> None:
    adapter = tmp_path / "ad"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    out = tmp_path / "m7.ipynb"
    assert main(["--out", str(out), "--kind", "m7", "--adapter-dir", str(adapter)]) == 0
    nb = json.loads(out.read_text(encoding="utf-8"))
    text = "".join("".join(c["source"]) for c in nb["cells"])
    assert "sitesight.serve.export" in text and "sitesight.serve.bench" in text
    assert "--n-regions 20" in text and "GITHUB_TOKEN" not in text and "--split test" not in text
    m = re.search(r'B64 = "([^"]+)"', "".join(nb["cells"][1]["source"]))
    assert m is not None
    z = zipfile.ZipFile(io.BytesIO(base64.b64decode(m.group(1))))
    assert z.read("adapter/epoch_4/adapter_model.safetensors") == b"weights"
    assert "sitesight/serve/onnx_backend.py" in z.namelist()
    with pytest.raises(SystemExit):  # the adapter is required: no silent export of the base model
        main(["--out", str(out), "--kind", "m7"])
