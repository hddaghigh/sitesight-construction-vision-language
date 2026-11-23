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

from scripts.build_colab_notebook import main


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
