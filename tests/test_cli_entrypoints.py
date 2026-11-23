"""Every command-line module must actually run when launched the way users (and Colab) do it:
``python -m <module>``. A ``main()`` without an ``if __name__ == "__main__"`` guard passes
unit tests that call it directly but silently does nothing from the command line (this happened
once with ``sitesight.data.splits``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CLI_MODULES = [
    "sitesight.data.via_parse",
    "sitesight.data.caption_fields",
    "sitesight.data.curate",
    "sitesight.data.splits",
    "sitesight.data.audit",
    "sitesight.data.download",
    "sitesight.eval.run",
    "sitesight.baselines.run",
]


@pytest.mark.parametrize("module", CLI_MODULES)
def test_module_prints_usage_when_run_with_dash_m(module: str) -> None:
    r = subprocess.run(
        [sys.executable, "-m", module, "--help"], capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0, r.stderr[-500:]
    assert "usage:" in r.stdout.lower(), f"{module} printed no usage (missing __main__ guard?)"


def test_splits_cli_does_real_work_from_the_command_line(tmp_path: Path) -> None:
    """End to end through ``python -m``: not just --help."""
    import json

    from sitesight.data.schema import AnnotationRecord, Dataset, ImageRecord

    ds = Dataset(images=[ImageRecord(id=1, file_name="1.jpg", width=10, height=10)])
    ds.annotations.append(
        AnnotationRecord(id=1, image_id=1, bbox=[0, 0, 5, 5], area=25, caption="x", caption_raw="x")
    )
    ann, spl, out = tmp_path / "a.json", tmp_path / "s.json", tmp_path / "o.json"
    ds.save(ann)
    spl.write_text(json.dumps({"1.jpg": {"split": "val", "group_id": 3}}))
    cmd = [sys.executable, "-m", "sitesight.data.splits", "--annotations", str(ann)]
    r = subprocess.run(
        [*cmd, "--splits", str(spl), "--out", str(out)], capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0 and "applied splits to 1 images" in r.stdout, r.stderr[-500:]
    assert Dataset.load(out).images[0].split == "val"
