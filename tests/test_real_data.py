"""Checks against the REAL dataset. Opt-in: ``SITESIGHT_DATA_DIR=... pytest -m real_data``.

The expected numbers were measured independently (a throwaway script, before the parser
existed) and are recorded in PLAN.md section 2. If the parser and that script disagree,
one of them is wrong; they currently agree.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sitesight.data.validate import validate_dataset
from sitesight.data.via_parse import parse_via

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def extracted() -> Path:
    root = os.environ.get("SITESIGHT_DATA_DIR")
    if not root:
        pytest.fail("set SITESIGHT_DATA_DIR to the folder that contains 'extracted/'")
    path = Path(root) / "extracted"
    assert (path / "via_region_data_final.json").is_file(), f"dataset not found in {path}"
    return path


def test_real_dataset_findings(extracted: Path) -> None:
    ds, report = parse_via(extracted / "via_region_data_final.json", extracted / "UAVdata")
    assert report.n_images == 1431
    # The Mendeley page says 8,601 captions; the file contains 8,428 regions (see PLAN.md).
    assert report.n_regions_in_source == 8428
    assert report.n_dropped == 0
    assert report.n_annotations_kept == 8428
    assert report.counts == {
        "clipped": 6,
        "duplicate": 12,
        "empty_caption": 1,
        "zero_regions": 19,
    }
    assert {(i.width, i.height) for i in ds.images} == {(960, 540)}
    assert validate_dataset(ds) == []
    # Whitespace normalisation matters: 795 raw strings -> 746 after strip() (trailing
    # spaces) -> 735 after collapsing internal double spaces / one newline.
    assert len({a.caption_raw for a in ds.annotations}) == 795
    assert len({a.caption_raw.strip() for a in ds.annotations}) == 746
    assert len({a.caption for a in ds.annotations}) == 735
    assert sum(a.caption_raw.strip() != a.caption for a in ds.annotations) == 13
    # Some regions carry two captions glued together (e.g. '...on soilA black dust ...').
    assert sum(len(a.caption.split()) > 12 for a in ds.annotations) == 9
