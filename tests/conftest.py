"""Shared fixture: a tiny VIA dataset with one example of every data-quality problem.

Built at test time with Pillow (real JPEG files, real JSON), so nothing binary is
committed and the whole fixture is a few KB.

Hand-derived expectations for this fixture (used by test_via_parse.py):

========  =====  ==========================================================
image     size   regions
========  =====  ==========================================================
1.jpg     100x60 r0 ok (10,10,20,20)          "A worker is located on soil"
                 r1 sticks out left: (-5,50,30,20) -> clipped (0,50,25,10),
                    caption has trailing space
                 r2 exact repeat of r0                       -> duplicate
                 r3 caption is only spaces                   -> empty_caption
                 r4 box fully outside the image (200,10,20,20) -> dropped
                 r5 zero width (5,5,0,10)                    -> dropped
                 r6 polygon, not a rect                      -> dropped
                 r7 no 'phrase' attribute                    -> dropped
2.jpg     100x60 no regions                                  -> zero_regions
10.jpg    80x40  r0 ok (0,0,80,40) full image; VIA 'size' field is wrong
========  =====  ==========================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

FIXTURE_IMAGE_SIZES = {"1.jpg": (100, 60), "2.jpg": (100, 60), "10.jpg": (80, 40)}
WRONG_SIZE_FOR_10 = 12345


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Do not even import the GUI test module unless the `viewer` marker is selected.

    It imports PyQt6 at module level, which needs system graphics libraries that a plain CI
    or laptop run may not have; marker deselection happens only AFTER collection.
    """
    if collection_path.name == "test_viewer_gui.py":
        expr = str(config.getoption("markexpr") or "")
        return "viewer" not in expr or "not viewer" in expr
    return None


def rect(x: int, y: int, w: int, h: int) -> dict[str, Any]:
    return {"name": "rect", "x": x, "y": y, "width": w, "height": h}


def region(shape: dict[str, Any], phrase: str | None) -> dict[str, Any]:
    attrs: dict[str, Any] = {} if phrase is None else {"phrase": phrase}
    return {"shape_attributes": shape, "region_attributes": attrs}


def build_fixture(root: Path) -> tuple[Path, Path]:
    """Write images + VIA JSON under ``root``; return ``(via_json, images_dir)``."""
    images = root / "UAVdata"
    images.mkdir(parents=True)
    for name, size in FIXTURE_IMAGE_SIZES.items():
        # Different colours so files differ; progressive for one to exercise SOF2.
        Image.new("RGB", size, color=(len(name) * 20, 80, 120)).save(
            images / name, "JPEG", progressive=(name == "2.jpg")
        )
    sizes = {n: (images / n).stat().st_size for n in FIXTURE_IMAGE_SIZES}

    via: dict[str, Any] = {}
    via[f"1.jpg{sizes['1.jpg']}"] = {
        "fileref": "",
        "size": sizes["1.jpg"],
        "filename": "1.jpg",
        "base64_img_data": "",
        "file_attributes": {},
        "regions": {
            "0": region(rect(10, 10, 20, 20), "A worker is located on soil"),
            "1": region(rect(-5, 50, 30, 20), "Fences are installed in a line "),
            "2": region(rect(10, 10, 20, 20), "A worker is located on soil"),
            "3": region(rect(30, 30, 10, 10), "   "),
            "4": region(rect(200, 10, 20, 20), "A box outside the image"),
            "5": region(rect(5, 5, 0, 10), "A zero width box"),
            "6": region({"name": "polygon", "all_points_x": [1], "all_points_y": [1]}, "poly"),
            "7": region(rect(1, 1, 5, 5), None),
        },
    }
    via[f"2.jpg{sizes['2.jpg']}"] = {
        "fileref": "",
        "size": sizes["2.jpg"],
        "filename": "2.jpg",
        "base64_img_data": "",
        "file_attributes": {},
        "regions": {},
    }
    via[f"10.jpg{WRONG_SIZE_FOR_10}"] = {
        "fileref": "",
        "size": WRONG_SIZE_FOR_10,
        "filename": "10.jpg",
        "base64_img_data": "",
        "file_attributes": {},
        "regions": {"0": region(rect(0, 0, 80, 40), "Two orange excavators are digging soil")},
    }
    via_path = root / "via_region_data_final.json"
    via_path.write_text(json.dumps(via), encoding="utf-8")
    return via_path, images


@pytest.fixture
def via_fixture(tmp_path: Path) -> tuple[Path, Path]:
    return build_fixture(tmp_path)
