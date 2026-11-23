"""Reader for the TUM RGB-D benchmark sequence layout.

A sequence folder holds ``rgb.txt`` and ``depth.txt`` (``timestamp relative/path.png``),
``groundtruth.txt`` (``timestamp tx ty tz qx qy qz qw``, the motion-capture pose of the camera) and
the ``rgb/`` and ``depth/`` image folders. Lines starting with ``#`` are comments. Depth PNGs are
16-bit with 5000 units per metre; 0 means "no measurement".

Timestamps of the three streams do not coincide, so entries are matched one-to-one by the TUM
rule: all pairs closer than ``max_diff`` seconds (0.02 by default), smallest difference first,
each entry used at most once.

The ground truth is **only for evaluation**; nothing in the odometry reads it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sitesight.geometry.se3 import pose_from_tum_row

DEPTH_SCALE = 5000.0


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


#: The parameter set the TUM page recommends for the pre-registered depth images (no undistortion).
ROS_DEFAULT = Intrinsics(525.0, 525.0, 319.5, 239.5)


@dataclass
class Frame:
    t: float
    rgb: Path
    depth: Path
    gt: np.ndarray | None  # 4x4 camera-to-world pose from motion capture, if one is close enough


def read_list(path: Path) -> list[tuple[float, list[str]]]:
    """``(timestamp, remaining columns)`` for every non-comment line."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            parts = line.split()
            rows.append((float(parts[0]), parts[1:]))
    return rows


def associate(a: list[float], b: list[float], max_diff: float = 0.02) -> list[tuple[int, int]]:
    """Index pairs ``(i, j)`` matched one-to-one, closest first, with ``|a_i - b_j| < max_diff``."""
    if not a or not b:
        return []
    diff = np.abs(np.asarray(a)[:, None] - np.asarray(b)[None, :])
    ii, jj = np.nonzero(diff < max_diff)
    order = np.argsort(diff[ii, jj], kind="stable")
    used_a: set[int] = set()
    used_b: set[int] = set()
    out = []
    for k in order:
        i, j = int(ii[k]), int(jj[k])
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            out.append((i, j))
    return sorted(out)


def load_sequence(root: Path, max_diff: float = 0.02) -> list[Frame]:
    """Frames that have both a colour and a depth image, with ground truth where one matches."""
    rgb = read_list(root / "rgb.txt")
    depth = read_list(root / "depth.txt")
    gt = read_list(root / "groundtruth.txt")
    frames: list[Frame] = []
    for i, j in associate([r[0] for r in rgb], [d[0] for d in depth], max_diff):
        frames.append(Frame(rgb[i][0], root / rgb[i][1][0], root / depth[j][1][0], None))
    for i, j in associate([f.t for f in frames], [g[0] for g in gt], max_diff):
        frames[i].gt = pose_from_tum_row([float(v) for v in gt[j][1]])
    return frames
