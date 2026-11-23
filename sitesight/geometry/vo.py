"""RGB-D visual odometry: ORB features, depth back-projection, PnP + RANSAC.

Pipeline per frame
------------------
1. Detect ORB keypoints and binary descriptors in the grey image.
2. For keypoints whose depth ``z`` (metres) is inside ``[min_depth, max_depth]`` back-project to 3D
   in the camera frame with the pinhole model: ``X = (u - cx) z / fx``, ``Y = (v - cy) z / fy``,
   ``Z = z``.
3. Match the current descriptors to the reference frame's (Hamming distance, Lowe ratio test:
   keep a match only if the best distance is below ``ratio`` times the second best).
4. The reference 3D points and their matched current pixels give a Perspective-n-Point problem:
   find ``T_cur_ref`` minimising the reprojection error ``sum |pi(K (R X + t)) - u|^2``.
   RANSAC (reprojection threshold ``reproj_error`` pixels) rejects wrong matches; the pose is then
   refined on the inliers by Levenberg-Marquardt.
5. World pose: ``T_world_cur = T_world_ref @ inverse(T_cur_ref)``. The first frame defines the
   world frame (identity); the evaluation aligns the trajectory to the ground truth afterwards.

Reference choice (``mode``)
---------------------------
* ``frame2frame``: the reference is always the previous frame. Errors of every step accumulate.
* ``keyframe``: the reference is a keyframe, replaced when tracking against it gets weak
  (fewer than ``kf_min_inliers`` inliers) or the camera has moved more than ``kf_max_translation``
  metres or ``kf_max_rotation_deg`` degrees from it. Fewer pose compositions means less drift.

Honesty rules: a frame that cannot be tracked (too few matches or inliers) keeps the previous pose
and is counted in ``n_failed``; the failure is reported, never hidden. The ground truth is never
read here. Parameter values are fixed defaults, not tuned on the evaluation sequence.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from sitesight.geometry.se3 import invert_pose, make_pose, rotation_angle
from sitesight.geometry.tum import DEPTH_SCALE, Frame, Intrinsics


@dataclass(frozen=True)
class VOConfig:
    mode: Literal["keyframe", "frame2frame"] = "keyframe"
    n_features: int = 1500
    ratio: float = 0.75
    min_depth: float = 0.3
    max_depth: float = 5.0
    reproj_error: float = 3.0
    ransac_iters: int = 200
    min_inliers: int = 20
    kf_min_inliers: int = 60
    kf_max_translation: float = 0.15
    kf_max_rotation_deg: float = 15.0


@dataclass
class Features:
    uv: np.ndarray  # (N, 2) pixel coordinates
    desc: np.ndarray | None  # (N, 32) uint8 ORB descriptors
    xyz: np.ndarray  # (N, 3) camera-frame points, NaN where the depth is missing or out of range

    @property
    def valid(self) -> np.ndarray:
        return np.asarray(~np.isnan(self.xyz[:, 2]), dtype=bool)


def backproject(uv: np.ndarray, z: np.ndarray, k: Intrinsics) -> np.ndarray:
    """Pixels ``(u, v)`` with depth ``z`` (metres) -> camera-frame points ``(X, Y, Z)``."""
    x = (uv[:, 0] - k.cx) * z / k.fx
    y = (uv[:, 1] - k.cy) * z / k.fy
    return np.asarray(np.column_stack([x, y, z]), dtype=np.float64)


def extract(
    gray: np.ndarray, depth_m: np.ndarray, k: Intrinsics, cfg: VOConfig, orb: cv2.ORB
) -> Features:
    kps, desc = orb.detectAndCompute(gray, None)
    if desc is None or len(kps) == 0:
        return Features(np.zeros((0, 2), np.float32), None, np.zeros((0, 3)))
    uv = np.array([kp.pt for kp in kps], dtype=np.float64)
    h, w = depth_m.shape
    cols = np.clip(np.rint(uv[:, 0]).astype(int), 0, w - 1)
    rows = np.clip(np.rint(uv[:, 1]).astype(int), 0, h - 1)
    z = depth_m[rows, cols]
    ok = (z >= cfg.min_depth) & (z <= cfg.max_depth)
    xyz = np.full((len(uv), 3), np.nan)
    xyz[ok] = backproject(uv[ok], z[ok], k)
    return Features(uv, desc, xyz)


def track(
    ref: Features, cur: Features, k: Intrinsics, cfg: VOConfig, matcher: cv2.BFMatcher
) -> tuple[np.ndarray | None, int, int]:
    """``(T_cur_ref or None, n_inliers, n_matches)``; None when there is too little to decide."""
    if ref.desc is None or cur.desc is None:
        return None, 0, 0
    keep = np.nonzero(ref.valid)[0]
    if len(keep) < 6 or len(cur.desc) < 6:
        return None, 0, 0
    pairs = matcher.knnMatch(cur.desc, ref.desc[keep], k=2)
    obj, img = [], []
    for p in pairs:
        if len(p) == 2 and p[0].distance < cfg.ratio * p[1].distance:
            obj.append(ref.xyz[keep[p[0].trainIdx]])
            img.append(cur.uv[p[0].queryIdx])
    n_matches = len(obj)
    if n_matches < cfg.min_inliers:
        return None, 0, n_matches
    obj_a, img_a = np.asarray(obj, dtype=np.float64), np.asarray(img, dtype=np.float64)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        obj_a,
        img_a,
        k.matrix(),
        None,
        iterationsCount=cfg.ransac_iters,
        reprojectionError=cfg.reproj_error,
        confidence=0.99,
    )
    if not ok or inl is None or len(inl) < cfg.min_inliers:
        return None, 0 if inl is None else len(inl), n_matches
    idx = inl[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(obj_a[idx], img_a[idx], k.matrix(), None, rvec, tvec)
    r, _ = cv2.Rodrigues(rvec)
    return make_pose(r, tvec[:, 0]), len(idx), n_matches


@dataclass
class VOResult:
    poses: list[np.ndarray]  # camera-to-world, one per frame, first is the identity
    n_failed: int = 0
    n_keyframes: int = 0
    seconds: list[float] = field(default_factory=list)
    failed_frames: list[int] = field(default_factory=list)


def load_frame(frame: Frame) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.imread(str(frame.rgb), cv2.IMREAD_GRAYSCALE)
    depth = cv2.imread(str(frame.depth), cv2.IMREAD_UNCHANGED)
    if gray is None or depth is None:
        raise FileNotFoundError(f"cannot read {frame.rgb} / {frame.depth}")
    return gray, depth.astype(np.float64) / DEPTH_SCALE


def run_vo(
    frames: list[Frame],
    k: Intrinsics,
    cfg: VOConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> VOResult:
    cfg = cfg or VOConfig()
    orb = cv2.ORB.create(nfeatures=cfg.n_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    res = VOResult(poses=[])
    ref: Features | None = None
    t_w_ref = np.eye(4)
    for n, frame in enumerate(frames):
        t0 = time.perf_counter()
        gray, depth = load_frame(frame)
        cur = extract(gray, depth, k, cfg, orb)
        if ref is None:
            res.poses.append(np.eye(4))
            ref, t_w_ref = cur, np.eye(4)
            res.n_keyframes += 1
        else:
            t_cur_ref, n_inl, _ = track(ref, cur, k, cfg, matcher)
            if t_cur_ref is None:
                # cannot track: keep the last pose, say so, and restart from this frame
                res.n_failed += 1
                res.failed_frames.append(n)
                pose = res.poses[-1]
                res.poses.append(pose)
                ref, t_w_ref = cur, pose
                res.n_keyframes += 1
            else:
                pose = t_w_ref @ invert_pose(t_cur_ref)
                res.poses.append(pose)
                moved = float(np.linalg.norm(t_cur_ref[:3, 3]))
                turned = float(np.degrees(rotation_angle(t_cur_ref[:3, :3])))
                new_kf = cfg.mode == "frame2frame" or (
                    n_inl < cfg.kf_min_inliers
                    or moved > cfg.kf_max_translation
                    or turned > cfg.kf_max_rotation_deg
                )
                if new_kf:
                    ref, t_w_ref = cur, pose
                    res.n_keyframes += 1
        res.seconds.append(time.perf_counter() - t0)
        if progress:
            progress(n + 1, len(frames))
    return res
