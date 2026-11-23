"""Rigid-body maths and trajectory error metrics (numpy only).

A pose is a 4x4 matrix ``T = [[R, t], [0, 1]]`` mapping camera coordinates to world coordinates.
Quaternions are ``(qx, qy, qz, qw)`` as in the TUM RGB-D ground-truth files.

Metrics (the TUM benchmark definitions):

* **ATE** (absolute trajectory error). The estimated trajectory lives in its own frame, so it is
  first aligned to the ground truth by the rigid transform ``(R, t)`` that minimises
  ``sum_i |g_i - (R p_i + t)|^2`` (Umeyama / Horn, closed form by SVD, no scale: RGB-D depth gives
  metric scale). ATE RMSE is ``sqrt(mean_i |g_i - (R p_i + t)|^2)`` over the positions.
* **RPE** (relative pose error) over a step of ``d`` frames:
  ``E_i = (Q_i^-1 Q_{i+d})^-1 (P_i^-1 P_{i+d})`` for ground truth ``Q`` and estimate ``P``; the
  translational error is ``|trans(E_i)|`` and the rotational error the angle of ``rot(E_i)``,
  ``arccos((trace(R_E) - 1) / 2)``. RPE needs no alignment and measures local drift.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def quat_to_rot(q: Array) -> Array:
    """Rotation matrix of the unit quaternion ``(qx, qy, qz, qw)`` (normalised first)."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(r: Array) -> Array:
    """Quaternion ``(qx, qy, qz, qw)`` with ``qw >= 0`` of a rotation matrix (Shepperd's method)."""
    m = np.asarray(r, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = np.array(
            [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
        )
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1.0)
        q = np.zeros(4)
        q[i] = s / 4
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
        q[3] = (m[k, j] - m[j, k]) / s
    return q if q[3] >= 0 else -q


def make_pose(r: Array, t: Array) -> Array:
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def invert_pose(t: Array) -> Array:
    r = t[:3, :3]
    return make_pose(r.T, -r.T @ t[:3, 3])


def pose_from_tum_row(row: list[float]) -> Array:
    """``[tx, ty, tz, qx, qy, qz, qw]`` -> 4x4 pose."""
    return make_pose(quat_to_rot(np.asarray(row[3:7])), np.asarray(row[:3]))


def rotation_angle(r: Array) -> float:
    """Angle (radians, 0..pi) of a rotation matrix."""
    return float(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)))


def umeyama(src: Array, dst: Array) -> tuple[Array, Array]:
    """Rigid ``(R, t)`` minimising ``sum |dst_i - (R src_i + t)|^2`` for (N, 3) point sets."""
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise ValueError("need two aligned (N, 3) point sets with N >= 3")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    u, _, vt = np.linalg.svd(cov)
    s = np.diag([1.0, 1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])  # no reflection
    r = u @ s @ vt
    return r, mu_d - r @ mu_s


def ate_rmse(est_xyz: Array, gt_xyz: Array) -> tuple[float, Array]:
    """(RMSE, per-frame errors) of the aligned estimate against the ground truth positions."""
    r, t = umeyama(est_xyz, gt_xyz)
    err = np.linalg.norm(gt_xyz - (est_xyz @ r.T + t), axis=1)
    return float(np.sqrt(np.mean(err**2))), err


def rpe(est: list[Array], gt: list[Array], delta: int = 1) -> tuple[float, float]:
    """(translational RMSE in metres, rotational RMSE in degrees) over steps of ``delta`` frames."""
    if len(est) != len(gt) or delta < 1 or len(est) <= delta:
        raise ValueError("need aligned pose lists longer than delta")
    trans, rot = [], []
    for i in range(len(est) - delta):
        dp = invert_pose(est[i]) @ est[i + delta]
        dq = invert_pose(gt[i]) @ gt[i + delta]
        e = invert_pose(dq) @ dp
        trans.append(np.linalg.norm(e[:3, 3]))
        rot.append(np.degrees(rotation_angle(e[:3, :3])))
    return float(np.sqrt(np.mean(np.square(trans)))), float(np.sqrt(np.mean(np.square(rot))))


def stationary_ate(gt_xyz: Array) -> float:
    """ATE of "the camera never moved": the RMS spread of the true positions around their mean."""
    return float(np.sqrt(np.mean(np.sum((gt_xyz - gt_xyz.mean(0)) ** 2, axis=1))))
