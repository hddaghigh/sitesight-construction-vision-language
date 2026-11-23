"""Run RGB-D odometry on a TUM sequence and score it against the motion-capture ground truth.

    python -m sitesight.geometry.run --root rgbd_dataset_freiburg1_desk --out-dir out/

Writes, per mode, ``est_<mode>.txt`` (TUM trajectory format, one line per frame that has ground
truth) next to ``gt.txt`` (the matched ground-truth poses, stamped with the colour-image time), so
that ``evo_ape tum gt.txt est_<mode>.txt -a`` scores the very same frames; and ``metrics.json``
with ATE (own Umeyama alignment), RPE over 1 and 30 frames, tracking failures, keyframes, speed and
the "camera never moved" baseline. The ground truth is read for scoring only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from sitesight.geometry.se3 import ate_rmse, rot_to_quat, rpe, stationary_ate
from sitesight.geometry.tum import ROS_DEFAULT, Frame, load_sequence
from sitesight.geometry.vo import VOConfig, VOResult, run_vo


def tum_line(t: float, pose: np.ndarray) -> str:
    q = rot_to_quat(pose[:3, :3])
    vals = [*pose[:3, 3], *q]
    return f"{t:.6f} " + " ".join(f"{v:.9f}" for v in vals)


def score(frames: list[Frame], res: VOResult) -> dict[str, Any]:
    idx = [i for i, f in enumerate(frames) if f.gt is not None]
    est = [res.poses[i] for i in idx]
    gt = [frames[i].gt for i in idx]
    est_xyz = np.array([p[:3, 3] for p in est])
    gt_xyz = np.array([g[:3, 3] for g in gt if g is not None])
    ate, err = ate_rmse(est_xyz, gt_xyz)
    rpe1 = rpe(est, [g for g in gt if g is not None], 1)
    rpe30 = rpe(est, [g for g in gt if g is not None], 30)
    return {
        "n_frames": len(frames),
        "n_with_ground_truth": len(idx),
        "n_tracking_failures": res.n_failed,
        "n_keyframes": res.n_keyframes,
        "ate_rmse_m": ate,
        "ate_median_m": float(np.median(err)),
        "ate_max_m": float(err.max()),
        "rpe_1frame_trans_rmse_m": rpe1[0],
        "rpe_1frame_rot_rmse_deg": rpe1[1],
        "rpe_30frames_trans_rmse_m": rpe30[0],
        "rpe_30frames_rot_rmse_deg": rpe30[1],
        "stationary_baseline_ate_rmse_m": stationary_ate(gt_xyz),
        "seconds_per_frame_mean": float(np.mean(res.seconds)),
        "seconds_per_frame_p95": float(np.percentile(res.seconds, 95)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="a TUM RGB-D sequence folder")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--modes", nargs="+", default=["keyframe", "frame2frame"])
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)

    frames = load_sequence(args.root)
    if args.max_frames:
        frames = frames[: args.max_frames]
    if sum(f.gt is not None for f in frames) < 31:
        raise SystemExit("need at least 31 frames with ground truth")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gt.txt").write_text(
        "\n".join(tum_line(f.t, f.gt) for f in frames if f.gt is not None) + "\n"
    )
    metrics: dict[str, Any] = {
        "sequence": args.root.name,
        "intrinsics": ROS_DEFAULT.__dict__,
        "modes": {},
    }
    for mode in args.modes:
        cfg = VOConfig(mode=mode)
        res = run_vo(frames, ROS_DEFAULT, cfg)
        (args.out_dir / f"est_{mode}.txt").write_text(
            "\n".join(tum_line(f.t, res.poses[i]) for i, f in enumerate(frames) if f.gt is not None)
            + "\n"
        )
        metrics["modes"][mode] = score(frames, res) | {"config": cfg.__dict__}
        print(mode, json.dumps(metrics["modes"][mode]), flush=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
