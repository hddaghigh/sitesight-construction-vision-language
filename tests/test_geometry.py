"""RGB-D odometry and trajectory metrics.

The end-to-end tests write a real TUM-format sequence (PNG images, rgb.txt / depth.txt /
groundtruth.txt with offset timestamps) of a textured plane rendered from known camera poses, then
run the same code that is used on the real sequence. Expected values come from the known poses.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from sitesight.geometry.run import main as run_main
from sitesight.geometry.se3 import (
    ate_rmse,
    invert_pose,
    make_pose,
    pose_from_tum_row,
    quat_to_rot,
    rot_to_quat,
    rotation_angle,
    rpe,
    stationary_ate,
    umeyama,
)
from sitesight.geometry.tum import ROS_DEFAULT, Intrinsics, associate, load_sequence, read_list
from sitesight.geometry.vo import VOConfig, backproject, extract, run_vo


def rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def test_quaternion_conversions_round_trip_in_every_branch() -> None:
    # 90 degrees about z: (0, 0, sin 45, cos 45)
    s = np.sqrt(0.5)
    assert quat_to_rot(np.array([0, 0, s, s])) == pytest.approx(
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]), abs=1e-12
    )
    for q in ([0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]):
        q_arr = np.array(q, dtype=float)
        back = rot_to_quat(quat_to_rot(q_arr))
        assert back == pytest.approx(q_arr if q_arr[3] >= 0 else -q_arr, abs=1e-12)
    assert rot_to_quat(quat_to_rot(np.array([-0.5, -0.5, -0.5, -0.5])))[3] >= 0  # canonical sign
    assert quat_to_rot(np.array([0, 0, 0, 5.0])) == pytest.approx(np.eye(3))  # normalised first


def test_pose_helpers() -> None:
    t = make_pose(rot_y(30), np.array([1.0, 2.0, 3.0]))
    assert invert_pose(t) @ t == pytest.approx(np.eye(4), abs=1e-12)
    p = pose_from_tum_row([1, 2, 3, 0, 0, 0, 1])
    assert p[:3, 3].tolist() == [1, 2, 3] and p[:3, :3].tolist() == np.eye(3).tolist()
    assert rotation_angle(rot_y(40)) == pytest.approx(np.radians(40))
    assert rotation_angle(np.eye(3)) == 0.0


def test_umeyama_recovers_a_known_rigid_transform() -> None:
    rng = np.random.default_rng(0)
    src = rng.normal(size=(30, 3))
    r, t = rot_y(25), np.array([0.5, -1.0, 2.0])
    dst = src @ r.T + t
    r_hat, t_hat = umeyama(src, dst)
    assert r_hat == pytest.approx(r, abs=1e-10) and t_hat == pytest.approx(t, abs=1e-10)
    assert np.linalg.det(r_hat) == pytest.approx(1.0)
    # a mirrored point set cannot be matched by a rotation: the result is still a proper rotation
    rm, _ = umeyama(src, src * np.array([-1.0, 1.0, 1.0]))
    assert np.linalg.det(rm) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="N >= 3"):
        umeyama(src[:2], dst[:2])
    with pytest.raises(ValueError, match="aligned"):
        umeyama(src, dst[:5])


def test_ate_by_hand() -> None:
    gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0.0]])
    assert ate_rmse(gt + np.array([5.0, -2.0, 1.0]), gt)[0] == pytest.approx(0.0, abs=1e-12)
    # the middle two points pushed 0.2 up and down: no rigid motion removes that exactly
    est = gt.copy()
    est[1, 1], est[2, 1] = 0.2, -0.2
    rmse, err = ate_rmse(est, gt)
    assert 0 < rmse < 0.2 and err.shape == (4,)
    assert rmse == pytest.approx(np.sqrt(np.mean(err**2)))
    assert stationary_ate(np.array([[-1.0, 0, 0], [1.0, 0, 0]])) == pytest.approx(1.0)


def test_rpe_by_hand() -> None:
    gt = [make_pose(np.eye(3), np.array([float(i), 0, 0])) for i in range(6)]
    est = [make_pose(np.eye(3), np.array([1.1 * i, 0, 0])) for i in range(6)]
    assert rpe(est, gt, 1) == pytest.approx((0.1, 0.0), abs=1e-12)
    assert rpe(est, gt, 3) == pytest.approx((0.3, 0.0), abs=1e-12)
    turned = [make_pose(rot_y(2.0 * i), np.array([float(i), 0, 0])) for i in range(6)]
    assert rpe(turned, gt, 1)[1] == pytest.approx(2.0)
    assert rpe(gt, gt, 2) == pytest.approx((0.0, 0.0), abs=1e-6)
    with pytest.raises(ValueError):
        rpe(gt[:3], gt[:3], 3)


def test_associate_matches_closest_pairs_one_to_one() -> None:
    assert associate([0.0, 1.0, 2.0], [0.015, 0.99, 1.02, 3.0]) == [(0, 0), (1, 1)]
    # 1.012 is 0.011 from 1.011 but 1.0 is 0.011 too; closest-first with ties by order
    assert associate([1.0, 1.012], [1.011]) == [(1, 0)]
    assert associate([0.0], [0.0], max_diff=0.0) == []  # strictly closer than max_diff
    assert associate([], [1.0]) == []


def test_read_list_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    f = tmp_path / "rgb.txt"
    f.write_text("# colour images\n# file\n\n1.5 rgb/a.png\n  2.5 rgb/b.png  \n")
    assert read_list(f) == [(1.5, ["rgb/a.png"]), (2.5, ["rgb/b.png"])]


def test_backproject_by_hand() -> None:
    uv = np.array([[319.5 + 525.0, 239.5], [319.5, 239.5 - 525.0]])
    xyz = backproject(uv, np.array([2.0, 4.0]), ROS_DEFAULT)
    assert xyz == pytest.approx(np.array([[2.0, 0.0, 2.0], [0.0, -4.0, 4.0]]))


def test_backproject_uses_each_focal_length_for_its_own_axis() -> None:
    k = Intrinsics(fx=500.0, fy=250.0, cx=100.0, cy=50.0)
    xyz = backproject(np.array([[600.0, 300.0]]), np.array([2.0]), k)
    assert xyz == pytest.approx(np.array([[2.0, 2.0, 2.0]]))  # x = 500 * 2 / 500, y = 250 * 2 / 250


def test_only_depths_inside_the_valid_range_become_3d_points() -> None:
    rng = np.random.default_rng(0)
    gray = cv2.GaussianBlur(rng.integers(0, 255, (480, 640), dtype=np.uint8), (3, 3), 0)
    depth = np.zeros((480, 640))
    depth[:, 160:320] = 0.1  # too close
    depth[:, 320:480] = 2.0  # valid
    depth[:, 480:] = 9.0  # too far; columns 0-159 stay 0 (no measurement)
    cfg = VOConfig()
    feats = extract(gray, depth, ROS_DEFAULT, cfg, cv2.ORB.create(nfeatures=1500))
    cols = np.rint(feats.uv[:, 0]).astype(int)
    assert feats.valid.sum() > 20 and (feats.valid == ((cols >= 320) & (cols < 480))).all()
    z = feats.xyz[feats.valid, 2]
    assert (z == 2.0).all()


PLANE_Z = 2.0
TEX_PX_PER_M = 340.0
TEX_HALF = 3.0  # texture covers x, y in [-3, 3] metres


def make_texture() -> np.ndarray:
    rng = np.random.default_rng(1)
    n = int(2 * TEX_HALF * TEX_PX_PER_M)
    tex = np.full((n, n, 3), 128, np.uint8)
    for _ in range(900):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y = (int(v) for v in rng.integers(0, n, 2))
        if rng.random() < 0.5:
            w, h = (int(v) for v in rng.integers(8, 60, 2))
            cv2.rectangle(tex, (x, y), (x + w, y + h), c, -1)
        else:
            cv2.circle(tex, (x, y), int(rng.integers(5, 30)), c, -1)
    return tex


def render(tex: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Colour and depth (metres, camera z) of the plane Z = PLANE_Z seen from ``pose``."""
    k = ROS_DEFAULT
    u, v = np.meshgrid(np.arange(640, dtype=np.float64), np.arange(480, dtype=np.float64))
    d = np.stack([(u - k.cx) / k.fx, (v - k.cy) / k.fy, np.ones_like(u)], axis=-1)
    rd = d @ pose[:3, :3].T
    s = (PLANE_Z - pose[2, 3]) / rd[..., 2]  # camera-frame depth, since d has z = 1
    p = pose[:3, 3] + s[..., None] * rd
    mx = ((p[..., 0] + TEX_HALF) * TEX_PX_PER_M).astype(np.float32)
    my = ((p[..., 1] + TEX_HALF) * TEX_PX_PER_M).astype(np.float32)
    rgb = cv2.remap(tex, mx, my, cv2.INTER_LINEAR)
    return rgb, s


def true_pose(i: int) -> np.ndarray:
    return make_pose(rot_y(0.8 * i), np.array([0.025 * i, 0.01 * i, 0.012 * i]))


def write_sequence(root: Path, n: int, blank: set[int] | None = None) -> None:
    tex = make_texture()
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    rgb_rows, depth_rows, gt_rows = [], [], []
    for i in range(n):
        t = 100.0 + i / 30.0
        rgb, z = render(tex, true_pose(i))
        if blank and i in blank:
            rgb = np.full_like(rgb, 100)
        cv2.imwrite(str(root / "rgb" / f"{t:.6f}.png"), rgb)
        cv2.imwrite(
            str(root / "depth" / f"{t + 0.004:.6f}.png"), np.rint(z * 5000).astype(np.uint16)
        )
        rgb_rows.append(f"{t:.6f} rgb/{t:.6f}.png")
        depth_rows.append(f"{t + 0.004:.6f} depth/{t + 0.004:.6f}.png")
        q = rot_to_quat(true_pose(i)[:3, :3])
        p = true_pose(i)[:3, 3]
        gt_rows.append(f"{t + 0.003:.6f} " + " ".join(f"{v:.9f}" for v in [*p, *q]))
    (root / "rgb.txt").write_text("# colour\n" + "\n".join(rgb_rows) + "\n")
    (root / "depth.txt").write_text("# depth\n" + "\n".join(depth_rows) + "\n")
    (root / "groundtruth.txt").write_text("# gt\n" + "\n".join(gt_rows) + "\n")


@pytest.fixture(scope="module")
def seq(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("tum") / "seq"
    write_sequence(root, 40)
    return root


def test_load_sequence_associates_streams_with_offset_timestamps(seq: Path) -> None:
    frames = load_sequence(seq)
    assert len(frames) == 40 and all(f.gt is not None for f in frames)
    assert frames[7].gt == pytest.approx(true_pose(7), abs=1e-6)
    assert frames[0].depth.name.endswith(".png") and frames[0].rgb.is_file()
    # a ground truth further than max_diff from every image is not attached
    assert all(f.gt is None for f in load_sequence(seq, max_diff=0.002))


@pytest.mark.parametrize("mode", ["keyframe", "frame2frame"])
def test_odometry_recovers_the_known_trajectory(seq: Path, mode: str) -> None:
    frames = load_sequence(seq)
    res = run_vo(frames, ROS_DEFAULT, VOConfig(mode=mode))  # type: ignore[arg-type]
    assert res.n_failed == 0 and len(res.poses) == 40 and res.seconds
    assert res.poses[0] == pytest.approx(np.eye(4))
    est = np.array([p[:3, 3] for p in res.poses])
    gt = np.array([f.gt[:3, 3] for f in frames if f.gt is not None])
    rmse, _ = ate_rmse(est, gt)
    assert rmse < 0.01  # 1 cm over a 1 m path, from sub-pixel-accurate synthetic data
    # the estimated rotation about y matches the 0.8 degrees per frame that was rendered
    final_turn = np.degrees(rotation_angle((invert_pose(res.poses[0]) @ res.poses[-1])[:3, :3]))
    # (observed error 0.0 to 0.6 degrees: sub-pixel keypoint noise on a single plane)
    assert final_turn == pytest.approx(0.8 * 39, abs=1.0)
    if mode == "keyframe":
        assert res.n_keyframes < 40
    else:
        assert res.n_keyframes == 40


def test_keyframes_are_replaced_when_the_camera_moves_far(seq: Path) -> None:
    frames = load_sequence(seq)
    # over the first 20 frames the first keyframe stays in view, so it is never replaced
    few = run_vo(
        frames[:20], ROS_DEFAULT, VOConfig(kf_max_translation=10.0, kf_max_rotation_deg=90)
    )
    many = run_vo(frames[:20], ROS_DEFAULT, VOConfig(kf_max_translation=0.05))
    # about 0.029 m per frame: a keyframe every second frame once the limit is 0.05 m
    assert few.n_keyframes == 1 and many.n_keyframes >= 8


def test_an_untrackable_frame_is_reported_and_keeps_the_previous_pose(tmp_path: Path) -> None:
    root = tmp_path / "seq"
    write_sequence(root, 12, blank={6})
    res = run_vo(load_sequence(root), ROS_DEFAULT, VOConfig())
    assert 6 in res.failed_frames and res.n_failed >= 1
    assert res.poses[6] == pytest.approx(res.poses[5])


def test_wrong_depth_scale_is_not_hidden(seq: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sitesight.geometry.vo as vo

    monkeypatch.setattr(vo, "DEPTH_SCALE", 1000.0)  # as if the PNGs were in the wrong unit
    frames = load_sequence(seq)
    res = run_vo(frames, ROS_DEFAULT, VOConfig())
    est = np.array([p[:3, 3] for p in res.poses])
    gt = np.array([f.gt[:3, 3] for f in frames if f.gt is not None])
    assert ate_rmse(est, gt)[0] > 0.1


def test_cli_writes_evo_ready_trajectories_and_metrics(seq: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert run_main(["--root", str(seq), "--out-dir", str(out), "--modes", "keyframe"]) == 0
    m = json.loads((out / "metrics.json").read_text())["modes"]["keyframe"]
    assert m["n_frames"] == 40 and m["n_with_ground_truth"] == 40 and m["n_tracking_failures"] == 0
    assert m["ate_rmse_m"] < 0.01 and m["rpe_1frame_trans_rmse_m"] < 0.01
    assert m["stationary_baseline_ate_rmse_m"] > 0.3  # the path is about a metre long
    gt_lines = (out / "gt.txt").read_text().splitlines()
    est_lines = (out / "est_keyframe.txt").read_text().splitlines()
    assert len(gt_lines) == len(est_lines) == 40
    assert gt_lines[0].split()[0] == est_lines[0].split()[0]  # same timestamps for evo
    assert len(est_lines[5].split()) == 8  # t tx ty tz qx qy qz qw
    with pytest.raises(SystemExit, match="31 frames"):
        run_main(["--root", str(seq), "--out-dir", str(out), "--max-frames", "20"])
