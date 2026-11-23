# M9: RGB-D visual odometry on TUM fr1/desk

Question: can the project's geometry side (depth + camera pose) be built from scratch and scored
honestly against motion-capture ground truth?

## What was built (`sitesight/geometry/`)
- `tum.py`: reader for the TUM RGB-D layout; streams matched one-to-one by timestamp (< 0.02 s).
- `vo.py`: ORB features, depth back-projection (`X = (u - cx) z / fx`, ...), descriptor matching
  with a ratio test, PnP + RANSAC (3 px) with Levenberg-Marquardt refinement; two reference
  strategies: `frame2frame` and `keyframe`. An untrackable frame keeps the previous pose and is
  counted as a failure.
- `se3.py`: quaternion / pose maths, Umeyama alignment, ATE and RPE (TUM definitions, own code).
- `run.py`: scores both strategies and writes TUM-format trajectories for `evo`.
- Ground truth is read only for scoring. Parameters are fixed defaults (`VOConfig`), not tuned on
  this sequence; there was no tuning round at all.

## Test of the code
17 unit and end-to-end tests (`tests/test_geometry.py`): hand-derived ATE/RPE/Umeyama/association
values, and a synthetic TUM-format sequence (PNG files, offset timestamps) rendered from known
poses of a textured plane that the odometry must recover (ATE under 1 cm; an untrackable frame
must be reported; a wrong depth scale must fail). Mutation check: 16 of 17 mutants of the
geometry code were caught; the survivor is a mutation in a docstring.

## Result (`metrics.json`; sequence `freiburg1_desk`, Colab CPU)
573 frames had colour, depth and ground truth within 0.02 s (of 613 colour and 595 depth images).
ROS-default intrinsics as the TUM page recommends (no undistortion).

| | ATE RMSE | ATE median | ATE max | RPE 1 frame (trans / rot) | RPE 30 frames (trans / rot) | failures | s/frame |
|---|---|---|---|---|---|---|---|
| frame-to-frame | 0.0893 m | 0.0840 m | 0.179 m | 0.0093 m / 0.59 deg | 0.0587 m / 3.36 deg | 0 | 0.070 |
| keyframe | 0.0905 m | 0.0784 m | 0.225 m | 0.0146 m / 0.84 deg | 0.0427 m / 2.36 deg | 0 | 0.075 |
| camera never moved | 0.871 m | | | | | | |

`evo_ape tum gt.txt est.txt -a` on the same frames gives 0.089283 m and 0.090531 m, matching our
own ATE to 1e-6 (different implementations, same alignment).

## What this says
- Both odometries follow the camera to about 9 cm RMS, about 10 times better
  than "the camera never moved" (0.87 m). No tracking failures.
- **The keyframe strategy did not improve ATE** (0.0905 vs 0.0893 m). It lowers local drift over a
  second (RPE30 translation 4.3 vs 5.9 cm, rotation 2.4 vs 3.4 deg) but is noisier from frame to
  frame (RPE1 1.5 vs 0.9 cm), and its worst error is larger. Both are single-run numbers; the two
  ATE values are not distinguishable.
- This is odometry only: no map, no loop closure, no bundle adjustment, so drift accumulates. A full
  SLAM system would be expected to do much better; that was not built or measured here.

## Caveats
- One sequence, one run per strategy, deterministic code but no repeated seeds (ORB and RANSAC use
  OpenCV's internal random sampling, so a re-run can differ slightly).
- Only the first-listed `freiburg1_desk` is used; conclusions about other scenes (fast motion,
  textureless walls) are not supported.
- 40 of 613 colour frames were not used because no depth image lay within 0.02 s.
- Depth is used as given (TUM's registered depth, scale 5000); no depth filtering beyond the
  0.3 to 5 m range check.
- The TUM dataset licence was not found on the pages that were read; the data stays in Colab and
  is not redistributed in this repository.
