# Table camera calibration — results

Source: `test/000grasp/grasp-clean-small.mov`, 858x644 @ 60 fps, 92.6 s.
Working image is `calib_plate.png`: a temporal median over 120 frames spanning
the clip, which removes the moving arm and leaves a clean view of the mat.

Reproduce with `extract_lines.py` -> `fit2.py` -> `solve_cam.py` (project venv
`.venv`).

## 1. Radial distortion: none detectable — use a pinhole model

Plumb-line fit over 12 vetted world-straight features:

| | rms residual |
|---|---|
| no correction | 1.045 px |
| fitted k1 = −0.071 | 1.018 px |

- Bootstrap over lines: k1 = −0.089 ± 0.087, **95% CI [−0.279, +0.038] — contains zero**.
- Leave-one-line-out: mean held-out improvement **+0.006 px** (8/12 lines improved,
  i.e. coin-flip).

Fitting distortion buys 0.03 px on a 1 px noise floor. The residual is dominated
by edge-extraction noise, not by the lens. **Conclusion: model this camera as a
pinhole.** Any k1 in the CI is visually indistinguishable at this resolution.

Caveat: the clip is downscaled to 858x644 from an unknown native resolution,
which shrinks pixel bow proportionally. Distortion may well be present at native
resolution — it is just below the noise floor in the footage we are matching,
which is the footage we care about.

### Correction to an earlier claim
The scene was first built assuming a strong fisheye, on the basis that "the table
edges bow". That was wrong. The curvature in frame is real geometry, not the lens:

- the printed reach-envelope arcs on the mat are genuinely circles;
- the worktable has genuinely rounded corners;
- the long "straight line" running top-to-bottom on the right is a **cable** lying
  across the table and running off the table edge — a catenary, not a straight
  edge, and not even on the table plane. It was carrying most of the apparent
  distortion signal and has been excluded.

## 2. Focal length / FOV: NOT recoverable from this footage

The centre printed feature is the only candidate with a usable orthogonality
constraint, and it fails on two counts.

It is not a square. Fitted corner intersections give side lengths
200.5 / 156.3 / 211.8 / 156.3 px — an aspect of ~1.32 with only a slight taper.
So the "it's a square" constraint that a single-rectangle focal solve needs is
simply false here.

It is also near fronto-parallel and near the image centre, which is the classic
degenerate case. Propagating only ±0.5 px of corner noise through the solve:

- 43% of perturbations yield **no real solution** (f² < 0);
- of the rest, f spans 990–4029 px, i.e. **HFOV 12°–47°** at the 5–95% range.

That is not a measurement. The scene config keeps ~100° HFOV as an assumption,
explicitly flagged, not as a calibrated value.

### What the FOV ambiguity costs
Holding the framing fixed (same world points landing on the same pixels), camera
distance scales as 1/tan(HFOV/2). So the current 100° assumption versus a
plausible 60° moves the camera by a factor of

    tan(50°) / tan(30°) = 2.06x

The *look* of the render is nearly unchanged; the camera's metric position is off
by 2x. This matters the moment the camera is used for anything metric — depth,
pose estimation, sim-to-real transfer — and not at all for eyeballing framing.

## 3. Extrinsics: blocked on (2)

Pose relative to the robot base cannot be made metric without either intrinsics
or a known length in the scene. Current `eye` / `look_at` in `scene_config.py`
reproduce the reference framing well but are one point on a one-parameter family.

## 4. Cheapest things that would unlock the rest

In descending order of value per effort:

1. **A few seconds of checkerboard or ChArUco footage from that camera.** Gold
   standard; gives intrinsics and distortion outright. ~2 minutes at the bench.
2. **The camera make/model.** Published sensor size + lens focal gives FOV
   directly, which collapses the whole ambiguity.
3. **One measured distance on the mat** — e.g. the true width of the centre
   printed rectangle, or base-centre to target-fixture. One metric length plus
   the robot's exactly-known geometry pins scale and makes pose solvable.
4. **Joint angle logs for the clip.** Would turn the robot itself into a moving
   calibration target with exactly-known 3D geometry — the strongest option, and
   it also gives us the trajectory to replay.

---

# Addendum — re-run against the push-task clip (higher resolution)

Source: `test/004push/push-clean-small.mov`, **1270x838** @ 60 fps, 31.9 s.
Clean plate: `calib_plate_push.png` (median over 150 frames).
Scripts: `transfer_candidates.py` -> `refit_push.py`.

## The two clips are the same camera

SIFT matching between the grasp plate and the push plate gives 552/594 RANSAC
inliers on a homography that is a **pure uniform scale of 1.481**, with no
rotation and a zero perspective row:

    [[1.4813  0.0002  -4.89]
     [0.0001  1.4809  -7.02]
     [0      0        1   ]]

So the push clip is the same optical setup, 1.48x more pixels, cropped ~120 px
at the bottom. That makes it strictly better calibration material.

## Inferred native resolution: 1280x960

- 858 x 644 is 1280 x 960 scaled by 0.670 (858.0 x 643.5 — rounds exactly)
- 1270 x 838 is 1280 x 960 cropped

Both published clips are consistent with a **1280x960 (1.2 MP, 4:3)** camera.
`scene_config.py` now uses that as the table camera resolution. Note this means
the sim renders a slightly *larger* field than either published clip shows.

## Distortion: null result confirmed and strengthened

| | rms |
|---|---|
| no correction | 1.449 px |
| fitted k1 = −0.032 | 1.425 px |

- Bootstrap: k1 = −0.062 ± 0.121, **95% CI [−0.382, +0.107]** — still contains zero.
- Leave-one-line-out: mean held-out improvement **−0.040 px**, 7/12 lines improved.
  The distortion model now generalises *worse* than no model at all.

Floating the principal point drops rms to 1.180 px but puts it at (254, 670) —
far bottom-left, physically implausible. Textbook overfitting of a parameter the
data cannot constrain.

**Pinhole remains the right model.** Raw residual scales with resolution
(1.045 x 1.481 = 1.548, observed 1.449), which says the residual is real
sub-pixel geometry — printed lines that are not perfectly straight — and not a
fixed-pixel noise floor that more resolution would beat down.

## The centre feature is definitively a rectangle

Re-measured at 1.48x: sides 297.4 / 231.5 / 313.7 / 231.0 px,
**aspect 1.321** — against 1.32 from the low-res clip. Scaled back down that is
206 x 156 px versus the original 200–212 x 156. Two independent measurements
agree, so the "assume it's a square" route to focal length stays closed.

(Its aspect is suspiciously close to 4:3. If it is a printed 4:3 target — say
200 x 150 mm — that single measured length would unlock scale. Worth checking at
the bench with a tape measure.)

---

# 5. SOLVED (2026-09-24) — focal length and pose, against the site scan

Sections 2–3 above are superseded. The footage alone could not fix scale because
nothing in it had known metric size. The Scaniverse site scan
(`CSI-site-digital-model.zip`, "204107.glb") supplies exactly that: a metric model
of the printed mat. `make_mat_texture.py --ortho` renders it as a 1 px = 1 mm
orthophoto, registered to the robot by the base-plate bolt pattern (J1 at pixel
(842, 780), scan yaw 1.37°).

17 hand-picked correspondences between `calib_plate_push.png` and the orthophoto —
the screw-target rectangle, both 2×3 grids and five BATTERY KIT screw-hole marks,
spread over the whole table — then a PnP solve with f free and the principal point
at the native 1280×960 centre (635, 477 in push-clip pixels):

    .venv/bin/python solve_tablecam_scan.py

| | |
|---|---|
| focal length | **995 ± 44 px** at 1280 wide |
| HFOV | **65.5°** (the ~100° used until now was an assumption) |
| camera centre | (0.659, −0.021, 0.847) m — 0.85 m above the mat, 0.66 m in front of J1 |
| looks at | (0.252, −0.038) on the mat, ~64° below horizontal, negligible roll |
| reprojection | 7.0 px RMS over 17 points |

Checks:
- `reference/tablecam_calibration_overlay.jpg`: edges of the real frame drawn over
  the sim render from the solved camera. Target, grids, BATTERY KIT, arcs and
  table edges coincide to a few pixels. (The robot differs only because it was
  moving in the clip.)
- Principal-point sensitivity: at the crop centre (635, 419) instead, f becomes
  1108 px (HFOV 60°). The native-centre assumption follows from the two clips
  being a pure scale of each other.
- The residual (7 px ≈ 7 mm on the mat) is consistent with hand-picking on a
  photogrammetry texture; it is not a sign of distortion, which stays at the
  null result of section 1.

The centre "rectangle" of sections 2 and the addendum is the screw target: 260 ×
205 mm on the mat (aspect 1.27; the 1.32 measured in the image included
perspective).
