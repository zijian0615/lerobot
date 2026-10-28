# Frames and calibration

| Frame | Definition |
|---|---|
| Robot base | x forward, y left, z up, metres. The work surface is at z = −0.0051 m. The twin's scenes, task cards and the camera calibration use this frame. |
| UF0 | The controller user frame used by `obs["tcp_pose"]` and by `"tcp"` actions, in millimetres. x and y equal the base frame's; z_UF0 = z_base − 330 mm, so the surface is at about −335 mm. |
| UTOOL1 (the TCP) | The fingertip: 223 mm past the flange face, which is 70 mm past the wrist centre, so 293 mm from the J6 origin. `w, p, r = −180, 0, R` points the fingers straight down, and R turns them about z. |
| Joints | J1–J6 in degrees, in the twin's convention: J3 is measured relative to the upper arm (`sim J3 = controller J3 + J2`). The gripper is 0 (open) or 1 (closed). |

`cameras.json` holds the real cameras, solved with ArUco plates on the cell:

| Camera | Size | Transform | Notes |
|---|---|---|---|
| table | 1920×1080 | `T_base_cam` | Fixed, overhead from the front. |
| side | 1920×1080 | `T_base_cam` | Fixed. Re-aimed and re-solved after the testbed rebuild. |
| table_depth | 640×400 | `T_base_cam` | The table camera's depth stream (`DEPTH = ["table"]`): pinhole with its own `fx, fy, cx, cy` and no distortion; values in mm along the optical axis. `depth_to_base` in `geometry.py` turns a depth image into base-frame points. Factory intrinsics and depth-to-colour extrinsics on the table colour camera's pose, levelled on the work surface (`rms_plane_mm`). |
| wrist_depth | 848×480 | `T_flange_cam` (J6_link) | The wrist camera's depth stream (`DEPTH = ["wrist"]`), the same view as the wrist colour image: pinhole with its own `fx, fy, cx, cy`, no distortion, mm along the optical axis. It moves with the arm: `depth_to_base(cams["wrist_depth"], depth, joints=obs["joints"])` (`flange_pose` in `geometry.py` is the arm's FK). Factory intrinsics on the wrist colour camera's flange pose, levelled on the work surface at `solved_at_joints`. |
| wrist | 1280×720 | `T_flange_cam` (J6_link) | Captured at 848×480 at 15 fps over USB 2.0, with the same field of view, and served at whatever size you request. |

The projection model and its inverse are in `geometry.py` (numpy only). Use `pixel_to_base` to map a
table-image pixel to a point on the table. These transforms match the twin's camera poses up to the residual
errors in `rms_px`, so pixels in the v0.5 twin and on the real cell line up.
