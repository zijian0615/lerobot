"""
Solve the overhead (TOP) camera's focal length and pose against the site scan.

    .venv/bin/python solve_tablecam_scan.py

The printed mat is a known plane once the site scan supplies its metric
geometry, so 2D correspondences between the push clip (calib_plate_push.png)
and the scan orthophoto (make_mat_texture.py --ortho) calibrate the camera --
which the footage alone could not (CALIBRATION.md sections 2-3).

Correspondences were picked by hand on feature corners that exist in both: the
screw-target rectangle, both 2x3 grids, and the BATTERY KIT screw-hole marks.
Principal point is assumed at the native 1280x960 centre, which is (635, 477) in
push-clip pixels (the push clip is a crop, and the grasp clip is a pure 1.481x
scale of it).
"""
import numpy as np
import cv2
from scipy.optimize import least_squares

# scan orthophoto pixel (1 px = 1 mm) -> robot frame; see make_mat_texture.py
J1_PX, SLOPE = np.array([842.0, 780.0]), 0.024


def ortho_to_robot(uv):
    uv = np.atleast_2d(np.asarray(uv, float))
    du, dv = uv[:, 0] - J1_PX[0], uv[:, 1] - J1_PX[1]
    return np.c_[(-du + SLOPE * dv) / 1000.0, (SLOPE * du + dv) / 1000.0]


PAIRS = [  # (orthophoto u, v)  ->  (push clip x, y)
    ((513, 660), (527, 552)), ((521, 919), (828, 550)),      # screw target
    ((315, 922), (838, 783)), ((307, 663), (525, 785)),
    ((913, 258), (200, 210)), ((913, 415), (340, 197)),      # thin 2x3 grid
    ((1027, 258), (215, 137)), ((1027, 415), (343, 125)),
    ((910, 1150), (1022, 208)), ((910, 1259), (1116, 213)),  # thick 2x3 grid
    ((1078, 1150), (998, 92)), ((1078, 1259), (1090, 98)),
    ((485, 1018), (937, 580)), ((605, 1063), (978, 462)),    # BATTERY KIT marks
    ((555, 1200), (1120, 515)), ((708, 1138), (1040, 375)),
    ((672, 1220), (1118, 402)),
]
CX, CY, SURFACE_Z = 635.0, 477.0, 0.002

W = ortho_to_robot([p[0] for p in PAIRS])
I = np.array([p[1] for p in PAIRS], float)
obj = np.c_[W, np.full(len(W), SURFACE_Z)]


def residual(p):
    K = np.array([[p[0], 0, CX], [0, p[0], CY], [0, 0, 1.0]])
    proj, _ = cv2.projectPoints(obj, p[1:4], p[4:7], K, None)
    return (proj[:, 0] - I).ravel()


K0 = np.array([[1000.0, 0, CX], [0, 1000.0, CY], [0, 0, 1]])
_, rv, tv = cv2.solvePnP(obj, I, K0, None)
sol = least_squares(residual, np.r_[1000.0, rv.ravel(), tv.ravel()])
r = sol.fun.reshape(-1, 2)
cov = np.linalg.inv(sol.jac.T @ sol.jac) * (r ** 2).sum() / (r.size - 7)
f = sol.x[0]
R, _ = cv2.Rodrigues(sol.x[1:4])
eye = -R.T @ sol.x[4:7]
axis, down = R[2], R[1]
look = eye + axis * (eye[2] - SURFACE_Z) / -axis[2]
print(f"f = {f:.0f} +/- {np.sqrt(cov[0, 0]):.0f} px at 1280 wide  "
      f"(HFOV {2 * np.degrees(np.arctan(640 / f)):.1f} deg)")
print(f"reprojection: rms {np.sqrt((r ** 2).sum(1).mean()):.1f} px, "
      f"max {np.linalg.norm(r, axis=1).max():.1f} px over {len(PAIRS)} points")
print(f"eye     {np.round(eye, 3)}   (above the table top)")
print(f"look_at {np.round(look, 3)}")
print(f"up      {np.round(-down, 3)}")
print(f"USD focal_length at horiz_aperture 6.0: {f * 6.0 / 1280:.3f} mm")
