"""FANUC pendant/RMI joint angles (degrees) <-> MuJoCo model joint angles (radians).

The model follows the ROS-Industrial URDF. FANUC reports J3 coupled to J2 (measured from the horizontal),
ROS-Industrial's driver converts with  J3_urdf = J3_fanuc + J2_fanuc.  Modes:
    "coupled": apply that conversion (default, what ROS-I does)
    "direct" : use the reported J3 as is
Verify on the real robot with  twin.py --check-cartesian  (compares forward kinematics with the controller's XYZW P R).
"""
import numpy as np

BASE_HEIGHT_M = 0.330   # FANUC world origin = intersection of J1/J2 axes, 330 mm above the base plate
FINGER_OPEN_M = 0.010   # finger travel per side when open (modelled 10 mm; closed pad gap 2 mm); 0 = closed
J3_MODES = ("coupled", "direct")


def fanuc_to_model(joints_deg, j3_mode="coupled"):
    if j3_mode not in J3_MODES:
        raise ValueError(f"j3_mode must be one of {J3_MODES}")
    q = np.radians(np.asarray(joints_deg, dtype=float)[:6])
    if j3_mode == "coupled":
        q[2] += q[1]
    return q


def model_to_fanuc(q, j3_mode="coupled"):
    if j3_mode not in J3_MODES:
        raise ValueError(f"j3_mode must be one of {J3_MODES}")
    q = np.array(q, dtype=float)[:6]
    if j3_mode == "coupled":
        q[2] -= q[1]
    return np.degrees(q)


def flange_world_mm(model, data):
    """Flange centre in the FANUC world frame (origin on the J1 axis at J2 height), millimetres."""
    return (data.site("flange").xpos - np.array([0.0, 0.0, BASE_HEIGHT_M])) * 1000.0


def limit_violations(model, q, tol_deg=0.5):
    """[(joint number, angle_deg, lo_deg, hi_deg)] for model-space angles outside the joint limits."""
    out = []
    for j in range(6):
        lo, hi = np.degrees(model.jnt_range[j])
        v = np.degrees(q[j])
        if v < lo - tol_deg or v > hi + tol_deg:
            out.append((j + 1, float(v), float(lo), float(hi)))
    return out
