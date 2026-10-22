"""xArm6 kinematics for the instrument-install cell. numpy only.

Joint chain from UFACTORY's xarm_ros2 `xarm_description` (as in
xarm_ems885_twin/twin/xarm_model.py, verified there against the controller):
each joint is a fixed (xyz, rpy) in its parent link, then a rotation about its
own +Z. The flange is link6 (`link_eef` coincides with it). The tool -- an
electric screwdriver (OnRobot Screwdriver) on the flange. It mounts by its
side: its axis is perpendicular to the flange's, so the tool point is
TOOL_TIP in the flange frame and the tool points along TOOL_AXIS there.

IK is damped least squares on the bit tip's position and the tool axis only:
a screwdriver is symmetric about its axis, so the roll is left free (it is
pulled toward the seed, which keeps J6 from winding).
"""
from __future__ import annotations

import math

import numpy as np

JOINTS = (
    ((0.0, 0.0, 0.267), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-1.5708, 0.0, 0.0)),
    ((0.0535, -0.2845, 0.0), (0.0, 0.0, 0.0)),
    ((0.0775, 0.3425, 0.0), (-1.5708, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.5708, 0.0, 0.0)),
    ((0.076, 0.097, 0.0), (-1.5708, 0.0, 0.0)),
)
LIMITS = np.array([
    (-2 * math.pi, 2 * math.pi),
    (-2.059, 2.0944),
    (-3.927, 0.19198),
    (-2 * math.pi, 2 * math.pi),
    (-1.69297, math.pi),
    (-2 * math.pi, 2 * math.pi),
])
# the OnRobot Screwdriver's nozzle tip and axis in the flange (link6) frame, from
# its CAD (assets/onrobot_screwdriver.npz, built by assets/build_driver.py)
TOOL_TIP = np.array([0.1594, 0.0, 0.0926])
TOOL_AXIS = np.array([1.0, 0.0, 0.0])


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


_FIXED = [(np.array(x, float), _rpy(*r)) for x, r in JOINTS]


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fk(q, upto=6):
    """Flange pose (R, p) in the base frame."""
    R, p = np.eye(3), np.zeros(3)
    for i in range(upto):
        x, Rf = _FIXED[i]
        p = p + R @ x
        R = R @ Rf @ _rz(float(q[i]))
    return R, p


def tip(q, tool=None):
    """Tool tip position and tool axis (unit) in the base frame. tool:
    (tip, axis) in the flange frame; default this cell's driver."""
    t, ax = (TOOL_TIP, TOOL_AXIS) if tool is None else tool
    R, p = fk(q)
    return p + R @ np.asarray(t, float), R @ np.asarray(ax, float)


def tool_to_base(q, pts):
    """Points in the flange frame -> base frame."""
    R, p = fk(q)
    return np.asarray(pts) @ R.T + p


def _side(q, local=(0.0, 0.0, 1.0)):
    """A flange-frame direction in the base frame (default the flange's +Z:
    from the side-mounted driver's axis toward the arm's wrist)."""
    return fk(q)[0] @ np.asarray(local, float)


def ik(q0, target, axis=(0.0, 0.0, -1.0), iters=150, tol=2e-4,
       damping=0.03, w_axis=0.3, w_roll=0.05, tool=None, side_local=(0.0, 0.0, 1.0),
       side_ref=None, symmetric=False):
    """Joints putting the tool tip at `target` with the tool along `axis`.

    The roll about the axis: flange direction `side_local` is pulled toward
    `side_ref` (weight w_roll), projected off the axis; symmetric=True accepts
    either sense of it (a two-cup bar). Default: the driver's wrist on the far
    side of the tool axis from the arm's base (the side the side-mounted driver
    reaches best from), a soft preference. Returns (q, ok)."""
    q = np.array(q0, float)
    a_des = np.asarray(axis, float) / np.linalg.norm(axis)
    tgt = np.asarray(target, float)
    lo, hi = LIMITS[:, 0], LIMITS[:, 1]
    ref = np.array([tgt[0], tgt[1], 0.0]) if side_ref is None else np.asarray(side_ref, float)
    ref = ref - a_des * np.dot(ref, a_des)
    n = np.linalg.norm(ref)
    ref = ref / n if n > 1e-6 and w_roll > 0 else None
    if ref is not None and symmetric and np.dot(_side(q, side_local), ref) < 0:
        ref = -ref

    def resid(qq):
        p, a = tip(qq, tool)
        e = np.r_[tgt - p, w_axis * np.cross(a, a_des)]
        if ref is not None:
            sd = _side(qq, side_local)
            e = np.r_[e, w_roll * (ref - (sd - a_des * np.dot(sd, a_des)))]
        return e

    for _ in range(iters):
        e = resid(q)
        if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:6]) < w_axis * 2e-3 \
                and (ref is None or w_roll < 0.1 or np.linalg.norm(e[6:]) < w_roll * 0.02):
            return q, True
        J = np.zeros((len(e), 6))
        for i in range(6):
            dq = q.copy()
            dq[i] += 1e-6
            J[:, i] = -(resid(dq) - e) / 1e-6       # d(p)/dq for the residual tgt - p
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(len(e)), e)
        q = np.clip(q + dq, lo, hi)
    p, a = tip(q, tool)
    return q, bool(np.linalg.norm(tgt - p) < 2e-3 and np.dot(a, a_des) > math.cos(math.radians(1.0)))
