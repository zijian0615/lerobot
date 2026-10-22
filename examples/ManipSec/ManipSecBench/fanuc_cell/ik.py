"""
Damped least-squares IK for the ER-4iA, on top of the verified fk.py.

Position plus approach-direction only: the tool's roll about its own axis is left
free, which is the right call for a 2-finger gripper on a symmetric cube and
keeps the problem well conditioned.
"""
import numpy as np
import fk


SPREAD_AXIS = (0.0, 1.0, 0.0)   # set by the caller to match scene_config


def set_spread_axis(a):
    global SPREAD_AXIS
    SPREAD_AXIS = tuple(a)


def solve(q0, p_target, approach, limits, iters=200, tol=1e-4, damping=0.05,
          tool_offset=(0, 0, 0), w_rot=0.35, spread_dir=None):
    """Return (q, converged). approach = desired world direction of TOOL_AXIS.

    spread_dir optionally pins the wrist roll by also demanding a world
    direction for FINGER_SPREAD_AXIS -- i.e. which way the jaws close. Leaving
    it None keeps roll free, which is fine for a cube but not for a cylinder:
    parallel jaws closing along a screw's length push it away instead of
    gripping it.
    """
    if spread_dir is not None:
        return _solve_full(q0, p_target, approach, spread_dir, limits, iters,
                           tol, damping, tool_offset, w_rot)
    q = np.array(q0, float)
    a_des = np.array(approach, float)
    a_des /= np.linalg.norm(a_des)
    lo, hi = limits[:, 0], limits[:, 1]

    for _ in range(iters):
        R, p = fk.tool_pose(q, tool_offset)
        e_pos = np.array(p_target) - p

        a_cur = R @ np.array(TOOL_AXIS)
        # rotation vector that swings a_cur onto a_des
        v = np.cross(a_cur, a_des)
        s, c = np.linalg.norm(v), float(np.dot(a_cur, a_des))
        e_rot = np.zeros(3) if s < 1e-9 else v / s * np.arctan2(s, c)

        err = np.concatenate([e_pos, w_rot * e_rot])
        if np.linalg.norm(e_pos) < tol and np.linalg.norm(e_rot) < 1e-2:
            return q, True

        J = fk.jacobian(q, tool_offset)
        J = np.vstack([J[:3], w_rot * J[3:]])
        dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(6), err)
        q = np.clip(q + dq, lo, hi)

    R, p = fk.tool_pose(q, tool_offset)
    return q, bool(np.linalg.norm(np.array(p_target) - p) < 5e-3)


TOOL_AXIS = (1.0, 0.0, 0.0)   # set by the caller to match scene_config


def set_tool_axis(a):
    global TOOL_AXIS
    TOOL_AXIS = tuple(a)


def _solve_full(q0, p_target, approach, spread_dir, limits, iters, tol, damping,
                tool_offset, w_rot):
    """IK with a fully constrained tool orientation."""
    q = np.array(q0, float)
    lo, hi = limits[:, 0], limits[:, 1]

    x_des = np.array(approach, float)
    x_des /= np.linalg.norm(x_des)
    y_des = np.array(spread_dir, float)
    y_des = y_des - np.dot(y_des, x_des) * x_des      # orthogonalise
    n = np.linalg.norm(y_des)
    if n < 1e-8:                                      # degenerate; fall back
        y_des = np.array([0.0, 1.0, 0.0])
        y_des = y_des - np.dot(y_des, x_des) * x_des
        n = np.linalg.norm(y_des)
    y_des /= n
    z_des = np.cross(x_des, y_des)

    # columns are where the tool's local axes must point in world
    tx = np.array(TOOL_AXIS, float)
    ty = np.array(SPREAD_AXIS, float)
    tz = np.cross(tx, ty)
    T_local = np.stack([tx, ty, tz], axis=1)
    T_world = np.stack([x_des, y_des, z_des], axis=1)
    R_des = T_world @ T_local.T

    for _ in range(iters):
        R, p = fk.tool_pose(q, tool_offset)
        e_pos = np.array(p_target) - p
        dR = R_des @ R.T
        w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0],
                      dR[1, 0] - dR[0, 1]])
        s_ = np.linalg.norm(w)
        c_ = (np.trace(dR) - 1.0) / 2.0
        e_rot = np.zeros(3) if s_ < 1e-9 else w / s_ * np.arctan2(s_ / 2.0, c_)

        if np.linalg.norm(e_pos) < tol and np.linalg.norm(e_rot) < 1e-2:
            return q, True

        J = fk.jacobian(q, tool_offset)
        J = np.vstack([J[:3], w_rot * J[3:]])
        err = np.concatenate([e_pos, w_rot * e_rot])
        dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(6), err)
        q = np.clip(q + dq, lo, hi)

    R, p = fk.tool_pose(q, tool_offset)
    return q, bool(np.linalg.norm(np.array(p_target) - p) < 5e-3)
