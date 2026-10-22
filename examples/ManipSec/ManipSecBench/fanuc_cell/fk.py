"""
Forward kinematics for the ER-4iA, derived from the joint chain in
er4ia_physics.usd. Verified against the simulator by verify_fk.py.

Chain, each step "translate in the parent frame, then rotate about the axis":
    base -> (0,0,0.33) R+z(q1) -> (0,0,0)    R+y(q2)
         -> (0,0,0.26) R-y(q3) -> (0,0,0.02) R-x(q4)
         -> (0.29,0,0) R-y(q5) -> (0,0,0)    R-x(q6) = flange (J6_link)

The sign flips on J3..J6 are not cosmetic. Each joint carries
localRot0 == localRot1 == a 180 deg rotation (about X for J3/J5, about Z for
J4/J6). Those cancel in the rest pose -- which is why the zero-pose link
positions look right -- but the joint axis is expressed in the *joint* frame, so
the effective axis in the parent link frame is localRot0 * physics:axis, i.e.
negated. Taking physics:axis at face value gives errors of up to 570 mm.
"""
import numpy as np

CHAIN = [((0.0, 0.0, 0.33), 'z',  +1),
         ((0.0, 0.0, 0.00), 'y',  +1),
         ((0.0, 0.0, 0.26), 'y',  -1),
         ((0.0, 0.0, 0.02), 'x',  -1),
         ((0.29, 0.0, 0.0), 'y',  -1),
         ((0.0, 0.0, 0.00), 'x',  -1)]


def _rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == 'x': return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'y': return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fk(q, upto=6):
    """Flange pose in the robot base frame. Returns (R, p)."""
    R = np.eye(3); p = np.zeros(3)
    for i in range(upto):
        off, axis, sign = CHAIN[i]
        p = p + R @ np.array(off)
        R = R @ _rot(axis, sign * q[i])
    return R, p


def tool_pose(q, tool_offset):
    """Pose of the tool tip, given its offset along TOOL_AXIS in flange coords."""
    R, p = fk(q)
    return R, p + R @ np.array(tool_offset)


def jacobian(q, tool_offset, eps=1e-6):
    """Numerical 6xN Jacobian of the tool pose (linear rows then angular rows)."""
    R0, p0 = tool_pose(q, tool_offset)
    J = np.zeros((6, len(q)))
    for i in range(len(q)):
        dq = np.array(q, float); dq[i] += eps
        R1, p1 = tool_pose(dq, tool_offset)
        J[:3, i] = (p1 - p0) / eps
        dR = R1 @ R0.T
        J[3:, i] = np.array([dR[2, 1] - dR[1, 2],
                             dR[0, 2] - dR[2, 0],
                             dR[1, 0] - dR[0, 1]]) / (2 * eps)
    return J
