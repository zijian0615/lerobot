"""ER-4iA FK, same chain as fanuc_er4ia_sim/fk.py, and the map from sim joints to the controller.

The policy is trained on sim joints. Two things differ on the real controller:

* J3. The sim chain measures J3 against the upper arm. FRC_ReadJointAngles reports it against
  the ground (J2/J3 interaction), so sim J3 = controller J3 + controller J2.
* Frames. The sim J6 frame sits on the wrist centre with the tool along +X. The controller pose is
  the UTOOL TCP in World, whose origin is 330 mm above the base mounting face.

Measured on the real arm, 2026-10-07 (fanuc_er4ia_sim/check_j3_convention.py, UF0/UT1): the faceplate
is 70 mm out from the wrist centre, with no lateral offset, and its frame is FACEPLATE_ROT in the sim J6 frame.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .pose import rotation_matrix_from_wpr

# Wrist centre to fingertip along the J6 +X tool axis in the sim (scene_config.TOOL_LENGTH).
TOOL_TIP_M = (0.293, 0.0, 0.0)

# Base mounting face in the controller World frame.
BASE_IN_WORLD_M = np.array([0.0, 0.0, -0.33])
# Wrist centre to faceplate, along the J6 +X axis.
FACEPLATE_M = 0.070
# Faceplate axes (columns) in the sim J6 frame: faceplate Z is the J6 +X axis.
FACEPLATE_ROT = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])

_CHAIN = (
    ((0.0, 0.0, 0.33), "z", +1),
    ((0.0, 0.0, 0.00), "y", +1),
    ((0.0, 0.0, 0.26), "y", -1),
    ((0.0, 0.0, 0.02), "x", -1),
    ((0.29, 0.0, 0.0), "y", -1),
    ((0.0, 0.0, 0.00), "x", -1),
)


def _rot(axis: str, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def tool_pose(
    q_rad: np.ndarray, tool_offset: tuple[float, float, float] = TOOL_TIP_M
) -> tuple[np.ndarray, np.ndarray]:
    """Tool tip pose in the robot base frame. q is sim joint radians."""
    rotation = np.eye(3)
    position = np.zeros(3)
    for index, (offset, axis, sign) in enumerate(_CHAIN):
        position = position + rotation @ np.array(offset)
        rotation = rotation @ _rot(axis, sign * float(q_rad[index]))
    return rotation, position + rotation @ np.array(tool_offset, dtype=float)


def sim_from_controller_deg(joints_deg: Sequence[float]) -> list[float]:
    """FRC_ReadJointAngles degrees -> the sim joint degrees the policy was trained on."""
    sim = [float(v) for v in joints_deg]
    sim[2] = sim[2] + sim[1]
    return sim


def controller_from_sim_deg(joints_deg: Sequence[float]) -> list[float]:
    """Inverse of ``sim_from_controller_deg``."""
    controller = [float(v) for v in joints_deg]
    controller[2] = controller[2] - controller[1]
    return controller


def _pose_matrix(xyzwpr_mm_deg: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x, y, z, w, p, r = (float(v) for v in xyzwpr_mm_deg)
    return np.asarray(rotation_matrix_from_wpr(w, p, r), dtype=float), np.array([x, y, z]) / 1000.0


def controller_tcp_pose(
    q_sim_rad: np.ndarray,
    utool_mm_deg: Sequence[float],
    uframe_mm_deg: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """UTOOL TCP pose (rotation, metres) in the user frame for sim joints, as FRC_ReadCartesianPosition reports it.

    ``utool_mm_deg`` / ``uframe_mm_deg`` are X, Y, Z, W, P, R as FRC_ReadUToolData / FRC_ReadUFrameData return
    them. ``None`` for the user frame means UFrame 0 (World).
    """
    wrist_r, wrist_p = tool_pose(q_sim_rad, (0.0, 0.0, 0.0))
    face_r = wrist_r @ FACEPLATE_ROT
    face_p = BASE_IN_WORLD_M + wrist_p + wrist_r @ np.array([FACEPLATE_M, 0.0, 0.0])
    tool_r, tool_p = _pose_matrix(utool_mm_deg)
    rotation, position = face_r @ tool_r, face_p + face_r @ tool_p
    if uframe_mm_deg is not None:
        frame_r, frame_p = _pose_matrix(uframe_mm_deg)
        rotation, position = frame_r.T @ rotation, frame_r.T @ (position - frame_p)
    return rotation, position
