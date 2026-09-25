#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any, Mapping

FANUC_CARTESIAN_NAMES = ("j0", "j1", "j2")
# Recorded state/action: x, y, z (mm), W, P, R (deg), gripper. W is kept in [0, 360): with the tool pointing down it
# sits near 180 deg, where the controller's (-180, 180] range would jump by 360 deg between frames.
FANUC_RAW_NAMES = ("j0", "j1", "j2", "j3", "j4", "j5", "j7")
FANUC_ORIENTATION_NAMES = (
    ("j3", "j3_sin", "j3_cos"),
    ("j4", "j4_sin", "j4_cos"),
    ("j5", "j5_sin", "j5_cos"),
)


def wrap_w_degrees(w_deg: float) -> float:
    """W in [0, 360) deg: continuous for tool-down poses (W near 180)."""
    return float(w_deg) % 360.0


def controller_degrees(angle_deg: float) -> float:
    """Angle in (-180, 180] deg, the range the controller reports and expects."""
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def raw_fanuc_pose_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    """Pose as j0..j2 (mm), j3 = W in [0, 360), j4 = P, j5 = R (deg); other keys (e.g. j7) pass through.

    Accepts raw j3..j5, the older sin/cos encoding, or position/rotation dicts.
    """
    decoded = decode_fanuc_pose_dict(encode_fanuc_pose_dict(values)) if not all(
        k in values for k in ("j3", "j4", "j5")
    ) else dict(values)
    out: dict[str, Any] = {k: v for k, v in decoded.items() if not (k.endswith("_sin") or k.endswith("_cos"))}
    out.pop("position", None)
    out.pop("rotation", None)
    out["j3"] = wrap_w_degrees(decoded["j3"])
    out["j4"] = controller_degrees(decoded["j4"])
    out["j5"] = controller_degrees(decoded["j5"])
    return out


def encode_angle_degrees(theta: float) -> tuple[float, float]:
    radians = math.radians(float(theta))
    return math.sin(radians), math.cos(radians)


def rotation_matrix_from_wpr(w_deg: float, p_deg: float, r_deg: float) -> list[list[float]]:
    """FANUC WPR as R = Rz(R) @ Ry(P) @ Rx(W), degrees."""
    w = math.radians(float(w_deg))
    p = math.radians(float(p_deg))
    r = math.radians(float(r_deg))
    cw, sw = math.cos(w), math.sin(w)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    return [
        [cr * cp, cr * sp * sw - sr * cw, cr * sp * cw + sr * sw],
        [sr * cp, sr * sp * sw + cr * cw, sr * sp * cw - cr * sw],
        [-sp, cp * sw, cp * cw],
    ]


def wpr_from_rotation_matrix(matrix: list[list[float]]) -> tuple[float, float, float]:
    """Inverse of ``rotation_matrix_from_wpr``. Returns W, P, R in degrees."""
    sin_p = max(-1.0, min(1.0, -float(matrix[2][0])))
    p = math.asin(sin_p)
    cp = math.cos(p)
    if abs(cp) < 1e-8:
        w = math.atan2(-float(matrix[0][1]), float(matrix[1][1]))
        r = 0.0
    else:
        w = math.atan2(float(matrix[2][1]), float(matrix[2][2]))
        r = math.atan2(float(matrix[1][0]), float(matrix[0][0]))
    return math.degrees(w), math.degrees(p), math.degrees(r)


def decode_angle_degrees(sin_value: float, cos_value: float, eps: float = 1e-8) -> float:
    norm = math.sqrt(float(sin_value) ** 2 + float(cos_value) ** 2)
    if norm < eps:
        return 0.0

    normalized_sin = float(sin_value) / norm
    normalized_cos = float(cos_value) / norm
    return math.degrees(math.atan2(normalized_sin, normalized_cos))


def encode_fanuc_pose_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    encoded: dict[str, Any] = {}

    for key, value in values.items():
        if key in {"j3", "j4", "j5", "position", "rotation"}:
            continue
        encoded[key] = value

    if "position" in values and isinstance(values["position"], Mapping):
        position = values["position"]
        encoded["j0"] = float(position["x"])
        encoded["j1"] = float(position["y"])
        encoded["j2"] = float(position["z"])
    else:
        for key in FANUC_CARTESIAN_NAMES:
            if key in values:
                encoded[key] = float(values[key])

    if all(trig_name in values for _, trig_name, _ in FANUC_ORIENTATION_NAMES) and all(
        cos_name in values for _, _, cos_name in FANUC_ORIENTATION_NAMES
    ):
        for _, sin_name, cos_name in FANUC_ORIENTATION_NAMES:
            encoded[sin_name] = float(values[sin_name])
            encoded[cos_name] = float(values[cos_name])
        return encoded

    if "rotation" in values and isinstance(values["rotation"], Mapping):
        source_angles = {
            "j3": float(values["rotation"]["w"]),
            "j4": float(values["rotation"]["p"]),
            "j5": float(values["rotation"]["r"]),
        }
    else:
        source_angles = {
            raw_name: float(values[raw_name])
            for raw_name, _, _ in FANUC_ORIENTATION_NAMES
            if raw_name in values
        }

    for raw_name, sin_name, cos_name in FANUC_ORIENTATION_NAMES:
        if raw_name not in source_angles:
            continue
        sin_value, cos_value = encode_angle_degrees(source_angles[raw_name])
        encoded[sin_name] = sin_value
        encoded[cos_name] = cos_value

    return encoded


def decode_fanuc_pose_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}

    for key, value in values.items():
        if key.endswith("_sin") or key.endswith("_cos"):
            continue
        decoded[key] = value

    for key in FANUC_CARTESIAN_NAMES:
        if key in values:
            decoded[key] = float(values[key])

    if all(raw_name in values for raw_name, _, _ in FANUC_ORIENTATION_NAMES):
        for raw_name, _, _ in FANUC_ORIENTATION_NAMES:
            decoded[raw_name] = float(values[raw_name])
        return decoded

    for raw_name, sin_name, cos_name in FANUC_ORIENTATION_NAMES:
        if sin_name not in values or cos_name not in values:
            continue
        decoded[raw_name] = decode_angle_degrees(values[sin_name], values[cos_name])

    return decoded
