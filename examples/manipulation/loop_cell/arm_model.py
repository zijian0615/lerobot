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

"""SO-100 inverse kinematics for the layout checks (same maths as the 3D page).

Pose = gripper pinch point (world, m) + tool angle ``phi`` in the arm's vertical plane
(-pi/2 = jaws pointing straight down). The base sits on the tile top at the arm's ``xy``
(= pan axis) with yaw ``yaw`` (rotation of the URDF base frame about z).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

MODEL = json.loads((Path(__file__).resolve().parent / "so100_model.json").read_text(encoding="utf-8"))
P = MODEL["plane"]
LIM = P["limits"]
TWO_PI = 2 * math.pi


def _into(q: float, lo: float, hi: float) -> float | None:
    for k in (0, -1, 1, -2, 2):
        v = q + k * TWO_PI
        if lo - 1e-6 <= v <= hi + 1e-6:
            return v
    return None


def solve(base_xy, base_z, yaw, target, phi):
    """Joint angles [pan, lift, elbow, wrist_flex] or None if out of reach / limits."""
    dx, dy = target[0] - base_xy[0], target[1] - base_xy[1]
    h = math.hypot(dx, dy)
    pan = _into(math.atan2(dy, dx) - (yaw + P["forward_yaw"]), *LIM["shoulder_pan"]) if h > 1e-4 else 0.0
    if pan is None:
        return None
    a3 = phi + (P["a3_0"] - P["phi0"])
    wr = h - P["LT"] * math.cos(a3) - P["p1"][0]
    wz = target[2] - base_z - P["LT"] * math.sin(a3) - P["p1"][1]
    d = math.hypot(wr, wz)
    L1, L2, s = P["L1"], P["L2"], P["sign"]
    if not (abs(L1 - L2) + 1e-4 < d < L1 + L2 - 1e-4):
        return None
    base_ang = math.atan2(wz, wr)
    off = math.acos(max(-1.0, min(1.0, (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d))))
    best = None
    for a1 in (base_ang + off, base_ang - off):
        ex, ez = L1 * math.cos(a1), L1 * math.sin(a1)
        a2 = math.atan2(wz - ez, wr - ex)
        q1 = _into((a1 - P["a1_0"]) / s, *LIM["shoulder_lift"])
        if q1 is None:
            continue
        q2 = _into((a2 - P["a2_0"]) / s - q1, *LIM["elbow_flex"])
        if q2 is None:
            continue
        q3 = _into((a3 - P["a3_0"]) / s - q1 - q2, *LIM["wrist_flex"])
        if q3 is None:
            continue
        cand = (ez, [pan, q1, q2, q3])
        if best is None or cand[0] > best[0]:  # prefer elbow up
            best = cand
    return best[1] if best else None


def tilt_needed(base_xy, base_z, yaw, target) -> float | None:
    """Smallest deviation (deg) of the tool from straight-down that still reaches ``target``."""
    for k in range(0, 81):
        for sgn in (1, -1):
            if solve(base_xy, base_z, yaw, target, -math.pi / 2 + sgn * math.radians(k)) is not None:
                return float(k)
    return None


# ---------------- forward kinematics (for self-checks) ----------------
def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]]


def _axis_angle(ax, q):
    x, y, z = ax
    c, s, t = math.cos(q), math.sin(q), 1 - math.cos(q)
    return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y], [t * x * y + s * z, t * y * y + c, t * y * z - s * x], [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]


def _mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _h(r, t):
    return [r[0] + [t[0]], r[1] + [t[1]], r[2] + [t[2]], [0, 0, 0, 1]]


def fk_tip(base_xy, base_z, yaw, q):
    """World pinch point for joints ``q`` = [pan, lift, elbow, wrist_flex, roll]."""
    px, py = P["pan_xy"]
    c, s = math.cos(yaw), math.sin(yaw)
    origin = (base_xy[0] - (c * px - s * py), base_xy[1] - (s * px + c * py), base_z)
    m = _h([[c, -s, 0], [s, c, 0], [0, 0, 1]], origin)
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    for j in MODEL["joints"]:
        if j["name"] not in names:
            continue
        m = _mul(m, _h(_rpy(*j["rpy"]), j["xyz"]))
        m = _mul(m, _h(_axis_angle(j["axis"], q[names.index(j["name"])]), (0, 0, 0)))
    t = MODEL["tip"]["xyz"]
    return [m[i][0] * t[0] + m[i][1] * t[1] + m[i][2] * t[2] + m[i][3] for i in range(3)]
