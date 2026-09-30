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

"""FANUC embodiment for a future Cosmos Edge post-train.

Domain 8 is DROID. Domain 22 is the published SO-101 post-train. This row is
neither of those. Upstream cosmos-framework has no FANUC action head; the id
has to be registered there before any training run.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

# Local id. Not DROID (8) and not the SO-101 checkpoint (22).
DROID_DOMAIN_ID = 8
SO101_DOMAIN_ID = 22
DOMAIN_ID = 23
DOMAIN_NAME = "fanuc"

ACTION_DIM = 7
ARM_JOINT_DIM = 6
CHUNK_STEPS = 32
CONDITIONING_FPS = 30

POLICY_PORT = 8001
REASONER_PORT = 8000

# Overhead matches the calibrated table camera. Wrist is the Innomaker on the gripper.
# Do not reuse the SO-101 "wrist on top, overhead on the bottom" sentence.
VIEW_DESCRIPTION = (
    "Two cameras: a fixed overhead camera looking down at the printed table, "
    "and a wrist camera on the gripper looking at the fingers."
)

# LR Mate 200iD/4S model ranges (radians), same numbers as
# fanuc_lrmate200id_smc.twin.json. J3 here is the model joint, not the
# FANUC pendant reading (pendant J3 = model J3 - J2).
MODEL_RANGE_RAD: tuple[tuple[float, float], ...] = (
    (-2.96706, 2.96706),
    (-1.919862, 2.094395),
    (-1.204277, 3.577925),
    (-3.316126, 3.316126),
    (-2.094395, 2.094395),
    (-6.283185, 6.283185),
)

GRIPPER_MIN = 0.0
GRIPPER_MAX = 1.0
GRIPPER_CLOSED_THRESHOLD = 0.5

STATS_PATH = Path(__file__).with_name("fanuc_lerobot_stats.json")
_TWIN_DIR = Path(__file__).resolve().parents[2] / "fanuc_lrmate200id_smc" / "twin"

_SO101_MARKERS = ("so101", "cosmos_edge_policy_so101")


def _degrees(radians: float) -> float:
    return radians * 180.0 / math.pi


def fanuc_joint_bounds_deg() -> list[tuple[float, float]]:
    """Outer min/max of pendant J1..J6 degrees used for minmax normalization.

    J1, J2, J4, J5, J6 match the URDF. J3 is the axis-aligned bound of
    ``J3_fanuc = J3_model - J2`` over the two URDF intervals. It is a
    normalization box, not a tighter controller soft limit.
    """
    bounds = [(_degrees(lo), _degrees(hi)) for lo, hi in MODEL_RANGE_RAD]
    j2_lo, j2_hi = bounds[1]
    j3_lo, j3_hi = bounds[2]
    bounds[2] = (j3_lo - j2_hi, j3_hi - j2_lo)
    return bounds


def action_bounds() -> tuple[list[float], list[float]]:
    joints = fanuc_joint_bounds_deg()
    lo = [round(lo, 4) for lo, _ in joints] + [GRIPPER_MIN]
    hi = [round(hi, 4) for _, hi in joints] + [GRIPPER_MAX]
    return lo, hi


def stats_document() -> dict[str, Any]:
    lo, hi = action_bounds()
    return {
        "_comment": (
            "Normalization box for a FANUC Cosmos Edge post-train. "
            "Not fitted to demonstrations (none recorded). "
            "J1..J6 are FANUC pendant degrees; gripper is 0 open / 1 closed. "
            "Do not point a server at so101_lerobot_stats.json."
        ),
        "domain_name": DOMAIN_NAME,
        "domain_id": DOMAIN_ID,
        "action_space": "joint_pos",
        "demonstrations": 0,
        "action": {"min": lo, "max": hi},
        "observation.state": {"min": lo, "max": hi},
    }


def load_stats(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or STATS_PATH).read_text())


def reject_checkpoint(path: str) -> None:
    """Refuse the published SO-101 weights. They are a different embodiment."""
    folded = path.replace("\\", "/").lower()
    for marker in _SO101_MARKERS:
        if marker in folded:
            raise ValueError(
                f"Refusing SO-101 checkpoint {path!r}. "
                "Those weights are domain 22, 6-D LeRobot .pos, wrist-over-overhead. "
                "FANUC needs its own post-train."
            )


def reject_port(port: int) -> None:
    if int(port) == REASONER_PORT:
        raise ValueError(
            f"Port {REASONER_PORT} is the Cosmos Nano reasoner "
            "(examples/CMD.md). The FANUC action policy uses "
            f"{POLICY_PORT}."
        )
    if int(port) != POLICY_PORT:
        raise ValueError(f"FANUC action policy port must be {POLICY_PORT}, got {port}.")


def _joint_map():
    if str(_TWIN_DIR) not in sys.path:
        sys.path.insert(0, str(_TWIN_DIR))
    import joint_map

    return joint_map


def model_limit_errors(joints_deg: Sequence[float], *, tol_deg: float = 0.5) -> list[str]:
    """URDF limit errors for a FANUC pendant J1..J6 sample."""
    if len(joints_deg) < ARM_JOINT_DIM:
        raise ValueError(f"Expected {ARM_JOINT_DIM} joints, got {len(joints_deg)}")
    q = _joint_map().fanuc_to_model(list(joints_deg)[:ARM_JOINT_DIM])
    tol = math.radians(tol_deg)
    errors = []
    for index, (lo, hi) in enumerate(MODEL_RANGE_RAD):
        if q[index] < lo - tol or q[index] > hi + tol:
            errors.append(
                f"J{index + 1} model {_degrees(float(q[index])):.2f} deg "
                f"outside [{_degrees(lo):.2f}, {_degrees(hi):.2f}]"
            )
    return errors
