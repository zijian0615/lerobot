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

"""VR-to-robot coordinate transforms adapted from telegrip (MIT license)."""

from __future__ import annotations

import numpy as np


def vr_to_robot_coordinates(vr_pos: dict[str, float], scale: float = 1.0) -> np.ndarray:
    """
    Convert VR controller position to robot coordinate system.

    VR: X=right, Y=up, Z=back (toward user)
    Robot: X=forward, Y=left, Z=up
    """
    return np.array(
        [
            -vr_pos["x"] * scale,
            vr_pos["z"] * scale,
            vr_pos["y"] * scale,
        ],
        dtype=float,
    )


def compute_relative_position(
    current_vr_pos: dict[str, float], origin_vr_pos: dict[str, float], scale: float = 1.0
) -> np.ndarray:
    """Compute relative position from VR origin to current position in robot frame."""
    delta_vr = {
        "x": current_vr_pos["x"] - origin_vr_pos["x"],
        "y": current_vr_pos["y"] - origin_vr_pos["y"],
        "z": current_vr_pos["z"] - origin_vr_pos["z"],
    }
    return vr_to_robot_coordinates(delta_vr, scale)
