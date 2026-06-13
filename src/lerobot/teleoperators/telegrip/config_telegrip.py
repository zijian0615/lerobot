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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..config import TeleoperatorConfig


class ControllerSide(Enum):
    LEFT = "left"
    RIGHT = "right"


@TeleoperatorConfig.register_subclass("telegrip")
@dataclass
class TelegripConfig(TeleoperatorConfig):
    """VR teleoperation via Meta Quest / WebXR (telegrip protocol)."""

    https_port: int = 8443
    websocket_port: int = 8442
    host_ip: str = "0.0.0.0"
    vr_to_robot_scale: float = 1.0

    # IK / robot mapping (required for end-effector teleop on SO arms)
    urdf_path: str = ""
    target_frame_name: str = "gripper_frame_link"

    # For single-arm robots: which VR controller drives the arm
    controller_side: ControllerSide = ControllerSide.RIGHT

    gripper_open_angle: float = 0.0
    gripper_closed_angle: float = 45.0

    # Safety limits for EE motion
    max_ee_step_m: float = 0.05
    max_joint_step_deg: float = 3.0
    end_effector_bounds_min: tuple[float, float, float] = (-0.5, -0.5, -0.1)
    end_effector_bounds_max: tuple[float, float, float] = (0.5, 0.5, 0.5)
