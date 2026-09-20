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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("fanuc")
@dataclass(kw_only=True)
class FanucConfig(RobotConfig):
    """Configuration for a FANUC controller over RMI / FRC JSON."""

    host: str = "172.30.109.22"
    port: int = 16001
    group: int = 1
    utool: int = 1
    uframe: int = 0
    speed: int = 250
    term_type: str = "CNT"
    term_value: int = 100

    gripper_lcb_type: str | None = "TA"
    gripper_lcb_value: int = 10
    gripper_port_type: int | None = 2
    gripper_state_port_number: int | None = None
    gripper_port_number: int | None = None
    gripper_open_port_number: int | None = 3
    gripper_close_port_number: int | None = 4
    gripper_open_value: str = "ON"
    gripper_close_value: str = "ON"

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
