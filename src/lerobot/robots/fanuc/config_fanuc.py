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

    # cartesian: FRC_LinearMotion with j0..j5 in mm / WPR degrees (existing live path).
    # joint: FRC_JointMotion with J1.pos..J6.pos in pendant degrees plus gripper.pos in {0, 1}.
    # Joint mode is what lerobot-rollout needs for a policy trained on joint targets.
    motion: str = "cartesian"
    joint_speed_percent: int = 10
    # Joint mode streams FRC_LinearMotion at this TCP speed. Keep it equal to the
    # sim demos' cap (fanuc_er4ia_sim/record_screw_episode.py --max-speed).
    joint_tcp_speed_mm_s: int = 50
    # A single policy step moves the TCP target at most this far from the previous target.
    joint_max_step_mm: float = 8.0
    # The TCP target never runs further than this ahead of the live pose.
    joint_max_lead_mm: float = 40.0
    # No joint target moves more than this from the previous one in one step (the demos peak at 3.45 deg
    # on J6). Chunk-to-chunk jumps are spread over several steps instead of faulting the controller.
    joint_max_step_deg: float = 5.0
    # Open / close the gripper only once the live TCP is within this many mm of that row's target (None: at
    # once). The real arm trails its targets by ~0.8 s, so without it a policy closes the jaws in the air.
    # The row is re-sent (gripper unchanged) until the arm arrives, or for joint_gripper_wait_s plus the time to travel
    # the remaining distance at joint_tcp_speed_mm_s, at most joint_gripper_wait_max_s.
    joint_gripper_wait_mm: float | None = None
    joint_gripper_wait_s: float = 2.0
    joint_gripper_wait_max_s: float = 15.0
    # Shift every joint-mode TCP target by this much in the user frame's Z (mm), e.g. -7 when the real work
    # surface sits 7 mm below the one the policy was trained on.
    joint_z_offset_mm: float = 0.0
    # Never command a joint-mode TCP target below this user-frame Z (mm); None = no floor.
    joint_z_min_mm: float | None = None

    gripper_lcb_type: str | None = "TA"
    gripper_lcb_value: int = 10
    gripper_port_type: int | None = 2
    gripper_state_port_number: int | None = None
    gripper_port_number: int | None = None
    gripper_open_port_number: int | None = 3
    gripper_close_port_number: int | None = 4
    gripper_open_value: str = "ON"
    gripper_close_value: str = "ON"

    # Optional MuJoCo twin: publish J1..J6 over UDP. Empty host disables it.
    # Twin must listen with `--source udp` (do not open a second RMI session).
    twin_udp_host: str | None = "127.0.0.1"
    twin_udp_port: int = 5005

    # Isaac Sim twin only (fanuc_lrmate200id_smc/twin/isaac_rmi_sim.py): between episodes, reset_episode() asks the
    # simulated controller to put the arm home and lay out the task objects again. Never enable on the real robot.
    sim_reset: bool = False

    # Quest / phone teleop workspace in the active user frame (mm).
    # Z min stays just above the calibrated table contact (UF Z=-335 mm).
    phone_ee_x_min_mm: float = -450.0
    phone_ee_y_min_mm: float = -450.0
    phone_ee_z_min_mm: float = -330.0
    phone_ee_x_max_mm: float = 450.0
    phone_ee_y_max_mm: float = 450.0
    phone_ee_z_max_mm: float = 250.0
    phone_max_ee_step_mm: float = 15.0

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
