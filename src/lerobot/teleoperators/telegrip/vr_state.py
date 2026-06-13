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

import math
import threading
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from lerobot.utils.rotation import Rotation

from .coordinates import compute_relative_position


@dataclass
class ArmVRState:
    """Latest VR controller state for one arm, consumed by get_action()."""

    enabled: bool = False
    target_delta: np.ndarray | None = None
    wrist_roll_deg: float = 0.0
    wrist_flex_deg: float = 0.0
    gripper_closed: bool | None = None
    reset_origin: bool = False


@dataclass
class VRControllerState:
    """Internal state for a single VR controller (adapted from telegrip)."""

    hand: Literal["left", "right"]
    grip_active: bool = False
    trigger_active: bool = False
    origin_position: dict[str, float] | None = None
    origin_quaternion: np.ndarray | None = None
    accumulated_rotation_quat: np.ndarray | None = None
    gripper_closed: bool | None = None
    reset_origin: bool = False

    def reset_grip(self) -> None:
        self.grip_active = False
        self.origin_position = None
        self.origin_quaternion = None
        self.accumulated_rotation_quat = None


@dataclass
class VRInputState:
    """Thread-safe aggregate of left/right VR controller states."""

    left: ArmVRState = field(default_factory=ArmVRState)
    right: ArmVRState = field(default_factory=ArmVRState)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update_arm(self, hand: Literal["left", "right"], arm_state: ArmVRState) -> None:
        with self._lock:
            if hand == "left":
                self.left = arm_state
            else:
                self.right = arm_state

    def snapshot(self) -> tuple[ArmVRState, ArmVRState]:
        with self._lock:
            return ArmVRState(**vars(self.left)), ArmVRState(**vars(self.right))


class VRControllerProcessor:
    """Processes raw WebXR controller JSON into ArmVRState (telegrip control logic)."""

    def __init__(self, vr_to_robot_scale: float = 1.0):
        self.vr_to_robot_scale = vr_to_robot_scale
        self.left = VRControllerState("left")
        self.right = VRControllerState("right")

    @staticmethod
    def _euler_to_quaternion(euler_deg: dict[str, float]) -> np.ndarray:
        euler_rad = [math.radians(euler_deg["x"]), math.radians(euler_deg["y"]), math.radians(euler_deg["z"])]
        return Rotation.from_euler("xyz", euler_rad).as_quat()

    @staticmethod
    def _extract_roll_from_quaternion(current_quat: np.ndarray, origin_quat: np.ndarray) -> float:
        origin_rotation = Rotation.from_quat(origin_quat)
        current_rotation = Rotation.from_quat(current_quat)
        relative_rotation = current_rotation * origin_rotation.inv()
        return float(-np.degrees(relative_rotation.as_rotvec()[2]))

    @staticmethod
    def _extract_pitch_from_quaternion(current_quat: np.ndarray, origin_quat: np.ndarray) -> float:
        origin_rotation = Rotation.from_quat(origin_quat)
        current_rotation = Rotation.from_quat(current_quat)
        relative_rotation = current_rotation * origin_rotation.inv()
        return float(np.degrees(relative_rotation.as_rotvec()[0]))

    def process_dual(self, data: dict) -> tuple[ArmVRState, ArmVRState]:
        left_data = data.get("leftController", {})
        right_data = data.get("rightController", {})
        left_state = self._process_single("left", left_data)
        right_state = self._process_single("right", right_data)
        return left_state, right_state

    def _process_single(self, hand: Literal["left", "right"], data: dict) -> ArmVRState:
        controller = self.left if hand == "left" else self.right
        position = data.get("position")
        rotation = data.get("rotation", {})
        quaternion = data.get("quaternion", {})
        grip_active = bool(data.get("gripActive", False))
        trigger = float(data.get("trigger", 0))

        arm_state = ArmVRState()
        arm_state.gripper_closed = controller.gripper_closed

        trigger_active = trigger > 0.5
        if trigger_active != controller.trigger_active:
            controller.trigger_active = trigger_active
            # Reversed: gripper open while trigger held, closed otherwise
            controller.gripper_closed = not trigger_active
            arm_state.gripper_closed = controller.gripper_closed

        if not position:
            arm_state.enabled = controller.grip_active
            return arm_state

        if grip_active:
            if not controller.grip_active:
                controller.grip_active = True
                controller.origin_position = dict(position)
                if quaternion and all(k in quaternion for k in ("x", "y", "z", "w")):
                    controller.origin_quaternion = np.array(
                        [quaternion["x"], quaternion["y"], quaternion["z"], quaternion["w"]]
                    )
                else:
                    controller.origin_quaternion = (
                        self._euler_to_quaternion(rotation) if rotation else None
                    )
                controller.accumulated_rotation_quat = controller.origin_quaternion
                arm_state.reset_origin = True

            if controller.origin_position is not None:
                relative_delta = compute_relative_position(
                    position, controller.origin_position, self.vr_to_robot_scale
                )
                wrist_roll = 0.0
                wrist_flex = 0.0
                if controller.origin_quaternion is not None:
                    if quaternion and all(k in quaternion for k in ("x", "y", "z", "w")):
                        current_quat = np.array(
                            [quaternion["x"], quaternion["y"], quaternion["z"], quaternion["w"]]
                        )
                    elif rotation:
                        current_quat = self._euler_to_quaternion(rotation)
                    else:
                        current_quat = None
                    if current_quat is not None:
                        wrist_roll = self._extract_roll_from_quaternion(
                            current_quat, controller.origin_quaternion
                        )
                        wrist_flex = self._extract_pitch_from_quaternion(
                            current_quat, controller.origin_quaternion
                        )

                arm_state.enabled = True
                arm_state.target_delta = relative_delta
                arm_state.wrist_roll_deg = wrist_roll
                arm_state.wrist_flex_deg = wrist_flex
        elif controller.grip_active:
            controller.reset_grip()
            arm_state.enabled = False

        arm_state.gripper_closed = controller.gripper_closed
        return arm_state
