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

"""Phone 6-DoF teleoperation → xArm Cartesian action (mm + 6D rotation + gripper)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.processor import (
    IdentityProcessorStep,
    ProcessorStepRegistry,
    RobotAction,
    RobotActionProcessorStep,
    RobotObservation,
    RobotProcessorPipeline,
    TransitionKey,
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots import Robot
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.types import RobotAction as RobotActionType

from .xarm import XArmRobot, _ROT6D_KEYS, _aa_to_rot6d, _rot6d_to_aa


def _phone_calib_to_xarm_delta(pos: np.ndarray, rotvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map calibrated phone pose (metres + rotvec in phone frame) to xArm base frame.

    Calibration pose: screen up (+Z), top edge forward (+X), phone right = robot -Y.
    HEBI / ARKit device frame: +X right, +Y toward top of phone, +Z out of screen.
    """
    delta_pos_m = np.array([pos[1], -pos[0], pos[2]], dtype=float)
    delta_rot = np.array([rotvec[1], -rotvec[0], rotvec[2]], dtype=float)
    return delta_pos_m, delta_rot


def _obs_pose_matrix(obs: RobotObservation) -> np.ndarray:
    """xArm observation (j0..j2 mm + r0..r5) → 4×4 pose."""
    if not all(k in obs for k in ("j0", "j1", "j2", *_ROT6D_KEYS)):
        raise KeyError("xArm Cartesian observation missing j0..j2 or r0..r5")
    pos = np.array([float(obs["j0"]), float(obs["j1"]), float(obs["j2"])], dtype=float)
    aa = _rot6d_to_aa(*[float(obs[k]) for k in _ROT6D_KEYS])
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = Rotation.from_rotvec(aa).as_matrix()
    pose[:3, 3] = pos
    return pose


def _pose_matrix_to_xarm_action(pose: np.ndarray, gripper_norm: float) -> RobotAction:
    aa = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
    rot6d = _aa_to_rot6d(float(aa[0]), float(aa[1]), float(aa[2]))
    action: RobotAction = {
        "j0": float(pose[0, 3]),
        "j1": float(pose[1, 3]),
        "j2": float(pose[2, 3]),
        "j7": float(np.clip(gripper_norm, 0.0, 1.0)),
    }
    for key, val in zip(_ROT6D_KEYS, rot6d, strict=True):
        action[key] = float(val)
    return action


@ProcessorStepRegistry.register("phone_xarm_skip_empty")
@dataclass
class PhoneXArmSkipEmpty(RobotActionProcessorStep):
    """Skip pipeline when phone is not calibrated / no pose yet."""

    def action(self, action: RobotAction) -> RobotAction:
        if not action or "phone.pos" not in action:
            return {}
        return action

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("map_phone_action_to_xarm_action")
@dataclass
class MapPhoneActionToXArmAction(RobotActionProcessorStep):
    """Phone calibrated pose → xArm base-frame position (m) and rotation (rotvec)."""

    platform: PhoneOS

    def action(self, action: RobotAction) -> RobotAction:
        if not action or "phone.enabled" not in action:
            return {}

        enabled = bool(action.pop("phone.enabled"))
        pos = action.pop("phone.pos")
        rot = action.pop("phone.rot")
        inputs = action.pop("phone.raw_inputs")

        if pos is None or rot is None:
            raise ValueError("pos and rot must be present in action")

        if self.platform == PhoneOS.IOS:
            gripper_vel = float(inputs.get("a3", 0.0))
        else:
            a = float(inputs.get("reservedButtonA", 0.0))
            b = float(inputs.get("reservedButtonB", 0.0))
            gripper_vel = a - b

        if enabled:
            delta_pos_m, delta_rot = _phone_calib_to_xarm_delta(pos, rot.as_rotvec())
            action["target_x"] = float(delta_pos_m[0])
            action["target_y"] = float(delta_pos_m[1])
            action["target_z"] = float(delta_pos_m[2])
            action["target_wx"] = float(delta_rot[0])
            action["target_wy"] = float(delta_rot[1])
            action["target_wz"] = float(delta_rot[2])
        else:
            action["target_x"] = 0.0
            action["target_y"] = 0.0
            action["target_z"] = 0.0
            action["target_wx"] = 0.0
            action["target_wy"] = 0.0
            action["target_wz"] = 0.0

        action["enabled"] = enabled
        action["gripper_vel"] = gripper_vel
        return action

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("phone_xarm_reference_and_delta")
@dataclass
class PhoneXArmReferenceAndDelta(RobotActionProcessorStep):
    """
    Map phone target deltas to absolute xArm Cartesian pose (mm + axis-angle internally).

    Same control model as SO-100 ``EEReferenceAndDelta``, but uses measured xArm pose
    instead of URDF forward kinematics.
    """

    end_effector_step_sizes_mm: dict[str, float] = field(
        default_factory=lambda: {"x": 1.0, "y": 1.0, "z": 1.0}
    )
    use_latched_reference: bool = True

    reference_ee_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _command_when_disabled: np.ndarray | None = field(default=None, init=False, repr=False)
    _gripper_vel: float = field(default=0.0, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        if not action:
            return {}

        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Robot observation is required for phone xArm teleoperation")

        t_curr = _obs_pose_matrix(observation)

        enabled = bool(action.pop("enabled"))
        tx = float(action.pop("target_x"))
        ty = float(action.pop("target_y"))
        tz = float(action.pop("target_z"))
        wx = float(action.pop("target_wx"))
        wy = float(action.pop("target_wy"))
        wz = float(action.pop("target_wz"))
        self._gripper_vel = float(action.pop("gripper_vel"))

        if enabled:
            ref = t_curr
            if self.use_latched_reference:
                if not self._prev_enabled or self.reference_ee_pose is None:
                    self.reference_ee_pose = t_curr.copy()
                ref = self.reference_ee_pose if self.reference_ee_pose is not None else t_curr

            step = self.end_effector_step_sizes_mm
            # target_* position is already in metres (phone delta mapped to xArm base frame).
            delta_p = np.array(
                [
                    tx * 1000.0 * step["x"],
                    ty * 1000.0 * step["y"],
                    tz * 1000.0 * step["z"],
                ],
                dtype=float,
            )
            r_delta = Rotation.from_rotvec([wx, wy, wz])
            desired = np.eye(4, dtype=float)
            # Phone rotation delta is in robot-base frame; pre-multiply latched EE orientation.
            desired[:3, :3] = r_delta.as_matrix() @ ref[:3, :3]
            desired[:3, 3] = ref[:3, 3] + delta_p
            self._command_when_disabled = desired.copy()
        else:
            if self._command_when_disabled is None:
                self._command_when_disabled = t_curr.copy()
            desired = self._command_when_disabled.copy()

        pos = desired[:3, 3]
        twist = Rotation.from_matrix(desired[:3, :3]).as_rotvec()
        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        action["ee.wx"] = float(twist[0])
        action["ee.wy"] = float(twist[1])
        action["ee.wz"] = float(twist[2])
        action["ee.gripper_vel"] = self._gripper_vel

        self._prev_enabled = enabled
        return action

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("phone_xarm_bounds_and_safety")
@dataclass
class PhoneXArmBoundsAndSafety(RobotActionProcessorStep):
    """Clip EE workspace (mm) and limit per-step Cartesian jumps."""

    end_effector_bounds_mm: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [150.0, -500.0, 50.0],
            "max": [650.0, 200.0, 600.0],
        }
    )
    max_ee_step_mm: float = 20.0
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        if not action:
            return {}

        pos = np.array([action["ee.x"], action["ee.y"], action["ee.z"]], dtype=float)
        pos = np.clip(pos, self.end_effector_bounds_mm["min"], self.end_effector_bounds_mm["max"])

        if self._last_pos is not None:
            dpos = pos - self._last_pos
            n = float(np.linalg.norm(dpos))
            if n > self.max_ee_step_mm and n > 0.0:
                pos = self._last_pos + dpos * (self.max_ee_step_mm / n)

        self._last_pos = pos
        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        return action

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("phone_xarm_gripper")
@dataclass
class PhoneXArmGripper(RobotActionProcessorStep):
    """Integrate phone gripper velocity into xArm normalized gripper j7 ∈ [0, 1]."""

    speed_factor: float = 0.04
    _gripper_norm: float | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        if not action:
            return {}

        observation = self.transition.get(TransitionKey.OBSERVATION)
        gripper_vel = float(action.pop("ee.gripper_vel"))
        if self._gripper_norm is None:
            self._gripper_norm = float(observation.get("j7", 0.0)) if observation else 0.0

        self._gripper_norm = float(
            np.clip(self._gripper_norm + gripper_vel * self.speed_factor, 0.0, 1.0)
        )

        pose = np.eye(4, dtype=float)
        pose[:3, 3] = [action.pop("ee.x"), action.pop("ee.y"), action.pop("ee.z")]
        twist = np.array([action.pop("ee.wx"), action.pop("ee.wy"), action.pop("ee.wz")], dtype=float)
        pose[:3, :3] = Rotation.from_rotvec(twist).as_matrix()

        return _pose_matrix_to_xarm_action(pose, self._gripper_norm)

    def transform_features(self, features):
        return features


def make_phone_xarm_processors(
    robot: Robot,
    teleop_config: PhoneConfig,
) -> tuple[
    RobotProcessorPipeline[tuple[RobotActionType, RobotObservation], RobotActionType],
    RobotProcessorPipeline[tuple[RobotActionType, RobotObservation], RobotActionType],
    RobotProcessorPipeline[RobotObservation, RobotObservation],
]:
    if not isinstance(robot, XArmRobot):
        raise TypeError(f"make_phone_xarm_processors expects XArmRobot, got {type(robot).__name__}")
    if robot.config.robot_mode not in (1, 7):
        raise ValueError(
            "Phone teleoperation for xArm requires Cartesian mode "
            f"(robot_mode=1 or 7), got {robot.config.robot_mode}."
        )

    teleop_action_processor = RobotProcessorPipeline[
        tuple[RobotActionType, RobotObservation], RobotActionType
    ](
        steps=[
            PhoneXArmSkipEmpty(),
            MapPhoneActionToXArmAction(platform=teleop_config.phone_os),
            PhoneXArmReferenceAndDelta(),
            PhoneXArmBoundsAndSafety(
                end_effector_bounds_mm={
                    "min": [
                        robot.config.phone_ee_x_min_mm,
                        robot.config.phone_ee_y_min_mm,
                        robot.config.phone_ee_z_min_mm,
                    ],
                    "max": [
                        robot.config.phone_ee_x_max_mm,
                        robot.config.phone_ee_y_max_mm,
                        robot.config.phone_ee_z_max_mm,
                    ],
                },
                max_ee_step_mm=robot.config.phone_max_ee_step_mm,
            ),
            PhoneXArmGripper(),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    robot_action_processor = RobotProcessorPipeline[
        tuple[RobotActionType, RobotObservation], RobotActionType
    ](
        steps=[IdentityProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    robot_observation_processor = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[IdentityProcessorStep()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    return teleop_action_processor, robot_action_processor, robot_observation_processor
