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

"""Quest / phone teleop → FANUC user-frame cartesian pose.

Same headset mapping as the xArm phone pipeline (iOS HEBI or Android/Quest
WebXR): calibrated ``phone.pos`` / ``phone.rot`` deltas plus gripper buttons.
Output is the FANUC action ``send_action`` already accepts: ``j0..j2`` in mm,
W/P/R as ``j3``..``j5`` in degrees (W in [0, 360)), and binary ``j7``.
"""

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

from .fanuc import Fanuc
from .pose import (
    decode_fanuc_pose_dict,
    raw_fanuc_pose_dict,
    rotation_matrix_from_wpr,
    wpr_from_rotation_matrix,
)


def _phone_calib_to_base_delta(pos: np.ndarray, rotvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Phone calibration frame → robot base. Same axis map as xArm Quest teleop."""
    delta_pos_m = np.array([pos[1], -pos[0], pos[2]], dtype=float)
    delta_rot = np.array([rotvec[1], -rotvec[0], rotvec[2]], dtype=float)
    return delta_pos_m, delta_rot


def _obs_pose_matrix(obs: RobotObservation) -> np.ndarray:
    if not all(k in obs for k in ("j0", "j1", "j2")):
        raise KeyError("FANUC observation missing j0..j2")
    decoded = decode_fanuc_pose_dict(obs)
    if not all(k in decoded for k in ("j3", "j4", "j5")):
        raise KeyError("FANUC observation missing W/P/R (j3..j5)")
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = np.array(
        rotation_matrix_from_wpr(decoded["j3"], decoded["j4"], decoded["j5"]),
        dtype=float,
    )
    pose[:3, 3] = [float(obs["j0"]), float(obs["j1"]), float(obs["j2"])]
    return pose


def _pose_to_fanuc_action(pose: np.ndarray, gripper_norm: float) -> RobotAction:
    w_deg, p_deg, r_deg = wpr_from_rotation_matrix(pose[:3, :3].tolist())
    return raw_fanuc_pose_dict(
        {
            "j0": float(pose[0, 3]),
            "j1": float(pose[1, 3]),
            "j2": float(pose[2, 3]),
            "j3": float(w_deg),
            "j4": float(p_deg),
            "j5": float(r_deg),
            "j7": 1.0 if float(gripper_norm) >= 0.5 else 0.0,
        }
    )


@ProcessorStepRegistry.register("phone_fanuc_skip_empty")
@dataclass
class PhoneFanucSkipEmpty(RobotActionProcessorStep):
    """Skip the pipeline until the headset reports a calibrated pose."""

    def action(self, action: RobotAction) -> RobotAction:
        if not action or "phone.pos" not in action:
            return {}
        return action

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("map_phone_action_to_fanuc_action")
@dataclass
class MapPhoneActionToFanucAction(RobotActionProcessorStep):
    """Phone calibrated pose → base-frame position (m) and rotation (rotvec)."""

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
            delta_pos_m, delta_rot = _phone_calib_to_base_delta(np.asarray(pos, dtype=float), rot.as_rotvec())
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


@ProcessorStepRegistry.register("phone_fanuc_reference_and_delta")
@dataclass
class PhoneFanucReferenceAndDelta(RobotActionProcessorStep):
    """Latch the user-frame pose when the headset enables, then apply the delta."""

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
            raise ValueError("Robot observation is required for phone FANUC teleoperation")

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

            delta_p = np.array([tx * 1000.0, ty * 1000.0, tz * 1000.0], dtype=float)
            desired = np.eye(4, dtype=float)
            desired[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix() @ ref[:3, :3]
            desired[:3, 3] = ref[:3, 3] + delta_p
            self._command_when_disabled = desired.copy()
        else:
            if self._command_when_disabled is None:
                self._command_when_disabled = t_curr.copy()
            desired = self._command_when_disabled.copy()

        w_deg, p_deg, r_deg = wpr_from_rotation_matrix(desired[:3, :3].tolist())
        action["ee.x"] = float(desired[0, 3])
        action["ee.y"] = float(desired[1, 3])
        action["ee.z"] = float(desired[2, 3])
        action["ee.w"] = float(w_deg)
        action["ee.p"] = float(p_deg)
        action["ee.r"] = float(r_deg)
        action["ee.gripper_vel"] = self._gripper_vel
        self._prev_enabled = enabled
        return action

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("phone_fanuc_bounds_and_safety")
@dataclass
class PhoneFanucBoundsAndSafety(RobotActionProcessorStep):
    """Clip the user-frame box and limit the Cartesian step between commands."""

    end_effector_bounds_mm: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-450.0, -450.0, -330.0],
            "max": [450.0, 450.0, 250.0],
        }
    )
    max_ee_step_mm: float = 15.0
    _last_pos: np.ndarray | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        if not action:
            return {}

        pos = np.array([action["ee.x"], action["ee.y"], action["ee.z"]], dtype=float)
        pos = np.clip(pos, self.end_effector_bounds_mm["min"], self.end_effector_bounds_mm["max"])
        if self._last_pos is not None:
            dpos = pos - self._last_pos
            step = float(np.linalg.norm(dpos))
            if step > self.max_ee_step_mm and step > 0.0:
                pos = self._last_pos + dpos * (self.max_ee_step_mm / step)

        self._last_pos = pos
        action["ee.x"] = float(pos[0])
        action["ee.y"] = float(pos[1])
        action["ee.z"] = float(pos[2])
        return action

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("phone_fanuc_gripper")
@dataclass
class PhoneFanucGripper(RobotActionProcessorStep):
    """Integrate headset gripper velocity, then binarize j7 for the RMI ports."""

    speed_factor: float = 0.04
    _gripper_norm: float | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        if not action:
            return {}

        observation = self.transition.get(TransitionKey.OBSERVATION)
        gripper_vel = float(action.pop("ee.gripper_vel"))
        if self._gripper_norm is None:
            current = float(observation.get("j7", 0.0)) if observation else 0.0
            self._gripper_norm = 1.0 if current >= 0.5 else 0.0

        self._gripper_norm = float(np.clip(self._gripper_norm + gripper_vel * self.speed_factor, 0.0, 1.0))
        pose = np.eye(4, dtype=float)
        pose[:3, 3] = [action.pop("ee.x"), action.pop("ee.y"), action.pop("ee.z")]
        pose[:3, :3] = np.array(
            rotation_matrix_from_wpr(action.pop("ee.w"), action.pop("ee.p"), action.pop("ee.r")),
            dtype=float,
        )
        action = _pose_to_fanuc_action(pose, self._gripper_norm)
        action["stream"] = True
        return action

    def transform_features(self, features):
        return features


def make_phone_fanuc_processors(
    robot: Robot,
    teleop_config: PhoneConfig,
) -> tuple[
    RobotProcessorPipeline[tuple[RobotActionType, RobotObservation], RobotActionType],
    RobotProcessorPipeline[tuple[RobotActionType, RobotObservation], RobotActionType],
    RobotProcessorPipeline[RobotObservation, RobotObservation],
]:
    """Build Quest/phone → FANUC pipelines for teleoperate and record."""
    if not isinstance(robot, Fanuc):
        raise TypeError(f"make_phone_fanuc_processors expects Fanuc, got {type(robot).__name__}")

    cfg = robot.config
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotActionType, RobotObservation], RobotActionType](
        steps=[
            PhoneFanucSkipEmpty(),
            MapPhoneActionToFanucAction(platform=teleop_config.phone_os),
            PhoneFanucReferenceAndDelta(),
            PhoneFanucBoundsAndSafety(
                end_effector_bounds_mm={
                    "min": [cfg.phone_ee_x_min_mm, cfg.phone_ee_y_min_mm, cfg.phone_ee_z_min_mm],
                    "max": [cfg.phone_ee_x_max_mm, cfg.phone_ee_y_max_mm, cfg.phone_ee_z_max_mm],
                },
                max_ee_step_mm=cfg.phone_max_ee_step_mm,
            ),
            PhoneFanucGripper(),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    robot_action_processor = RobotProcessorPipeline[tuple[RobotActionType, RobotObservation], RobotActionType](
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
