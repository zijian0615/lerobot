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

"""Quest WebXR controller (telegrip) → FANUC user-frame cartesian pose.

``target_delta`` comes from telegrip as ``[-vr_x, vr_z, vr_y]`` (WebXR: x right, y up, z back toward the user).
``_VR_TO_FANUC`` maps WebXR axes to the user frame, for both position and wrist orientation: on this cell,
controller right increases user-frame X, controller forward increases user-frame Y, and up stays Z.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from lerobot.utils.rotation import Rotation
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
from lerobot.teleoperators.telegrip.config_telegrip import TelegripConfig
from lerobot.types import RobotAction as RobotActionType

from .fanuc import Fanuc
from .pose import decode_fanuc_pose_dict, raw_fanuc_pose_dict, rotation_matrix_from_wpr, wpr_from_rotation_matrix

# Columns are the VR axes (right, up, back) expressed in the Fanuc user frame.
# Controller right decreases X. Controller forward (negative back) decreases Y. Up stays Z.
# rows: user-frame X, Y, Z in WebXR axes (x right, y up, z back): X = right, Y = forward = -back, Z = up
_VR_TO_FANUC = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)

logger = logging.getLogger(__name__)


def _wpr_from_controller(origin_wpr: np.ndarray, wrist_quat: np.ndarray | None) -> np.ndarray:
    """Apply the controller's relative orientation, in the Fanuc user frame, to the latched WPR."""
    if wrist_quat is None:
        return origin_wpr.copy()
    quat = np.asarray(wrist_quat, dtype=float).reshape(4)
    if float(np.linalg.norm(quat)) < 1e-8:
        return origin_wpr.copy()
    relative_vr = Rotation.from_quat(quat).as_matrix()
    relative_fanuc = _VR_TO_FANUC @ relative_vr @ _VR_TO_FANUC.T
    latched = np.array(rotation_matrix_from_wpr(origin_wpr[0], origin_wpr[1], origin_wpr[2]), dtype=float)
    return np.array(wpr_from_rotation_matrix((relative_fanuc @ latched).tolist()), dtype=float)


def _obs_pose(observation: RobotObservation) -> tuple[np.ndarray, np.ndarray]:
    if not all(k in observation for k in ("j0", "j1", "j2")):
        raise KeyError("FANUC observation missing j0..j2")
    decoded = decode_fanuc_pose_dict(observation)
    if not all(k in decoded for k in ("j3", "j4", "j5")):
        raise KeyError("FANUC observation missing W/P/R (j3..j5)")
    pos = np.array([float(observation["j0"]), float(observation["j1"]), float(observation["j2"])], dtype=float)
    wpr = np.array([float(decoded["j3"]), float(decoded["j4"]), float(decoded["j5"])], dtype=float)
    return pos, wpr


@ProcessorStepRegistry.register("telegrip_vr_to_fanuc_pose")
@dataclass
class TelegripVRToFanucPose(RobotActionProcessorStep):
    """Hold grip to track the controller; trigger opens the gripper."""

    controller_side: str = "right"
    bounds_min_mm: tuple[float, float, float] = (-450.0, -450.0, -330.0)
    bounds_max_mm: tuple[float, float, float] = (450.0, 450.0, 250.0)
    max_step_mm: float = 15.0

    _enabled: bool = field(default=False, init=False, repr=False)
    _origin_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _origin_wpr: np.ndarray | None = field(default=None, init=False, repr=False)
    _target_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    _held_wpr: np.ndarray | None = field(default=None, init=False, repr=False)
    _gripper: float | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Robot observation is required for telegrip FANUC teleoperation")

        cur_pos, cur_wpr = _obs_pose(observation)
        prefix = f"vr.{self.controller_side}."
        enabled = bool(action.get(f"{prefix}enabled", False))
        target_delta = action.get(f"{prefix}target_delta")
        wrist_quat = action.get(f"{prefix}wrist_quat")
        gripper_closed = action.get(f"{prefix}gripper_closed")
        reset_origin = bool(action.get(f"{prefix}reset_origin", False))

        if gripper_closed is not None:
            # telegrip sets gripper_closed False while the trigger is held.
            # On this arm, holding that button closes the gripper.
            self._gripper = 0.0 if bool(gripper_closed) else 1.0
        if self._gripper is None:
            self._gripper = 1.0 if float(observation.get("j7", 0.0)) >= 0.5 else 0.0

        if not enabled:
            self._enabled = False
            self._origin_pos = None
            self._origin_wpr = None
            self._target_pos = None
            self._held_wpr = None
            return {}

        rising = not self._enabled
        if reset_origin or rising or self._origin_pos is None:
            self._origin_pos = cur_pos.copy()
            self._origin_wpr = cur_wpr.copy()
            self._target_pos = cur_pos.copy()
            self._held_wpr = cur_wpr.copy()
            logger.info(
                "Quest grip held, tracking from xyz=(%.1f, %.1f, %.1f) wpr=(%.1f, %.1f, %.1f)",
                cur_pos[0],
                cur_pos[1],
                cur_pos[2],
                cur_wpr[0],
                cur_wpr[1],
                cur_wpr[2],
            )

        if enabled and target_delta is not None and self._origin_pos is not None and self._origin_wpr is not None:
            telegrip_mm = np.asarray(target_delta, dtype=float) * 1000.0
            # telegrip target_delta is [-vr_x, vr_z, vr_y]; back to WebXR axes, then the same map as the wrist
            vr_mm = np.array([-telegrip_mm[0], telegrip_mm[2], telegrip_mm[1]], dtype=float)
            delta_mm = _VR_TO_FANUC @ vr_mm
            target = self._origin_pos + delta_mm
            if self._target_pos is not None:
                delta = target - self._target_pos
                step = float(np.linalg.norm(delta))
                if step > self.max_step_mm and step > 0.0:
                    target = self._target_pos + delta * (self.max_step_mm / step)
            self._target_pos = target
            self._held_wpr = _wpr_from_controller(self._origin_wpr, wrist_quat)

        pos = self._target_pos if self._target_pos is not None else cur_pos
        wpr = self._held_wpr if self._held_wpr is not None else cur_wpr
        self._enabled = enabled

        sent = raw_fanuc_pose_dict(
            {
                "j0": float(pos[0]),
                "j1": float(pos[1]),
                "j2": float(pos[2]),
                "j3": float(wpr[0]),
                "j4": float(wpr[1]),
                "j5": float(wpr[2]),
                "j7": float(self._gripper),
            }
        )
        sent["stream"] = True
        return sent

    def reset(self) -> None:
        self._enabled = False
        self._origin_pos = None
        self._origin_wpr = None
        self._target_pos = None
        self._held_wpr = None
        self._gripper = None

    def transform_features(self, features):
        return features


def make_telegrip_fanuc_processors(
    robot: Fanuc,
    teleop_config: TelegripConfig,
) -> tuple[
    RobotProcessorPipeline[tuple[RobotActionType, RobotObservation], RobotActionType],
    RobotProcessorPipeline[tuple[RobotActionType, RobotObservation], RobotActionType],
    RobotProcessorPipeline[RobotObservation, RobotObservation],
]:
    """Quest controller → FANUC pipelines. Does not use a URDF."""
    cfg = robot.config
    teleop_action_processor = RobotProcessorPipeline[tuple[RobotActionType, RobotObservation], RobotActionType](
        steps=[
            TelegripVRToFanucPose(
                controller_side=teleop_config.controller_side.value,
                bounds_min_mm=(cfg.phone_ee_x_min_mm, cfg.phone_ee_y_min_mm, cfg.phone_ee_z_min_mm),
                bounds_max_mm=(cfg.phone_ee_x_max_mm, cfg.phone_ee_y_max_mm, cfg.phone_ee_z_max_mm),
                max_step_mm=max(cfg.phone_max_ee_step_mm, 40.0),
            ),
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
