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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.model.kinematics import RobotKinematics
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
from lerobot.robots.bi_so_follower import BiSOFollower
from lerobot.robots.so_follower import SOFollower
from lerobot.types import RobotAction as RobotActionType

from .config_telegrip import TelegripConfig

WRIST_FLEX_JOINT = "wrist_flex"
WRIST_ROLL_JOINT = "wrist_roll"
GRIPPER_JOINT = "gripper"


@dataclass
class _ArmRuntimeState:
    enabled: bool = False
    origin_position: np.ndarray | None = None
    origin_wrist_roll: float = 0.0
    origin_wrist_flex: float = 0.0
    target_position: np.ndarray | None = None
    wrist_roll: float = 0.0
    wrist_flex: float = 0.0
    gripper_closed: bool = True
    q_curr: np.ndarray | None = None


@dataclass
class ArmKinematicsLayout:
    vr_side: Literal["left", "right"]
    motor_prefix: str
    motor_names: list[str]


@ProcessorStepRegistry.register("telegrip_vr_to_joints")
@dataclass
class TelegripVRToJoints(RobotActionProcessorStep):
    """
    Converts telegrip VR controller state into joint position commands.

    Mirrors telegrip control-loop logic: position IK on the first three joints,
    direct wrist roll/flex control, and trigger-based gripper open/close.
    """

    kinematics: RobotKinematics
    arms: list[ArmKinematicsLayout]
    gripper_open_angle: float = 0.0
    gripper_closed_angle: float = 45.0
    max_ee_step_m: float = 0.10
    max_joint_step_deg: float = 3.0
    end_effector_bounds_min: tuple[float, float, float] = (-0.5, -0.5, -0.1)
    end_effector_bounds_max: tuple[float, float, float] = (0.5, 0.5, 0.5)

    _arm_states: dict[str, _ArmRuntimeState] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        for arm in self.arms:
            key = f"{arm.motor_prefix}{arm.vr_side}"
            self._arm_states[key] = _ArmRuntimeState()

    def _obs_joint_positions(self, observation: RobotAction, arm: ArmKinematicsLayout) -> np.ndarray:
        values = []
        for name in arm.motor_names:
            key = f"{arm.motor_prefix}{name}.pos"
            if key not in observation:
                raise KeyError(f"Missing observation key '{key}' required for telegrip IK")
            values.append(float(observation[key]))
        return np.array(values, dtype=float)

    @staticmethod
    def _strip_vr_keys(action: RobotAction) -> None:
        """Remove any leftover VR teleop keys (e.g. unused controller side)."""
        for side in ("left", "right"):
            prefix = f"vr.{side}."
            for key in list(action.keys()):
                if key.startswith(prefix):
                    action.pop(key)

    def _read_vr_arm(self, action: RobotAction, vr_side: str):
        prefix = f"vr.{vr_side}."
        enabled = bool(action.pop(f"{prefix}enabled", False))
        target_delta = action.pop(f"{prefix}target_delta", None)
        wrist_roll_deg = float(action.pop(f"{prefix}wrist_roll_deg", 0.0))
        wrist_flex_deg = float(action.pop(f"{prefix}wrist_flex_deg", 0.0))
        gripper_closed = action.pop(f"{prefix}gripper_closed", None)
        reset_origin = bool(action.pop(f"{prefix}reset_origin", False))
        return enabled, target_delta, wrist_roll_deg, wrist_flex_deg, gripper_closed, reset_origin

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required for telegrip VR to joints conversion")

        for arm in self.arms:
            state_key = f"{arm.motor_prefix}{arm.vr_side}"
            arm_state = self._arm_states[state_key]
            q_raw = self._obs_joint_positions(observation, arm)

            enabled, target_delta, wrist_roll_deg, wrist_flex_deg, gripper_closed, reset_origin = (
                self._read_vr_arm(action, arm.vr_side)
            )

            if gripper_closed is not None:
                arm_state.gripper_closed = bool(gripper_closed)

            rising_edge = enabled and not arm_state.enabled
            if reset_origin or rising_edge:
                t_curr = self.kinematics.forward_kinematics(q_raw)
                arm_state.origin_position = t_curr[:3, 3].copy()
                arm_state.target_position = arm_state.origin_position.copy()
                wrist_flex_idx = arm.motor_names.index(WRIST_FLEX_JOINT)
                wrist_roll_idx = arm.motor_names.index(WRIST_ROLL_JOINT)
                arm_state.origin_wrist_flex = float(q_raw[wrist_flex_idx])
                arm_state.origin_wrist_roll = float(q_raw[wrist_roll_idx])
                arm_state.wrist_flex = arm_state.origin_wrist_flex
                arm_state.wrist_roll = arm_state.origin_wrist_roll

            if enabled and target_delta is not None and arm_state.origin_position is not None:
                target = arm_state.origin_position + np.asarray(target_delta, dtype=float)
                target = np.clip(
                    target,
                    self.end_effector_bounds_min,
                    self.end_effector_bounds_max,
                )
                if arm_state.target_position is not None:
                    delta = target - arm_state.target_position
                    dist = float(np.linalg.norm(delta))
                    if dist > self.max_ee_step_m and dist > 0:
                        target = arm_state.target_position + delta * (self.max_ee_step_m / dist)
                arm_state.target_position = target
                arm_state.wrist_roll = arm_state.origin_wrist_roll + wrist_roll_deg
                arm_state.wrist_flex = arm_state.origin_wrist_flex + wrist_flex_deg
            elif not enabled:
                arm_state.target_position = None

            arm_state.enabled = enabled

            if arm_state.enabled and arm_state.target_position is not None:
                if arm_state.q_curr is None:
                    arm_state.q_curr = q_raw.copy()
                else:
                    arm_state.q_curr = q_raw.copy()

                t_des = self.kinematics.forward_kinematics(arm_state.q_curr)
                t_des[:3, 3] = arm_state.target_position
                q_target = self.kinematics.inverse_kinematics(
                    arm_state.q_curr, t_des, orientation_weight=0.0
                )
                arm_state.q_curr = q_target

                gripper_angle = (
                    self.gripper_closed_angle if arm_state.gripper_closed else self.gripper_open_angle
                )
                wrist_flex_idx = arm.motor_names.index(WRIST_FLEX_JOINT)
                wrist_roll_idx = arm.motor_names.index(WRIST_ROLL_JOINT)
                gripper_idx = arm.motor_names.index(GRIPPER_JOINT)
                q_target[wrist_flex_idx] = arm_state.wrist_flex
                q_target[wrist_roll_idx] = arm_state.wrist_roll
                q_target[gripper_idx] = gripper_angle

                for i in range(len(q_target)):
                    joint_delta = float(q_target[i] - q_raw[i])
                    if abs(joint_delta) > self.max_joint_step_deg:
                        q_target[i] = q_raw[i] + np.sign(joint_delta) * self.max_joint_step_deg

                for i, name in enumerate(arm.motor_names):
                    action[f"{arm.motor_prefix}{name}.pos"] = float(q_target[i])
            elif gripper_closed is not None:
                gripper_idx = arm.motor_names.index(GRIPPER_JOINT)
                q_target = q_raw.copy()
                q_target[gripper_idx] = (
                    self.gripper_closed_angle if arm_state.gripper_closed else self.gripper_open_angle
                )
                for i, name in enumerate(arm.motor_names):
                    action[f"{arm.motor_prefix}{name}.pos"] = float(q_target[i])

        self._strip_vr_keys(action)
        return action

    def reset(self):
        for state in self._arm_states.values():
            state.enabled = False
            state.origin_position = None
            state.target_position = None
            state.q_curr = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for side in ("left", "right"):
            for feat in (
                "enabled",
                "target_delta",
                "wrist_roll_deg",
                "wrist_flex_deg",
                "gripper_closed",
                "reset_origin",
            ):
                features[PipelineFeatureType.ACTION].pop(f"vr.{side}.{feat}", None)

        for arm in self.arms:
            for name in arm.motor_names:
                features[PipelineFeatureType.ACTION][f"{arm.motor_prefix}{name}.pos"] = PolicyFeature(
                    type=FeatureType.ACTION, shape=(1,)
                )
        return features


_STANDARD_SO_MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
_TELEGrip_SO100_URDF_JOINTS = ["1", "2", "3", "4", "5", "6"]
_EE_FRAME_CANDIDATES = ("gripper_frame_link", "Fixed_Jaw_tip")


def _infer_urdf_kinematics_settings(
    urdf_path: str, motor_names: list[str], target_frame_name: str
) -> tuple[list[str], str]:
    """Match placo joint / frame names to the loaded URDF (SO100 vs SO101)."""
    import placo

    robot = placo.RobotWrapper(urdf_path)
    urdf_joints = list(robot.joint_names())

    if all(name in urdf_joints for name in motor_names):
        kinematics_joint_names = motor_names
    elif (
        urdf_joints == _TELEGrip_SO100_URDF_JOINTS
        and motor_names == _STANDARD_SO_MOTOR_NAMES
    ):
        kinematics_joint_names = _TELEGrip_SO100_URDF_JOINTS
    else:
        kinematics_joint_names = urdf_joints[: len(motor_names)]

    resolved_frame = target_frame_name
    frame_candidates = [target_frame_name, *_EE_FRAME_CANDIDATES]
    seen: set[str] = set()
    for frame in frame_candidates:
        if frame in seen:
            continue
        seen.add(frame)
        try:
            robot.get_T_world_frame(frame)
            resolved_frame = frame
            break
        except Exception:
            continue

    return kinematics_joint_names, resolved_frame


_SO101_URDF_SETUP_HINT = (
    "Download the SO101 simulation folder from SO-ARM100 and point --teleop.urdf_path "
    "to the .urdf file inside it (the folder must also contain the assets/ meshes).\n"
    "  git clone https://github.com/TheRobotStudio/SO-ARM100.git\n"
    "  # then use: --teleop.urdf_path=SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
)


def resolve_urdf_path(urdf_path: str) -> str:
    """Resolve and validate a URDF path for placo (file + sibling assets/)."""
    path = Path(urdf_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    if path.is_dir():
        candidates = sorted(path.glob("*.urdf"))
        if not candidates:
            raise FileNotFoundError(
                f"No .urdf file found in directory '{path}'.\n{_SO101_URDF_SETUP_HINT}"
            )
        path = candidates[0]

    if not path.is_file():
        raise FileNotFoundError(
            f"URDF file not found: '{path}'.\n{_SO101_URDF_SETUP_HINT}"
        )

    assets_dir = path.parent / "assets"
    if not assets_dir.is_dir():
        raise FileNotFoundError(
            f"URDF mesh folder not found: '{assets_dir}'.\n"
            f"Copy the full SO101 directory (URDF + assets/) from SO-ARM100, not just the .urdf file.\n"
            f"{_SO101_URDF_SETUP_HINT}"
        )

    return str(path.resolve())


def _motor_names_from_robot(robot: Robot) -> list[str]:
    if isinstance(robot, SOFollower):
        return list(robot.bus.motors.keys())
    if isinstance(robot, BiSOFollower):
        return list(robot.left_arm.bus.motors.keys())
    raise ValueError(
        f"Robot type '{robot.name}' is not supported for telegrip teleoperation. "
        "Supported: so100_follower, so101_follower, bi_so_follower."
    )


def _arm_layouts(robot: Robot, teleop_config: TelegripConfig) -> list[ArmKinematicsLayout]:
    motor_names = _motor_names_from_robot(robot)
    if isinstance(robot, BiSOFollower):
        return [
            ArmKinematicsLayout(vr_side="left", motor_prefix="left_", motor_names=motor_names),
            ArmKinematicsLayout(vr_side="right", motor_prefix="right_", motor_names=motor_names),
        ]
    return [
        ArmKinematicsLayout(
            vr_side=teleop_config.controller_side.value,
            motor_prefix="",
            motor_names=motor_names,
        )
    ]


def make_telegrip_processors(
    robot: Robot, teleop_config: TelegripConfig
) -> tuple[
    RobotProcessorPipeline[tuple[RobotActionType, RobotObservation], RobotActionType],
    RobotProcessorPipeline[tuple[RobotActionType, RobotObservation], RobotActionType],
    RobotProcessorPipeline[RobotObservation, RobotObservation],
]:
    if not teleop_config.urdf_path:
        raise ValueError(
            "telegrip teleoperation requires --teleop.urdf_path pointing to the robot URDF "
            "(e.g. SO101/so101_new_calib.urdf from SO-ARM100)."
        )

    motor_names = _motor_names_from_robot(robot)
    urdf_path = resolve_urdf_path(teleop_config.urdf_path)
    urdf_joint_names, target_frame_name = _infer_urdf_kinematics_settings(
        urdf_path, motor_names, teleop_config.target_frame_name
    )
    kinematics = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name=target_frame_name,
        joint_names=urdf_joint_names,
    )

    teleop_action_processor = RobotProcessorPipeline[
        tuple[RobotActionType, RobotObservation], RobotActionType
    ](
        steps=[
            TelegripVRToJoints(
                kinematics=kinematics,
                arms=_arm_layouts(robot, teleop_config),
                gripper_open_angle=teleop_config.gripper_open_angle,
                gripper_closed_angle=teleop_config.gripper_closed_angle,
                max_ee_step_m=teleop_config.max_ee_step_m,
                max_joint_step_deg=teleop_config.max_joint_step_deg,
                end_effector_bounds_min=teleop_config.end_effector_bounds_min,
                end_effector_bounds_max=teleop_config.end_effector_bounds_max,
            ),
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
