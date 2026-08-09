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

"""Per-arm blocking executor. No models. All motion via ``move_to_pose``."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

Pose = tuple[float, float, float, float]  # x, y, z, yaw
MoveToPose = Callable[[Pose], None]
GripperCmd = Callable[[str], None]  # "open" | "close"
WidthReader = Callable[[], float]
# Returns geometric_view-like dict with objects[{name, xy, ...}]
PerceiveFn = Callable[[], Mapping[str, Any]]


class ExecutionResult(TypedDict):
    step: int
    status: str  # "success" | "fail"
    reason: str
    observed: dict[str, Any]


@dataclass
class ExecutorConfig:
    approach_offset: float = 0.08
    lift_offset: float = 0.08
    gripper_open_width: float = 0.08
    gripper_closed_width: float = 0.01
    place_xy_tol: float = 0.03


class ArmExecutor:
    """
    One executor instance per arm.

    Grasp(pose): open → above → descend → close → ascend
    Place(pose): above → descend → open → ascend
    LiftUp():    up by lift_offset
    """

    def __init__(
        self,
        arm: str,
        *,
        move_to_pose: MoveToPose,
        gripper: GripperCmd,
        read_gripper_width: WidthReader,
        perceive: PerceiveFn | None = None,
        config: ExecutorConfig | None = None,
        get_current_pose: Callable[[], Pose] | None = None,
    ) -> None:
        self.arm = arm
        self.move_to_pose = move_to_pose
        self.gripper = gripper
        self.read_gripper_width = read_gripper_width
        self.perceive = perceive
        self.config = config or ExecutorConfig()
        self.get_current_pose = get_current_pose
        self._last_pose: Pose | None = None

    def _above(self, pose: Sequence[float]) -> Pose:
        x, y, z, yaw = (float(pose[0]), float(pose[1]), float(pose[2]), float(pose[3]))
        return (x, y, z + self.config.approach_offset, yaw)

    def _go(self, pose: Pose) -> None:
        self.move_to_pose(pose)
        self._last_pose = pose

    def _width_ok(self) -> tuple[bool, float]:
        w = float(self.read_gripper_width())
        lo = self.config.gripper_closed_width
        hi = self.config.gripper_open_width
        return (lo < w < hi), w

    def execute(self, bound_step: Mapping[str, Any]) -> ExecutionResult:
        sid = int(bound_step["step"])
        prim = str(bound_step["primitive"])
        params = dict(bound_step.get("params") or {})

        try:
            if prim == "Grasp":
                return self._grasp(sid, params)
            if prim == "Place":
                return self._place(sid, params)
            if prim == "LiftUp":
                return self._lift_up(sid)
            return {
                "step": sid,
                "status": "fail",
                "reason": f"unknown_primitive:{prim}",
                "observed": {},
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "step": sid,
                "status": "fail",
                "reason": f"exception:{type(exc).__name__}:{exc}",
                "observed": {},
            }

    def _grasp(self, step: int, params: Mapping[str, Any]) -> ExecutionResult:
        pose = tuple(float(v) for v in params["pose"])
        assert len(pose) == 4
        self.gripper("open")
        self._go(self._above(pose))
        self._go(pose)  # type: ignore[arg-type]
        self.gripper("close")
        self._go(self._above(pose))

        ok, width = self._width_ok()
        observed = {"gripper_width": width, "object": params.get("object")}
        if not ok:
            return {
                "step": step,
                "status": "fail",
                "reason": "grasp_empty",
                "observed": observed,
            }
        return {"step": step, "status": "success", "reason": "", "observed": observed}

    def _place(self, step: int, params: Mapping[str, Any]) -> ExecutionResult:
        pose = tuple(float(v) for v in params["pose"])
        assert len(pose) == 4
        obj = str(params.get("object", ""))
        self._go(self._above(pose))
        self._go(pose)  # type: ignore[arg-type]
        self.gripper("open")
        self._go(self._above(pose))

        observed: dict[str, Any] = {
            "place_pose": list(pose),
            "object": obj,
        }
        if self.perceive is None:
            return {
                "step": step,
                "status": "fail",
                "reason": "place_offset",
                "observed": {**observed, "note": "no perceive callback"},
            }

        geo = self.perceive()
        xy = None
        for item in geo.get("objects", []):
            if str(item.get("name")) == obj:
                xy = (float(item["xy"][0]), float(item["xy"][1]))
                break
        observed["observed_xy"] = list(xy) if xy is not None else None
        if xy is None:
            return {
                "step": step,
                "status": "fail",
                "reason": "place_offset",
                "observed": observed,
            }
        err = ((xy[0] - pose[0]) ** 2 + (xy[1] - pose[1]) ** 2) ** 0.5
        observed["xy_error"] = err
        if err > self.config.place_xy_tol:
            return {
                "step": step,
                "status": "fail",
                "reason": "place_offset",
                "observed": observed,
            }
        return {"step": step, "status": "success", "reason": "", "observed": observed}

    def _lift_up(self, step: int) -> ExecutionResult:
        if self.get_current_pose is not None:
            x, y, z, yaw = self.get_current_pose()
        elif self._last_pose is not None:
            x, y, z, yaw = self._last_pose
        else:
            return {
                "step": step,
                "status": "fail",
                "reason": "exception:no_current_pose",
                "observed": {},
            }
        target = (x, y, z + self.config.lift_offset, yaw)
        self._go(target)
        ok, width = self._width_ok()
        observed = {"gripper_width": width, "pose": list(target)}
        if not ok:
            return {
                "step": step,
                "status": "fail",
                "reason": "grasp_empty",
                "observed": observed,
            }
        return {"step": step, "status": "success", "reason": "", "observed": observed}
