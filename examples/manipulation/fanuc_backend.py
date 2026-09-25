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

"""Table-frame poses → Fanuc UF LinearMotion (conservative XY + wrist)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from tabletop_perception.fanuc.jog_xy import move_fanuc_xyz
from tabletop_perception.fanuc.run_fanuc_grasp import (
    _clamp_z,
    _pick_reachable_r,
    _reach_from_samples,
    _read_tcp,
    _table_yaw_to_fanuc_r_deg,
    _travel_xy,
    _workspace_from_samples,
    _wrap_signed_deg,
)


class FanucMotionBackend:
    """Blocking Cartesian moves + digital gripper for ``ArmExecutor``."""

    def __init__(
        self,
        robot: Any,
        *,
        calib: dict[str, Any],
        arm_name: str = "fanuc",
        samples_file: Path | None = None,
        speed: float = 40.0,
        yaw_speed: float = 10.0,
    ) -> None:
        exe = dict(calib.get("execution_by_arm") or {}).get(arm_name) or dict(
            calib.get("execution") or {}
        )
        self.robot = robot
        self.speed = float(speed)
        self.yaw_speed = float(yaw_speed)
        self.base_xy = tuple(float(v) for v in exe.get("table_base_xy_m", [0.0, 0.0]))
        self.table_z_base_m = float(exe.get("table_z_base_m", -0.335))
        self.flip_x = bool(exe.get("flip_x", False))
        self.flip_y = bool(exe.get("flip_y", False))
        off = exe.get("xy_offset_base_mm") or [0.0, 0.0]
        self.xy_off = (float(off[0]), float(off[1]))
        self.taught_wpr = tuple(float(v) for v in exe.get("topdown_wpr_deg", [180.0, 0.0, 0.0]))
        self.use_object_yaw = bool(exe.get("use_object_yaw", True))
        self.yaw_offset = float(exe.get("grasp_yaw_offset_rad", 0.0))
        self.approach_z_mm = (self.table_z_base_m + float(exe.get("approach_offset_m", 0.08))) * 1000.0
        samples = samples_file or (
            Path(__file__).resolve().parents[1]
            / "tabletop_perception"
            / "calib"
            / "table_xy_calib_samples_fanuc.json"
        )
        self.workspace = _workspace_from_samples(samples)
        self.reach = _reach_from_samples(samples)
        self._gripper_closed = False
        self.home_tcp = dict(_read_tcp(self.robot))
        print(
            "home TCP = "
            f"({self.home_tcp['x_mm']:.1f}, {self.home_tcp['y_mm']:.1f}, {self.home_tcp['z_mm']:.1f}) "
            f"WPR=({self.home_tcp['w_deg']:.1f}, {self.home_tcp['p_deg']:.1f}, {self.home_tcp['r_deg']:.1f})",
            flush=True,
        )

    def _xy_mm(self, x_t: float, y_t: float) -> tuple[float, float]:
        x = (x_t - self.base_xy[0]) * 1000.0
        y = (y_t - self.base_xy[1]) * 1000.0
        if self.flip_x:
            x = -x
        if self.flip_y:
            y = -y
        return x + self.xy_off[0], y + self.xy_off[1]

    def _z_mm(self, z_t: float) -> float:
        return _clamp_z((self.table_z_base_m + float(z_t)) * 1000.0)

    def _xy_reachable(self, x_mm: float, y_mm: float) -> bool:
        """Loose envelope around calib samples. Rejects laptop-corner ghosts."""
        box = self.workspace
        if box is None:
            return True
        extra = 150.0
        x0, x1, y0, y1 = box
        return (x0 - extra) <= x_mm <= (x1 + extra) and (y0 - extra) <= y_mm <= (y1 + extra)

    def _facing_wpr(self, yaw_t: float, current_r: float) -> tuple[float, float, float]:
        w, p, taught_r = self.taught_wpr
        if not self.use_object_yaw:
            return (w, p, taught_r)
        target_r = _table_yaw_to_fanuc_r_deg(
            float(yaw_t),
            flip_x=self.flip_x,
            flip_y=self.flip_y,
            offset_rad=self.yaw_offset,
            taught_r_deg=taught_r,
        )
        return (w, p, _pick_reachable_r(target_r, current_r, taught_r))

    def get_current_pose_table(self) -> tuple[float, float, float, float]:
        tcp = _read_tcp(self.robot)
        x_t = tcp["x_mm"] / 1000.0 + self.base_xy[0]
        y_t = tcp["y_mm"] / 1000.0 + self.base_xy[1]
        if self.flip_x:
            x_t = -tcp["x_mm"] / 1000.0 + self.base_xy[0]
        if self.flip_y:
            y_t = -tcp["y_mm"] / 1000.0 + self.base_xy[1]
        z_t = tcp["z_mm"] / 1000.0 - self.table_z_base_m
        yaw = math.radians(_wrap_signed_deg(tcp["r_deg"] - self.taught_wpr[2]))
        return (x_t, y_t, z_t, yaw)

    def go_home(self) -> None:
        """Lift from the current TCP, then return to the home captured at connect."""
        recover = getattr(self.robot, "recover_after_fault", None)
        if recover is not None:
            print("go home: Abort/Reset after fault, then move", flush=True)
            recover()
            seq = getattr(self.robot, "seq_id", None)
            if seq is not None:
                print(f"go home: RMI SequenceID reset to {seq}", flush=True)
        home = self.home_tcp
        if not home:
            self.home_tcp = dict(_read_tcp(self.robot))
            home = self.home_tcp
        tcp = _read_tcp(self.robot)
        hold = (tcp["w_deg"], tcp["p_deg"], tcp["r_deg"])
        home_wpr = (home["w_deg"], home["p_deg"], home["r_deg"])
        travel_z = max(float(tcp["z_mm"]), float(home["z_mm"]), float(self.approach_z_mm))
        print(
            f"go home → ({home['x_mm']:.1f}, {home['y_mm']:.1f}, {home['z_mm']:.1f}) "
            f"travel_z={travel_z:.1f}",
            flush=True,
        )
        if abs(tcp["z_mm"] - travel_z) > 1.0:
            move_fanuc_xyz(
                self.robot,
                x_mm=tcp["x_mm"],
                y_mm=tcp["y_mm"],
                z_mm=travel_z,
                wpr_deg=hold,
                speed=self.speed,
            )
            tcp = _read_tcp(self.robot)
            hold = (tcp["w_deg"], tcp["p_deg"], tcp["r_deg"])
        if math.hypot(home["x_mm"] - tcp["x_mm"], home["y_mm"] - tcp["y_mm"]) > 1.0:
            _travel_xy(
                self.robot,
                x0=tcp["x_mm"],
                y0=tcp["y_mm"],
                x1=home["x_mm"],
                y1=home["y_mm"],
                z_mm=travel_z,
                wpr_deg=hold,
                speed=self.speed,
                reach=self.reach,
            )
            tcp = _read_tcp(self.robot)
            hold = (tcp["w_deg"], tcp["p_deg"], tcp["r_deg"])
        delta_r = _wrap_signed_deg(home_wpr[2] - hold[2])
        if abs(delta_r) > 2.0:
            print(
                f"go home: skip in-place ΔR={delta_r:.1f}° "
                f"(hold R={hold[2]:.1f}, home R={home_wpr[2]:.1f})",
                flush=True,
            )
        if abs(tcp["z_mm"] - home["z_mm"]) > 1.0:
            move_fanuc_xyz(
                self.robot,
                x_mm=home["x_mm"],
                y_mm=home["y_mm"],
                z_mm=home["z_mm"],
                wpr_deg=hold,
                speed=min(self.speed, 20.0),
            )

    def move_to_pose(self, pose_table: tuple[float, float, float, float]) -> None:
        x_t, y_t, z_t, yaw_t = (float(v) for v in pose_table)
        cmd_x, cmd_y = self._xy_mm(x_t, y_t)
        cmd_z = self._z_mm(z_t)
        if not self._xy_reachable(cmd_x, cmd_y):
            box = self.workspace
            raise RuntimeError(
                f"refuse XY ({cmd_x:.1f}, {cmd_y:.1f}) outside calib envelope {box}"
            )
        tcp = _read_tcp(self.robot)
        face_wpr = self._facing_wpr(yaw_t, tcp["r_deg"])
        hold = (tcp["w_deg"], tcp["p_deg"], tcp["r_deg"])
        travel_z = max(float(tcp["z_mm"]), cmd_z)
        if abs(tcp["z_mm"] - travel_z) > 1.0:
            move_fanuc_xyz(
                self.robot,
                x_mm=tcp["x_mm"],
                y_mm=tcp["y_mm"],
                z_mm=travel_z,
                wpr_deg=hold,
                speed=self.speed,
            )
            tcp = _read_tcp(self.robot)
            hold = (tcp["w_deg"], tcp["p_deg"], tcp["r_deg"])
        # Long XY keeps the current wrist. New R rides the Z descend that
        # already follows, so we do not add a slow extra yaw hop.
        if math.hypot(cmd_x - tcp["x_mm"], cmd_y - tcp["y_mm"]) > 1.0:
            delta_r = _wrap_signed_deg(face_wpr[2] - hold[2])
            if abs(delta_r) > 2.0:
                print(
                    f"[yaw] hold R={hold[2]:.1f}° on XY; ΔR={delta_r:.1f}° on Z/settle",
                    flush=True,
                )
            _travel_xy(
                self.robot,
                x0=tcp["x_mm"],
                y0=tcp["y_mm"],
                x1=cmd_x,
                y1=cmd_y,
                z_mm=travel_z,
                wpr_deg=hold,
                speed=self.speed,
                reach=self.reach,
            )
            tcp = _read_tcp(self.robot)
        if abs(travel_z - cmd_z) > 1.0:
            move_fanuc_xyz(
                self.robot,
                x_mm=cmd_x,
                y_mm=cmd_y,
                z_mm=cmd_z,
                wpr_deg=face_wpr,
                speed=min(self.speed, 20.0),
            )
        elif abs(_wrap_signed_deg(face_wpr[2] - tcp["r_deg"])) > 2.0:
            move_fanuc_xyz(
                self.robot,
                x_mm=cmd_x,
                y_mm=cmd_y,
                z_mm=cmd_z,
                wpr_deg=face_wpr,
                speed=self.yaw_speed,
            )

    def gripper(self, cmd: str) -> None:
        close = str(cmd).lower() in {"close", "1", "closed"}
        tcp = _read_tcp(self.robot)
        wpr = (tcp["w_deg"], tcp["p_deg"], tcp["r_deg"])
        move_fanuc_xyz(
            self.robot,
            x_mm=tcp["x_mm"],
            y_mm=tcp["y_mm"],
            z_mm=tcp["z_mm"],
            wpr_deg=wpr,
            speed=20.0,
            gripper=1.0 if close else 0.0,
        )
        self._gripper_closed = close

    def read_gripper_width(self) -> float:
        return 0.03 if self._gripper_closed else 0.08
