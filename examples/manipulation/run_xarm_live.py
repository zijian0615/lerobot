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

"""
Live perceive → plan → solve → execute on xArm + /dev/video2.

Dry-run (default, no motion)::

    export GOOGLE_API_KEY=...
    cd examples
    python -m manipulation.run_xarm_live \\
        --instruction "move the grey stuffed animal to free space"

Execute motions (requires --execute)::

    python -m manipulation.run_xarm_live \\
        --instruction "move the grey stuffed animal to free space" \\
        --execute

Dual-arm collaboration (plan with both arms, two executors)::

    python -m manipulation.run_xarm_live --collab --execute --skip-place-verify \\
        --instruction "put the black screw in the container"
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import cv2  # noqa: E402

from manipulation.coordination import (  # noqa: E402
    OverlapGuard,
    exclusive_retract_xy,
    overlap_from_workspaces,
)
from manipulation.executor import ArmExecutor, ExecutorConfig  # noqa: E402
from manipulation.leaphand import LeapHandEE  # noqa: E402
from manipulation.orchestrator import run_manipulation  # noqa: E402
from manipulation.solver import SolverConfig  # noqa: E402
from planner import Planner  # noqa: E402
from tabletop_perception.perception import Perception  # noqa: E402
from tabletop_perception.run_xarm_live import (  # noqa: E402
    DEFAULT_CALIB,
    _capture_from_camera,
    _connect_xarm,
    _draw_image_overlay,
    _execution_cfg_for_arm,
    _load_calib,
    _table_xy_affine_from_calib,
    _load_image,
    _polygon_from_xy,
    _serialize_geometric,
    _workspace_from_spec,
)
from tabletop_perception.visualize import visualize_table_plane  # noqa: E402
from tabletop_perception.vlm import parse_vlm_detections  # noqa: E402

PERCEPTION_RUNS = Path(__file__).resolve().parents[1] / "tabletop_perception" / "runs"

logger = logging.getLogger(__name__)


def _arm_table_xy_affine_from_exe(
    exe_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Optional per-arm residual: true_table = A @ perception_table + b."""
    cfg = exe_cfg.get("table_xy_affine")
    if not cfg:
        return None
    a = np.asarray(cfg["A"], dtype=float)
    b = np.asarray(cfg["b"], dtype=float)
    if a.shape != (2, 2) or b.shape != (2,):
        raise ValueError("execution table_xy_affine needs A 2x2 and b length-2")
    return a, b


def _resolve_arm_names(args: argparse.Namespace, calib: dict[str, Any]) -> list[str]:
    """Single arm (default) or collaborative multi-arm list."""
    if bool(getattr(args, "collab", False)):
        return ["xarm", "xarm2"]
    raw = getattr(args, "arms", None)
    if raw:
        names = [a.strip() for a in str(raw).split(",") if a.strip()]
        if not names:
            raise ValueError("--arms is empty")
        return names
    legacy = dict(calib.get("execution") or {})
    return [str(args.arm or legacy.get("arm_name", "xarm"))]


def _ee_name_for_arm(calib: dict[str, Any], arm_name: str, exe_cfg: dict[str, Any]) -> str:
    return str(
        exe_cfg.get("ee")
        or dict(calib.get("robots") or {}).get(arm_name, {}).get("ee")
        or "parallel_gripper"
    )


def _build_backend(
    robot: Any,
    *,
    calib: dict[str, Any],
    arm_name: str,
) -> XArmMotionBackend:
    exe_cfg = _execution_cfg_for_arm(calib, arm_name)
    base_xy = tuple(float(v) for v in exe_cfg.get("table_base_xy_m", [-0.45, -0.35]))
    table_z = float(exe_cfg.get("table_z_base_m", 0.0))
    topdown = tuple(float(v) for v in exe_cfg.get("topdown_rpy_rad", [math.pi, 0.0, 0.0]))
    flip_x = bool(exe_cfg.get("flip_x", False))
    flip_y = bool(exe_cfg.get("flip_y", True))
    xy_off = exe_cfg.get("xy_offset_base_mm", [0.0, 0.0])
    arm_affine = _arm_table_xy_affine_from_exe(exe_cfg)
    if arm_affine is not None:
        logger.info("Using per-arm table_xy_affine for %s", arm_name)
    logger.info(
        "Backend arm=%s flip_x=%s flip_y=%s base_xy=%s",
        arm_name,
        flip_x,
        flip_y,
        base_xy,
    )
    return XArmMotionBackend(
        robot,
        base_xy_table=base_xy,  # type: ignore[arg-type]
        table_z_base_m=table_z,
        topdown_rpy=topdown,  # type: ignore[arg-type]
        speed=float(exe_cfg.get("move_speed_mm_s", 80.0)),
        acc=float(exe_cfg.get("move_acc_mm_s2", 500.0)),
        flip_x=flip_x,
        flip_y=flip_y,
        xy_offset_base_mm=(float(xy_off[0]), float(xy_off[1])),
        table_xy_affine=arm_affine,
    )


def _apply_arm_table_affine(
    pose_table: tuple[float, float, float, float] | list[float],
    affine: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[float, float, float, float]:
    x_t, y_t, z_t, yaw_t = (float(v) for v in pose_table)
    if affine is None:
        return (x_t, y_t, z_t, yaw_t)
    a, b = affine
    xy = a @ np.array([x_t, y_t], dtype=float) + b
    return (float(xy[0]), float(xy[1]), z_t, yaw_t)


def _table_pose_to_base_mm(
    pose_table: tuple[float, float, float, float] | list[float],
    *,
    base_xy_table: tuple[float, float],
    table_z_base_m: float,
    topdown_rpy: tuple[float, float, float],
    flip_x: bool = False,
    flip_y: bool = True,
    use_object_yaw: bool = False,
    table_xy_affine: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[float]:
    """
    Table-frame grasp pose (m, yaw) → xArm base pose [x,y,z,roll,pitch,yaw]
    with mm + radians.

    ``base_xy_table`` is the arm base expressed in the table frame.
    ``flip_y`` matches the usual xArm mount (base +y opposite table +y).
    ``flip_x`` is for the mirrored Robot2 mount (base +X along −table-x).
    ``table_xy_affine`` is an optional per-arm residual on table XY.
    """
    x_t, y_t, z_t, yaw_t = _apply_arm_table_affine(pose_table, table_xy_affine)
    x_b = (x_t - base_xy_table[0]) * 1000.0
    y_b = (y_t - base_xy_table[1]) * 1000.0
    if flip_x:
        x_b = -x_b
    if flip_y:
        y_b = -y_b
    z_b = (table_z_base_m + z_t) * 1000.0
    roll, pitch, yaw0 = topdown_rpy
    yaw = yaw0 + (yaw_t if use_object_yaw else 0.0)
    return [x_b, y_b, z_b, roll, pitch, yaw]


class XArmMotionBackend:
    """Blocking Cartesian moves + gripper via xArm SDK (mode 0)."""

    def __init__(
        self,
        robot: Any,
        *,
        base_xy_table: tuple[float, float],
        table_z_base_m: float,
        topdown_rpy: tuple[float, float, float],
        speed: float,
        acc: float,
        flip_x: bool = False,
        flip_y: bool = True,
        xy_offset_base_mm: tuple[float, float] = (0.0, 0.0),
        table_xy_affine: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.robot = robot
        self.arm = robot.real_arm
        self.base_xy_table = base_xy_table
        self.table_z_base_m = table_z_base_m
        self.topdown_rpy = topdown_rpy
        self.speed = speed
        self.acc = acc
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.xy_offset_base_mm = (float(xy_offset_base_mm[0]), float(xy_offset_base_mm[1]))
        self.table_xy_affine = table_xy_affine
        self._ensure_mode0()

    def _ensure_mode0(self) -> None:
        # Clear residual faults from a previous failed move.
        if getattr(self.arm, "error_code", 0) != 0:
            self.arm.clean_error()
            self.arm.clean_warn()
            time.sleep(0.1)
        self.arm.motion_enable(True)
        if self.arm.mode != 0:
            self.arm.set_mode(0)
        self.arm.set_state(0)
        time.sleep(0.1)

    def move_to_pose(self, pose_table: tuple[float, float, float, float]) -> None:
        self._ensure_mode0()
        xyzrpy = _table_pose_to_base_mm(
            pose_table,
            base_xy_table=self.base_xy_table,
            table_z_base_m=self.table_z_base_m,
            topdown_rpy=self.topdown_rpy,
            flip_x=self.flip_x,
            flip_y=self.flip_y,
            use_object_yaw=False,
            table_xy_affine=self.table_xy_affine,
        )
        xyzrpy[0] += self.xy_offset_base_mm[0]
        xyzrpy[1] += self.xy_offset_base_mm[1]
        logger.info(
            "move_to_pose table=%s → base_mm=%s (xy_offset=%s flip_x=%s flip_y=%s)",
            [round(v, 3) for v in pose_table],
            [round(v, 1) if i < 3 else round(v, 3) for i, v in enumerate(xyzrpy)],
            [round(v, 1) for v in self.xy_offset_base_mm],
            self.flip_x,
            self.flip_y,
        )
        code = self.arm.set_position(
            *xyzrpy,
            speed=self.speed,
            mvacc=self.acc,
            wait=True,
            is_radian=True,
        )
        if code != 0:
            raise RuntimeError(f"set_position failed code={code} pose={xyzrpy}")

    def get_current_pose_table(self) -> tuple[float, float, float, float]:
        code, pose = self.arm.get_position(is_radian=True)
        if code != 0 or pose is None:
            raise RuntimeError(f"get_position failed code={code}")
        x_mm, y_mm, z_mm, _r, _p, yaw = pose[:6]
        y_rel = -y_mm / 1000.0 if self.flip_y else y_mm / 1000.0
        x_t = x_mm / 1000.0 + self.base_xy_table[0]
        y_t = y_rel + self.base_xy_table[1]
        z_t = z_mm / 1000.0 - self.table_z_base_m
        return (x_t, y_t, z_t, float(yaw - self.topdown_rpy[2]))

    def gripper(self, cmd: str) -> None:
        # j7 norm: 0 open, 1 closed — reuse robot helper.
        norm = 0.0 if cmd == "open" else 1.0
        self.robot._send_gripper(norm)
        time.sleep(0.6)

    def read_gripper_width(self) -> float:
        """Map gripper norm to a pseudo width in metres for executor checks."""
        obs = self.robot.get_observation()
        # j7: 0 open → width~0.08, 1 closed → width~0.01; partial = success band.
        g = float(obs.get("j7", self.robot._latest_gripper_norm))
        # After close, SDK often reports binary 0/1; use latest commanded norm.
        g = float(self.robot._latest_gripper_norm)
        return 0.08 - 0.07 * g


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live xArm tabletop manipulation stack")
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--camera", type=str, default=None)
    p.add_argument("--image", type=Path, default=None, help="Still image instead of live camera")
    p.add_argument("--robot-ip", type=str, default=None)
    p.add_argument(
        "--arm",
        type=str,
        default=None,
        help="Arm name in calib (default: execution.arm_name / xarm). e.g. xarm2",
    )
    p.add_argument(
        "--arms",
        type=str,
        default=None,
        help="Comma-separated arms for multi-arm planning, e.g. xarm,xarm2",
    )
    p.add_argument(
        "--collab",
        action="store_true",
        help="Shortcut: plan/execute with both xarm and xarm2",
    )
    p.add_argument("--instruction", type=str, required=True)
    p.add_argument("--execute", action="store_true", help="Actually move the arm (default: dry-run)")
    p.add_argument("--no-robot", action="store_true", help="Skip robot connect (plan/solve only)")
    p.add_argument("--lookahead", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-place-verify", action="store_true", help="Skip VLM re-perceive after Place")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_argparser().parse_args(argv)
    calib = _load_calib(args.calib)
    arm_names = _resolve_arm_names(args, calib)
    logger.info("Active arms=%s collab=%s", arm_names, bool(args.collab))
    grasp_height = float(calib["grasp_height_m"])
    grasp_height_offsets = {
        str(k): float(v)
        for k, v in dict(calib.get("grasp_height_offset_m") or {}).items()
        if not str(k).startswith("_")
    }
    place_height_offsets = {
        str(k): float(v)
        for k, v in dict(calib.get("place_height_offset_m") or {}).items()
        if not str(k).startswith("_")
    }
    # EE-specific extras (e.g. leaphand bear=-0.05) → per-arm map for the solver.
    ee_height_offs = {
        str(ee).lower(): {
            str(k): float(v)
            for k, v in dict(offs or {}).items()
            if not str(k).startswith("_")
        }
        for ee, offs in dict(calib.get("grasp_height_offset_by_ee") or {}).items()
        if not str(ee).startswith("_")
    }
    grasp_height_offsets_by_arm: dict[str, dict[str, float]] = {}
    place_height_offsets_by_arm: dict[str, dict[str, float]] = {}
    for name in arm_names:
        exe_i = _execution_cfg_for_arm(calib, name)
        ee = _ee_name_for_arm(calib, name, exe_i).lower()
        merged: dict[str, float] = {}
        if ee in {"leaphand", "leap_hand", "leap"}:
            merged.update(ee_height_offs.get("leaphand") or {})
        elif ee in ee_height_offs:
            merged.update(ee_height_offs[ee])
        for k, v in dict(exe_i.get("grasp_height_offset_m") or {}).items():
            if not str(k).startswith("_"):
                merged[str(k)] = float(v)
        if merged:
            grasp_height_offsets_by_arm[name] = merged
        place_merged = {
            str(k): float(v)
            for k, v in dict(exe_i.get("place_height_offset_m") or {}).items()
            if not str(k).startswith("_")
        }
        if place_merged:
            place_height_offsets_by_arm[name] = place_merged

    cam_cfg = calib["camera"]
    camera_path = args.camera or cam_cfg["index_or_path"]
    k = np.asarray(calib["K"], dtype=float)
    t_cam_table = np.asarray(calib["T_cam_table"], dtype=float)
    table_polygon = _polygon_from_xy(calib["table_polygon_xy"])
    arm_workspaces = {
        name: _workspace_from_spec(spec) for name, spec in calib["arm_workspaces_xy"].items()
    }

    robots: dict[str, Any] = {}
    leaps: dict[str, LeapHandEE] = {}
    if not args.no_robot:
        for i, name in enumerate(arm_names):
            # --robot-ip only overrides the first arm (legacy single-arm CLI).
            ip = args.robot_ip if (args.robot_ip and i == 0 and len(arm_names) == 1) else None
            logger.info("Connecting xArm arm=%s …", name)
            robots[name] = _connect_xarm(calib, ip, arm_name=name)
            logger.info("xArm connected (%s).", name)
            exe_cfg = _execution_cfg_for_arm(calib, name)
            ee_name = _ee_name_for_arm(calib, name, exe_cfg)
            if ee_name.lower() in {"leaphand", "leap_hand", "leap"}:
                logger.info("Connecting LeapHand EE for %s …", name)
                leap = LeapHandEE.from_calib(calib, name)
                leap.connect()
                leap.open()
                leaps[name] = leap
                logger.info(
                    "LeapHand ready arm=%s port=%s curr_lim=%s",
                    name,
                    leap.port,
                    leap.curr_lim,
                )
    else:
        logger.info("Skipping robot (--no-robot).")

    table_xy_affine = _table_xy_affine_from_calib(calib)
    if table_xy_affine is not None:
        logger.info("Using table_xy_affine correction from calib.")
    if grasp_height_offsets:
        logger.info("Per-object grasp height offsets (global): %s", grasp_height_offsets)
    if grasp_height_offsets_by_arm:
        logger.info(
            "Per-arm grasp height offsets (EE-specific): %s",
            grasp_height_offsets_by_arm,
        )
    if place_height_offsets:
        logger.info("Per-destination place height offsets: %s", place_height_offsets)
    if place_height_offsets_by_arm:
        logger.info("Per-arm place height offsets: %s", place_height_offsets_by_arm)
    perception = Perception(
        footprint_buffer_m=float(calib.get("footprint_buffer_m", 0.02)),
        table_xy_affine=table_xy_affine,
        grasp_height_offsets_m=grasp_height_offsets,
    )
    planner = Planner(max_attempts=2)

    # Cache last geometric view for place-verify shortcut.
    last_views: dict[str, Any] = {"symbolic": None, "geometric": None}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / stamp
    perception_dir = PERCEPTION_RUNS / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    perception_dir.mkdir(parents=True, exist_ok=True)
    perceive_count = {"n": 0}

    def capture_image() -> np.ndarray:
        if args.image is not None:
            return _load_image(args.image)
        return _capture_from_camera(
            camera_path,
            width=int(cam_cfg["width"]),
            height=int(cam_cfg["height"]),
            fps=int(cam_cfg["fps"]),
        )

    def _dump_perception_logs(
        image: np.ndarray,
        symbolic: dict[str, Any],
        geometric: dict[str, Any],
        detections: list[dict[str, Any]],
    ) -> Path:
        perceive_count["n"] += 1
        tag = f"perceive_{perceive_count['n']:02d}"
        run_dir = perception_dir / tag
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "instruction.txt").write_text(args.instruction + "\n")
        (run_dir / "symbolic_view.json").write_text(json.dumps(symbolic, indent=2))
        (run_dir / "geometric_view.json").write_text(
            json.dumps(_serialize_geometric(geometric), indent=2)
        )
        cv2.imwrite(str(run_dir / "capture_rgb.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        overlay = _draw_image_overlay(image, geometric, detections)
        cv2.imwrite(str(run_dir / "capture_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        visualize_table_plane(
            geometric,
            show=False,
            save_path=run_dir / "table_plane.png",
            title=f"perception {tag}",
        )
        # Mirror latest perception into manipulation out_dir for convenience.
        for name in (
            "instruction.txt",
            "symbolic_view.json",
            "geometric_view.json",
            "capture_rgb.png",
            "capture_overlay.png",
            "table_plane.png",
        ):
            src = run_dir / name
            if src.exists():
                (out_dir / name).write_bytes(src.read_bytes())
        logger.info("Perception logs → %s (and %s)", run_dir, out_dir)
        return run_dir

    def perceive_pair():
        image = capture_image()
        logger.info("Perceiving image shape=%s", image.shape)
        # One VLM call, reuse for Perception + overlay boxes.
        raw = perception.vlm_caller(image, args.instruction, perception.prompt or "")
        detections = parse_vlm_detections(raw, image_hw=(image.shape[0], image.shape[1]))
        perception_once = Perception(
            footprint_buffer_m=float(calib.get("footprint_buffer_m", 0.02)),
            prompt=perception.prompt,
            vlm_caller=lambda _img, _ins, _p: raw,
            table_xy_affine=table_xy_affine,
            grasp_height_offsets_m=grasp_height_offsets,
        )
        symbolic, geometric = perception_once(
            image=image,
            K=k,
            T_cam_table=t_cam_table,
            grasp_height=grasp_height,
            instruction=args.instruction,
            arm_workspaces=arm_workspaces,
            table_polygon=table_polygon,
        )
        ov_cfg = dict(calib.get("overlap_xy") or {})
        ov_arms = ov_cfg.get("arms")
        ov = overlap_from_workspaces(
            geometric["workspaces"],
            arms=ov_arms,
            table_polygon=table_polygon if ov_cfg.get("clip_to_table", True) else None,
        )
        geometric = dict(geometric)
        geometric["overlap"] = ov
        last_views["symbolic"] = symbolic
        last_views["geometric"] = geometric
        _dump_perception_logs(image, symbolic, geometric, detections)
        logger.info("symbolic_view=%s", json.dumps(symbolic, indent=2))
        return symbolic, geometric

    def plan_fn(symbolic_view, instruction, arms):
        logger.info("Planning for arms=%s instruction=%r", list(arms), instruction)
        plan = planner(symbolic_view, instruction, arms)
        logger.info("plan=%s", json.dumps(plan, indent=2))
        (out_dir / "plan.json").write_text(json.dumps(plan, indent=2))
        return plan

    try:
        # Solver margin: use first arm's config (same default for both today).
        primary_exe = _execution_cfg_for_arm(calib, arm_names[0])
        solver_margin = float(primary_exe.get("solver_margin_m", 0.03))

        if not args.execute:
            # Dry-run: perceive → plan → solve only.
            symbolic, geometric = perceive_pair()
            plan = plan_fn(symbolic, args.instruction, arm_names)
            from manipulation.solver import solve

            bound = solve(
                plan,
                geometric,
                SolverConfig(
                    grasp_height=grasp_height,
                    margin=solver_margin,
                    place_height_offsets_m=place_height_offsets,
                    place_height_offsets_by_arm=place_height_offsets_by_arm,
                    grasp_height_offsets_by_arm=grasp_height_offsets_by_arm,
                ),
                lookahead=args.lookahead,
            )
            (out_dir / "symbolic_view.json").write_text(json.dumps(symbolic, indent=2))
            (out_dir / "plan.json").write_text(json.dumps(plan, indent=2))
            (out_dir / "bound.json").write_text(json.dumps(bound, indent=2))
            print(json.dumps({"plan": plan, "bound": bound}, indent=2))
            print(f"\nDry-run only. Outputs → {out_dir}")
            print("Re-run with --execute to move the arm(s).")
            return 0

        if not robots:
            raise RuntimeError("--execute requires a connected robot (omit --no-robot)")

        # Last commanded Place pose — used when --skip-place-verify trusts the command.
        last_place_cmd: dict[str, Any] = {}

        def perceive_for_place():
            if args.skip_place_verify:
                geo = dict(last_views["geometric"] or {"objects": []})
                obj = last_place_cmd.get("object")
                pose = last_place_cmd.get("pose")
                if obj and pose:
                    objs = []
                    for item in geo.get("objects", []):
                        row = dict(item)
                        if str(row.get("name")) == obj:
                            row["xy"] = (float(pose[0]), float(pose[1]))
                        objs.append(row)
                    geo["objects"] = objs
                return geo
            _sym, geo = perceive_pair()
            return geo

        executors: dict[str, ArmExecutor] = {}
        backends: dict[str, XArmMotionBackend] = {}
        for name in arm_names:
            robot = robots[name]
            exe_cfg = _execution_cfg_for_arm(calib, name)
            backend = _build_backend(robot, calib=calib, arm_name=name)
            backends[name] = backend
            leap = leaps.get(name)
            if leap is not None:
                gripper_fn = leap.gripper
                width_fn = leap.read_gripper_width
            else:

                def _make_width_fn(r: Any = robot) -> Any:
                    def width_fn() -> float:
                        g = float(r._latest_gripper_norm)
                        if g >= 0.5:
                            return 0.03
                        return 0.08

                    return width_fn

                gripper_fn = backend.gripper
                width_fn = _make_width_fn()

            executor = ArmExecutor(
                name,
                move_to_pose=backend.move_to_pose,
                gripper=gripper_fn,
                read_gripper_width=width_fn,
                perceive=perceive_for_place,
                get_current_pose=backend.get_current_pose_table,
                config=ExecutorConfig(
                    approach_offset=float(exe_cfg.get("approach_offset_m", 0.08)),
                    lift_offset=float(exe_cfg.get("lift_offset_m", 0.08)),
                    gripper_open_width=0.085,
                    gripper_closed_width=0.005,
                    place_xy_tol=0.03,
                ),
            )
            raw_execute = executor.execute

            def _make_tracking(raw=raw_execute):
                def _execute_tracking(step: dict[str, Any]):
                    if str(step.get("primitive")) == "Place":
                        params = step.get("params") or {}
                        last_place_cmd["object"] = params.get("object")
                        last_place_cmd["pose"] = params.get("pose")
                    return raw(step)

                return _execute_tracking

            executor.execute = _make_tracking()  # type: ignore[method-assign]
            executors[name] = executor

        # Overlap = ∩ workspaces (clipped to table). Mutex + retract after in-zone steps.
        ov_cfg = dict(calib.get("overlap_xy") or {})
        ov_poly = overlap_from_workspaces(
            arm_workspaces,
            arms=ov_cfg.get("arms"),
            table_polygon=table_polygon if ov_cfg.get("clip_to_table", True) else None,
        )
        overlap_guard = OverlapGuard(
            ov_poly,
            retract={},
            enabled=bool(ov_cfg) and not ov_poly.is_empty,
        )
        for name in arm_names:
            exe_cfg = _execution_cfg_for_arm(calib, name)
            retract_z = grasp_height + float(exe_cfg.get("lift_offset_m", 0.08)) + 0.05
            ws_poly = arm_workspaces[name]
            backend = backends[name]

            def _make_retract(
                arm: str = name,
                ws=ws_poly,
                z: float = retract_z,
                be=backend,
            ):
                def _retract_arm() -> None:
                    xy = exclusive_retract_xy(ws, overlap_guard.overlap)
                    if xy is None:
                        logger.warning("retract skipped: no exclusive point for %s", arm)
                        return
                    logger.info(
                        "retract %s → exclusive xy=(%.3f, %.3f) z=%.3f",
                        arm,
                        xy[0],
                        xy[1],
                        z,
                    )
                    be.move_to_pose((xy[0], xy[1], z, 0.0))

                return _retract_arm

            overlap_guard.retract[name] = _make_retract()

        if overlap_guard.enabled:
            logger.info(
                "Overlap mutex enabled area=%.3f m^2 centroid=(%.3f, %.3f)",
                float(ov_poly.area),
                float(ov_poly.centroid.x),
                float(ov_poly.centroid.y),
            )

        result = run_manipulation(
            args.instruction,
            perceive=perceive_pair,
            plan_fn=plan_fn,
            executors=executors,
            solver_config=SolverConfig(
                grasp_height=grasp_height,
                margin=solver_margin,
                place_height_offsets_m=place_height_offsets,
                place_height_offsets_by_arm=place_height_offsets_by_arm,
                grasp_height_offsets_by_arm=grasp_height_offsets_by_arm,
            ),
            arms=arm_names,
            lookahead=args.lookahead,
            overlap_guard=overlap_guard,
        )

        payload = {
            "status": result.status,
            "reason": result.reason,
            "plan": result.plan,
            "bound": result.bound,
            "results": result.results,
            "symbolic_view": result.symbolic_view,
            "arms": arm_names,
        }
        (out_dir / "result.json").write_text(json.dumps(payload, indent=2, default=str))
        print(json.dumps(payload, indent=2, default=str))
        print(f"\nOutputs → {out_dir}")
        return 0 if result.status == "success" else 1
    finally:
        for name, leap in list(leaps.items()):
            try:
                leap.disconnect()
            except Exception as exc:  # noqa: BLE001
                logger.warning("LeapHand disconnect %s: %s", name, exc)
        for name, robot in list(robots.items()):
            try:
                robot.disconnect()
            except Exception as exc:  # noqa: BLE001
                logger.warning("robot disconnect %s: %s", name, exc)


if __name__ == "__main__":
    raise SystemExit(main())
