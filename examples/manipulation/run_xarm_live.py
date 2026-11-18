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

GPT-6 Astra instead of Gemini (needs OPENAI_API_KEY)::

    export OPENAI_API_KEY=...
    python -m manipulation.run_xarm_live --model gpt-6 --thinking-budget 0 \\
        --instruction "place all screws into the container"

Execute motions (requires --execute)::

    python -m manipulation.run_xarm_live \\
        --instruction "move the grey stuffed animal to free space" \\
        --execute

Dual-arm collaboration (plan with both arms, two executors)::

    python -m manipulation.run_xarm_live --collab --execute --skip-place-verify \\
        --instruction "put the black screw in the container"

Keep running the same instruction for 10 minutes (robot stays connected)::

    python -m manipulation.run_xarm_live --collab --execute --skip-place-verify \\
        --loop-minutes 10 \\
        --instruction "put all the screws into the container" \\
        --camera /dev/video0

Interactive multi-turn (games, take-turns, until-done)::

    python -m manipulation.run_xarm_live --execute --skip-place-verify \\
        --instruction "put all the screws into the container" \\
        --camera /dev/video0

Live UI (top + xarm1 wrist + timer) opens at http://127.0.0.1:8765/ ::

    # default on; disable with --no-ui
    # wrist camera defaults to /dev/video2 (Innomaker)

Serve mode: connect once, then type instructions into the live UI page (no restart)::

    python -m manipulation.run_xarm_live --serve --execute --skip-place-verify
    # open http://127.0.0.1:8765/ → type an instruction → RUN; repeat as often as needed.
    # --instruction is optional here: if given, it runs immediately.

Voice (Boson Higgs Realtime) on top of serve mode — talk while the arm works::

    export BOSON_API_KEY=...
    python -m manipulation.run_xarm_live --serve --voice --execute --skip-place-verify
    # Mic/speaker device (e.g. a laptop, via ssh -L 8765:localhost:8765): open
    #   http://localhost:8765/voice   (audio only, no video) → START VOICE.
    # Robot machine: open http://127.0.0.1:8765/  (camera + task state + mirrored voice log).
    # The model can ask for a task / status / go-home; any motion needs Enter (or the CONFIRM
    # button) on either page, Esc cancels. Use headphones; the mic needs localhost/127.0.0.1.

Fixed plan (no planner Gemini; all steps on xarm / arm1)::

    python -m manipulation.run_xarm_live --execute --skip-place-verify --fixed-plan \\
        --loop-minutes 10 \\
        --instruction "put all the screws into the container" \\
        --camera /dev/video0
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections.abc import Callable
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
from manipulation.fixed_plan import plan_screws_to_container_xarm  # noqa: E402
from manipulation.leaphand import LeapHandEE  # noqa: E402
from manipulation.interactive import run_interactive  # noqa: E402
from manipulation.live_ui import LiveUI  # noqa: E402
from manipulation.orchestrator import ManipulationLoop, run_manipulation  # noqa: E402
from manipulation.solver import SolverConfig  # noqa: E402
from planner import Planner  # noqa: E402
from openai_backend import resolve_base_model  # noqa: E402
from tabletop_perception.perception import Perception, object_top_z_from_calib  # noqa: E402
from tabletop_perception.run_xarm_live import (  # noqa: E402
    DEFAULT_CALIB,
    _capture_from_camera,
    undistort_with_calib,
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


def _check_leap_ready(calib: dict[str, Any], arm_names: list[str]) -> None:
    """Fail before any arm moves if a LEAP-hand arm cannot possibly connect its hand."""
    problems = []
    for name in arm_names:
        ee = _ee_name_for_arm(calib, name, _execution_cfg_for_arm(calib, name)).lower()
        if ee not in {"leaphand", "leap_hand", "leap"}:
            continue
        leap = LeapHandEE.from_calib(calib, name)
        if not (Path(leap.api_dir) / "leap_hand_utils").is_dir():
            problems.append(
                f"{name}: LEAP Hand API (leap_hand_utils) not found under {leap.api_dir}; "
                "clone LEAP_Hand_API and set leaphand.api_dir in the calib"
            )
        import importlib.util

        if importlib.util.find_spec("dynamixel_sdk") is None:
            problems.append(f"{name}: Python package dynamixel_sdk missing (uv pip install dynamixel_sdk)")
        ports = [str(leap.port), "/dev/ttyUSB0", "/dev/ttyUSB1"]
        found = [p for p in ports if os.path.exists(p)]
        if found and not any(os.access(p, os.R_OK | os.W_OK) for p in found):
            problems.append(
                f"{name}: no permission on {found[0]}; you are in group dialout only after a new "
                'login, or start with: sg dialout -c "uv run python -m manipulation.run_xarm_live ..."'
            )
        if not found:
            problems.append(
                f"{name}: LEAP Hand USB serial port not found ({leap.port}); plug the hand's "
                "USB (FTDI) cable into this computer"
            )
    if problems:
        raise RuntimeError("LEAP Hand not ready:\n  " + "\n  ".join(problems))


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
        "Backend arm=%s flip_x=%s flip_y=%s base_xy=%s use_object_yaw=%s yaw_offset=%.3f",
        arm_name,
        flip_x,
        flip_y,
        base_xy,
        bool(exe_cfg.get("use_object_yaw", True)),
        float(exe_cfg.get("grasp_yaw_offset_rad", math.pi / 2.0)),
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
        use_object_yaw=bool(exe_cfg.get("use_object_yaw", True)),
        grasp_yaw_offset_rad=float(exe_cfg.get("grasp_yaw_offset_rad", math.pi / 2.0)),
        ee_center_offset_base_mm=tuple(float(v) for v in exe_cfg.get("ee_center_offset_base_mm", [0.0, 0.0])),  # type: ignore[arg-type]
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
    # Rotate yaw by the linear part of the affine.
    direction = a @ np.array([math.cos(yaw_t), math.sin(yaw_t)], dtype=float)
    yaw_out = float(math.atan2(direction[1], direction[0]))
    return (float(xy[0]), float(xy[1]), z_t, yaw_out)


def _map_table_yaw_to_base(yaw_t: float, *, flip_x: bool, flip_y: bool) -> float:
    """Map a table-plane yaw through the same XY flips used for position."""
    c = math.cos(yaw_t)
    s = math.sin(yaw_t)
    if flip_x:
        c = -c
    if flip_y:
        s = -s
    return float(math.atan2(s, c))


def _table_pose_to_base_mm(
    pose_table: tuple[float, float, float, float] | list[float],
    *,
    base_xy_table: tuple[float, float],
    table_z_base_m: float,
    topdown_rpy: tuple[float, float, float],
    flip_x: bool = False,
    flip_y: bool = True,
    use_object_yaw: bool = False,
    grasp_yaw_offset_rad: float = 0.0,
    table_xy_affine: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[float]:
    """
    Table-frame grasp pose (m, yaw) → xArm base pose [x,y,z,roll,pitch,yaw]
    with mm + radians.

    ``base_xy_table`` is the arm base expressed in the table frame.
    ``flip_y`` matches the usual xArm mount (base +y opposite table +y).
    ``flip_x`` is for the mirrored Robot2 mount (base +X along −table-x).
    ``table_xy_affine`` is an optional per-arm residual on table XY.
    ``grasp_yaw_offset_rad`` rotates jaws relative to object long axis
    (π/2 ≈ fingers close across the short axis of a screw).
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
    if use_object_yaw:
        yaw_mapped = _map_table_yaw_to_base(yaw_t, flip_x=flip_x, flip_y=flip_y)
        yaw = yaw0 + yaw_mapped + float(grasp_yaw_offset_rad)
    else:
        yaw = yaw0
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
        use_object_yaw: bool = True,
        grasp_yaw_offset_rad: float = math.pi / 2.0,
        ee_center_offset_base_mm: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.robot = robot
        # Grasp centre vs flange axis, base mm as seen at yaw 0 (measure_hand_center.py).
        self.ee_center_offset_base_mm = (
            float(ee_center_offset_base_mm[0]),
            float(ee_center_offset_base_mm[1]),
        )
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
        self.use_object_yaw = bool(use_object_yaw)
        self.grasp_yaw_offset_rad = float(grasp_yaw_offset_rad)
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
            use_object_yaw=self.use_object_yaw,
            grasp_yaw_offset_rad=self.grasp_yaw_offset_rad,
            table_xy_affine=self.table_xy_affine,
        )
        xyzrpy[0] += self.xy_offset_base_mm[0]
        xyzrpy[1] += self.xy_offset_base_mm[1]
        vx, vy = self.ee_center_offset_base_mm
        if vx or vy:
            # The XY calibration was taken at wrist yaw 0, so it already contains v once.
            # At yaw psi the grasp centre sits at P + R(psi) v: move the flange by
            # -(R(psi) - I) v to keep it where the yaw-0 calibration put it.
            c, s = math.cos(xyzrpy[5]), math.sin(xyzrpy[5])
            xyzrpy[0] -= (c * vx - s * vy) - vx
            xyzrpy[1] -= (s * vx + c * vy) - vy
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
    p.add_argument(
        "--wrist-camera",
        type=str,
        default=None,
        help="xarm1 wrist camera (default: calib wrist_camera or /dev/video2).",
    )
    p.add_argument(
        "--ui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the live dashboard (top + wrist + timer). --no-ui to disable.",
    )
    p.add_argument("--ui-port", type=int, default=8765)
    p.add_argument(
        "--voice",
        action="store_true",
        help="With --serve: voice control via Boson Higgs Realtime (needs BOSON_API_KEY). Open "
        "/voice on the device with the mic (audio only); the main page mirrors it. Motion still "
        "needs Enter/CONFIRM.",
    )
    p.add_argument("--voice-name", type=str, default="default", help="Higgs Realtime voice preset.")
    p.add_argument(
        "--post-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --serve --execute: after each task (arm homed) photograph the table and ask "
        "the VLM whether the instruction is really done; the page/voice report that verdict, "
        "not just 'moves finished'.",
    )
    p.add_argument("--voice-model", type=str, default="higgs-realtime")
    p.add_argument(
        "--ui-host",
        type=str,
        default=None,
        help="Bind address of the live UI (default 0.0.0.0, or 127.0.0.1 with --serve because "
        "the page can then start robot motion; use ssh -L to reach it remotely).",
    )
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
    p.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Task instruction (required unless --serve).",
    )
    p.add_argument(
        "--serve",
        action="store_true",
        help="Stay running after a task: the live UI gets an instruction box, and every "
        "instruction submitted there runs on the already-connected robot/cameras. "
        "Ctrl-C to quit.",
    )
    p.add_argument("--execute", action="store_true", help="Actually move the arm (default: dry-run)")
    p.add_argument("--no-robot", action="store_true", help="Skip robot connect (plan/solve only)")
    p.add_argument("--lookahead", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-place-verify", action="store_true", help="Skip VLM re-perceive after Place")
    p.add_argument(
        "--loop-minutes",
        type=float,
        default=0.0,
        help="With --execute: keep re-running the same instruction for this many minutes "
        "(robot stays connected). 0 = single shot.",
    )
    p.add_argument(
        "--loop-pause-s",
        type=float,
        default=2.0,
        help="Seconds to wait between loop cycles (default: 2).",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="Multi-turn: go home → photograph → plan VLM. Repeats after each move.",
    )
    p.add_argument(
        "--watch-s",
        type=float,
        default=2.0,
        help="With --interactive: seconds to wait on planner status=wait before "
        "the next home/photo/plan cycle (default: 2).",
    )
    p.add_argument(
        "--turn-loop",
        action="store_true",
        help="Alias for --interactive.",
    )
    p.add_argument(
        "--fixed-plan",
        action="store_true",
        help="Skip planner Gemini; build Grasp→Place on xarm (arm1) from perception names.",
    )
    p.add_argument(
        "--fixed-plan-arm",
        type=str,
        default="xarm",
        help="Arm used by --fixed-plan (default: xarm = arm1).",
    )
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    p.add_argument(
        "--model",
        "--vlm",
        dest="model",
        default="gemini",
        help="Base VLM for perceive + plan: gemini, gpt-6, or cosmos / cosmos3-nano.",
    )
    p.add_argument(
        "--thinking-budget",
        type=int,
        default=-1,
        help="Gemini thinking budget (0=off, -1=dynamic, or token cap). "
        "With --model gpt-6 this maps to reasoning effort low/medium/high.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = build_argparser()
    args = parser.parse_args(argv)
    if args.turn_loop:
        args.interactive = True
    if args.serve:
        if not args.ui:
            parser.error("--serve needs the live UI (drop --no-ui).")
        if args.interactive or args.loop_minutes > 0:
            parser.error("--serve cannot be combined with --interactive/--turn-loop/--loop-minutes.")
    if args.voice and not args.serve:
        parser.error("--voice requires --serve.")
    if not args.serve and not args.instruction:
        parser.error("--instruction is required unless --serve is given.")
    calib = _load_calib(args.calib)
    if args.fixed_plan:
        # Arm1-only fixed planner: ignore --collab / multi-arm selection.
        arm_names = [str(args.fixed_plan_arm)]
        logger.info("Fixed-plan mode: forcing arms=%s (planner VLM disabled)", arm_names)
    else:
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
    place_clearance = float(calib.get("place_clearance_m", 0.01))
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
    wrist_cfg = dict(calib.get("wrist_camera") or {})
    wrist_path = args.wrist_camera or wrist_cfg.get("index_or_path") or "/dev/video2"
    # Two threads cannot stream the same V4L2 node: it makes both stall ("select() timeout").
    # Compare the resolved devices (by-path/by-id symlinks and /dev/videoN can alias).
    if wrist_path and os.path.realpath(str(wrist_path)) == os.path.realpath(str(camera_path)):
        logger.warning(
            "wrist camera %s is the same device as the top camera %s → wrist disabled",
            wrist_path,
            camera_path,
        )
        wrist_path = None
    elif wrist_path and str(wrist_path).startswith("/") and not os.path.exists(str(wrist_path)):
        logger.warning("wrist camera %s not found → wrist disabled", wrist_path)
        wrist_path = None

    ui: LiveUI | None = None
    if args.ui:
        ui = LiveUI(
            instruction=args.instruction or "",
            top_path=str(camera_path),
            wrist_path=str(wrist_path) if wrist_path else None,
            width=int(cam_cfg.get("width", 640)),
            height=int(cam_cfg.get("height", 480)),
            fps=int(cam_cfg.get("fps", 30)),
            host=args.ui_host or ("127.0.0.1" if args.serve else "0.0.0.0"),
            port=int(args.ui_port),
        )
        if args.serve:
            ui.enable_serve()
        if args.voice:
            from manipulation.voice_ui import build_session

            ui.enable_voice(
                build_session(
                    model=args.voice_model,
                    voice=args.voice_name,
                    arms=[
                        (n, _ee_name_for_arm(calib, n, _execution_cfg_for_arm(calib, n)))
                        for n in arm_names
                    ],
                )
            )
            if not os.environ.get("BOSON_API_KEY"):
                logger.warning("--voice: BOSON_API_KEY is not set; START VOICE will fail until it is.")
            if args.ui_host and args.ui_host not in ("127.0.0.1", "localhost", "::1"):
                logger.warning(
                    "--voice on a non-loopback --ui-host: anyone who can reach the page can spend "
                    "your Boson credit and start robot tasks (browsers also block the mic off localhost)."
                )
        ui.start()
        if not args.serve:
            ui.start_task(args.instruction)

    def _on_phase(phase: str, detail: str = "") -> None:
        if ui is not None:
            ui.phase(phase, detail)
    k = np.asarray(calib["K"], dtype=float)
    t_cam_table = np.asarray(calib["T_cam_table"], dtype=float)
    table_polygon = _polygon_from_xy(calib["table_polygon_xy"])
    arm_workspaces = {
        name: _workspace_from_spec(spec) for name, spec in calib["arm_workspaces_xy"].items()
    }

    robots: dict[str, Any] = {}
    leaps: dict[str, LeapHandEE] = {}
    guard_ref: dict[str, Any] = {"g": None}  # overlap mutex, once built (execute mode)
    task_arms: dict[str, list[str] | None] = {"v": None}  # serve: arms allowed for this task
    task_rules: dict[str, list[dict[str, Any]]] = {"v": []}  # serve: per-object arm rules

    def _arms_not_home(tol_rad: float = 0.05) -> list[str]:
        """Arms that are in an error state or not at start_joints (e.g. after an e-stop the
        SDK refuses the home move with code 9 and the arm stays where it was)."""
        out = []
        for name, robot in robots.items():
            api = getattr(robot, "real_arm", None)
            if api is None:
                continue
            err = int(getattr(api, "error_code", 0) or 0)
            if err:
                out.append(f"{name} has controller error {err} (1 = emergency stop pressed)")
                continue
            code, ang = api.get_servo_angle(is_radian=True)
            home = list(getattr(getattr(robot, "config", None), "start_joints", []) or [])
            if code == 0 and ang and home:
                dev = max(abs(float(a) - float(b)) for a, b in zip(list(ang)[: len(home)], home, strict=False))
                if dev > tol_rad:
                    out.append(f"{name} is not at home (off by {dev:.2f} rad)")
        return out

    if not args.no_robot:
        _check_leap_ready(calib, arm_names)
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

    if ui is not None and robots:
        def _ui_go_home() -> None:
            for name, robot in robots.items():
                api = getattr(robot, "real_arm", None)
                if api is not None and int(getattr(api, "error_code", 0) or 0) != 0:
                    # e.g. after an emergency stop (released): GO HOME is an explicit operator action.
                    logger.info("UI go home: clearing error %s on arm=%s", api.error_code, name)
                    robot._clear_and_enable()
                logger.info("UI go home arm=%s", name)
                robot.move_to_home(speed_deg_s=20.0)
            _wait_arms_still()
            problems = _arms_not_home()
            if problems:
                raise RuntimeError("; ".join(problems))
            if guard_ref["g"] is not None:
                for name in robots:
                    guard_ref["g"].release_after(name)

        ui.set_go_home(_ui_go_home)

    table_xy_affine = _table_xy_affine_from_calib(calib)
    object_top_z_default, object_top_z = object_top_z_from_calib(calib)
    if table_xy_affine is not None:
        logger.info("Using table_xy_affine correction from calib.")
    if object_top_z or object_top_z_default:
        logger.info(
            "Object top-face heights for pixel projection: default=%.3f m %s",
            object_top_z_default,
            object_top_z,
        )
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
        object_top_z_m=object_top_z,
        object_top_z_m_default=object_top_z_default,
        thinking_budget=int(args.thinking_budget),
        model=str(args.model),
    )
    backend, api_model = resolve_base_model(args.model)
    logger.info(
        "Base model backend=%s api=%s thinking_budget=%s",
        backend,
        api_model,
        args.thinking_budget,
    )
    planner = Planner(
        max_attempts=2,
        model=str(args.model),
        thinking_budget=int(args.thinking_budget),
    )

    def _apply_arm_rules(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Force the arm on every step whose object matches an operator rule (per task)."""
        from tabletop_perception.perception import matches_height_key

        rules = task_rules["v"] or []
        for step in plan:
            obj = str((step.get("args") or {}).get("object") or "")
            for r in rules:
                if obj and matches_height_key(obj, r["object"]):
                    if step.get("arm") not in r["arms"]:
                        logger.info(
                            "arm rule %s → %s: step %s %s %s (was %s)",
                            r["object"], r["arms"][0], step.get("step"), step.get("primitive"),
                            obj, step.get("arm"),
                        )
                        step["arm"] = r["arms"][0]
                    break
        return plan

    def plan_fn(symbolic_view, instruction, arms):
        rules = task_rules["v"] or []
        if rules and not args.fixed_plan:
            # Tell the planner too, so ordering/handovers fit; enforcement is _apply_arm_rules.
            instruction = instruction + " Arm rules (must follow): " + "; ".join(
                f"anything that is a {r['object']} is handled only by {r['arms'][0]}" for r in rules
            ) + "."
        if args.fixed_plan:
            plan = plan_screws_to_container_xarm(
                symbolic_view,
                instruction,
                arms,
                arm=str(args.fixed_plan_arm),
            )
            logger.info("Fixed plan (no planner Gemini): %s", json.dumps(plan, indent=2))
        else:
            logger.info("Planning for arms=%s instruction=%r", list(arms), instruction)
            plan = planner(symbolic_view, instruction, arms)
            plan = _apply_arm_rules(plan)
            logger.info("plan=%s", json.dumps(plan, indent=2))
        (out_dir / "plan.json").write_text(json.dumps(plan, indent=2))
        return plan

    # Cache last geometric view for place-verify shortcut.
    last_views: dict[str, Any] = {"symbolic": None, "geometric": None}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / stamp
    perception_dir = PERCEPTION_RUNS / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    perception_dir.mkdir(parents=True, exist_ok=True)
    perceive_count = {"n": 0}

    def capture_image(newer_than: float | None = None) -> np.ndarray:
        if args.image is not None:
            return _load_image(args.image)
        if ui is not None:
            frame = ui.snapshot("top", newer_than=newer_than)
            # Right after the camera opens it can emit dark warm-up frames; wait briefly.
            for _ in range(20):
                if frame is None or float(frame.mean()) >= 30.0:
                    break
                time.sleep(0.15)
                frame = ui.snapshot("top")
            if frame is not None:
                return undistort_with_calib(frame, calib)
        return undistort_with_calib(
            _capture_from_camera(
                camera_path,
                width=int(cam_cfg["width"]),
                height=int(cam_cfg["height"]),
                fps=int(cam_cfg["fps"]),
            ),
            calib,
        )

    check_count = {"n": 0}

    def _wait_arms_still(timeout_s: float = 20.0, settle_s: float = 0.5) -> bool:
        """Block until every connected arm reports no motion and its joints have not changed
        for ``settle_s``. ``move_to_home(wait=True)`` can return before the arm has settled."""
        deadline = time.monotonic() + timeout_s
        for name, robot in robots.items():
            api = getattr(robot, "real_arm", None)
            if api is None:
                continue
            last: list[float] | None = None
            still_since: float | None = None
            while True:
                if time.monotonic() > deadline:
                    logger.warning("arm %s did not settle within %.0f s", name, timeout_s)
                    return False
                moving = getattr(api, "state", None) == 1  # xArm state 1 = in motion
                code, ang = api.get_servo_angle(is_radian=True)
                if code == 0 and ang:
                    cur = [float(a) for a in list(ang)[:6]]
                    if last is not None and not moving and max(abs(a - b) for a, b in zip(cur, last, strict=False)) < 0.002:
                        still_since = still_since or time.monotonic()
                        if time.monotonic() - still_since >= settle_s:
                            break
                    else:
                        still_since = None
                    last = cur
                time.sleep(0.05)
        return True

    def _scene_check(question: str | None = None, *, wait_still: bool = False) -> dict[str, Any]:
        """Top-camera photo → VLM verdict on the current instruction (saved in out_dir).
        With ``wait_still`` the photo is taken only after the arm(s) stopped moving."""
        from manipulation.scene_check import check_scene

        arm_still = None
        newer_than = None
        if wait_still and robots:
            arm_still = _wait_arms_still()
            newer_than = time.monotonic() + 0.15  # skip frames already in flight
        image = capture_image(newer_than=newer_than)
        result = check_scene(image, instruction=args.instruction or "", question=question)
        if arm_still is False:
            result["answer"] = "(arm had not fully stopped) " + str(result.get("answer") or "")
        result["arm_still"] = arm_still
        check_count["n"] += 1
        tag = f"check_{check_count['n']:02d}"
        cv2.imwrite(str(out_dir / f"{tag}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        (out_dir / f"{tag}.json").write_text(
            json.dumps({"instruction": args.instruction, "question": question, **result}, indent=2)
        )
        logger.info("Camera check (%s): %s", question or "task done?", result)
        return result

    def _preflight(instruction: str) -> dict[str, Any]:
        """Before a spoken task: do the objects it names exist on the table? (saved in out_dir)"""
        from manipulation.scene_check import preflight_objects

        image = capture_image()
        result = preflight_objects(image, instruction=instruction)
        check_count["n"] += 1
        tag = f"precheck_{check_count['n']:02d}"
        cv2.imwrite(str(out_dir / f"{tag}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        (out_dir / f"{tag}.json").write_text(json.dumps({"said": instruction, **result}, indent=2))
        logger.info("Object pre-check %r → %s", instruction, result)
        return result

    if ui is not None:
        ui.set_scene_check(_scene_check)
        ui.arms = list(arm_names)
        roles: dict[str, list[str]] = {}
        for n in arm_names:
            ee = _ee_name_for_arm(calib, n, _execution_cfg_for_arm(calib, n)).lower()
            role = "hand" if ee in {"leaphand", "leap_hand", "leap"} else "gripper"
            roles.setdefault(role, []).append(n)
        ui.arm_roles = roles
        ui.post_check = bool(args.serve and args.execute and args.post_check)
        ui.set_preflight(_preflight)

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
        logger.info(
            "Perceiving image shape=%s mean=%.1f instruction=%r",
            image.shape,
            float(image.mean()),
            args.instruction,
        )
        # Save the exact frame sent to the VLM first, so a failed perception can be inspected.
        cv2.imwrite(str(out_dir / "capture_rgb.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # One VLM call, reuse for Perception + overlay boxes.
        raw = perception.vlm_caller(image, args.instruction, perception.prompt or "")
        detections = parse_vlm_detections(raw, image_hw=(image.shape[0], image.shape[1]))
        perception_once = Perception(
            footprint_buffer_m=float(calib.get("footprint_buffer_m", 0.02)),
            prompt=perception.prompt,
            vlm_caller=lambda _img, _ins, _p: raw,
            table_xy_affine=table_xy_affine,
            grasp_height_offsets_m=grasp_height_offsets,
            object_top_z_m=object_top_z,
            object_top_z_m_default=object_top_z_default,
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
        detections = perception_once.last_detections or detections
        _dump_perception_logs(image, symbolic, geometric, detections)
        logger.info("symbolic_view=%s", json.dumps(symbolic, indent=2))
        return symbolic, geometric

    try:
        # Solver margin: use first arm's config (same default for both today).
        primary_exe = _execution_cfg_for_arm(calib, arm_names[0])
        solver_margin = float(primary_exe.get("solver_margin_m", 0.03))

        def _serve_forever(
            run_one: Callable[[], str],
            *,
            reset: Callable[[], None] | None = None,
        ) -> int:
            """Keep the process (robot, cameras, VLM clients) alive and run each
            instruction submitted through the live UI."""
            nonlocal out_dir, perception_dir
            assert ui is not None
            pending = (args.instruction or "").strip() or None
            if pending is None:
                ui.phase("idle", "waiting for an instruction")
                print("\nServe mode: waiting for an instruction at the live UI.\n")
            else:
                ui.set_task_busy(True)
            first = True
            while True:
                while pending is None:
                    pending = ui.wait_task(timeout=0.5)
                args.instruction = pending
                pending = None
                task_arms["v"] = [a for a in (ui.last_task_arms or []) if a in arm_names] or None
                task_rules["v"] = [
                    r for r in (ui.last_task_rules or []) if all(a in arm_names for a in r["arms"])
                ]
                if task_rules["v"]:
                    logger.info("Serve: per-object arm rules=%s", task_rules["v"])
                if task_arms["v"]:
                    logger.info("Serve: task restricted to arms=%s", task_arms["v"])
                if not first:
                    # Fresh run dir per task so earlier logs are not overwritten.
                    stamp_i = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_dir = args.out_dir / stamp_i
                    perception_dir = PERCEPTION_RUNS / stamp_i
                    out_dir.mkdir(parents=True, exist_ok=True)
                    perception_dir.mkdir(parents=True, exist_ok=True)
                    perceive_count["n"] = 0
                    last_views["symbolic"] = None
                    last_views["geometric"] = None
                    if reset is not None:
                        reset()
                first = False
                logger.info("Serve: task %r", args.instruction)
                ui.start_task(args.instruction)
                try:
                    status = run_one()
                except Exception as exc:  # noqa: BLE001
                    # Keep serving; the arm is NOT auto-homed here (state unknown) — use
                    # GO HOME on the page once the workspace is clear.
                    logger.exception("Serve: task failed")
                    ui.phase("failed", f"{type(exc).__name__}: {exc}")
                else:
                    if args.execute and args.post_check:
                        # "success" from the executor = every move ran, NOT that the object is
                        # where it should be. Report what the camera sees instead.
                        from manipulation.scene_check import verdict_text

                        ui.phase("verifying", "waiting for the arm to stop, then camera check")
                        try:
                            verdict = verdict_text(_scene_check(None, wait_still=True))
                        except Exception as exc:  # noqa: BLE001
                            logger.warning("Serve: camera check failed: %s", exc)
                            verdict = f"camera check failed: {exc}"
                        ui.phase("result", f"{verdict} (arm moves: {status})")
                    else:
                        ui.stop_task(status)
                finally:
                    ui.set_task_busy(False)

        def _dry_run_once() -> str:
            from manipulation.solver import solve

            _on_phase("perceiving", "vision VLM")
            symbolic, geometric = perceive_pair()
            _on_phase("planning", args.instruction)
            plan = plan_fn(symbolic, args.instruction, task_arms["v"] or arm_names)

            _on_phase("solving", f"{len(plan)} step(s)")
            bound = solve(
                plan,
                geometric,
                SolverConfig(
                    grasp_height=grasp_height,
                    margin=solver_margin,
                    place_height_offsets_m=place_height_offsets,
                    place_height_offsets_by_arm=place_height_offsets_by_arm,
                    grasp_height_offsets_by_arm=grasp_height_offsets_by_arm,
                    place_clearance_m=place_clearance,
                ),
                lookahead=args.lookahead,
            )
            (out_dir / "symbolic_view.json").write_text(json.dumps(symbolic, indent=2))
            (out_dir / "plan.json").write_text(json.dumps(plan, indent=2))
            (out_dir / "bound.json").write_text(json.dumps(bound, indent=2))
            print(json.dumps({"plan": plan, "bound": bound}, indent=2))
            print(f"\nDry-run only. Outputs → {out_dir}")
            return "dry-run"

        if not args.execute:
            if args.interactive:
                _on_phase("perceiving", "vision VLM")
                symbolic, geometric = perceive_pair()
                _on_phase("planning", args.instruction)
                decision = planner.decide(symbolic, args.instruction, arm_names)
                _on_phase("done", str(decision.get("status") or ""))
                print(json.dumps(decision, indent=2, default=str))
                print(f"\nInteractive dry-run. Outputs → {out_dir}")
                print("Re-run with --execute --interactive to play turns.")
                return 0
            # Dry-run: perceive → plan → solve only.
            if args.serve:
                return _serve_forever(_dry_run_once)
            _dry_run_once()
            print("Re-run with --execute to move the arm(s).")
            _on_phase("done", "dry-run")
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

        # Overlap = ∩ workspaces. Hold mutex while an arm is in-zone; retract
        # the occupant only when the other arm needs to enter.
        ov_cfg = dict(calib.get("overlap_xy") or {})
        ov_poly = overlap_from_workspaces(
            arm_workspaces,
            arms=ov_cfg.get("arms"),
            table_polygon=table_polygon if ov_cfg.get("clip_to_table", True) else None,
        )
        overlap_guard = guard_ref["g"] = OverlapGuard(
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
                        raise RuntimeError(f"no exclusive retract point for {arm}")
                    z_cmd = z
                    try:
                        cur = be.get_current_pose_table()
                        z_cmd = max(z_cmd, float(cur[2]))
                    except Exception:  # noqa: BLE001
                        pass
                    logger.info(
                        "retract %s → exclusive xy=(%.3f, %.3f) z=%.3f",
                        arm,
                        xy[0],
                        xy[1],
                        z_cmd,
                    )
                    be.move_to_pose((xy[0], xy[1], z_cmd, 0.0))

                return _retract_arm

            overlap_guard.retract[name] = _make_retract()

        if overlap_guard.enabled:
            logger.info(
                "Overlap mutex enabled area=%.3f m^2 centroid=(%.3f, %.3f)",
                float(ov_poly.area),
                float(ov_poly.centroid.x),
                float(ov_poly.centroid.y),
            )

        if args.interactive:
            loop = ManipulationLoop(
                perceive=perceive_pair,
                plan_fn=plan_fn,
                executors=executors,
                solver_config=SolverConfig(
                    grasp_height=grasp_height,
                    margin=solver_margin,
                    place_height_offsets_m=place_height_offsets,
                    place_height_offsets_by_arm=place_height_offsets_by_arm,
                    grasp_height_offsets_by_arm=grasp_height_offsets_by_arm,
                    place_clearance_m=place_clearance,
                ),
                arms=arm_names,
                lookahead=args.lookahead,
                overlap_guard=overlap_guard,
                on_phase=_on_phase,
            )

            def _decide(symbolic, instruction, arms, history):
                return planner.decide(symbolic, instruction, arms, history=history)

            def _execute(plan, geometric):
                result = loop.execute_bound_plan(plan, geometric)
                payload = {
                    "status": result.status,
                    "reason": result.reason,
                    "plan": result.plan,
                    "bound": result.bound,
                    "results": result.results,
                }
                (out_dir / "result.json").write_text(
                    json.dumps(payload, indent=2, default=str)
                )
                print(json.dumps(payload, indent=2, default=str))
                return result.status

            def _go_home() -> None:
                for name, robot in robots.items():
                    try:
                        logger.info("interactive → go home arm=%s", name)
                        robot.move_to_home(speed_deg_s=20.0)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("go home %s failed: %s", name, exc)

            return run_interactive(
                args.instruction,
                arms=arm_names,
                perceive=perceive_pair,
                decide=_decide,
                execute=_execute,
                go_home=_go_home,
                wait_s=float(args.watch_s if args.watch_s is not None else 2.0),
                on_phase=_on_phase,
            )

        def _run_one_execute(cycle: int = 1) -> str:
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
                    place_clearance_m=place_clearance,
                ),
                arms=task_arms["v"] or arm_names,
                lookahead=args.lookahead,
                overlap_guard=overlap_guard,
                on_phase=_on_phase,
            )

            payload = {
                "status": result.status,
                "reason": result.reason,
                "plan": result.plan,
                "bound": result.bound,
                "results": result.results,
                "symbolic_view": result.symbolic_view,
                "arms": task_arms["v"] or arm_names,
                "cycle": cycle,
            }
            (out_dir / "result.json").write_text(json.dumps(payload, indent=2, default=str))
            print(json.dumps(payload, indent=2, default=str))
            print(f"\nOutputs → {out_dir}")

            for name, robot in robots.items():
                try:
                    logger.info("Query finished → go home arm=%s", name)
                    robot.move_to_home(speed_deg_s=20.0)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("go home %s failed: %s", name, exc)
            _wait_arms_still()  # move_to_home(wait=True) can return before the arm settles
            problems = _arms_not_home()
            if problems:
                # Keep the overlap lock as is (an arm may still be in the zone) and say so.
                raise RuntimeError(
                    "moves finished but " + "; ".join(problems)
                    + ". Release the e-stop if pressed, then press GO HOME."
                )
            # Home is outside the shared zone: without this the next task would "retract" an
            # arm that already sits at home, because the lock still names it as the holder.
            for name in robots:
                overlap_guard.release_after(name)
            return result.status

        if args.serve:
            return _serve_forever(_run_one_execute, reset=last_place_cmd.clear)

        loop_minutes = float(args.loop_minutes or 0.0)
        deadline = time.perf_counter() + loop_minutes * 60.0 if loop_minutes > 0 else None
        cycle = 0
        last_status = "success"
        while True:
            cycle += 1
            if deadline is not None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    logger.info("Loop time budget exhausted after %d cycle(s).", cycle - 1)
                    break
                logger.info(
                    "Loop cycle %d starting (%.0fs remaining of %.1f min)…",
                    cycle,
                    remaining,
                    loop_minutes,
                )
            else:
                logger.info("Single-shot cycle starting…")

            # Fresh run dir per cycle so later rounds don't overwrite earlier logs.
            if cycle > 1:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_dir = args.out_dir / stamp
                perception_dir = PERCEPTION_RUNS / stamp
                out_dir.mkdir(parents=True, exist_ok=True)
                perception_dir.mkdir(parents=True, exist_ok=True)
                perceive_count["n"] = 0
                last_views["symbolic"] = None
                last_views["geometric"] = None
                last_place_cmd.clear()

            last_status = _run_one_execute(cycle)

            if deadline is None:
                break

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                logger.info("Loop time budget exhausted after cycle %d.", cycle)
                break
            pause = min(float(args.loop_pause_s), max(remaining, 0.0))
            if pause > 0:
                logger.info("Loop pause %.1fs before next cycle…", pause)
                time.sleep(pause)

        return 0 if last_status == "success" else 1
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
        if ui is not None:
            ui.close()


if __name__ == "__main__":
    raise SystemExit(main())
