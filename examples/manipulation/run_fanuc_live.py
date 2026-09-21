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
Fanuc perceive → plan → solve → execute.

Same stack as ``manipulation.run_xarm_live`` (Gemini-ER / Cosmos planner
emits Grasp / Place / LiftUp). The arm is Fanuc.

    cd examples
    UV_NO_SYNC=1 uv run python -m manipulation.run_fanuc_live \\
        --vlm cosmos --camera /dev/video0 --execute \\
        --instruction "pick all screws and put into the yellow container"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from manipulation.executor import ArmExecutor, ExecutorConfig  # noqa: E402
from manipulation.fanuc_backend import FanucMotionBackend  # noqa: E402
from manipulation.fixed_plan import plan_screws_to_container_xarm  # noqa: E402
from manipulation.orchestrator import run_manipulation  # noqa: E402
from manipulation.solver import SolverConfig, solve  # noqa: E402
from openai_backend import resolve_base_model  # noqa: E402
from planner import Planner  # noqa: E402
from tabletop_perception.calibrate_table_xy import _samples_path  # noqa: E402
from tabletop_perception.fanuc.run_fanuc_live import _connect_fanuc  # noqa: E402
from tabletop_perception.perception import Perception, object_top_z_from_calib  # noqa: E402
from tabletop_perception.run_xarm_live import (  # noqa: E402
    _capture_from_camera,
    _draw_image_overlay,
    _execution_cfg_for_arm,
    _load_calib,
    _polygon_from_xy,
    _serialize_geometric,
    _table_xy_affine_from_calib,
    _workspace_from_spec,
)
from tabletop_perception.visualize import visualize_table_plane  # noqa: E402
from tabletop_perception.vlm import parse_vlm_detections  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CALIB = Path(__file__).resolve().parents[1] / "tabletop_perception" / "calib" / "fanuc_overhead.json"
PERCEPTION_RUNS = Path(__file__).resolve().parents[1] / "tabletop_perception" / "runs"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Fanuc perceive → plan → execute")
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--camera", type=str, default=None)
    p.add_argument("--robot-host", dest="robot_host", type=str, default=None)
    p.add_argument("--instruction", type=str, default=None)
    p.add_argument("--execute", action="store_true", help="Move the arm (default: plan/solve only)")
    p.add_argument("--skip-place-verify", action="store_true", default=True)
    p.add_argument("--lookahead", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fixed-plan", action="store_true", help="Rule-based screws→container, no planner VLM")
    p.add_argument("--model", "--vlm", dest="model", default="cosmos")
    p.add_argument("--thinking-budget", type=int, default=-1)
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    args = p.parse_args(argv)

    instruction = (args.instruction or "").strip()
    if not instruction:
        instruction = input("任务指令：").strip()
    if not instruction:
        raise SystemExit("需要任务指令，例如：pick all screws and put into the yellow container")

    calib = _load_calib(args.calib)
    arm_name = str((calib.get("execution") or {}).get("arm_name") or "fanuc")
    exe = _execution_cfg_for_arm(calib, arm_name)

    def _height_map(raw: Any) -> dict[str, float]:
        return {
            str(k): float(v)
            for k, v in dict(raw or {}).items()
            if not str(k).startswith("_")
        }

    # Fanuc heights live under execution_by_arm.fanuc, not the xArm tables.
    grasp_height = float(exe.get("grasp_height_m", calib.get("grasp_height_m", 0.0)))
    grasp_height_offsets = _height_map(
        exe.get("grasp_height_offset_m") or calib.get("grasp_height_offset_m")
    )
    place_height_offsets = _height_map(
        exe.get("place_height_offset_m") or calib.get("place_height_offset_m")
    )
    cam_cfg = calib["camera"]
    camera_path = args.camera or cam_cfg["index_or_path"]
    k = np.asarray(calib["K"], dtype=float)
    t_cam_table = np.asarray(calib["T_cam_table"], dtype=float)
    table_polygon = _polygon_from_xy(calib["table_polygon_xy"])
    arm_workspaces = {
        name: _workspace_from_spec(spec) for name, spec in calib["arm_workspaces_xy"].items()
    }
    table_xy_affine = _table_xy_affine_from_calib(calib)
    object_top_z_default, object_top_z = object_top_z_from_calib(calib, arm=arm_name)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / stamp
    perception_dir = PERCEPTION_RUNS / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    perception_dir.mkdir(parents=True, exist_ok=True)
    print(f"task={instruction!r}")
    print(f"logs → {out_dir}")
    backend_name, api_model = resolve_base_model(args.model)
    print(f"VLM backend={backend_name} model={api_model} arm={arm_name}")
    print(
        f"Fanuc heights grasp={grasp_height:.3f} m offsets={grasp_height_offsets} "
        f"place={place_height_offsets} object_top={object_top_z}",
        flush=True,
    )

    perception = Perception(
        footprint_buffer_m=float(calib.get("footprint_buffer_m", 0.02)),
        table_xy_affine=table_xy_affine,
        grasp_height_offsets_m=grasp_height_offsets,
        object_top_z_m=object_top_z,
        object_top_z_m_default=object_top_z_default,
        thinking_budget=int(args.thinking_budget),
        model=str(args.model),
    )
    planner = Planner(
        max_attempts=2,
        model=str(args.model),
        thinking_budget=int(args.thinking_budget),
    )
    last_views: dict[str, Any] = {"symbolic": None, "geometric": None}
    last_place_cmd: dict[str, Any] = {}
    perceive_n = {"n": 0}

    def perceive_pair():
        image = _capture_from_camera(
            camera_path,
            width=int(cam_cfg["width"]),
            height=int(cam_cfg["height"]),
            fps=int(cam_cfg["fps"]),
            fourcc=cam_cfg.get("fourcc") or "MJPG",
        )
        raw = perception.vlm_caller(image, instruction, perception.prompt or "")
        detections = parse_vlm_detections(raw, image_hw=(image.shape[0], image.shape[1]))
        perception_once = Perception(
            footprint_buffer_m=float(calib.get("footprint_buffer_m", 0.02)),
            prompt=perception.prompt,
            vlm_caller=lambda _img, _ins, _p: raw,
            table_xy_affine=table_xy_affine,
            grasp_height_offsets_m=grasp_height_offsets,
            object_top_z_m=object_top_z,
            object_top_z_m_default=object_top_z_default,
            model=str(args.model),
        )
        symbolic, geometric = perception_once(
            image=image,
            K=k,
            T_cam_table=t_cam_table,
            grasp_height=grasp_height,
            instruction=instruction,
            arm_workspaces=arm_workspaces,
            table_polygon=table_polygon,
        )
        detections = perception_once.last_detections or detections
        last_views["symbolic"] = symbolic
        last_views["geometric"] = geometric
        perceive_n["n"] += 1
        run_dir = perception_dir / f"perceive_{perceive_n['n']:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "instruction.txt").write_text(instruction + "\n")
        (run_dir / "symbolic_view.json").write_text(json.dumps(symbolic, indent=2))
        (run_dir / "geometric_view.json").write_text(
            json.dumps(_serialize_geometric(geometric), indent=2)
        )
        cv2.imwrite(str(run_dir / "capture_rgb.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        overlay = _draw_image_overlay(image, geometric, detections)
        cv2.imwrite(str(run_dir / "capture_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        visualize_table_plane(
            geometric, show=False, save_path=run_dir / "table_plane.png", title=run_dir.name
        )
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
        print(f"symbolic={json.dumps(symbolic, ensure_ascii=False)}")
        return symbolic, geometric

    def plan_fn(symbolic_view, instr, arms):
        if args.fixed_plan:
            plan = plan_screws_to_container_xarm(
                symbolic_view, instr, arms, arm=arm_name
            )
            print("fixed plan (no planner VLM)")
        else:
            print(f"planning arms={list(arms)} instruction={instr!r}")
            plan = planner(symbolic_view, instr, arms)
        (out_dir / "plan.json").write_text(json.dumps(plan, indent=2))
        print(json.dumps(plan, indent=2))
        return plan

    solver_cfg = SolverConfig(
        grasp_height=grasp_height,
        margin=float(exe.get("solver_margin_m", 0.03)),
        place_height_offsets_m=place_height_offsets,
        place_height_offsets_by_arm={arm_name: place_height_offsets},
        grasp_height_offsets_by_arm={arm_name: grasp_height_offsets},
        place_clearance_m=float(calib.get("place_clearance_m", 0.01)),
    )

    if not args.execute:
        symbolic, geometric = perceive_pair()
        plan = plan_fn(symbolic, instruction, [arm_name])
        bound = solve(plan, geometric, solver_cfg, lookahead=args.lookahead)
        (out_dir / "bound.json").write_text(json.dumps(bound, indent=2))
        print(json.dumps({"plan": plan, "bound": bound}, indent=2))
        print(f"\nDry-run. Re-run with --execute to move. Outputs → {out_dir}")
        return 0

    print("示教器请保持可急停。")
    confirm = input("输入 go 开始感知+规划+执行，其它键取消：").strip().lower()
    if confirm != "go":
        print("已取消。")
        return 0

    robot = _connect_fanuc(calib, args.robot_host, arm_name=arm_name)
    try:
        backend = FanucMotionBackend(
            robot,
            calib=calib,
            arm_name=arm_name,
            samples_file=_samples_path(args.calib, "fanuc"),
            speed=float(exe.get("move_speed_mm_s", 40.0)),
        )
        backend.use_object_yaw = True

        def perceive_for_place():
            geo = dict(last_views["geometric"] or {"objects": []})
            obj = last_place_cmd.get("object")
            pose = last_place_cmd.get("pose")
            if args.skip_place_verify and obj and pose:
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

        executor = ArmExecutor(
            arm_name,
            move_to_pose=backend.move_to_pose,
            gripper=backend.gripper,
            read_gripper_width=backend.read_gripper_width,
            perceive=perceive_for_place,
            get_current_pose=backend.get_current_pose_table,
            config=ExecutorConfig(
                approach_offset=float(exe.get("approach_offset_m", 0.08)),
                lift_offset=float(exe.get("lift_offset_m", 0.08)),
                gripper_open_width=0.085,
                gripper_closed_width=0.005,
                place_xy_tol=0.05,
            ),
        )
        raw_execute = executor.execute

        def _execute_tracking(step: dict[str, Any]):
            if str(step.get("primitive")) == "Place":
                params = step.get("params") or {}
                last_place_cmd["object"] = params.get("object")
                last_place_cmd["pose"] = params.get("pose")
            return raw_execute(step)

        executor.execute = _execute_tracking  # type: ignore[method-assign]

        result = run_manipulation(
            instruction,
            perceive=perceive_pair,
            plan_fn=plan_fn,
            executors={arm_name: executor},
            solver_config=solver_cfg,
            arms=[arm_name],
            lookahead=args.lookahead,
            recover=False,
        )
        payload = {
            "status": result.status,
            "reason": result.reason,
            "plan": result.plan,
            "bound": result.bound,
            "results": result.results,
            "symbolic_view": result.symbolic_view,
        }
        (out_dir / "result.json").write_text(json.dumps(payload, indent=2, default=str))
        print(json.dumps(payload, indent=2, default=str))
        print(f"\nOutputs → {out_dir}")
        return 0 if result.status == "success" else 1
    finally:
        robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
