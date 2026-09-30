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

"""
Live monocular tabletop perception on Fanuc + overhead OpenCV camera.

Same pipeline as ``tabletop_perception.run_xarm_live``: capture → VLM →
symbolic/geometric views. The robot handle is Fanuc RMI; connect is read-only
(no ``FRC_LinearMotion``).

Example (real hardware)::

    export GOOGLE_API_KEY=...
    cd examples
    python -m tabletop_perception.fanuc.run_fanuc_live \\
        --instruction "pick up the black panther plush" \\
        --robot-host 172.30.109.22 \\
        --camera /dev/video2

Offline smoke test on a previously captured frame::

    python -m tabletop_perception.fanuc.run_fanuc_live \\
        --image ../outputs/captured_images/opencv__dev_video2.png \\
        --no-robot \\
        --instruction "list graspable objects on the table"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_EXAMPLES_DIR = Path(__file__).resolve().parents[2]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from tabletop_perception.perception import Perception, object_top_z_from_calib  # noqa: E402
from tabletop_perception.run_xarm_live import (  # noqa: E402
    _capture_from_camera,
    _draw_image_overlay,
    _execution_cfg_for_arm,
    _load_calib,
    _load_image,
    _mock_detections_for_scene,
    _polygon_from_xy,
    _serialize_geometric,
    _table_xy_affine_from_calib,
    _workspace_from_spec,
)
from tabletop_perception.visualize import visualize_table_plane  # noqa: E402

DEFAULT_CALIB = Path(__file__).resolve().parents[1] / "calib" / "fanuc_overhead.json"

_FANUC_CFG_KEYS = {
    "host",
    "port",
    "group",
    "utool",
    "uframe",
    "speed",
    "term_type",
    "term_value",
    "gripper_lcb_type",
    "gripper_lcb_value",
    "gripper_port_type",
    "gripper_state_port_number",
    "gripper_port_number",
    "gripper_open_port_number",
    "gripper_close_port_number",
    "gripper_open_value",
    "gripper_close_value",
    "id",
    "calibration_dir",
    "twin_udp_host",
    "twin_udp_port",
}


def _robot_cfg_from_calib(calib: dict[str, Any], arm_name: str | None = None) -> dict[str, Any]:
    robots = dict(calib.get("robots") or {})
    if arm_name and arm_name in robots:
        raw = dict(robots[arm_name])
    else:
        raw = dict(calib.get("robot") or {})
    if "robot_ip" in raw and "host" not in raw:
        raw["host"] = raw["robot_ip"]
    return {k: v for k, v in raw.items() if k in _FANUC_CFG_KEYS and not str(k).startswith("_")}


def _connect_fanuc(
    calib: dict[str, Any],
    robot_host: str | None,
    *,
    arm_name: str | None = "fanuc",
) -> Any:
    from lerobot.robots.fanuc import Fanuc, FanucConfig

    robot_cfg = _robot_cfg_from_calib(calib, arm_name)
    if robot_host:
        robot_cfg["host"] = robot_host
    elif not robot_cfg.get("host"):
        raise ValueError(f"No host for arm {arm_name!r} in calib")
    robot_cfg["cameras"] = {}

    config = FanucConfig(**robot_cfg)
    robot = Fanuc(config)
    robot.connect()
    return robot


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fanuc + overhead camera tabletop perception")
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--camera", type=str, default=None, help="Override camera path, e.g. /dev/video2")
    p.add_argument("--image", type=Path, default=None, help="Use a still RGB/BGR image instead of live camera")
    p.add_argument("--robot-host", "--robot-ip", dest="robot_host", type=str, default=None)
    p.add_argument("--no-robot", action="store_true", help="Skip Fanuc connect (camera/VLM only)")
    p.add_argument("--instruction", type=str, default="detect all graspable objects on the table")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "runs")
    p.add_argument("--prompt-file", type=Path, default=None, help="Optional prompt text with {instruction}")
    p.add_argument(
        "--mock-vlm",
        action="store_true",
        help="Skip Gemini and inject a small hardcoded detection list (pipeline dry-run)",
    )
    p.add_argument(
        "--thinking-budget",
        type=int,
        default=-1,
        help="Gemini thinking budget: 0=off (faster), -1=dynamic (default), or a positive token cap. "
        "With --model gpt-6 this maps to reasoning effort.",
    )
    p.add_argument(
        "--model",
        "--vlm",
        dest="model",
        default="gemini",
        help="Base VLM: gemini, gpt-6, or cosmos / cosmos3-nano.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    calib = _load_calib(args.calib)

    cam_cfg = calib["camera"]
    camera_path: str | int = args.camera or cam_cfg["index_or_path"]
    k = np.asarray(calib["K"], dtype=float)
    t_cam_table = np.asarray(calib["T_cam_table"], dtype=float)
    table_polygon = _polygon_from_xy(calib["table_polygon_xy"])
    arm_workspaces = {
        name: _workspace_from_spec(spec) for name, spec in calib["arm_workspaces_xy"].items()
    }
    exe = _execution_cfg_for_arm(calib, "fanuc")
    grasp_height = float(exe.get("grasp_height_m", calib.get("grasp_height_m", 0.0)))
    grasp_height_offsets = {
        str(key): float(val)
        for key, val in dict(
            exe.get("grasp_height_offset_m") or calib.get("grasp_height_offset_m") or {}
        ).items()
        if not str(key).startswith("_")
    }
    object_top_z_default, object_top_z = object_top_z_from_calib(calib, arm="fanuc")
    footprint_buffer = float(calib.get("footprint_buffer_m", 0.02))

    prompt = None
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text()

    robot = None
    if not args.no_robot:
        host = args.robot_host or _robot_cfg_from_calib(calib, "fanuc").get("host")
        print(f"Connecting Fanuc at {host} ...")
        robot = _connect_fanuc(calib, args.robot_host)
        print("Fanuc connected (no motion commanded).")
        try:
            obs = robot.get_observation()
            print(
                "Fanuc TCP mm/deg: "
                f"X={obs['j0']:.1f} Y={obs['j1']:.1f} Z={obs['j2']:.1f} "
                f"W={obs['j3']:.1f} P={obs['j4']:.1f} R={obs['j5']:.1f}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not read Fanuc observation: {exc}")
    else:
        print("Skipping robot connect (--no-robot).")

    try:
        if args.image is not None:
            print(f"Loading still image: {args.image}")
            image = _load_image(args.image)
        else:
            print(f"Capturing from camera: {camera_path}")
            image = _capture_from_camera(
                camera_path,
                width=int(cam_cfg["width"]),
                height=int(cam_cfg["height"]),
                fps=int(cam_cfg["fps"]),
                fourcc=cam_cfg.get("fourcc") or "MJPG",
            )
        print(f"Image shape: {image.shape}")

        from tabletop_perception.vlm import parse_vlm_detections

        if args.mock_vlm:
            print("Using --mock-vlm (no Gemini call).")
            raw = _mock_detections_for_scene()
        else:
            perception = Perception(
                footprint_buffer_m=footprint_buffer,
                prompt=prompt,
                thinking_budget=int(args.thinking_budget),
                model=str(args.model),
            )
            raw = perception.vlm_caller(image, args.instruction, prompt or "")
        detections = parse_vlm_detections(raw, image_hw=(image.shape[0], image.shape[1]))
        affine = _table_xy_affine_from_calib(calib)
        if affine is not None:
            print("Using table_xy_affine correction from calib.")
        if grasp_height_offsets:
            print(f"Per-object grasp height offsets: {grasp_height_offsets}")
        if object_top_z or object_top_z_default:
            print(
                f"Object top-face heights for projection: "
                f"default={object_top_z_default:.3f} m {object_top_z}"
            )
        perception_once = Perception(
            footprint_buffer_m=footprint_buffer,
            prompt=prompt,
            vlm_caller=lambda _img, _ins, _p: raw,
            table_xy_affine=affine,
            grasp_height_offsets_m=grasp_height_offsets,
            object_top_z_m=object_top_z,
            object_top_z_m_default=object_top_z_default,
            model=str(args.model),
        )
        symbolic_view, geometric_view = perception_once(
            image=image,
            K=k,
            T_cam_table=t_cam_table,
            grasp_height=grasp_height,
            instruction=args.instruction,
            arm_workspaces=arm_workspaces,
            table_polygon=table_polygon,
        )
        try:
            from manipulation.coordination import overlap_from_workspaces

            ov_cfg = dict(calib.get("overlap_xy") or {})
            geometric_view = dict(geometric_view)
            geometric_view["overlap"] = overlap_from_workspaces(
                geometric_view["workspaces"],
                arms=ov_cfg.get("arms"),
                table_polygon=(table_polygon if ov_cfg.get("clip_to_table", True) else None),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"overlap attach skipped: {exc}")

        stamp_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        stamp_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = stamp_dir / "capture_rgb.png"
        overlay_path = stamp_dir / "capture_overlay.png"
        table_path = stamp_dir / "table_plane.png"
        symbolic_path = stamp_dir / "symbolic_view.json"
        geometric_path = stamp_dir / "geometric_view.json"

        cv2.imwrite(str(rgb_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        detections = perception_once.last_detections or detections
        overlay = _draw_image_overlay(image, geometric_view, detections)
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        visualize_table_plane(
            geometric_view,
            show=False,
            save_path=table_path,
            title="Fanuc tabletop perception",
        )

        symbolic_path.write_text(json.dumps(symbolic_view, indent=2))
        geometric_path.write_text(json.dumps(_serialize_geometric(geometric_view), indent=2))

        print("\nsymbolic_view:")
        print(json.dumps(symbolic_view, indent=2))
        print("\ngeometric objects:")
        for obj in geometric_view["objects"]:
            print(
                f"  {obj['name']}: xy=({obj['xy'][0]:.3f}, {obj['xy'][1]:.3f}) "
                f"yaw={obj['yaw']:.3f} grasp_pose={tuple(round(v, 3) for v in obj['grasp_pose'])}"
            )
        print(f"\nWrote outputs to {stamp_dir}")
        print(f"  {rgb_path.name}, {overlay_path.name}, {table_path.name}")
        print(f"  {symbolic_path.name}, {geometric_path.name}")
        return 0
    finally:
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: robot disconnect failed: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
