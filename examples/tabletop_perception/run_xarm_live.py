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
Live monocular tabletop perception on xArm + overhead OpenCV camera.

Example (real hardware)::

    export GOOGLE_API_KEY=...
    cd examples
    python -m tabletop_perception.run_xarm_live \\
        --instruction "pick up the black panther plush" \\
        --robot-ip 192.168.1.204 \\
        --camera /dev/video2

Offline smoke test on a previously captured frame::

    python -m tabletop_perception.run_xarm_live \\
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
from shapely.geometry import Polygon

# Allow ``python -m tabletop_perception.run_xarm_live`` from ``examples/``.
_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from tabletop_perception.perception import Perception  # noqa: E402
from tabletop_perception.visualize import visualize_table_plane  # noqa: E402

DEFAULT_CALIB = Path(__file__).resolve().parent / "calib" / "xarm_overhead.json"


def _load_calib(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _table_xy_affine_from_calib(
    calib: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Optional ``true = A @ raw_table_xy + b`` correction from calib JSON."""
    cfg = calib.get("table_xy_affine")
    if not cfg:
        return None
    a = np.asarray(cfg["A"], dtype=float).reshape(2, 2)
    b = np.asarray(cfg["b"], dtype=float).reshape(2)
    return a, b


def _polygon_from_xy(coords: list[list[float]]) -> Polygon:
    return Polygon([(float(x), float(y)) for x, y in coords])


def _opening_to_mid_angle(opening: str | float | int) -> float:
    """Map opening label / degrees to the semicircle mid-angle (radians, CCW from +x)."""
    if isinstance(opening, (int, float)):
        return float(np.deg2rad(float(opening)))
    key = str(opening).strip().lower()
    named = {
        "+x": 0.0,
        "-x": np.pi,
        "+y": np.pi / 2.0,
        "-y": -np.pi / 2.0,
        # Diagonals (useful for corner-mounted bases).
        "+x+y": np.pi / 4.0,
        "ne": np.pi / 4.0,
        "-x+y": 3.0 * np.pi / 4.0,
        "nw": 3.0 * np.pi / 4.0,
        "-x-y": -3.0 * np.pi / 4.0,
        "sw": -3.0 * np.pi / 4.0,
        "+x-y": -np.pi / 4.0,
        "se": -np.pi / 4.0,
    }
    if key not in named:
        raise ValueError(
            f"Unsupported semicircle opening {opening!r}; "
            "use ±x/±y, ne/nw/se/sw, or an angle in degrees"
        )
    return float(named[key])


def _semicircle_polygon(
    center: tuple[float, float] | list[float],
    radius: float,
    opening: str | float | int = "-x",
    n_arc: int = 48,
) -> Polygon:
    """
    Disk-sector workspace: semicircle of ``radius`` around ``center``.

    ``opening`` is the inward direction onto the table (named axis/diagonal,
    or degrees CCW from +x).
    """
    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    n = max(8, int(n_arc))
    mid = _opening_to_mid_angle(opening)

    angles = mid + np.linspace(-np.pi / 2.0, np.pi / 2.0, n)
    arc = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in angles]
    # Close along the diameter through the base center.
    return Polygon(arc)


def _workspace_from_spec(spec: Any) -> Polygon:
    """Accept either a raw vertex list or a semicircle descriptor."""
    if isinstance(spec, list):
        return _polygon_from_xy(spec)
    if not isinstance(spec, dict):
        raise TypeError(f"Workspace spec must be list or dict, got {type(spec)!r}")
    kind = str(spec.get("type", "polygon")).lower()
    if kind == "semicircle":
        opening: str | float | int
        if "opening_deg" in spec:
            opening = float(spec["opening_deg"])
        else:
            opening = spec.get("opening", "-x")
        return _semicircle_polygon(
            center=spec["center"],
            radius=float(spec["radius"]),
            opening=opening,
            n_arc=int(spec.get("n_arc", 48)),
        )
    if kind == "polygon":
        return _polygon_from_xy(spec["xy"])
    raise ValueError(f"Unknown workspace type {kind!r}")


def _capture_from_camera(
    index_or_path: str | int,
    *,
    width: int,
    height: int,
    fps: int,
) -> np.ndarray:
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    cfg = OpenCVCameraConfig(
        index_or_path=index_or_path,
        width=width,
        height=height,
        fps=fps,
    )
    cam = OpenCVCamera(cfg)
    cam.connect()
    try:
        # Warm up a few frames — UVC devices often return a dark first frame.
        frame = None
        for _ in range(8):
            frame = cam.read()
        if frame is None:
            raise RuntimeError(f"Failed to read a frame from {index_or_path}")
        return np.asarray(frame)
    finally:
        cam.disconnect()


def _load_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


_XARM_CFG_KEYS = {
    "robot_ip",
    "robot_dof",
    "robot_mode",
    "robot_speed",
    "robot_acc",
    "gripper_type",
    "gripper_port",
    "gripper_speed",
    "gripper_force",
    "start_joints",
    "move_to_start_on_connect",
    "ema_alpha",
    "approach_first_frame",
    "approach_speed",
    "approach_acc",
    "approach_pos_threshold_mm",
    "id",
    "calibration_dir",
}


def _robot_cfg_from_calib(calib: dict[str, Any], arm_name: str | None = None) -> dict[str, Any]:
    """Pick Robot1 legacy ``robot`` or ``robots[arm]`` entry; strip unknown keys."""
    robots = dict(calib.get("robots") or {})
    raw: dict[str, Any]
    if arm_name and arm_name in robots:
        raw = dict(robots[arm_name])
    else:
        raw = dict(calib.get("robot") or {})
    return {k: v for k, v in raw.items() if k in _XARM_CFG_KEYS and not str(k).startswith("_")}


def _execution_cfg_for_arm(calib: dict[str, Any], arm_name: str) -> dict[str, Any]:
    by_arm = dict(calib.get("execution_by_arm") or {})
    if arm_name in by_arm:
        return {k: v for k, v in dict(by_arm[arm_name]).items() if not str(k).startswith("_")}
    exe = dict(calib.get("execution") or {})
    return {k: v for k, v in exe.items() if not str(k).startswith("_")}


def _connect_xarm(
    calib: dict[str, Any],
    robot_ip: str | None,
    *,
    arm_name: str | None = None,
) -> Any:
    from lerobot.robots.xarm import XArmConfig, XArmRobot

    robot_cfg = _robot_cfg_from_calib(calib, arm_name)
    if robot_ip:
        robot_cfg["robot_ip"] = robot_ip
    elif arm_name and not robot_cfg.get("robot_ip"):
        raise ValueError(f"No robot_ip for arm {arm_name!r} in calib")
    robot_cfg.setdefault("move_to_start_on_connect", False)
    # Perception does not need cameras on the robot handle; we open video2 separately.
    robot_cfg["cameras"] = {}

    config = XArmConfig(**robot_cfg)
    robot = XArmRobot(config)
    robot.connect()
    return robot


def _draw_image_overlay(
    image_rgb: np.ndarray,
    geometric_view: dict[str, Any],
    detections_px: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Draw 2D boxes / grasp points on the camera image for quick sanity checks."""
    vis = image_rgb.copy()
    if detections_px:
        for det in detections_px:
            xmin, ymin, xmax, ymax = [int(round(v)) for v in det["box_2d_px"]]
            u, v = [int(round(c)) for c in det["grasp_point_px"]]
            cv2.rectangle(vis, (xmin, ymin), (xmax, ymax), (0, 200, 0), 2)
            cv2.circle(vis, (u, v), 5, (255, 0, 0), -1)
            cv2.putText(
                vis,
                det["name"],
                (xmin, max(15, ymin - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 220, 0),
                1,
                cv2.LINE_AA,
            )
    else:
        # Fallback: project grasp_xy back is unavailable without inverse; just label names.
        for i, obj in enumerate(geometric_view.get("objects", [])):
            cv2.putText(
                vis,
                f"{i}:{obj['name']}",
                (10, 24 + 18 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
    return vis


def _serialize_geometric(geometric_view: dict[str, Any]) -> dict[str, Any]:
    objects = []
    for obj in geometric_view["objects"]:
        objects.append(
            {
                "name": obj["name"],
                "xy": list(obj["xy"]),
                "yaw": float(obj["yaw"]),
                "grasp_pose": list(obj["grasp_pose"]),
                "footprint_wkt": obj["footprint"].wkt,
            }
        )
    out = {
        "objects": objects,
        "free_space_wkt": geometric_view["free_space"].wkt,
        "table_polygon_wkt": geometric_view["table_polygon"].wkt,
        "workspaces_wkt": {k: v.wkt for k, v in geometric_view["workspaces"].items()},
    }
    ov = geometric_view.get("overlap")
    if ov is not None and hasattr(ov, "wkt") and not ov.is_empty:
        out["overlap_wkt"] = ov.wkt
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="xArm + /dev/video2 tabletop perception")
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--camera", type=str, default=None, help="Override camera path, e.g. /dev/video2")
    p.add_argument("--image", type=Path, default=None, help="Use a still RGB/BGR image instead of live camera")
    p.add_argument("--robot-ip", type=str, default=None)
    p.add_argument("--no-robot", action="store_true", help="Skip xArm connect (camera/VLM only)")
    p.add_argument("--instruction", type=str, default="detect all graspable objects on the table")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    p.add_argument("--prompt-file", type=Path, default=None, help="Optional prompt text with {instruction}")
    p.add_argument(
        "--mock-vlm",
        action="store_true",
        help="Skip Gemini and inject a small hardcoded detection list (pipeline dry-run)",
    )
    return p


def _mock_detections_for_scene() -> list[dict[str, Any]]:
    """Rough boxes for the current multi-xArm table scene (normalized 0-1000)."""
    return [
        {
            "name": "black_panther_plush",
            "box_2d": [420, 280, 720, 620],
            "grasp_point": [560, 450],
            "blocked_by": None,
        },
        {
            "name": "grey_plush_pair",
            "box_2d": [430, 620, 620, 820],
            "grasp_point": [520, 720],
            "blocked_by": None,
        },
        {
            "name": "quest_headset",
            "box_2d": [180, 620, 420, 900],
            "grasp_point": [300, 760],
            "blocked_by": None,
        },
        {
            "name": "pliers",
            "box_2d": [250, 120, 420, 280],
            "grasp_point": [330, 200],
            "blocked_by": None,
        },
    ]


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
    grasp_height = float(calib["grasp_height_m"])
    grasp_height_offsets = {
        str(k): float(v)
        for k, v in dict(calib.get("grasp_height_offset_m") or {}).items()
        if not str(k).startswith("_")
    }
    footprint_buffer = float(calib.get("footprint_buffer_m", 0.02))

    prompt = None
    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text()

    robot = None
    if not args.no_robot:
        print(f"Connecting xArm at {args.robot_ip or calib.get('robot', {}).get('robot_ip')} ...")
        robot = _connect_xarm(calib, args.robot_ip)
        print("xArm connected (no motion commanded).")
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
            )
        print(f"Image shape: {image.shape}")

        from tabletop_perception.vlm import parse_vlm_detections

        if args.mock_vlm:
            print("Using --mock-vlm (no Gemini call).")
            raw = _mock_detections_for_scene()
        else:
            perception = Perception(footprint_buffer_m=footprint_buffer, prompt=prompt)
            raw = perception.vlm_caller(image, args.instruction, prompt or "")
        detections = parse_vlm_detections(raw, image_hw=(image.shape[0], image.shape[1]))
        # Reuse the same VLM result (no second API call).
        affine = _table_xy_affine_from_calib(calib)
        if affine is not None:
            print("Using table_xy_affine correction from calib.")
        if grasp_height_offsets:
            print(f"Per-object grasp height offsets: {grasp_height_offsets}")
        perception_once = Perception(
            footprint_buffer_m=footprint_buffer,
            prompt=prompt,
            vlm_caller=lambda _img, _ins, _p: raw,
            table_xy_affine=affine,
            grasp_height_offsets_m=grasp_height_offsets,
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
        # Attach multi-arm overlap (workspace ∩) for viz / downstream mutex.
        try:
            from manipulation.coordination import overlap_from_workspaces

            ov_cfg = dict(calib.get("overlap_xy") or {})
            geometric_view = dict(geometric_view)
            geometric_view["overlap"] = overlap_from_workspaces(
                geometric_view["workspaces"],
                arms=ov_cfg.get("arms"),
                table_polygon=(
                    table_polygon if ov_cfg.get("clip_to_table", True) else None
                ),
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
        overlay = _draw_image_overlay(image, geometric_view, detections)
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        visualize_table_plane(
            geometric_view,
            show=False,
            save_path=table_path,
            title="xArm tabletop perception",
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
