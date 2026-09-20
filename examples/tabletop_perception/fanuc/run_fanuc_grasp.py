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

"""Conservative one-shot Fanuc grasp: perceive → confirm → approach → down → close → lift."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[2]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from tabletop_perception.calibrate_table_xy import (  # noqa: E402
    _pick_detection,
    _samples_path,
    _table_to_base_xy,
    _uv_zone,
)
from tabletop_perception.fanuc.jog_xy import move_fanuc_xyz  # noqa: E402
from tabletop_perception.fanuc.run_fanuc_live import (  # noqa: E402
    DEFAULT_CALIB,
    _connect_fanuc,
)
from tabletop_perception.geometry import (  # noqa: E402
    apply_table_xy_affine,
    to_table,
    yaw_from_axis,
)
from tabletop_perception.perception import (  # noqa: E402
    Perception,
    object_top_z_from_calib,
    resolve_grasp_height_m,
    resolve_object_top_z_m,
)
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
from tabletop_perception.vlm import call_detection_vlm, parse_vlm_detections  # noqa: E402

DEFAULT_RUNS = Path(__file__).resolve().parents[1] / "runs"

_TABLE_CONTACT_Z_MM = -320.0
_Z_FLOOR_MM = _TABLE_CONTACT_Z_MM


def _read_tcp(robot) -> dict[str, float]:
    obs = robot.get_observation()
    return {
        "x_mm": float(obs["j0"]),
        "y_mm": float(obs["j1"]),
        "z_mm": float(obs["j2"]),
        "w_deg": float(obs["j3"]),
        "p_deg": float(obs["j4"]),
        "r_deg": float(obs["j5"]),
    }


def _clamp_z(z_mm: float) -> float:
    if z_mm < _Z_FLOOR_MM:
        print(f"  Z={z_mm:.1f} mm 低于安全下限 {_Z_FLOOR_MM:.1f}，夹到下限。")
        return _Z_FLOOR_MM
    return float(z_mm)


def _set_gripper(robot, tcp: dict[str, float], wpr: tuple[float, float, float], *, close: bool) -> None:
    move_fanuc_xyz(
        robot,
        x_mm=tcp["x_mm"],
        y_mm=tcp["y_mm"],
        z_mm=tcp["z_mm"],
        wpr_deg=wpr,
        speed=20.0,
        gripper=1.0 if close else 0.0,
    )
    time.sleep(0.6)


def _workspace_from_samples(samples_file: Path, margin_mm: float = 50.0) -> tuple[float, float, float, float] | None:
    if not samples_file.exists():
        return None
    samples = list(json.loads(samples_file.read_text()).get("samples") or [])
    pts = [s.get("true_base_mm") for s in samples if s.get("true_base_mm")]
    if len(pts) < 2:
        return None
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return (min(xs) - margin_mm, max(xs) + margin_mm, min(ys) - margin_mm, max(ys) + margin_mm)


def _inside_workspace(x_mm: float, y_mm: float, box: tuple[float, float, float, float] | None) -> bool:
    if box is None:
        return True
    x0, x1, y0, y1 = box
    return x0 <= x_mm <= x1 and y0 <= y_mm <= y1


def _wrap_signed_deg(angle: float) -> float:
    """Wrap to (-180, 180]. Fanuc WPR R is not valid at 268°."""
    return ((float(angle) + 180.0) % 360.0) - 180.0


def _angle_diff_deg(a: float, b: float) -> float:
    return abs(_wrap_signed_deg(float(a) - float(b)))


# Parallel-jaw grasp is 180°-periodic. Do not clamp yaw to a 150° box.
# With W≈178 / P≈-1.6 we have only seen SystemFault in two pockets:
#   R ≈ +80…+90 (in-place wrist flip) and |R| ≳ 160.
_R_SINGULAR_LO_DEG = 70.0
_R_SINGULAR_HI_DEG = 100.0
_R_ABS_LIMIT_DEG = 155.0
_MIN_YAW_PATH_MM = 40.0


def _r_in_bad_pocket(r: float) -> bool:
    wrapped = _wrap_signed_deg(r)
    if abs(wrapped) > _R_ABS_LIMIT_DEG:
        return True
    return _R_SINGULAR_LO_DEG <= wrapped <= _R_SINGULAR_HI_DEG


def _r_equivalents(target_r: float) -> list[float]:
    unique: list[float] = []
    for raw in (target_r, target_r + 180.0, target_r - 180.0):
        r = _wrap_signed_deg(raw)
        if not any(_angle_diff_deg(r, seen) < 1e-6 for seen in unique):
            unique.append(r)
    return unique


def _pick_reachable_r(target_r: float, current_r: float, taught_r: float) -> float:
    """Pick R or R±180 that avoids known-bad wrist pockets.

    Object heading only needs 180° of unique R. Prefer the equivalent
    closest to the live wrist, not a hard [-110, 40] box.
    """
    cands = _r_equivalents(target_r)
    ok = [r for r in cands if not _r_in_bad_pocket(r)]
    pool = ok or cands
    chosen = min(
        pool,
        key=lambda r: (
            _angle_diff_deg(r, current_r),
            _angle_diff_deg(r, taught_r),
            abs(r),
        ),
    )
    print(
        f"[yaw] R cands={[round(r, 1) for r in cands]}  "
        f"bad_pockets=[±>{_R_ABS_LIMIT_DEG:.0f}, "
        f"{_R_SINGULAR_LO_DEG:.0f}…{_R_SINGULAR_HI_DEG:.0f}]  "
        f"ok={[round(r, 1) for r in ok]}  pick={chosen:.1f}"
        + ("  (no pocket-free equivalent)" if not ok else ""),
        flush=True,
    )
    return chosen


def _yaw_via_xy(
    x0: float,
    y0: float,
    workspace: tuple[float, float, float, float] | None,
) -> tuple[float, float] | None:
    for dx, dy in ((40.0, 0.0), (-40.0, 0.0), (0.0, 40.0), (0.0, -40.0)):
        vx, vy = x0 + dx, y0 + dy
        if _inside_workspace(vx, vy, workspace):
            return vx, vy
    return None


def _object_table_yaw_rad(
    det: dict,
    *,
    k: np.ndarray,
    t_ct: np.ndarray,
    z_top: float,
    affine: tuple[np.ndarray, np.ndarray] | None,
) -> float | None:
    axis = det.get("long_axis_px")
    print(f"[yaw] det={det.get('name')!r} long_axis_px={axis}", flush=True)
    if not axis or len(axis) != 2:
        print("[yaw] skip: missing long_axis_px", flush=True)
        return None
    p0 = to_table(tuple(axis[0]), k, t_ct, z_plane=z_top)
    p1 = to_table(tuple(axis[1]), k, t_ct, z_plane=z_top)
    print(f"[yaw] axis table raw  p0=({p0[0]:.4f},{p0[1]:.4f}) p1=({p1[0]:.4f},{p1[1]:.4f})", flush=True)
    if affine is not None:
        p0 = apply_table_xy_affine(p0, affine[0], affine[1])
        p1 = apply_table_xy_affine(p1, affine[0], affine[1])
        print(f"[yaw] axis table aff  p0=({p0[0]:.4f},{p0[1]:.4f}) p1=({p1[0]:.4f},{p1[1]:.4f})", flush=True)
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    yaw_t = yaw_from_axis((p0[0], p0[1]), (p1[0], p1[1]))
    print(
        f"[yaw] axis_len={length * 1000:.1f} mm  "
        f"yaw_t={None if yaw_t is None else f'{math.degrees(yaw_t):.2f} deg'}",
        flush=True,
    )
    return yaw_t


def _table_yaw_to_fanuc_r_deg(
    yaw_t: float,
    *,
    flip_x: bool,
    flip_y: bool,
    offset_rad: float,
    taught_r_deg: float,
) -> float:
    yaw_plus = float(yaw_t) + float(offset_rad)
    # Do not wrap to ±90° here. wrap_half_pi(yaw+90°) maps a Y-aligned object
    # (typical after this cell's affine) back onto taught R, so the wrist never turns.
    c, s = math.cos(yaw_plus), math.sin(yaw_plus)
    if flip_x:
        c = -c
    if flip_y:
        s = -s
    delta = math.degrees(math.atan2(s, c))
    target = ((float(taught_r_deg) + delta + 180.0) % 360.0) - 180.0
    print(
        f"[yaw] yaw_t={math.degrees(yaw_t):.2f}° + offset={math.degrees(offset_rad):.1f}° "
        f"= {math.degrees(yaw_plus):.2f}°  delta={delta:.2f}°  target_R={target:.2f}° "
        f"(no half-pi wrap)",
        flush=True,
    )
    return target


def _facing_wpr(
    det: dict,
    taught_wpr: tuple[float, float, float],
    *,
    k: np.ndarray,
    t_ct: np.ndarray,
    z_top: float,
    affine: tuple[np.ndarray, np.ndarray] | None,
    flip_x: bool,
    flip_y: bool,
    offset_rad: float,
    current_r_deg: float,
) -> tuple[tuple[float, float, float], str]:
    w, p, taught_r = taught_wpr
    print(
        f"[yaw] taught_wpr=({w:.2f},{p:.2f},{taught_r:.2f}) "
        f"current_R={current_r_deg:.2f} offset={math.degrees(offset_rad):.1f}° "
        f"flip_x={flip_x} flip_y={flip_y}",
        flush=True,
    )
    yaw_t = _object_table_yaw_rad(det, k=k, t_ct=t_ct, z_top=z_top, affine=affine)
    if yaw_t is None:
        print("[yaw] no usable long axis → keep taught R", flush=True)
        return (w, p, taught_r), "no long_axis, keep taught R"
    target_r = _table_yaw_to_fanuc_r_deg(
        yaw_t,
        flip_x=flip_x,
        flip_y=flip_y,
        offset_rad=offset_rad,
        taught_r_deg=taught_r,
    )
    r = _pick_reachable_r(target_r, current_r_deg, taught_r)
    delta = _wrap_signed_deg(r - current_r_deg)
    print(
        f"[yaw] target_R={target_r:.2f} chosen_R={r:.2f} "
        f"from live {current_r_deg:.1f} ΔR={delta:.2f}",
        flush=True,
    )
    if abs(r) > 180.0 + 1e-6:
        raise RuntimeError(f"Fanuc R={r:.1f} out of (-180, 180], refuse to send")
    return (w, p, r), (
        f"long-axis yaw={math.degrees(yaw_t):.1f}° → R={r:.1f}° "
        f"(perp. long edge, offset={math.degrees(offset_rad):.0f}°)"
    )


def _dump_perceive_dir(
    run_dir: Path,
    *,
    image: np.ndarray,
    instruction: str,
    raw: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    symbolic: dict[str, Any],
    geometric: dict[str, Any],
) -> Path:
    """Same bundle as run_fanuc_live / manipulation perceive_NN."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "instruction.txt").write_text(instruction + "\n")
    (run_dir / "vlm_raw.json").write_text(json.dumps(raw, indent=2))
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
        title=f"Fanuc grasp {run_dir.name}",
    )
    return run_dir


def _mirror_latest(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "instruction.txt",
        "vlm_raw.json",
        "symbolic_view.json",
        "geometric_view.json",
        "capture_rgb.png",
        "capture_overlay.png",
        "table_plane.png",
    ):
        path = src / name
        if path.exists():
            (dest / name).write_bytes(path.read_bytes())


def main() -> int:
    p = argparse.ArgumentParser(description="Conservative Fanuc single grasp")
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--camera", type=str, default="/dev/video0")
    p.add_argument("--robot-host", dest="robot_host", type=str, default=None)
    p.add_argument("--object", type=str, default=None)
    p.add_argument("--instruction", type=str, default="detect all graspable objects on the table")
    p.add_argument("--speed", type=float, default=40.0)
    p.add_argument("--descend-speed", type=float, default=20.0)
    p.add_argument("--yaw-speed", type=float, default=10.0)
    p.add_argument(
        "--model",
        "--vlm",
        dest="model",
        default="gemini",
        help="Base VLM: gemini, gpt-6, or cosmos / cosmos3-nano.",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_RUNS)
    args = p.parse_args()

    calib = _load_calib(args.calib)
    cam_cfg = calib["camera"]
    camera_path = args.camera or cam_cfg["index_or_path"]
    k = np.asarray(calib["K"], dtype=float)
    t_ct = np.asarray(calib["T_cam_table"], dtype=float)
    exe = _execution_cfg_for_arm(calib, "fanuc")
    affine = _table_xy_affine_from_calib(calib)
    if affine is None:
        raise SystemExit("No table_xy_affine in calib. Run calibrate_table_xy first.")

    base_xy = tuple(float(v) for v in exe.get("table_base_xy_m", [0.0, 0.0]))
    flip_x = bool(exe.get("flip_x", False))
    flip_y = bool(exe.get("flip_y", False))
    xy_off = [float(v) for v in exe.get("xy_offset_base_mm", [0.0, 0.0])]
    wpr = tuple(float(v) for v in exe.get("topdown_wpr_deg", [180.0, 0.0, 0.0]))
    yaw_offset = float(exe.get("grasp_yaw_offset_rad", 0.0))
    table_z = float(exe.get("table_z_base_m", -0.320))
    approach = float(exe.get("approach_offset_m", 0.08))
    lift = float(exe.get("lift_offset_m", 0.08))
    grasp_h = float(calib["grasp_height_m"])
    grasp_off = {
        str(key): float(val)
        for key, val in dict(calib.get("grasp_height_offset_m") or {}).items()
        if not str(key).startswith("_")
    }
    z_default, z_heights = object_top_z_from_calib(calib)
    table_polygon = _polygon_from_xy(calib["table_polygon_xy"])
    arm_workspaces = {
        name: _workspace_from_spec(spec) for name, spec in calib["arm_workspaces_xy"].items()
    }
    footprint_buffer = float(calib.get("footprint_buffer_m", 0.02))
    stamp_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    stamp_dir.mkdir(parents=True, exist_ok=True)
    perceive_n = 0

    print(f"calib={args.calib} camera={camera_path}")
    print(f"logs → {stamp_dir}")
    print(
        f"table_z_base={table_z * 1000:.1f} mm  grasp_height={grasp_h * 1000:.1f} mm  "
        f"approach={approach * 1000:.1f} mm  Z_floor={_Z_FLOOR_MM:.1f} mm"
    )
    print(f"taught WPR={wpr}  yaw_offset={math.degrees(yaw_offset):.1f}°", flush=True)
    print("示教器请保持可急停。输入 go 后才会运动。")

    robot = _connect_fanuc(calib, args.robot_host)
    try:
        tcp = _read_tcp(robot)
        print(
            f"current TCP=({tcp['x_mm']:.1f}, {tcp['y_mm']:.1f}, {tcp['z_mm']:.1f}) "
            f"WPR=({tcp['w_deg']:.1f}, {tcp['p_deg']:.1f}, {tcp['r_deg']:.1f})"
        )

        det = None
        while det is None:
            print("摆好物体后，按 Enter 拍照。")
            input(">>> 按 Enter 拍照：")
            image = _capture_from_camera(
                camera_path,
                width=int(cam_cfg["width"]),
                height=int(cam_cfg["height"]),
                fps=int(cam_cfg["fps"]),
                fourcc=cam_cfg.get("fourcc") or "MJPG",
            )
            print(f"图像 {image.shape[1]}x{image.shape[0]}，正在调用 VLM ({args.model})…")
            raw = call_detection_vlm(image, args.instruction, model=args.model)
            dets = parse_vlm_detections(raw, image_hw=(image.shape[0], image.shape[1]))
            perception_once = Perception(
                footprint_buffer_m=footprint_buffer,
                prompt=None,
                vlm_caller=lambda _img, _ins, _p: raw,
                table_xy_affine=affine,
                grasp_height_offsets_m=grasp_off,
                object_top_z_m=z_heights,
                object_top_z_m_default=z_default,
                model=str(args.model),
            )
            symbolic_view, geometric_view = perception_once(
                image=image,
                K=k,
                T_cam_table=t_ct,
                grasp_height=grasp_h,
                instruction=args.instruction,
                arm_workspaces=arm_workspaces,
                table_polygon=table_polygon,
            )
            dets = perception_once.last_detections or dets
            perceive_n += 1
            perceive_dir = stamp_dir / f"perceive_{perceive_n:02d}"
            _dump_perceive_dir(
                perceive_dir,
                image=image,
                instruction=args.instruction,
                raw=list(raw),
                detections=dets,
                symbolic=symbolic_view,
                geometric=geometric_view,
            )
            _mirror_latest(perceive_dir, stamp_dir)
            print(f"Wrote perception logs → {perceive_dir}", flush=True)
            for i, d in enumerate(dets):
                print(
                    f"[yaw] det[{i}] name={d.get('name')} "
                    f"grasp={d.get('grasp_point_px')} long_axis_px={d.get('long_axis_px')}",
                    flush=True,
                )
            det = _pick_detection(dets, args.object)

        uv = tuple(float(v) for v in det["grasp_point_px"])
        z_top = resolve_object_top_z_m(det["name"], z_heights, z_default)
        raw_t = to_table(uv, k, t_ct, z_plane=z_top)
        pred_t = apply_table_xy_affine(raw_t, *affine)
        cmd_x, cmd_y = _table_to_base_xy(pred_t[0], pred_t[1], base_xy=base_xy, flip_x=flip_x, flip_y=flip_y)
        cmd_x += xy_off[0]
        cmd_y += xy_off[1]
        z_t = resolve_grasp_height_m(det["name"], grasp_h, grasp_off)
        grasp_z = _clamp_z((table_z + z_t) * 1000.0)
        approach_z = _clamp_z(grasp_z + approach * 1000.0)
        lift_z = _clamp_z(grasp_z + lift * 1000.0)
        face_wpr, yaw_note = _facing_wpr(
            det,
            wpr,
            k=k,
            t_ct=t_ct,
            z_top=z_top,
            affine=affine,
            flip_x=flip_x,
            flip_y=flip_y,
            offset_rad=yaw_offset,
            current_r_deg=tcp["r_deg"],
        )

        workspace = _workspace_from_samples(_samples_path(args.calib, "fanuc"))
        print()
        print("========== GRASP PLAN ==========")
        print(f"  object={det['name']} uv={uv} zone={_uv_zone(uv, image.shape[0])}")
        print(f"  pred_table=({pred_t[0]:.3f}, {pred_t[1]:.3f}) m")
        print(f"  target UF XY=({cmd_x:.1f}, {cmd_y:.1f}) mm")
        if workspace is not None:
            print(
                f"  calib workspace X=[{workspace[0]:.0f},{workspace[1]:.0f}] "
                f"Y=[{workspace[2]:.0f},{workspace[3]:.0f}] mm"
            )
        print(f"  taught WPR=({wpr[0]:.1f}, {wpr[1]:.1f}, {wpr[2]:.1f})", flush=True)
        print(f"  grasp  WPR=({face_wpr[0]:.1f}, {face_wpr[1]:.1f}, {face_wpr[2]:.1f})  {yaw_note}", flush=True)
        print(
            f"  [yaw] will send R={face_wpr[2]:.2f} (taught R={wpr[2]:.2f}, "
            f"live R={tcp['r_deg']:.2f}, Δ={face_wpr[2] - tcp['r_deg']:+.2f})",
            flush=True,
        )
        print(f"  travel/approach Z={approach_z:.1f} mm")
        print(f"  grasp    Z={grasp_z:.1f} mm  (table contact {_TABLE_CONTACT_Z_MM:.1f})")
        print(f"  lift     Z={lift_z:.1f} mm")
        print("动作：张开 → 当前高度一次平移 → 再下降 → 闭合 → 抬起。不放置。")
        in_ws = _inside_workspace(cmd_x, cmd_y, workspace)
        plan = {
            "object": det["name"],
            "uv": list(uv),
            "zone": _uv_zone(uv, image.shape[0]),
            "pred_table_m": [pred_t[0], pred_t[1]],
            "target_uf_xy_mm": [cmd_x, cmd_y],
            "workspace_mm": list(workspace) if workspace is not None else None,
            "inside_workspace": in_ws,
            "taught_wpr_deg": list(wpr),
            "grasp_wpr_deg": list(face_wpr),
            "yaw_note": yaw_note,
            "travel_z_mm": approach_z,
            "grasp_z_mm": grasp_z,
            "lift_z_mm": lift_z,
            "vlm": args.model,
            "perceive_dir": str(stamp_dir / f"perceive_{perceive_n:02d}"),
        }
        (stamp_dir / "grasp_plan.json").write_text(json.dumps(plan, indent=2))
        print(f"Wrote grasp plan → {stamp_dir / 'grasp_plan.json'}", flush=True)
        if not in_ws:
            print("目标 XY 超出标定包络，拒绝自动运动，避免再触发 SystemFault。")
            print("请换一个更靠近标定区域的物体，或再补几个标定的点。")
            return 1
        confirm = input("输入 go 开始运动，其它键取消：").strip().lower()
        if confirm != "go":
            print("已取消。")
            return 0

        tcp = _read_tcp(robot)
        travel_z = max(float(tcp["z_mm"]), float(approach_z))
        hold_wpr = (tcp["w_deg"], tcp["p_deg"], tcp["r_deg"])
        print("1/6 张开夹爪（先不转腕）")
        _set_gripper(robot, tcp, hold_wpr, close=False)
        tcp = _read_tcp(robot)

        if abs(tcp["z_mm"] - travel_z) > 1.0:
            print(f"2/6 先到平移高度 {travel_z:.1f}（当前 XY，不转腕）")
            move_fanuc_xyz(
                robot,
                x_mm=tcp["x_mm"],
                y_mm=tcp["y_mm"],
                z_mm=travel_z,
                wpr_deg=hold_wpr,
                speed=args.speed,
            )
            tcp = _read_tcp(robot)

        path_mm = math.hypot(cmd_x - tcp["x_mm"], cmd_y - tcp["y_mm"])
        delta_r = _wrap_signed_deg(face_wpr[2] - tcp["r_deg"])
        travel_speed = args.yaw_speed if abs(delta_r) > 15.0 and path_mm < 80.0 else args.speed
        waypoints: list[tuple[float, float]] = []
        if abs(delta_r) > 5.0 and path_mm < _MIN_YAW_PATH_MM:
            via = _yaw_via_xy(tcp["x_mm"], tcp["y_mm"], workspace)
            if via is not None:
                waypoints.append(via)
                print(
                    f"[yaw] path {path_mm:.0f} mm too short for ΔR={delta_r:.1f}°, "
                    f"via=({via[0]:.1f},{via[1]:.1f})",
                    flush=True,
                )
        waypoints.append((cmd_x, cmd_y))
        print(
            f"3/6 平移中连续转腕 → ({cmd_x:.1f}, {cmd_y:.1f}) Z={travel_z:.1f} "
            f"R={tcp['r_deg']:.1f}→{face_wpr[2]:.1f} @ {travel_speed:.0f} mm/s"
        )
        for wx, wy in waypoints:
            move_fanuc_xyz(
                robot,
                x_mm=wx,
                y_mm=wy,
                z_mm=travel_z,
                wpr_deg=face_wpr,
                speed=travel_speed,
            )
        after_xy = _read_tcp(robot)
        print(
            f"[yaw] after travel TCP WPR=({after_xy['w_deg']:.2f},{after_xy['p_deg']:.2f},"
            f"{after_xy['r_deg']:.2f})  commanded_R={face_wpr[2]:.2f} "
            f"live_ΔR={((after_xy['r_deg'] - face_wpr[2] + 180) % 360) - 180:.2f}",
            flush=True,
        )

        if abs(travel_z - approach_z) > 1.0:
            print(f"4/6 降到接近高度 {approach_z:.1f}")
            move_fanuc_xyz(
                robot,
                x_mm=cmd_x,
                y_mm=cmd_y,
                z_mm=approach_z,
                wpr_deg=face_wpr,
                speed=args.descend_speed,
            )
        print(f"5/6 下降到抓取高度 {grasp_z:.1f}")
        move_fanuc_xyz(
            robot,
            x_mm=cmd_x,
            y_mm=cmd_y,
            z_mm=grasp_z,
            wpr_deg=face_wpr,
            speed=args.descend_speed,
        )
        print("6/6 闭合并抬起")
        _set_gripper(
            robot,
            {"x_mm": cmd_x, "y_mm": cmd_y, "z_mm": grasp_z},
            face_wpr,
            close=True,
        )
        move_fanuc_xyz(
            robot,
            x_mm=cmd_x,
            y_mm=cmd_y,
            z_mm=lift_z,
            wpr_deg=face_wpr,
            speed=args.speed,
        )
        now = _read_tcp(robot)
        print(
            f"完成。TCP=({now['x_mm']:.1f}, {now['y_mm']:.1f}, {now['z_mm']:.1f}) "
            f"WPR=({now['w_deg']:.1f}, {now['p_deg']:.1f}, {now['r_deg']:.1f})  "
            f"commanded={face_wpr}  物体应还在夹爪里。",
            flush=True,
        )
        return 0
    finally:
        robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
