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
Fanuc table calibration.

Teach a table-normal TCP on the pendant first (W≈±180, P≈0). This script
never changes WPR during calibration; WASD only jogs XY.

Seed (no motion) — record current TCP / taught WPR::

    cd examples
    python -m tabletop_perception.fanuc.calibrate_table_xy --seed --camera /dev/video0

Interactive multi-point XY affine (WASD only)::

    python -m tabletop_perception.fanuc.calibrate_table_xy \\
        --n-points 10 --camera /dev/video0 --object box

Hold-out hover test (uses saved affine, does not rewrite it)::

    python -m tabletop_perception.fanuc.calibrate_table_xy \\
        --verify --n-points 3 --camera /dev/video0 --vlm gemini

Tilt the overhead camera down first so the yellow diamond sits in the
lower half of the frame. Do not move the camera during the 10 points.
Cover NEAR / MID / FAR and left / center / right.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[2]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from tabletop_perception.calibrate_table_xy import (  # noqa: E402
    _base_to_table_xy,
    _pick_detection,
    _sample_image_h,
    _samples_path,
    _table_to_base_xy,
    _uv_zone,
    _write_affine_block,
)
from tabletop_perception.fanuc.jog_xy import jog_xy_wasd, move_fanuc_xyz  # noqa: E402
from tabletop_perception.fanuc.run_fanuc_live import (  # noqa: E402
    DEFAULT_CALIB,
    _connect_fanuc,
    _robot_cfg_from_calib,
)
from tabletop_perception.geometry import apply_table_xy_affine, fit_table_xy_affine, to_table  # noqa: E402
from tabletop_perception.perception import object_top_z_from_calib, resolve_object_top_z_m  # noqa: E402
from tabletop_perception.run_xarm_live import (  # noqa: E402
    _capture_from_camera,
    _execution_cfg_for_arm,
    _load_calib,
    _table_xy_affine_from_calib,
)
from tabletop_perception.vlm import call_detection_vlm, parse_vlm_detections  # noqa: E402


def _tcp_wpr(tcp: dict[str, float]) -> tuple[float, float, float]:
    return (float(tcp["w_deg"]), float(tcp["p_deg"]), float(tcp["r_deg"]))


def _workspace_from_true_samples(
    samples: list[dict], margin_mm: float = 50.0
) -> tuple[float, float, float, float] | None:
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


def _print_fit_residuals(samples: list[dict], a: np.ndarray, b: np.ndarray, *, title: str) -> None:
    if not samples:
        return
    print(f"\n========== {title} ==========")
    errs: list[float] = []
    for i, s in enumerate(samples, 1):
        pred = np.asarray(s["pred_table_xy"], dtype=float)
        true = np.asarray(s["true_table_xy"], dtype=float)
        hat = a @ pred + b
        e = (hat - true) * 1000.0
        nrm = float(np.hypot(e[0], e[1]))
        errs.append(nrm)
        uv = s.get("uv") or [0.0, 0.0]
        print(
            f"  [{i}] {s.get('name')} zone={_uv_zone(tuple(uv), _sample_image_h(s, 1080.0))} "
            f"true_uf=({s['true_base_mm'][0]:.1f},{s['true_base_mm'][1]:.1f}) "
            f"e=({e[0]:+.1f},{e[1]:+.1f}) |{nrm:.1f}| mm"
        )
    print(f"  n={len(errs)}  mean={float(np.mean(errs)):.1f} mm  max={max(errs):.1f} mm")


def _store_taught_wpr(calib: dict, tcp: dict[str, float]) -> list[float]:
    wpr = [round(tcp["w_deg"], 3), round(tcp["p_deg"], 3), round(tcp["r_deg"], 3)]
    exe = calib.setdefault("execution", {})
    exe["topdown_wpr_deg"] = wpr
    exe["use_object_yaw"] = False
    fanuc_exe = calib.setdefault("execution_by_arm", {}).setdefault("fanuc", {})
    fanuc_exe["topdown_wpr_deg"] = wpr
    fanuc_exe["use_object_yaw"] = False
    return wpr


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


def _write_seed(calib: dict, tcp: dict[str, float], *, camera: str, host: str) -> None:
    x_m, y_m, z_m = tcp["x_mm"] / 1000.0, tcp["y_mm"] / 1000.0, tcp["z_mm"] / 1000.0
    reach = max(0.45, (x_m**2 + y_m**2) ** 0.5 + 0.25)

    calib["camera"]["index_or_path"] = camera
    calib["measured_tcp"] = {
        "_comment": "Seed snapshot. Table frame = current Fanuc UF (mm→m). No affine yet.",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "host": host,
        **{k: round(v, 3) for k, v in tcp.items()},
    }
    calib["grasp_height_m"] = round(max(0.02, z_m), 3)
    calib["arm_workspaces_xy"]["fanuc"] = {
        "_role": f"Fanuc RMI @ {host}",
        "type": "semicircle",
        "center": [round(x_m, 3), round(y_m, 3)],
        "radius": round(reach, 3),
        "opening": "+x",
        "n_arc": 48,
    }
    half = max(0.35, reach)
    calib["table_polygon_xy"] = [
        [round(x_m - half, 3), round(y_m - half, 3)],
        [round(x_m + half, 3), round(y_m - half, 3)],
        [round(x_m + half, 3), round(y_m + half, 3)],
        [round(x_m - half, 3), round(y_m + half, 3)],
    ]

    exe = {
        "table_base_xy_m": [0.0, 0.0],
        "table_z_base_m": 0.0,
        "flip_x": False,
        "flip_y": False,
        "xy_offset_base_mm": [0.0, 0.0],
        "topdown_wpr_deg": [round(tcp["w_deg"], 3), round(tcp["p_deg"], 3), round(tcp["r_deg"], 3)],
        "use_object_yaw": False,
        "grasp_yaw_offset_rad": 0.0,
        "move_speed_mm_s": 80.0,
        "solver_margin_m": 0.03,
        "approach_offset_m": 0.08,
        "lift_offset_m": 0.08,
    }
    calib["execution"] = {"arm_name": "fanuc", **exe}
    calib.setdefault("execution_by_arm", {})["fanuc"] = dict(exe)


def _run_seed(args) -> int:
    calib = _load_calib(args.calib)
    host = args.robot_host or _robot_cfg_from_calib(calib, "fanuc").get("host")
    camera = args.camera or calib["camera"]["index_or_path"]
    print(f"Seed calibration (no motion). host={host} camera={camera}")
    robot = _connect_fanuc(calib, args.robot_host)
    try:
        tcp = _read_tcp(robot)
        print(
            f"TCP mm/deg: X={tcp['x_mm']:.1f} Y={tcp['y_mm']:.1f} Z={tcp['z_mm']:.1f} "
            f"W={tcp['w_deg']:.1f} P={tcp['p_deg']:.1f} R={tcp['r_deg']:.1f}"
        )
        _write_seed(calib, tcp, camera=str(camera), host=str(host))
        args.calib.write_text(json.dumps(calib, indent=2) + "\n")
        print(f"Wrote seed into {args.calib}")
        print("Convention: table XY (m) = Fanuc UF XY (mm) / 1000, table_base = (0, 0).")
        print(
            f"Recorded taught WPR as topdown_wpr_deg="
            f"[{tcp['w_deg']:.1f}, {tcp['p_deg']:.1f}, {tcp['r_deg']:.1f}]. "
            "Calibration will hold this pose and only jog XY."
        )
        print("Next: hover+jog affine with --n-points 10 (this WILL move the arm).")
        return 0
    finally:
        robot.disconnect()


def _run_points(args) -> int:
    verify = bool(getattr(args, "verify", False))
    if args.n_points is None:
        args.n_points = 3 if verify else 10
    if args.n_points < 1:
        raise SystemExit("--n-points must be >= 1")
    if not verify and not args.append and args.n_points < 3:
        raise SystemExit("--n-points must be >= 3 (or use --append)")

    calib = _load_calib(args.calib)
    cam_cfg = calib["camera"]
    camera_path = args.camera or cam_cfg["index_or_path"]
    k = np.asarray(calib["K"], dtype=float)
    t_ct = np.asarray(calib["T_cam_table"], dtype=float)
    exe = _execution_cfg_for_arm(calib, "fanuc")
    host = args.robot_host or _robot_cfg_from_calib(calib, "fanuc").get("host")
    if not host:
        raise SystemExit("No Fanuc host in calib")

    base_xy = tuple(float(v) for v in exe.get("table_base_xy_m", [0.0, 0.0]))
    flip_x = bool(exe.get("flip_x", False))
    flip_y = bool(exe.get("flip_y", False))
    xy_off = [float(v) for v in exe.get("xy_offset_base_mm", [0.0, 0.0])]
    wpr = tuple(float(v) for v in exe.get("topdown_wpr_deg", [180.0, 0.0, 0.0]))
    global_affine = _table_xy_affine_from_calib(calib)
    z_default, z_heights = object_top_z_from_calib(calib, arm="fanuc")
    if verify:
        if global_affine is None:
            raise SystemExit("--verify 需要先完成 table_xy_affine 标定")
        use_global = True
    else:
        use_global = bool(args.use_global_affine) and global_affine is not None

    print(f"arm=fanuc host={host} camera={camera_path}")
    image_h = float(cam_cfg["height"])
    print(
        f"capture={cam_cfg.get('width')}x{cam_cfg.get('height')} "
        f"fourcc={cam_cfg.get('fourcc') or 'MJPG'}"
    )
    print(f"base_xy={base_xy} flip_x={flip_x} flip_y={flip_y} calib_file_wpr={wpr}")
    if verify:
        print("精度抽检：检测 → 当前高度平移到仿射预测 XY（不改 Z/WPR）→ WASD 微调，残差就是误差。")
        print("不会改写 fanuc_overhead.json 里的 affine。")
    else:
        print("请先用示教器把末端摆成垂直桌面。标定只锁这个姿态，WASD 只动 XY，不改 yaw。")
        print(
            "相机请先手动向下压，让黄菱形和近端桌面在画面中下部，少拍窗户。"
            "支架固定后再采点，中途不要再动相机。"
        )
        print(
            "10 个点请铺开：NEAR(画面下)左/中/右，MID 左/中/右，"
            "FAR(画面上、靠近臂座)左/中，再补黄菱形附近 1–2 点。"
        )

    calib_samples_file = _samples_path(args.calib, "fanuc")
    workspace = None
    if calib_samples_file.exists():
        calib_samples = list(json.loads(calib_samples_file.read_text()).get("samples") or [])
        workspace = _workspace_from_true_samples(calib_samples)
        if verify and global_affine is not None and calib_samples:
            a, b = global_affine
            _print_fit_residuals(calib_samples, a, b, title="IN-SAMPLE FIT (标定 10 点)")

    if verify:
        samples_file = args.samples or args.calib.with_name("table_xy_calib_verify_fanuc.json")
        samples: list[dict] = []
    else:
        samples_file = args.samples or calib_samples_file
        samples = []
        if args.append:
            if not samples_file.exists():
                raise SystemExit(f"--append but no samples file: {samples_file}")
            samples = list(json.loads(samples_file.read_text()).get("samples") or [])
            print(f"Loaded {len(samples)} previous samples from {samples_file}")

    robot = _connect_fanuc(calib, args.robot_host)
    tcp0 = _read_tcp(robot)
    hold_wpr = _tcp_wpr(tcp0)
    taught = _store_taught_wpr(calib, tcp0)
    print(
        f"current TCP mm: X={tcp0['x_mm']:.1f} Y={tcp0['y_mm']:.1f} Z={tcp0['z_mm']:.1f} "
        f"WPR=({hold_wpr[0]:.1f}, {hold_wpr[1]:.1f}, {hold_wpr[2]:.1f})"
    )
    if abs(hold_wpr[1]) > 10.0:
        print(f"  注意：当前 P={hold_wpr[1]:.1f}°，末端可能不是垂直桌面。请先在示教器摆正后再继续。")
    print(f"  锁定示教姿态 topdown_wpr_deg={taught}，点动不改 WPR。")
    if not args.no_write and not verify:
        args.calib.write_text(json.dumps(calib, indent=2) + "\n")
    if args.hover_z_mm is not None:
        print(f"Note: --hover-z-mm={args.hover_z_mm} is ignored; jog keeps the current Z.")

    def _checkpoint() -> None:
        samples_file.write_text(
            json.dumps({"arm": "fanuc", "camera": str(camera_path), "samples": samples, "partial": True}, indent=2)
            + "\n"
        )
        print(f"  checkpoint {len(samples)} sample(s) → {samples_file}")

    interrupted = False
    try:
        for i in range(args.n_points):
            print()
            print(f"===== [{i + 1}/{args.n_points}] =====")
            det = None
            while det is None:
                print("摆好物体后，只按一次 Enter 拍照。现在不要按 WASD。")
                input(">>> 按 Enter 拍照：")
                image = _capture_from_camera(
                    camera_path,
                    width=int(cam_cfg["width"]),
                    height=int(cam_cfg["height"]),
                    fps=int(cam_cfg["fps"]),
                    fourcc=cam_cfg.get("fourcc") or "MJPG",
                )
                print(f"图像 {image.shape[1]}x{image.shape[0]}，正在调用 VLM…")
                raw = call_detection_vlm(image, args.instruction, model=args.model)
                dets = parse_vlm_detections(raw, image_hw=(image.shape[0], image.shape[1]))
                det = _pick_detection(dets, args.object)
            uv = tuple(float(v) for v in det["grasp_point_px"])
            image_h = float(image.shape[0])
            z_top = resolve_object_top_z_m(det["name"], z_heights, z_default)
            raw_t = to_table(uv, k, t_ct, z_plane=z_top)
            pred_t = apply_table_xy_affine(raw_t, *global_affine) if use_global else raw_t
            cmd_x, cmd_y = _table_to_base_xy(pred_t[0], pred_t[1], base_xy=base_xy, flip_x=flip_x, flip_y=flip_y)
            cmd_x += xy_off[0]
            cmd_y += xy_off[1]
            tcp = _read_tcp(robot)
            start_x, start_y = tcp["x_mm"], tcp["y_mm"]
            start_z = tcp["z_mm"]
            print(
                f"  detected={det['name']} uv={uv} zone={_uv_zone(uv, image_h)} "
                f"pred_table=({pred_t[0]:.3f},{pred_t[1]:.3f}) pred_uf_mm=({cmd_x:.1f},{cmd_y:.1f})"
            )
            print(
                f"  当前 TCP=({start_x:.1f}, {start_y:.1f}, {start_z:.1f}) "
                f"WPR=({tcp['w_deg']:.1f}, {tcp['p_deg']:.1f}, {tcp['r_deg']:.1f})（保持不变）"
            )
            if workspace is not None:
                print(
                    f"  calib workspace X=[{workspace[0]:.0f},{workspace[1]:.0f}] "
                    f"Y=[{workspace[2]:.0f},{workspace[3]:.0f}] mm"
                )
            if verify:
                in_ws = _inside_workspace(cmd_x, cmd_y, workspace)
                if not in_ws:
                    print("  预测 XY 超出标定包络，跳过这一点，避免大跨度 SystemFault。")
                    continue
                print(
                    f"  将保持 Z={start_z:.1f} 平移到预测 XY=({cmd_x:.1f},{cmd_y:.1f})，不改 WPR。"
                )
                confirm = input("  输入 go 平移，其它键跳过这一点：").strip().lower()
                if confirm != "go":
                    print("  已跳过。")
                    continue
                move_fanuc_xyz(
                    robot,
                    x_mm=cmd_x,
                    y_mm=cmd_y,
                    z_mm=start_z,
                    wpr_deg=hold_wpr,
                    speed=args.speed,
                )
                start_x, start_y = float(cmd_x), float(cmd_y)
                print("  用 WASD 微调到物体正上方，ok 后的残差就是这次精度。")
            else:
                print("  用 WASD 点到物体上方，ok 确认。")
            if args.type_delta:
                parts = input("  dx dy (相对当前 UF mm) > ").replace(",", " ").split()
                dx, dy = float(parts[0]), float(parts[1])
            else:
                dx, dy = jog_xy_wasd(
                    robot,
                    start_xy_mm=(start_x, start_y),
                    z_mm=start_z,
                    wpr_deg=hold_wpr,
                    speed=args.speed,
                    step_mm=args.step_mm,
                )
            true_base = (start_x + dx, start_y + dy)
            residual = (true_base[0] - cmd_x, true_base[1] - cmd_y)
            true_t = _base_to_table_xy(
                true_base[0] - xy_off[0],
                true_base[1] - xy_off[1],
                base_xy=base_xy,
                flip_x=flip_x,
                flip_y=flip_y,
            )
            samples.append(
                {
                    "arm": "fanuc",
                    "name": det["name"],
                    "uv": list(uv),
                    "image_hw": [int(image.shape[0]), int(image.shape[1])],
                    "z_plane": z_top,
                    "raw_table_xy": list(raw_t),
                    "pred_table_xy": list(pred_t),
                    "true_table_xy": list(true_t),
                    "cmd_base_mm": [cmd_x, cmd_y],
                    "start_base_mm": [start_x, start_y],
                    "delta_base_mm": [dx, dy],
                    "true_base_mm": list(true_base),
                    "residual_vs_pred_mm": list(residual),
                }
            )
            print(
                f"  jog Δ=({dx:+.1f},{dy:+.1f}) mm  residual_vs_pred=({residual[0]:+.1f},{residual[1]:+.1f}) mm "
                f"→ true_table=({true_t[0]:.3f},{true_t[1]:.3f})"
            )
            _checkpoint()
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n停止采集。已确认 {len(samples)} 个点仍会保存。")
    finally:
        robot.disconnect()

    if verify:
        if not samples:
            print("没有确认的抽检点。")
            return 1 if interrupted else 0
        res = np.asarray([s["residual_vs_pred_mm"] for s in samples], dtype=float)
        nrm = np.linalg.norm(res, axis=1)
        rmse = float(np.sqrt(np.mean(np.sum(res**2, axis=1))))
        print("\n========== HOLD-OUT VERIFY ==========")
        for i, s in enumerate(samples, 1):
            e = s["residual_vs_pred_mm"]
            print(
                f"  [{i}] {s.get('name')} zone={_uv_zone(tuple(s['uv']), _sample_image_h(s, image_h))} "
                f"pred_uf=({s['cmd_base_mm'][0]:.1f},{s['cmd_base_mm'][1]:.1f}) "
                f"true_uf=({s['true_base_mm'][0]:.1f},{s['true_base_mm'][1]:.1f}) "
                f"e=({e[0]:+.1f},{e[1]:+.1f}) |{float(np.hypot(e[0], e[1])):.1f}| mm"
            )
        print(f"  n={len(samples)}  rmse={rmse:.1f} mm  max={float(np.max(nrm)):.1f} mm")
        samples_file.write_text(
            json.dumps(
                {
                    "arm": "fanuc",
                    "camera": str(camera_path),
                    "mode": "verify",
                    "samples": samples,
                    "rmse_mm": rmse,
                    "max_err_mm": float(np.max(nrm)),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Wrote {samples_file}（未改写 affine）")
        return 0

    if len(samples) < 3:
        if samples:
            _checkpoint()
        print(f"Need at least 3 samples to fit affine, got {len(samples)}.")
        return 1 if interrupted else 0

    pred = [tuple(s["pred_table_xy"]) for s in samples]
    true = [tuple(s["true_table_xy"]) for s in samples]
    a, b, metrics = fit_table_xy_affine(pred, true)
    print("\n========== TABLE XY AFFINE ==========")
    print(
        f"arm=fanuc  n={int(metrics['n'])}  "
        f"rmse={metrics['rmse_m'] * 1000:.1f} mm  max={metrics['max_err_m'] * 1000:.1f} mm"
    )
    print(f"A = {a.tolist()}")
    print(f"b = {b.tolist()}")
    n_far = sum(
        1 for s in samples if _uv_zone(tuple(s["uv"]), _sample_image_h(s, image_h)) == "FAR (image top)"
    )
    n_near = sum(
        1
        for s in samples
        if _uv_zone(tuple(s["uv"]), _sample_image_h(s, image_h)) == "NEAR (image bottom)"
    )
    print(
        f"Coverage: n={len(samples)}  far/top={n_far}  mid="
        f"{len(samples) - n_far - n_near}  near/bottom={n_near}"
    )
    if n_far < 2 or n_near < 2:
        print("覆盖不够：FAR 和 NEAR 都至少要 2 个点，否则桌面远端会外推不准。")
    samples_file.write_text(
        json.dumps(
            {
                "arm": "fanuc",
                "camera": str(camera_path),
                "samples": samples,
                "partial": False,
                "A": a.tolist(),
                "b": b.tolist(),
                "metrics": metrics,
            },
            indent=2,
        )
        + "\n"
    )
    if args.no_write:
        return 0
    block = _write_affine_block(a, b, comment="true_table = A @ pred_table + b (metres). pred = to_table(uv).")
    calib["table_xy_affine"] = block
    calib.setdefault("execution_by_arm", {}).setdefault("fanuc", {})["table_xy_affine"] = block
    args.calib.write_text(json.dumps(calib, indent=2) + "\n")
    print(f"Wrote {args.calib}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Fanuc table XY calibration")
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--camera", type=str, default="/dev/video0")
    p.add_argument("--robot-host", "--robot-ip", dest="robot_host", type=str, default=None)
    p.add_argument("--seed", action="store_true", help="Read current TCP and write UF=table seed (no motion)")
    p.add_argument("--object", type=str, default=None)
    p.add_argument(
        "--n-points",
        type=int,
        default=None,
        help="Calib default 10; --verify default 3.",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Hover to saved affine XY and measure WASD residual. Does not rewrite A.",
    )
    p.add_argument("--append", action="store_true")
    p.add_argument("--samples", type=Path, default=None)
    p.add_argument("--hover-z-mm", type=float, default=None)
    p.add_argument("--speed", type=float, default=40.0)
    p.add_argument("--step-mm", type=float, default=5.0)
    p.add_argument("--type-delta", action="store_true")
    p.add_argument("--use-global-affine", action="store_true")
    p.add_argument("--instruction", type=str, default="detect all graspable objects on the table")
    p.add_argument(
        "--model",
        "--vlm",
        dest="model",
        default="gemini",
        help="Base VLM: gemini, gpt-6, or cosmos / cosmos3-nano.",
    )
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args()
    if args.seed or (args.n_points is not None and args.n_points <= 0):
        return _run_seed(args)
    return _run_points(args)


if __name__ == "__main__":
    raise SystemExit(main())
