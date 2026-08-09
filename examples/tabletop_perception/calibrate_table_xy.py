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
Multi-point table XY calibration (fixes position-dependent grasp bias).

Place the SAME object at ≥3 spread-out spots. At each spot the arm hovers at
the predicted table XY; WASD-jog onto the object and confirm
(w=+X上, s=-X下, a=+Y左, d=-Y右).

Robot1 (shared camera affine)::

    python -m tabletop_perception.calibrate_table_xy \\
      --arm xarm --object black_plush_bear --n-points 4

Robot2 (per-arm residual affine; does NOT overwrite Robot1)::

    python -m tabletop_perception.calibrate_table_xy \\
      --arm xarm2 --object black_plush_bear --n-points 4 --hover-z-mm 320
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from manipulation.jog_xy import jog_xy_wasd
from tabletop_perception.geometry import (
    apply_table_xy_affine,
    fit_table_xy_affine,
    to_table,
)
from tabletop_perception.run_xarm_live import (
    DEFAULT_CALIB,
    _capture_from_camera,
    _connect_xarm,
    _execution_cfg_for_arm,
    _load_calib,
    _robot_cfg_from_calib,
    _table_xy_affine_from_calib,
)
from tabletop_perception.vlm import call_gemini_robotics_er, parse_vlm_detections


def _table_to_base_xy(
    x_t: float,
    y_t: float,
    *,
    base_xy: tuple[float, float],
    flip_x: bool,
    flip_y: bool,
) -> tuple[float, float]:
    x_b = (x_t - base_xy[0]) * 1000.0
    y_b = (y_t - base_xy[1]) * 1000.0
    if flip_x:
        x_b = -x_b
    if flip_y:
        y_b = -y_b
    return x_b, y_b


def _base_to_table_xy(
    x_b_mm: float,
    y_b_mm: float,
    *,
    base_xy: tuple[float, float],
    flip_x: bool,
    flip_y: bool,
) -> tuple[float, float]:
    if flip_x:
        x_b_mm = -x_b_mm
    if flip_y:
        y_b_mm = -y_b_mm
    return x_b_mm / 1000.0 + base_xy[0], y_b_mm / 1000.0 + base_xy[1]


def _clear(arm) -> None:
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.2)


def _parse_offset(line: str) -> tuple[float, float]:
    parts = line.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("Need two numbers: dx dy  (base-frame mm)")
    return float(parts[0]), float(parts[1])


def _pick_detection(dets: list[dict], object_name: str | None) -> dict:
    if not dets:
        raise RuntimeError("VLM returned no detections")
    if object_name:
        key = object_name.lower().replace(" ", "_")
        matches = [
            d for d in dets if key in d["name"].lower().replace(" ", "_")
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"Multiple matches for {object_name!r}:")
            dets = matches
        else:
            print(
                f"Object {object_name!r} not found; pick from detections "
                f"(or Ctrl-C and re-run with --object <name>):"
            )
    if len(dets) == 1:
        print(f"Using sole detection: {dets[0]['name']}")
        return dets[0]
    print("Detections:")
    for i, d in enumerate(dets):
        print(f"  [{i}] {d['name']} grasp_px={d['grasp_point_px']}")
    idx = int(input("Pick index > ").strip())
    return dets[idx]


def _write_affine_block(a: np.ndarray, b: np.ndarray, *, comment: str) -> dict:
    return {
        "_comment": comment,
        "A": [
            [round(float(a[0, 0]), 6), round(float(a[0, 1]), 6)],
            [round(float(a[1, 0]), 6), round(float(a[1, 1]), 6)],
        ],
        "b": [round(float(b[0]), 6), round(float(b[1]), 6)],
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-point table XY affine calibration")
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--arm", type=str, default="xarm", help="xarm | xarm2")
    p.add_argument("--camera", type=str, default=None)
    p.add_argument("--robot-ip", type=str, default=None)
    p.add_argument("--object", type=str, default=None, help="Target object name substring")
    p.add_argument("--n-points", type=int, default=4)
    p.add_argument("--hover-z-mm", type=float, default=280.0)
    p.add_argument("--speed", type=float, default=40.0)
    p.add_argument("--roll", type=float, default=math.pi)
    p.add_argument("--step-mm", type=float, default=5.0, help="WASD jog step (mm)")
    p.add_argument(
        "--type-delta",
        action="store_true",
        help="Type dx dy instead of WASD jog",
    )
    p.add_argument(
        "--raw-keys",
        action="store_true",
        help="Single-keypress WASD (needs real TTY; line mode is default for SSH)",
    )
    p.add_argument(
        "--ignore-global-affine",
        action="store_true",
        help="Fit against raw pinhole table XY (default for --arm xarm). "
        "For xarm2 default is to keep global affine and fit a residual.",
    )
    p.add_argument("--instruction", type=str, default="detect all graspable objects on the table")
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args()
    if args.n_points < 3:
        raise SystemExit("--n-points must be >= 3")

    calib = _load_calib(args.calib)
    cam_cfg = calib["camera"]
    camera_path = args.camera or cam_cfg["index_or_path"]
    k = np.asarray(calib["K"], dtype=float)
    t_ct = np.asarray(calib["T_cam_table"], dtype=float)
    exe = _execution_cfg_for_arm(calib, args.arm)
    rcfg = _robot_cfg_from_calib(calib, args.arm)
    ip = args.robot_ip or rcfg.get("robot_ip")
    if not ip:
        raise SystemExit(f"No robot_ip for arm {args.arm!r}")

    base_xy = tuple(float(v) for v in exe.get("table_base_xy_m", [-0.45, -0.35]))
    flip_x = bool(exe.get("flip_x", False))
    flip_y = bool(exe.get("flip_y", True))
    xy_off = [float(v) for v in exe.get("xy_offset_base_mm", [0.0, 0.0])]
    global_affine = _table_xy_affine_from_calib(calib)

    # xarm: refit shared camera affine (ignore previous).
    # xarm2: keep shared affine; fit residual so Robot1 is not overwritten.
    use_global = (args.arm != "xarm") and (not args.ignore_global_affine)
    if args.ignore_global_affine:
        use_global = False
    if args.arm == "xarm" and not args.ignore_global_affine:
        # Legacy default: collect against raw.
        use_global = False

    print(f"arm={args.arm} ip={ip}")
    print(f"base_xy={base_xy} flip_x={flip_x} flip_y={flip_y} xy_offset={xy_off}")
    if use_global and global_affine is not None:
        print("Pred XY = global table_xy_affine(raw)  → fit per-arm residual affine")
    else:
        print("Pred XY = raw pinhole to_table(uv)  → fit affine")
        if global_affine is not None and args.arm == "xarm":
            print("Note: ignoring existing table_xy_affine while collecting samples.")

    print(
        f"\nPlace the SAME object at {args.n_points} spread-out locations "
        f"inside {args.arm} workspace.\n"
        "At each hover, WASD-jog the gripper onto the object, then ok/Enter.\n"
        "  w=+X(上)  s=-X(下)  a=+Y(左)  d=-Y(右)\n"
    )

    robot = _connect_xarm(calib, ip, arm_name=args.arm)
    arm = robot.real_arm
    samples: list[dict] = []
    try:
        for i in range(args.n_points):
            input(f"[{i+1}/{args.n_points}] Place object, then Enter to capture … ")
            image = _capture_from_camera(
                camera_path,
                width=int(cam_cfg["width"]),
                height=int(cam_cfg["height"]),
                fps=int(cam_cfg["fps"]),
            )
            raw = call_gemini_robotics_er(image, args.instruction)
            dets = parse_vlm_detections(raw, image_hw=(image.shape[0], image.shape[1]))
            det = _pick_detection(dets, args.object)
            uv = tuple(float(v) for v in det["grasp_point_px"])
            raw_t = to_table(uv, k, t_ct)
            if use_global and global_affine is not None:
                a_g, b_g = global_affine
                pred_t = apply_table_xy_affine(raw_t, a_g, b_g)
            else:
                pred_t = raw_t
            cmd_x, cmd_y = _table_to_base_xy(
                pred_t[0],
                pred_t[1],
                base_xy=base_xy,
                flip_x=flip_x,
                flip_y=flip_y,
            )
            cmd_x += xy_off[0]
            cmd_y += xy_off[1]
            print(
                f"  detected={det['name']} uv={uv} "
                f"pred_table=({pred_t[0]:.3f},{pred_t[1]:.3f}) "
                f"cmd_base_mm=({cmd_x:.1f},{cmd_y:.1f})"
            )

            _clear(arm)
            target = [cmd_x, cmd_y, args.hover_z_mm, args.roll, 0.0, 0.0]
            code = arm.set_position(
                *target, speed=args.speed, mvacc=300, wait=True, is_radian=True
            )
            if code != 0:
                raise RuntimeError(f"Hover failed code={code} err={arm.error_code}")

            if args.type_delta:
                line = input("  dx dy (base mm) > ").strip()
                dx, dy = _parse_offset(line)
            else:
                dx, dy = jog_xy_wasd(
                    arm,
                    start_xy_mm=(cmd_x, cmd_y),
                    z_mm=args.hover_z_mm,
                    roll=args.roll,
                    speed=args.speed,
                    step_mm=args.step_mm,
                    raw_keys=args.raw_keys,
                )
            true_base = (cmd_x + dx, cmd_y + dy)
            # Remove constant xy_offset before inverting to table.
            true_t = _base_to_table_xy(
                true_base[0] - xy_off[0],
                true_base[1] - xy_off[1],
                base_xy=base_xy,
                flip_x=flip_x,
                flip_y=flip_y,
            )
            samples.append(
                {
                    "arm": args.arm,
                    "name": det["name"],
                    "uv": list(uv),
                    "raw_table_xy": list(raw_t),
                    "pred_table_xy": list(pred_t),
                    "true_table_xy": list(true_t),
                    "cmd_base_mm": [cmd_x, cmd_y],
                    "delta_base_mm": [dx, dy],
                    "true_base_mm": list(true_base),
                }
            )
            print(
                f"  residual=({dx:+.1f},{dy:+.1f}) mm → "
                f"true_table=({true_t[0]:.3f},{true_t[1]:.3f})"
            )
    finally:
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass

    pred = [tuple(s["pred_table_xy"]) for s in samples]
    true = [tuple(s["true_table_xy"]) for s in samples]
    a, b, metrics = fit_table_xy_affine(pred, true)

    print("\n========== TABLE XY AFFINE ==========")
    print(
        f"arm={args.arm}  n={int(metrics['n'])}  "
        f"rmse={metrics['rmse_m']*1000:.1f} mm  "
        f"max={metrics['max_err_m']*1000:.1f} mm"
    )
    print(f"A = {a.tolist()}")
    print(f"b = {b.tolist()}")
    print("=====================================")

    out_samples = Path(args.calib).with_name(f"table_xy_calib_samples_{args.arm}.json")
    out_samples.write_text(
        json.dumps(
            {
                "arm": args.arm,
                "use_global_affine_as_pred": use_global,
                "samples": samples,
                "A": a.tolist(),
                "b": b.tolist(),
                "metrics": metrics,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Samples → {out_samples}")

    if args.no_write:
        return 0

    block = _write_affine_block(
        a,
        b,
        comment=(
            "true_table = A @ pred_table + b (metres). "
            + (
                "pred = global table_xy_affine(raw); per-arm residual."
                if use_global
                else "pred = raw to_table(uv)."
            )
        ),
    )
    if args.arm == "xarm" and not use_global:
        calib["table_xy_affine"] = block
        legacy = calib.setdefault("execution", {})
        legacy["xy_offset_base_mm"] = [0.0, 0.0]
        by = calib.setdefault("execution_by_arm", {})
        arm_exe = by.setdefault("xarm", {})
        arm_exe["xy_offset_base_mm"] = [0.0, 0.0]
        print(f"Updated global table_xy_affine + zeroed xarm xy_offset")
    else:
        by = calib.setdefault("execution_by_arm", {})
        arm_exe = by.setdefault(args.arm, {})
        arm_exe["table_xy_affine"] = block
        arm_exe["xy_offset_base_mm"] = [0.0, 0.0]
        print(
            f"Updated execution_by_arm.{args.arm}.table_xy_affine "
            f"(global table_xy_affine left unchanged)"
        )

    args.calib.write_text(json.dumps(calib, indent=2) + "\n")
    print(f"Wrote {args.calib}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
