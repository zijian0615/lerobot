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

Place the SAME object at ≥3 spread-out spots. Cover the **image top**
(far table edge) — that is where the current 4-point affine is weakest.
At each spot the arm hovers at the predicted table XY; WASD-jog onto the
object and confirm (w=+X上, s=-X下, a=+Y左, d=-Y右).

Predictions use the same ``z=h`` top-face projection as live perception.

Robot1 (shared camera affine, 6 points, 2–3 at image top)::

    python -m tabletop_perception.calibrate_table_xy \\
      --arm xarm --camera /dev/video0 --object box --n-points 6

Keep previous samples and add more (recomputes pred from stored uv)::

    python -m tabletop_perception.calibrate_table_xy \\
      --arm xarm --camera /dev/video0 --object box --n-points 3 --append

Robot2 (per-arm residual affine; does NOT overwrite Robot1)::

    python -m tabletop_perception.calibrate_table_xy \\
      --arm xarm2 --camera /dev/video0 --object box --n-points 6 --hover-z-mm 320
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
from tabletop_perception.perception import (
    object_top_z_from_calib,
    resolve_object_top_z_m,
)
from tabletop_perception.vlm import call_detection_vlm, parse_vlm_detections

_FAR_EDGE_FRAC = 0.33


def _uv_zone(uv: tuple[float, float], image_h: float | None = None) -> str:
    h = float(image_h) if image_h and float(image_h) > 1.0 else 480.0
    v = float(uv[1])
    band = _FAR_EDGE_FRAC * h
    if v <= band:
        return "FAR (image top)"
    if v >= h - band:
        return "NEAR (image bottom)"
    return "MID"


def _sample_image_h(sample: dict, default: float | None = None) -> float | None:
    hw = sample.get("image_hw")
    if isinstance(hw, (list, tuple)) and hw:
        return float(hw[0])
    return default


def _samples_path(calib_path: Path, arm: str) -> Path:
    named = Path(calib_path).with_name(f"table_xy_calib_samples_{arm}.json")
    legacy = Path(calib_path).with_name("table_xy_calib_samples.json")
    if named.exists():
        return named
    if arm == "xarm" and legacy.exists():
        return legacy
    return named


def _recompute_pred(
    sample: dict,
    *,
    k: np.ndarray,
    t_ct: np.ndarray,
    z_heights: dict[str, float],
    z_default: float,
    use_global: bool,
    global_affine: tuple[np.ndarray, np.ndarray] | None,
) -> dict:
    """Refresh raw/pred table XY from stored uv with current z=h projection."""
    uv = tuple(float(v) for v in sample["uv"])
    z_top = resolve_object_top_z_m(str(sample.get("name", "")), z_heights, z_default)
    raw_t = to_table(uv, k, t_ct, z_plane=z_top)
    if use_global and global_affine is not None:
        pred_t = apply_table_xy_affine(raw_t, global_affine[0], global_affine[1])
    else:
        pred_t = raw_t
    out = dict(sample)
    out["z_plane"] = z_top
    out["raw_table_xy"] = [float(raw_t[0]), float(raw_t[1])]
    out["pred_table_xy"] = [float(pred_t[0]), float(pred_t[1])]
    return out


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


def _pick_detection(dets: list[dict], object_name: str | None) -> dict | None:
    """Return a detection, or ``None`` to recapture this round.

    SSH: type an index and Enter, or ``r`` to retry, ``q`` to stop.
    Never Ctrl-C — previous ``ok`` samples are checkpointed.
    """
    if object_name and dets:
        key = object_name.lower().replace(" ", "_")
        matches = [
            d for d in dets if key in d["name"].lower().replace(" ", "_")
        ]
        if len(matches) == 1:
            print(f"Using match: {matches[0]['name']}")
            return matches[0]
        if len(matches) > 1:
            print(f"Multiple matches for {object_name!r}:")
            dets = matches
        else:
            print(f"没有名叫 {object_name!r} 的检测，选编号或重拍。")

    if not dets:
        print("VLM 这一帧没有检测。")
    else:
        print("Detections:")
        for i, d in enumerate(dets):
            print(
                f"  [{i}] {d['name']} grasp_px={d['grasp_point_px']} "
                f"long_axis_px={d.get('long_axis_px')}"
            )

    while True:
        sys.stdout.flush()
        line = input("Pick index（r=重拍这一轮, q=结束并保存已有点）> ").strip().lower()
        if line in ("r", "retry", "rest"):
            print("重拍这一轮（已确认的点还在）。")
            return None
        if line in ("q", "quit"):
            raise KeyboardInterrupt("user quit pick")
        if not line:
            continue
        try:
            idx = int(line)
            return dets[idx]
        except (ValueError, IndexError):
            print("  输入编号（如 0）、r 重拍、或 q 结束。不要 Ctrl-C。")


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
    p.add_argument(
        "--n-points",
        type=int,
        default=6,
        help="New samples to collect this run (default 6; put 2–3 at image top)",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help="Keep previous samples and add --n-points more (recomputes pred from uv)",
    )
    p.add_argument(
        "--samples",
        type=Path,
        default=None,
        help="Existing samples JSON for --append (default: table_xy_calib_samples_<arm>.json)",
    )
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
    p.add_argument(
        "--model",
        "--vlm",
        dest="model",
        default="gemini",
        help="Base VLM: gemini, gpt-6, or cosmos / cosmos3-nano.",
    )
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args()
    if args.n_points < 1:
        raise SystemExit("--n-points must be >= 1")
    if not args.append and args.n_points < 3:
        raise SystemExit("--n-points must be >= 3 (or use --append)")

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
    z_default, z_heights = object_top_z_from_calib(calib)

    # xarm: refit shared camera affine (ignore previous).
    # xarm2: keep shared affine; fit residual so Robot1 is not overwritten.
    use_global = (args.arm != "xarm") and (not args.ignore_global_affine)
    if args.ignore_global_affine:
        use_global = False
    if args.arm == "xarm" and not args.ignore_global_affine:
        # Legacy default: collect against raw.
        use_global = False

    print(f"arm={args.arm} ip={ip} camera={camera_path}")
    print(f"base_xy={base_xy} flip_x={flip_x} flip_y={flip_y} xy_offset={xy_off}")
    print(
        f"Projection z=h from calib object_top_z_m (default={z_default:.3f} m): "
        f"{z_heights}"
    )
    if str(camera_path) not in {"/dev/video0", "0"}:
        print(
            "WARNING: live runs use --camera /dev/video0 (Aoni overhead). "
            "Pass the same camera here or the affine will not match."
        )
    if use_global and global_affine is not None:
        print("Pred XY = global table_xy_affine(to_table(uv, z=h))  → fit per-arm residual")
    else:
        print("Pred XY = to_table(uv, z=h)  → fit affine")
        if global_affine is not None and args.arm == "xarm":
            print("Note: ignoring existing table_xy_affine while collecting samples.")

    samples_file = args.samples or _samples_path(args.calib, args.arm)
    samples: list[dict] = []
    if args.append:
        if not samples_file.exists():
            raise SystemExit(f"--append but no samples file: {samples_file}")
        prev = json.loads(samples_file.read_text())
        raw_prev = list(prev.get("samples") or [])
        samples = [
            _recompute_pred(
                s,
                k=k,
                t_ct=t_ct,
                z_heights=z_heights,
                z_default=z_default,
                use_global=use_global,
                global_affine=global_affine,
            )
            for s in raw_prev
        ]
        print(f"Loaded {len(samples)} previous samples from {samples_file}")
        for i, s in enumerate(samples):
            uv = tuple(float(v) for v in s["uv"])
            print(
                f"  prev[{i}] {s.get('name')} uv=({uv[0]:.0f},{uv[1]:.0f}) "
                f"{_uv_zone(uv, _sample_image_h(s))}"
            )
        n_far = sum(
            1
            for s in samples
            if _uv_zone(tuple(s["uv"]), _sample_image_h(s)) == "FAR (image top)"
        )
        if n_far < 2:
            print(
                f"Far-edge coverage is thin ({n_far} point(s) in the top third of the image). "
                "Put the new samples near the image TOP."
            )

    print(
        f"\nPlace the SAME object at {args.n_points} spread-out locations "
        f"inside {args.arm} workspace.\n"
        "Suggested layout (image coords: top = far table edge):\n"
        "  2× image TOP / far edge   ← currently the weak region\n"
        "  2× image CENTER\n"
        "  2× image BOTTOM / near robots\n"
        "Use a box if you will grasp boxes (same top height as live z=h).\n"
        "\nSSH / line-mode (do NOT hold WASD like a game):\n"
        "  1) Place the object, then press Enter once to capture.\n"
        "  2) After the arm hovers, type a command and Enter:\n"
        "       w      one step +X (上)\n"
        "       d 3    three steps -Y (右)\n"
        "       ok     confirm this sample\n"
        "  w=+X(上)  s=-X(下)  a=+Y(左)  d=-Y(右)\n"
    )
    sys.stdout.flush()

    robot = _connect_xarm(calib, ip, arm_name=args.arm)
    arm = robot.real_arm
    time.sleep(0.4)
    out_samples = args.samples or Path(args.calib).with_name(
        f"table_xy_calib_samples_{args.arm}.json"
    )

    def _checkpoint() -> None:
        payload = {
            "arm": args.arm,
            "camera": str(camera_path),
            "use_global_affine_as_pred": use_global,
            "object_top_z_m": z_heights,
            "object_top_z_m_default": z_default,
            "samples": samples,
            "partial": True,
        }
        out_samples.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"  checkpoint {len(samples)} sample(s) → {out_samples}")

    interrupted = False
    try:
        for i in range(args.n_points):
            print()
            print(f"===== [{i+1}/{args.n_points}] =====")
            det = None
            while det is None:
                print("摆好物体后，只按一次 Enter 拍照。现在不要按 WASD。")
                sys.stdout.flush()
                input(">>> 按 Enter 拍照：")
                print("拍照中…")
                sys.stdout.flush()
                image = _capture_from_camera(
                    camera_path,
                    width=int(cam_cfg["width"]),
                    height=int(cam_cfg["height"]),
                    fps=int(cam_cfg["fps"]),
                    fourcc=cam_cfg.get("fourcc") or "MJPG",
                )
                print(f"图像 {image.shape[1]}x{image.shape[0]}，正在调用 VLM（SSH 下可能要等几秒）…")
                sys.stdout.flush()
                raw = call_detection_vlm(image, args.instruction, model=args.model)
                dets = parse_vlm_detections(raw, image_hw=(image.shape[0], image.shape[1]))
                det = _pick_detection(dets, args.object)
            uv = tuple(float(v) for v in det["grasp_point_px"])
            image_h = float(image.shape[0])
            z_top = resolve_object_top_z_m(det["name"], z_heights, z_default)
            raw_t = to_table(uv, k, t_ct, z_plane=z_top)
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
                f"  detected={det['name']} uv={uv} zone={_uv_zone(uv, image_h)} "
                f"z_h={z_top:.3f}m "
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
                    "image_hw": [int(image.shape[0]), int(image.shape[1])],
                    "z_plane": z_top,
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
            _checkpoint()
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n停止采集。已确认 {len(samples)} 个点仍会保存。")
    finally:
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass

    if len(samples) < 3:
        if samples:
            _checkpoint()
        print(
            f"Need at least 3 samples to fit affine, got {len(samples)}. "
            f"Resume with --append --n-points {max(1, 3 - len(samples))}"
        )
        return 1 if interrupted else 0

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

    n_far = sum(
        1 for s in samples if _uv_zone(tuple(s["uv"]), _sample_image_h(s)) == "FAR (image top)"
    )
    n_near = sum(
        1
        for s in samples
        if _uv_zone(tuple(s["uv"]), _sample_image_h(s)) == "NEAR (image bottom)"
    )
    print(
        f"Coverage: n={len(samples)}  far/top={n_far}  mid="
        f"{len(samples) - n_far - n_near}  near/bottom={n_near}"
    )
    if n_far < 2:
        print(
            "WARNING: fewer than 2 far-edge (image-top) samples. "
            "Re-run with --append --n-points 3 and place those at the far edge."
        )

    out_samples.write_text(
        json.dumps(
            {
                "arm": args.arm,
                "camera": str(camera_path),
                "use_global_affine_as_pred": use_global,
                "object_top_z_m": z_heights,
                "object_top_z_m_default": z_default,
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
                else "pred = to_table(uv, z=object_top)."
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
