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
Measure how far the hand's grasp centre sits off the wrist (flange) axis.

Why: table-XY calibration hovers at yaw 0. Grasps rotate the wrist to the object's yaw, so
an off-axis grasp centre (LEAP hand: between thumb and fingers) swings around the flange and
misses by ``|(R(yaw) - I) v|`` — a translation-only calibration cannot remove that.

How: put one small rigid object (3-4 cm cube) on the table. The hand hovers above it at
several wrist yaws (default 0, +90, -90 deg); each time jog (WASD, line mode) until the
**grasp centre** is right above the cube centre, then ``ok``. With flange positions P_k at
yaws psi_k and the fixed cube point G:  G = P_k + R(psi_k) v  → least squares for (G, v).
``v`` (base mm, as seen at yaw 0) goes to ``execution_by_arm.<arm>.ee_center_offset_base_mm``
with ``--write``; run_xarm_live then moves the flange by ``-(R(yaw) - I) v`` so the grasp
centre stays where the yaw-0 calibration put it.

Stop the serve process first (it holds the arm and the LEAP hand)::

    cd ~/lerobot/examples
    uv run python -m manipulation.measure_hand_center --arm xarm2            # measure only
    uv run python -m manipulation.measure_hand_center --arm xarm2 --write    # and save
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_EXAMPLES = Path(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from tabletop_perception.run_xarm_live import DEFAULT_CALIB, _connect_xarm, _load_calib  # noqa: E402

from manipulation.jog_xy import jog_xy_wasd  # noqa: E402
from manipulation.leaphand import LeapHandEE  # noqa: E402
from manipulation.run_xarm_live import (  # noqa: E402
    _arm_table_xy_affine_from_exe,
    _execution_cfg_for_arm,
    _table_pose_to_base_mm,
)


def rot(psi: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s], [s, c]])


def solve_center(samples: list[tuple[float, tuple[float, float]]]) -> dict:
    """``samples`` = [(yaw_rad, (Px_mm, Py_mm))]. Returns G, v (mm) and per-sample residuals."""
    rows, rhs = [], []
    for psi, p in samples:
        r = rot(psi)
        # P = G - R v   →   [I, -R] [G; v] = P
        rows.append(np.hstack([np.eye(2), -r]))
        rhs.append(np.asarray(p, dtype=float))
    a = np.vstack(rows)
    b = np.concatenate(rhs)
    x, *_ = np.linalg.lstsq(a, b, rcond=None)
    g, v = x[:2], x[2:]
    res = (a @ x - b).reshape(-1, 2)
    return {
        "G_mm": g.tolist(),
        "v_mm": v.tolist(),
        "residual_mm": [float(np.linalg.norm(r)) for r in res],
        "n": len(samples),
    }


def miss_at(v: np.ndarray, yaw_rad: float) -> float:
    """How far the grasp centre lands from target at ``yaw`` without compensation (mm)."""
    return float(np.linalg.norm((rot(yaw_rad) - np.eye(2)) @ v))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--arm", default="xarm2")
    p.add_argument("--start-table", default="-0.20,0.15", help="table x,y (m) of the first hover")
    p.add_argument("--hover-z-mm", type=float, default=250.0, help="flange height while aligning")
    p.add_argument("--yaws-deg", default="0,90,-90", help="wrist yaws to align at (first must be 0)")
    p.add_argument("--speed", type=float, default=60.0)
    p.add_argument("--step-mm", type=float, default=2.0)
    p.add_argument("--write", action="store_true", help="save v into the calib")
    p.add_argument(
        "--no-close-hand",
        action="store_true",
        help="LEAP-hand arms: do not close the hand while aligning (closed shows the pinch point)",
    )
    args = p.parse_args(argv)

    calib = _load_calib(args.calib)
    exe = _execution_cfg_for_arm(calib, args.arm)
    yaws = [math.radians(float(y)) for y in args.yaws_deg.split(",")]
    if len(yaws) < 2:
        raise SystemExit("need at least two yaws")
    sx, sy = (float(v) for v in args.start_table.split(","))
    base = _table_pose_to_base_mm(
        (sx, sy, 0.0, 0.0),
        base_xy_table=tuple(float(v) for v in exe.get("table_base_xy_m", [-0.45, -0.35])),
        table_z_base_m=float(exe.get("table_z_base_m", 0.0)),
        topdown_rpy=tuple(float(v) for v in exe.get("topdown_rpy_rad", [math.pi, 0.0, 0.0])),
        flip_x=bool(exe.get("flip_x", False)),
        flip_y=bool(exe.get("flip_y", True)),
        use_object_yaw=False,
        table_xy_affine=_arm_table_xy_affine_from_exe(exe),
    )
    roll, pitch = float(base[3]), float(base[4])
    x, y = float(base[0]), float(base[1])

    print(f"Connecting {args.arm} …")
    robot = _connect_xarm(calib, None, arm_name=args.arm)
    arm = robot.real_arm
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    hand = None
    if not args.no_close_hand and str(exe.get("ee", "")).lower() in {"leaphand", "leap_hand", "leap"}:
        print("Closing the LEAP hand so the pinch point (grasp centre) is visible …")
        hand = LeapHandEE.from_calib(calib, args.arm)
        hand.connect()
        hand.close()
    samples: list[tuple[float, tuple[float, float]]] = []
    try:
        for i, psi in enumerate(yaws):
            print(f"\n=== yaw {math.degrees(psi):+.0f} deg ({i + 1}/{len(yaws)}) ===")
            code = arm.set_position(
                x, y, args.hover_z_mm, roll, pitch, psi, speed=args.speed, mvacc=300, wait=True, is_radian=True
            )
            if code != 0:
                raise RuntimeError(f"hover failed code={code} err={arm.error_code}")
            if i == 0:
                print("Place the small cube under the hand now (roughly), then align precisely.")
            print("Jog until the GRASP CENTRE (where thumb and fingertips meet) is right above the cube centre.")
            dx, dy = jog_xy_wasd(
                arm, start_xy_mm=(x, y), z_mm=args.hover_z_mm, roll=roll, pitch=pitch, yaw=psi,
                speed=args.speed, step_mm=args.step_mm,
            )
            x, y = x + dx, y + dy
            samples.append((psi, (x, y)))
            print(f"  recorded flange P=({x:.1f}, {y:.1f}) mm at yaw {math.degrees(psi):+.0f} deg")
    finally:
        try:
            arm.set_position(x, y, args.hover_z_mm, roll, pitch, 0.0, speed=args.speed, mvacc=300, wait=True, is_radian=True)
            robot.move_to_home(speed_deg_s=20.0)
        finally:
            robot.disconnect()
            if hand is not None:
                hand.open()
                hand.disconnect()

    out = solve_center(samples)
    v = np.asarray(out["v_mm"])
    print("\n=== result ===")
    print(f"grasp-centre offset v (base mm at yaw 0) = ({v[0]:+.1f}, {v[1]:+.1f})  |v| = {np.linalg.norm(v):.1f} mm")
    print("per-yaw residual (mm):", [round(r, 1) for r in out["residual_mm"]],
          "(only meaningful with >2 yaws; > ~5 mm means the alignments disagree)")
    for deg in (15, 30, 45, 90):
        print(f"  uncompensated miss at yaw {deg:>3} deg: {miss_at(v, math.radians(deg)):5.1f} mm")
    if args.write:
        text = args.calib.read_text()
        data = json.loads(text)
        blk = data.setdefault("execution_by_arm", {}).setdefault(args.arm, {})
        blk["ee_center_offset_base_mm"] = [round(float(v[0]), 1), round(float(v[1]), 1)]
        blk["_ee_center_note"] = (
            f"measured by measure_hand_center (yaws {args.yaws_deg} deg, residual "
            f"{[round(r, 1) for r in out['residual_mm']]} mm). Flange is moved by -(R(yaw)-I)v."
        )
        args.calib.write_text(json.dumps(data, indent=2) + "\n")
        print(f"written to {args.calib} → execution_by_arm.{args.arm}.ee_center_offset_base_mm")
    else:
        print("(not saved; rerun with --write to store it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
