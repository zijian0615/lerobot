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
Slowly descend above a target XY until you hit STOP at the grasp height.

The script keeps the last successful pose. After STOP / controller error it
prints the measured base-frame Z (mm) and suggested calib values.

```bash
cd examples
# 1) optional: get target from last dry-run bound.json, or pass --x/--y mm
python -m manipulation.measure_grasp_height \\
    --robot-ip 192.168.1.204 \\
    --from-bound runs/20260808_151625/bound.json \\
    --start-z-mm 300 --step-mm 5 --speed 30
```
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from tabletop_perception.run_xarm_live import DEFAULT_CALIB, _connect_xarm, _load_calib


def _read_pose(arm) -> list[float] | None:
    code, pose = arm.get_position(is_radian=True)
    if code != 0 or pose is None:
        return None
    return list(pose[:6])


def _clear(arm) -> None:
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.2)


def main() -> int:
    p = argparse.ArgumentParser(description="Measure grasp height by descending until STOP")
    p.add_argument("--robot-ip", default="192.168.1.204")
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--from-bound", type=Path, default=None, help="bound.json from dry-run")
    p.add_argument("--step-index", type=int, default=0, help="which bound step pose to use")
    p.add_argument("--x", type=float, default=None, help="base X mm override")
    p.add_argument("--y", type=float, default=None, help="base Y mm override")
    p.add_argument("--start-z-mm", type=float, default=300.0)
    p.add_argument("--min-z-mm", type=float, default=20.0, help="do not descend below this")
    p.add_argument("--step-mm", type=float, default=5.0)
    p.add_argument("--speed", type=float, default=30.0, help="mm/s")
    p.add_argument("--roll", type=float, default=math.pi)
    p.add_argument("--pitch", type=float, default=0.0)
    p.add_argument("--yaw", type=float, default=0.0)
    args = p.parse_args()

    calib = _load_calib(args.calib)
    exe = calib.get("execution", {})
    base_xy = tuple(float(v) for v in exe.get("table_base_xy_m", [-0.45, -0.35]))
    flip_y = bool(exe.get("flip_y", True))

    x_mm, y_mm = args.x, args.y
    if (x_mm is None or y_mm is None) and args.from_bound is not None:
        bound = json.loads(args.from_bound.read_text())
        pose_t = bound[args.step_index]["params"]["pose"]
        # table m → base mm (same as run_xarm_live)
        x_mm = (float(pose_t[0]) - base_xy[0]) * 1000.0
        y_rel = (float(pose_t[1]) - base_xy[1]) * 1000.0
        y_mm = -y_rel if flip_y else y_rel
        print(f"Target from bound step {args.step_index}: table={pose_t[:2]} → base_xy=({x_mm:.1f}, {y_mm:.1f})")

    if x_mm is None or y_mm is None:
        # Fall back: use current XY, only measure Z descent under current hover.
        print("No XY given — will use the arm's current XY and only descend in Z.")

    print(f"Connecting {args.robot_ip} …")
    robot = _connect_xarm(calib, args.robot_ip)
    arm = robot.real_arm
    last_ok: list[float] | None = None

    try:
        _clear(arm)
        cur = _read_pose(arm)
        if cur is None:
            raise RuntimeError("Cannot read current pose")
        last_ok = cur
        print(f"Current pose: {[round(v, 1) if i < 3 else round(v, 3) for i, v in enumerate(cur)]}")

        if x_mm is None:
            x_mm = cur[0]
        if y_mm is None:
            y_mm = cur[1]

        # 1) Move high above target XY.
        hover = [x_mm, y_mm, args.start_z_mm, args.roll, args.pitch, args.yaw]
        print(f"Moving to hover {[round(v, 1) for v in hover[:3]]} …")
        print("Press the robot STOP when the gripper is at a good grasp height during descent.")
        code = arm.set_position(
            *hover, speed=args.speed, mvacc=300, wait=True, is_radian=True
        )
        if code != 0:
            raise RuntimeError(f"Hover move failed code={code} err={arm.error_code}")
        last_ok = _read_pose(arm) or hover

        # 2) Descend step by step.
        z = float(args.start_z_mm)
        while z - args.step_mm >= args.min_z_mm - 1e-6:
            z = max(args.min_z_mm, z - args.step_mm)
            target = [x_mm, y_mm, z, args.roll, args.pitch, args.yaw]
            print(f"  ↓ z={z:.1f} mm …", flush=True)
            code = arm.set_position(
                *target, speed=args.speed, mvacc=200, wait=True, is_radian=True
            )
            if code != 0 or arm.error_code != 0 or arm.state == 4:
                print(
                    f"\nStopped (code={code}, error={arm.error_code}, state={arm.state})."
                )
                break
            pose = _read_pose(arm)
            if pose is not None:
                last_ok = pose
                print(
                    f"    ok  pose z={pose[2]:.1f} mm  "
                    f"xy=({pose[0]:.1f}, {pose[1]:.1f})",
                    flush=True,
                )
        else:
            print("Reached min-z without STOP.")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt — using last successful pose.")
    finally:
        # Try one last read (may fail after E-stop).
        try:
            pose = _read_pose(arm)
            if pose is not None:
                last_ok = pose
        except Exception:  # noqa: BLE001
            pass
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass

    if last_ok is None:
        print("No successful pose recorded.")
        return 1

    z_mm = float(last_ok[2])
    z_m = z_mm / 1000.0
    print("\n========== MEASUREMENT ==========")
    print(f"Last OK pose (base mm): x={last_ok[0]:.1f}  y={last_ok[1]:.1f}  z={z_mm:.1f}")
    print(f"Last OK rpy (rad): {[round(v, 4) for v in last_ok[3:6]]}")
    print()
    print("Suggested calib (table plane ≈ base z=0 assumption):")
    print(f'  "grasp_height_m": {z_m:.3f}')
    print(f'  "execution.table_z_base_m": 0.0')
    print()
    print("If the table surface is NOT at base z=0, set:")
    print('  table_z_base_m = <table surface z in base frame, metres>')
    print("  grasp_height_m = tip_z_above_table  (usually ~0.03–0.08)")
    print("  # with tip_z_base = table_z_base_m + grasp_height_m")
    print(f"  # measured tip_z_base = {z_m:.3f} m")
    print("=================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
