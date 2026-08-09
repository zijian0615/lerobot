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
Measure / enter XY bias (no teach pendant required).

1. Arm moves to predicted grasp XY at a safe hover height.
2. Look at how the gripper misses the object.
3. Type the correction in **base-frame mm**:
      dx dy
   meaning: add (dx, dy) to future commands so the gripper moves
   toward the real object.
   Example: if gripper is 30mm too far in +X and 20mm too far in -Y
   relative to the object, you want the next command more -X and +Y,
   so enter:  -30  20

```bash
cd ~/lerobot/examples
python -m manipulation.measure_xy_offset \\
  --from-bound manipulation/runs/20260808_160900/result.json \\
  --hover-z-mm 280
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

from manipulation.jog_xy import jog_xy_wasd
from tabletop_perception.run_xarm_live import DEFAULT_CALIB, _connect_xarm, _load_calib


def _table_to_base_xy(
    x_t: float,
    y_t: float,
    *,
    base_xy: tuple[float, float],
    flip_y: bool,
) -> tuple[float, float]:
    x_b = (x_t - base_xy[0]) * 1000.0
    y_b = (y_t - base_xy[1]) * 1000.0
    if flip_y:
        y_b = -y_b
    return x_b, y_b


def _clear(arm) -> None:
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.2)


def _load_target_table_xy(path: Path) -> tuple[float, float, str]:
    data = json.loads(path.read_text())
    if "bound" in data:
        steps = data["bound"]
    elif isinstance(data, list):
        steps = data
    else:
        raise ValueError(f"Unrecognized bound/result file: {path}")
    grasp = next(s for s in steps if s.get("primitive") == "Grasp")
    pose = grasp["params"]["pose"]
    return float(pose[0]), float(pose[1]), str(grasp["params"].get("object", "?"))


def _parse_offset(line: str) -> tuple[float, float]:
    parts = line.replace(",", " ").split()
    if len(parts) != 2:
        raise ValueError("Need two numbers: dx dy  (base-frame mm)")
    return float(parts[0]), float(parts[1])


def main() -> int:
    p = argparse.ArgumentParser(description="Enter XY offset after hover preview")
    p.add_argument("--robot-ip", default="192.168.1.204")
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--from-bound", type=Path, required=True)
    p.add_argument("--hover-z-mm", type=float, default=280.0)
    p.add_argument("--speed", type=float, default=40.0)
    p.add_argument("--roll", type=float, default=math.pi)
    p.add_argument("--dx", type=float, default=None, help="Skip prompt; set dx mm directly")
    p.add_argument("--dy", type=float, default=None, help="Skip prompt; set dy mm directly")
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
    p.add_argument("--no-move", action="store_true", help="Only write offset, do not move arm")
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args()

    calib = _load_calib(args.calib)
    exe = calib.get("execution", {})
    base_xy = tuple(float(v) for v in exe.get("table_base_xy_m", [-0.45, -0.35]))
    flip_y = bool(exe.get("flip_y", True))
    prev = [float(v) for v in exe.get("xy_offset_base_mm", [0.0, 0.0])]

    x_t, y_t, name = _load_target_table_xy(args.from_bound)
    cmd_x, cmd_y = _table_to_base_xy(x_t, y_t, base_xy=base_xy, flip_y=flip_y)
    cmd_x += prev[0]
    cmd_y += prev[1]

    print(f"Target object: {name}")
    print(f"Predicted table xy=({x_t:.3f}, {y_t:.3f})")
    print(f"Commanded base xy mm=({cmd_x:.1f}, {cmd_y:.1f})  prev_offset={prev}")

    dx = dy = 0.0
    if args.dx is not None and args.dy is not None:
        dx, dy = float(args.dx), float(args.dy)
        print(f"Using CLI offset: {dx} {dy}")
    elif args.no_move or args.type_delta:
        print("\nEnter correction to ADD (base mm): dx dy")
        line = input("dx dy > ").strip()
        dx, dy = _parse_offset(line)
    else:
        robot = _connect_xarm(calib, args.robot_ip)
        arm = robot.real_arm
        try:
            _clear(arm)
            target = [cmd_x, cmd_y, args.hover_z_mm, args.roll, 0.0, 0.0]
            print(f"Moving to hover { [round(v,1) for v in target[:3]] } …")
            code = arm.set_position(
                *target, speed=args.speed, mvacc=300, wait=True, is_radian=True
            )
            if code != 0:
                raise RuntimeError(f"Hover failed code={code} err={arm.error_code}")
            dx, dy = jog_xy_wasd(
                arm,
                start_xy_mm=(cmd_x, cmd_y),
                z_mm=args.hover_z_mm,
                roll=args.roll,
                speed=args.speed,
                step_mm=args.step_mm,
                raw_keys=args.raw_keys,
            )
        finally:
            try:
                robot.disconnect()
            except Exception:  # noqa: BLE001
                pass

    new_offset = [prev[0] + dx, prev[1] + dy]
    print("\n========== XY OFFSET ==========")
    print(f"Added this run: ({dx:+.1f}, {dy:+.1f}) mm")
    print(f'Calib:\n  "xy_offset_base_mm": [{new_offset[0]:.1f}, {new_offset[1]:.1f}]')
    print("================================")

    if not args.no_write:
        exe["xy_offset_base_mm"] = [round(new_offset[0], 1), round(new_offset[1], 1)]
        calib["execution"] = exe
        args.calib.write_text(json.dumps(calib, indent=2) + "\n")
        print(f"Updated {args.calib}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
