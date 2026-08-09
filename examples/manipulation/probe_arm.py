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
Connect an arm from calib, print pose, optional hover at a table XY (for flip check).

```bash
cd ~/lerobot/examples
# SDK smoke (no motion):
python -m manipulation.probe_arm --arm xarm2

# Hover above table center-ish for Robot2 (safe Z):
python -m manipulation.probe_arm --arm xarm2 --hover-table -0.05 -0.05 --hover-z-mm 320
```
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from manipulation.run_xarm_live import (
    _arm_table_xy_affine_from_exe,
    _table_pose_to_base_mm,
)
from tabletop_perception.run_xarm_live import (
    DEFAULT_CALIB,
    _connect_xarm,
    _execution_cfg_for_arm,
    _load_calib,
    _robot_cfg_from_calib,
)


def _clear(arm) -> None:
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.2)


def main() -> int:
    p = argparse.ArgumentParser(description="Probe / hover an arm from calib")
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--arm", type=str, default="xarm2")
    p.add_argument("--robot-ip", type=str, default=None)
    p.add_argument(
        "--hover-table",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Table-frame XY (m) to hover",
    )
    p.add_argument("--hover-z-mm", type=float, default=320.0)
    p.add_argument("--speed", type=float, default=30.0)
    p.add_argument("--no-move", action="store_true", help="Connect + print only")
    p.add_argument(
        "--flip-x",
        choices=("auto", "true", "false"),
        default="auto",
        help="Override calib flip_x for this probe",
    )
    p.add_argument(
        "--flip-y",
        choices=("auto", "true", "false"),
        default="auto",
        help="Override calib flip_y for this probe",
    )
    p.add_argument(
        "--print-flips",
        action="store_true",
        help="Print all 4 flip combos for --hover-table (no motion unless also hovering)",
    )
    args = p.parse_args()

    calib = _load_calib(args.calib)
    exe = _execution_cfg_for_arm(calib, args.arm)
    rcfg = _robot_cfg_from_calib(calib, args.arm)
    ip = args.robot_ip or rcfg.get("robot_ip")
    if not ip:
        raise SystemExit(f"No robot_ip for arm {args.arm!r}")

    base_xy = tuple(float(v) for v in exe.get("table_base_xy_m", [0.45, -0.35]))
    flip_x = bool(exe.get("flip_x", False))
    flip_y = bool(exe.get("flip_y", True))
    if args.flip_x != "auto":
        flip_x = args.flip_x == "true"
    if args.flip_y != "auto":
        flip_y = args.flip_y == "true"
    topdown = tuple(float(v) for v in exe.get("topdown_rpy_rad", [math.pi, 0.0, 0.0]))
    table_z = float(exe.get("table_z_base_m", 0.0))
    xy_off = [float(v) for v in exe.get("xy_offset_base_mm", [0.0, 0.0])]

    uncal = bool(dict(calib.get("robots") or {}).get(args.arm, {}).get("_uncalibrated"))
    print(f"arm={args.arm} ip={ip}")
    print(f"base_xy={base_xy} flip_x={flip_x} flip_y={flip_y} xy_offset={xy_off}")
    print(f"uncalibrated={uncal}")

    if args.hover_table is not None and args.print_flips:
        x_t, y_t = float(args.hover_table[0]), float(args.hover_table[1])
        z_t = args.hover_z_mm / 1000.0 - table_z
        print(f"flip preview for table=({x_t:.3f},{y_t:.3f}) z={args.hover_z_mm:.0f}mm:")
        for fx in (False, True):
            for fy in (False, True):
                p = _table_pose_to_base_mm(
                    (x_t, y_t, z_t, 0.0),
                    base_xy_table=base_xy,
                    table_z_base_m=table_z,
                    topdown_rpy=topdown,
                    flip_x=fx,
                    flip_y=fy,
                )
                p[0] += xy_off[0]
                p[1] += xy_off[1]
                p[2] = args.hover_z_mm
                mark = " <-- calib/override" if (fx == flip_x and fy == flip_y) else ""
                print(
                    f"  flip_x={fx!s:5} flip_y={fy!s:5} → "
                    f"[{p[0]:7.1f}, {p[1]:7.1f}, {p[2]:5.1f}] mm{mark}"
                )

    if args.no_move and args.hover_table is None:
        print("No hover requested (--no-move or no --hover-table).")
        return 0
    if args.print_flips and args.no_move:
        return 0

    robot = _connect_xarm(calib, ip, arm_name=args.arm)
    try:
        arm = robot.real_arm
        _clear(arm)
        code, pose = arm.get_position(is_radian=True)
        print(f"current pose code={code} xyzrpy={pose}")

        if args.no_move or args.hover_table is None:
            print("No hover requested (--no-move or no --hover-table).")
            return 0

        x_t, y_t = float(args.hover_table[0]), float(args.hover_table[1])
        z_t = args.hover_z_mm / 1000.0 - table_z
        arm_affine = _arm_table_xy_affine_from_exe(exe)
        xyzrpy = _table_pose_to_base_mm(
            (x_t, y_t, z_t, 0.0),
            base_xy_table=base_xy,
            table_z_base_m=table_z,
            topdown_rpy=topdown,
            flip_x=flip_x,
            flip_y=flip_y,
            table_xy_affine=arm_affine,
        )
        xyzrpy[0] += xy_off[0]
        xyzrpy[1] += xy_off[1]
        # Force hover Z in base mm.
        xyzrpy[2] = args.hover_z_mm
        print(f"hover table=({x_t:.3f},{y_t:.3f}) → base_mm={ [round(v,1) for v in xyzrpy[:3]] }")
        print("Expect image direction toward bear ≈ left/up from Robot2 home (not right/out of FOV).")
        code = arm.set_position(
            *xyzrpy, speed=args.speed, mvacc=300, wait=True, is_radian=True
        )
        if code != 0:
            raise RuntimeError(f"hover failed code={code} err={arm.error_code}")
        code, pose = arm.get_position(is_radian=True)
        print(f"hover reached code={code} pose={pose}")
        print(
            "Check: does the EE sit above the intended table point?\n"
            "  If still image-right → try --flip-y false / --flip-x true\n"
            "  If close but biased → jog_arm + update xy_offset_base_mm"
        )
        return 0
    finally:
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
