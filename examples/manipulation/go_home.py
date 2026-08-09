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

"""Slowly move xArm to configured ``start_joints`` (home).

```bash
cd examples
python -m manipulation.go_home --arm xarm --speed 20
python -m manipulation.go_home --arm xarm2 --speed 20
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

from tabletop_perception.run_xarm_live import (
    DEFAULT_CALIB,
    _connect_xarm,
    _load_calib,
    _robot_cfg_from_calib,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Move xArm to home / start_joints")
    p.add_argument("--arm", default="xarm", help="Calib arm name: xarm | xarm2")
    p.add_argument("--robot-ip", default=None, help="Override calib robot_ip")
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--speed", type=float, default=20.0, help="deg/s (slow is safer)")
    args = p.parse_args()

    calib = _load_calib(args.calib)
    ip = args.robot_ip or _robot_cfg_from_calib(calib, args.arm).get("robot_ip")
    if not ip:
        raise SystemExit(f"No robot_ip for arm {args.arm!r}")
    print(f"Connecting arm={args.arm} ip={ip} …")
    robot = _connect_xarm(calib, ip, arm_name=args.arm)
    try:
        arm = robot.real_arm
        print(
            f"state={arm.state} mode={arm.mode} "
            f"error={arm.error_code} warn={arm.warn_code}"
        )
        if arm.error_code != 0:
            print("Cleaning controller error …")
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(True)
            arm.set_mode(0)
            arm.set_state(0)
            time.sleep(0.3)

        joints = list(robot.config.start_joints)
        print("Home joints (rad):", [round(j, 4) for j in joints])
        print("Home joints (deg):", [round(math.degrees(j), 1) for j in joints])
        print(f"Moving to home at {args.speed:.0f} deg/s …")
        robot.move_to_home(speed_deg_s=args.speed)

        code, pose = arm.get_position(is_radian=True)
        code_j, angles = arm.get_servo_angle(is_radian=True)
        print("Done.")
        print("position code/pose:", code, pose)
        print("joints code/angles:", code_j, angles)
        return 0
    finally:
        robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
