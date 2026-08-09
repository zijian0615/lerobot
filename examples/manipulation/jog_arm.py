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
WASD-jog from the arm's **current** pose (no replan / no table hover).

```bash
cd ~/lerobot/examples
python -m manipulation.jog_arm --arm xarm2
```

Line mode (SSH): ``w`` / ``a`` / ``s`` / ``d`` / ``w 3`` / ``step 10`` / ``ok``
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from manipulation.jog_xy import jog_xy_wasd
from tabletop_perception.run_xarm_live import (
    DEFAULT_CALIB,
    _connect_xarm,
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
    p = argparse.ArgumentParser(description="WASD jog from current EE pose")
    p.add_argument("--arm", default="xarm2")
    p.add_argument("--robot-ip", default=None)
    p.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    p.add_argument("--speed", type=float, default=30.0)
    p.add_argument("--step-mm", type=float, default=5.0)
    p.add_argument("--raw-keys", action="store_true")
    p.add_argument(
        "--hold-rpy",
        action="store_true",
        help="Keep current roll/pitch/yaw (default: keep current)",
    )
    args = p.parse_args()

    calib = _load_calib(args.calib)
    ip = args.robot_ip or _robot_cfg_from_calib(calib, args.arm).get("robot_ip")
    if not ip:
        raise SystemExit(f"No robot_ip for {args.arm!r}")

    print(f"Connecting arm={args.arm} ip={ip} …")
    robot = _connect_xarm(calib, ip, arm_name=args.arm)
    try:
        arm = robot.real_arm
        _clear(arm)
        code, pose = arm.get_position(is_radian=True)
        if code != 0 or pose is None:
            raise RuntimeError(f"get_position failed code={code}")
        x0, y0, z0 = float(pose[0]), float(pose[1]), float(pose[2])
        roll, pitch, yaw = float(pose[3]), float(pose[4]), float(pose[5])
        print(f"start pose xy=({x0:.1f},{y0:.1f}) z={z0:.1f} rpy=({roll:.3f},{pitch:.3f},{yaw:.3f})")

        dx, dy = jog_xy_wasd(
            arm,
            start_xy_mm=(x0, y0),
            z_mm=z0,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=args.speed,
            step_mm=args.step_mm,
            raw_keys=args.raw_keys,
        )
        code, pose2 = arm.get_position(is_radian=True)
        print(f"\nDone. Δbase_xy=({dx:+.1f}, {dy:+.1f}) mm")
        print(f"final pose={pose2}")
        print(
            "Note: this does not write calib. Tell me the Δ / where it sits "
            "to update xarm2 table_base / flip / offset."
        )
        return 0
    finally:
        try:
            robot.disconnect()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
