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

"""WASD jog in Fanuc UF XY after a hover (SSH-friendly line mode)."""

from __future__ import annotations

import sys

from lerobot.robots.fanuc import Fanuc

_KEY_DELTA_BASE_MM = {
    "w": (+1.0, 0.0),
    "s": (-1.0, 0.0),
    "a": (0.0, +1.0),
    "d": (0.0, -1.0),
}

_HELP = (
    "WASD jog (Fanuc UF mm):\n"
    "  w = +X   s = -X   a = +Y   d = -Y\n"
    "Line mode — type a line and Enter:\n"
    "  w          one step +X\n"
    "  d 3        three steps -Y\n"
    "  step 10    set step mm\n"
    "  r          reset to start hover\n"
    "  ok / done  confirm\n"
    "  q          abort\n"
)


def wait_motion_ack(robot: Fanuc, sequence_id: int, timeout_s: float = 60.0) -> int:
    return robot.wait_for_ack(int(sequence_id), timeout_s=timeout_s)


def move_fanuc_xyz(
    robot: Fanuc,
    *,
    x_mm: float,
    y_mm: float,
    z_mm: float,
    wpr_deg: tuple[float, float, float],
    speed: float,
    term_type: str = "FINE",
    term_value: int = 0,
    gripper: float | None = None,
) -> None:
    action = {
        "j0": float(x_mm),
        "j1": float(y_mm),
        "j2": float(z_mm),
        "j3": float(wpr_deg[0]),
        "j4": float(wpr_deg[1]),
        "j5": float(wpr_deg[2]),
        "speed": int(speed),
        "speed_type": "mmSec",
        "term_type": term_type,
        "term_value": int(term_value),
    }
    if gripper is not None:
        action["j7"] = float(gripper)
    sent = robot.send_action(action)
    print(
        f"  sent FRC_LinearMotion seq={sent['sequence_id']} "
        f"xyz=({x_mm:.1f},{y_mm:.1f},{z_mm:.1f}) "
        f"wpr=({wpr_deg[0]:.1f},{wpr_deg[1]:.1f},{wpr_deg[2]:.1f})"
    )
    wait_motion_ack(robot, int(sent["sequence_id"]))
    robot.commit_commanded_pose(
        float(x_mm),
        float(y_mm),
        float(z_mm),
        float(wpr_deg[0]),
        float(wpr_deg[1]),
        float(wpr_deg[2]),
    )


def jog_xy_wasd(
    robot: Fanuc,
    *,
    start_xy_mm: tuple[float, float],
    z_mm: float,
    wpr_deg: tuple[float, float, float],
    speed: float = 40.0,
    step_mm: float = 5.0,
) -> tuple[float, float]:
    x0, y0 = float(start_xy_mm[0]), float(start_xy_mm[1])
    x, y = x0, y0
    step = float(step_mm)

    print()
    print(_HELP)
    print("Mode: line (type then Enter)")

    def _goto(nx: float, ny: float) -> None:
        move_fanuc_xyz(
            robot,
            x_mm=nx,
            y_mm=ny,
            z_mm=z_mm,
            wpr_deg=wpr_deg,
            speed=speed,
        )

    def _status() -> None:
        print(f"  pose=({x:.1f}, {y:.1f})  Δ=({x - x0:+.1f}, {y - y0:+.1f}) mm  step={step:.1f}")

    def _apply_key(ch: str) -> None:
        nonlocal x, y
        sx, sy = _KEY_DELTA_BASE_MM[ch]
        x += sx * step
        y += sy * step
        _goto(x, y)

    _status()
    while True:
        try:
            line = input("jog> ").strip().lower()
        except EOFError as exc:
            raise KeyboardInterrupt("jog aborted (EOF)") from exc
        if not line:
            continue
        tokens = line.replace(",", " ").split()
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in ("ok", "done", "enter", "yes", "y"):
                return x - x0, y - y0
            if tok in ("q", "quit", "abort"):
                raise KeyboardInterrupt("jog aborted")
            if tok == "r":
                x, y = x0, y0
                _goto(x, y)
                _status()
                i += 1
                continue
            if tok == "step" and i + 1 < len(tokens):
                step = max(1.0, min(50.0, float(tokens[i + 1])))
                print(f"  step → {step:.1f} mm")
                i += 2
                continue
            if tok in _KEY_DELTA_BASE_MM:
                n = 1
                if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    n = max(1, int(tokens[i + 1]))
                    i += 1
                for _ in range(n):
                    _apply_key(tok)
                _status()
                i += 1
                continue
            if all(c in _KEY_DELTA_BASE_MM for c in tok):
                for c in tok:
                    _apply_key(c)
                _status()
                i += 1
                continue
            print(f"  unknown token {tok!r}; try: d / w 3 / step 10 / r / ok / q")
            i += 1
