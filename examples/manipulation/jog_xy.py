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

"""WASD jog in xArm base XY after a hover (SSH-friendly line mode)."""

from __future__ import annotations

import sys
import termios
import tty


# Measured on this cell (operator view at the table):
#   +X ≈ up,  -X ≈ down,  +Y ≈ left,  -Y ≈ right.
# WASD follows that view (not the camera-frame guess).
_KEY_DELTA_BASE_MM = {
    "w": (+1.0, 0.0),  # +X  up
    "s": (-1.0, 0.0),  # -X  down
    "a": (0.0, +1.0),  # +Y  left
    "d": (0.0, -1.0),  # -Y  right
}

_HELP = (
    "WASD jog (xArm base mm, operator view at table):\n"
    "  w = +X (上)   s = -X (下)   a = +Y (左)   d = -Y (右)\n"
    "Line mode (SSH-safe) — type a line and Enter:\n"
    "  w          one step up (+X)\n"
    "  d 3        three steps right (-Y)\n"
    "  w w a      multiple keys on one line\n"
    "  step 10    set step mm\n"
    "  r          reset to start hover\n"
    "  ok / done  confirm\n"
    "  q          abort\n"
)


def _getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _stdin_is_tty() -> bool:
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


def jog_xy_wasd(
    arm,
    *,
    start_xy_mm: tuple[float, float],
    z_mm: float,
    roll: float,
    speed: float = 40.0,
    step_mm: float = 5.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
    raw_keys: bool = False,
) -> tuple[float, float]:
    """
    Interactive base-XY jog. Returns ``(dx, dy)`` relative to ``start_xy_mm``.

    Default is **line mode** (works over SSH). Pass ``raw_keys=True`` only on a
    local TTY for single-keypress WASD.
    """
    x0, y0 = float(start_xy_mm[0]), float(start_xy_mm[1])
    x, y = x0, y0
    step = float(step_mm)
    use_raw = bool(raw_keys) and _stdin_is_tty()

    print()
    print(_HELP)
    if raw_keys and not use_raw:
        print("stdin is not a TTY → falling back to line mode.")
    print("Mode:", "raw keys" if use_raw else "line (type then Enter)")

    def _goto(nx: float, ny: float) -> None:
        code = arm.set_position(
            nx,
            ny,
            z_mm,
            roll,
            pitch,
            yaw,
            speed=speed,
            mvacc=300,
            wait=True,
            is_radian=True,
        )
        if code != 0:
            raise RuntimeError(f"jog move failed code={code} err={arm.error_code}")

    def _status() -> None:
        dx, dy = x - x0, y - y0
        print(f"  pose=({x:.1f}, {y:.1f})  Δ=({dx:+.1f}, {dy:+.1f}) mm  step={step:.1f}")

    def _apply_key(ch: str) -> None:
        nonlocal x, y
        sx, sy = _KEY_DELTA_BASE_MM[ch]
        x += sx * step
        y += sy * step
        _goto(x, y)

    _status()

    if use_raw:
        print("Raw: WASD / +/- / r / Enter=ok / q=quit")
        while True:
            ch = _getch().lower()
            if ch in ("\r", "\n"):
                return x - x0, y - y0
            if ch in ("q", "\x03"):
                raise KeyboardInterrupt("jog aborted")
            if ch == "+":
                step = min(50.0, step + 1.0)
                _status()
                continue
            if ch == "-":
                step = max(1.0, step - 1.0)
                _status()
                continue
            if ch == "r":
                x, y = x0, y0
                _goto(x, y)
                _status()
                continue
            if ch in _KEY_DELTA_BASE_MM:
                _apply_key(ch)
                _status()
        # unreachable

    # Line mode — reliable over SSH / IDE terminals.
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
            # Allow glued keys: "ddaw"
            if all(c in _KEY_DELTA_BASE_MM for c in tok):
                for c in tok:
                    _apply_key(c)
                _status()
                i += 1
                continue
            print(f"  unknown token {tok!r}; try: d / w 3 / step 10 / r / ok / q")
            i += 1
