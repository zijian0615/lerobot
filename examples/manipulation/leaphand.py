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

"""LeapHand end-effector for tabletop grasp/place (open / close poses).

Uses the official LEAP_Hand_API Dynamixel client under ``~/LEAP_Hand_API/python``.

```bash
cd ~/lerobot/examples
python -m manipulation.leaphand --open
python -m manipulation.leaphand --close
python -m manipulation.leaphand --cycle
```
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_API = Path.home() / "LEAP_Hand_API" / "python"
_DEFAULT_PORT = (
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO3U2-if00-port0"
)

# Dynamixel raw ↔ rad (same as LEAP_Hand_API DEFAULT_POS_SCALE).
POSITION_SCALE = 2.0 * np.pi / 4096


def leap_rad_to_raw(leap_rad: float) -> int:
    return int(round(float(leap_rad) / POSITION_SCALE))


# Mechanical open stop in Dynamixel raw counts (not teleop's single-motor
# "safe" ID6=-169). Allegro offset = (raw - 2048) * POSITION_SCALE, then
# set_allegro adds π. Tuned from open act on 2026-08-08:
#   act leap idx_mcp/mid_mcp/mid_pip ≈ 4.54 / 4.95 / 5.76
# Tuning: ↑ raw → same max_joint_rad closes deeper; ↓ raw → more open / weaker fist.
# Nudge ±50..150 per joint while watching index/middle.
MEASURED_OPEN_RAW = {1: 2978, 5: 3234, 6: 3772}
MEASURED_ALLEGRO_OFFSETS = {
    motor_id: (open_raw - 2048) * POSITION_SCALE
    for motor_id, open_raw in MEASURED_OPEN_RAW.items()
}

# Allegro convention: 0 = open, positive = curl closed.
# With measured calibration, teleop used max_joint_rad <= 1.2 (relative curl).
_OPEN_ALLEGRO = np.zeros(16, dtype=float)
_DEFAULT_MAX_JOINT_RAD = 1.2


def build_close_allegro(*, max_joint_rad: float = _DEFAULT_MAX_JOINT_RAD) -> np.ndarray:
    """
    Grasp close in Allegro space (before measured open offsets).

    Matches quest_leap_real --use-measured-calibration: relative curl capped at
    ``max_joint_rad`` (default 1.2), then offsets are added in ``_set_allegro``.
    """
    c = float(np.clip(max_joint_rad, 0.05, 1.2))
    out = np.zeros(16, dtype=float)
    # index / middle / ring: MCP-fwd, PIP, DIP
    for base in (0, 4, 8):
        out[base + 1] = c
        out[base + 2] = c
        out[base + 3] = c * 0.9
    # thumb
    out[12] = c * 0.9
    out[13] = c * 0.85
    out[14] = c
    out[15] = c * 0.8
    return out


def _ensure_leap_api(api_dir: Path) -> None:
    api = str(api_dir.resolve())
    if api not in sys.path:
        sys.path.insert(0, api)


def _allegro_offsets_from_raw(open_raw: dict[Any, Any]) -> dict[int, float]:
    """``offset = (open_raw - 2048) * POSITION_SCALE`` — keep negative raw as-is."""
    out: dict[int, float] = {}
    for k, v in open_raw.items():
        if str(k).startswith("_"):
            continue
        out[int(k)] = (float(v) - 2048.0) * POSITION_SCALE
    return out


def _allegro_offsets_from_calib(merged: dict[str, Any]) -> dict[int, float]:
    if merged.get("measured_open_raw"):
        return _allegro_offsets_from_raw(dict(merged["measured_open_raw"]))
    if merged.get("measured_allegro_offsets"):
        return {
            int(k): float(v)
            for k, v in dict(merged["measured_allegro_offsets"]).items()
            if not str(k).startswith("_")
        }
    return dict(MEASURED_ALLEGRO_OFFSETS)


def apply_allegro_offsets(
    joints: np.ndarray,
    offsets: dict[int, float],
) -> np.ndarray:
    """Apply per-motor software reference offsets without touching EEPROM."""
    calibrated = np.asarray(joints, dtype=float).copy()
    for motor_id, offset in offsets.items():
        calibrated[int(motor_id)] += float(offset)
    return calibrated


class LeapHandEE:
    """Minimal open/close wrapper around LEAP ``LeapNode``."""

    def __init__(
        self,
        *,
        port: str = _DEFAULT_PORT,
        baud: int = 4_000_000,
        curr_lim: int = 100,
        settle_s: float = 0.8,
        close_duration_s: float = 1.8,
        open_duration_s: float = 0.6,
        ramp_hz: float = 30.0,
        api_dir: Path | str = _DEFAULT_API,
        kP: int = 600,
        kI: int = 0,
        kD: int = 200,
        max_joint_rad: float = _DEFAULT_MAX_JOINT_RAD,
        allegro_open_offsets: dict[int, float] | None = None,
        use_measured_calibration: bool = True,
    ) -> None:
        self.port = port
        self.baud = int(baud)
        # Measured-calibration path was validated at 100 mA (quest_leap_real).
        self.curr_lim = int(curr_lim)
        self.settle_s = float(settle_s)
        self.close_duration_s = float(max(0.0, close_duration_s))
        self.open_duration_s = float(max(0.0, open_duration_s))
        self.ramp_hz = float(max(1.0, ramp_hz))
        self.api_dir = Path(api_dir)
        self.kP = int(kP)
        self.kI = int(kI)
        self.kD = int(kD)
        self.max_joint_rad = float(np.clip(max_joint_rad, 0.05, 1.2))
        self.use_measured_calibration = bool(use_measured_calibration)
        self.allegro_open_offsets = (
            dict(allegro_open_offsets)
            if allegro_open_offsets is not None
            else (dict(MEASURED_ALLEGRO_OFFSETS) if self.use_measured_calibration else {})
        )
        self._node: Any | None = None
        self._norm: float = 0.0  # 0 open, 1 closed
        self._allegro_cmd = _OPEN_ALLEGRO.copy()

    def close_pose_allegro(self) -> np.ndarray:
        return build_close_allegro(max_joint_rad=self.max_joint_rad)

    @classmethod
    def from_calib(cls, calib: dict[str, Any], arm_name: str = "xarm2") -> LeapHandEE:
        robots = dict(calib.get("robots") or {})
        exe_by = dict(calib.get("execution_by_arm") or {})
        arm_robot = dict(robots.get(arm_name) or {})
        arm_exe = dict(exe_by.get(arm_name) or {})
        cfg = dict(calib.get("leaphand") or {})
        # Prefer nested per-arm block, then top-level leaphand{}, then defaults.
        nested = dict(arm_exe.get("leaphand") or arm_robot.get("leaphand") or {})
        merged = {**cfg, **nested}
        use_meas = bool(merged.get("use_measured_calibration", True))
        max_j = float(merged.get("max_joint_rad", _DEFAULT_MAX_JOINT_RAD))
        if use_meas and max_j > 1.2:
            raise ValueError(
                "leaphand use_measured_calibration requires max_joint_rad <= 1.2 "
                f"(got {max_j})"
            )
        return cls(
            port=str(merged.get("port", _DEFAULT_PORT)),
            baud=int(merged.get("baud", 4_000_000)),
            curr_lim=int(merged.get("curr_lim", 100)),
            settle_s=float(merged.get("settle_s", 0.8)),
            close_duration_s=float(merged.get("close_duration_s", 1.8)),
            open_duration_s=float(merged.get("open_duration_s", 0.6)),
            ramp_hz=float(merged.get("ramp_hz", 30.0)),
            api_dir=Path(merged.get("api_dir", _DEFAULT_API)),
            kP=int(merged.get("kP", 600)),
            kI=int(merged.get("kI", 0)),
            kD=int(merged.get("kD", 200)),
            max_joint_rad=max_j,
            allegro_open_offsets=(
                _allegro_offsets_from_calib(merged) if use_meas else {}
            ),
            use_measured_calibration=use_meas,
        )

    def connect(self) -> None:
        if self._node is not None:
            return
        _ensure_leap_api(self.api_dir)
        # Import after path inject (official package layout).
        from leap_hand_utils.dynamixel_client import DynamixelClient
        import leap_hand_utils.leap_hand_utils as lhu

        motors = list(range(16))
        ports = [self.port]
        if self.port != "/dev/ttyUSB0":
            ports.append("/dev/ttyUSB0")
        if self.port != "/dev/ttyUSB1":
            ports.append("/dev/ttyUSB1")

        last_err: Exception | None = None
        client = None
        for p in ports:
            try:
                logger.info("LeapHand connecting port=%s baud=%s", p, self.baud)
                client = DynamixelClient(motors, p, self.baud)
                client.connect()
                self.port = p
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                client = None
        if client is None:
            raise RuntimeError(
                f"LeapHand connect failed (tried {ports}): {last_err}"
            )

        # Mirror LeapNode init (position-current mode + gains).
        client.sync_write(motors, np.ones(len(motors)) * 5, 11, 1)
        client.set_torque_enabled(motors, True)
        client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2)
        client.sync_write([0, 4, 8], np.ones(3) * (self.kP * 0.75), 84, 2)
        client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2)
        client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2)
        client.sync_write([0, 4, 8], np.ones(3) * (self.kD * 0.75), 80, 2)
        client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)

        open_allegro = apply_allegro_offsets(_OPEN_ALLEGRO, self.allegro_open_offsets)
        self._node = {
            "client": client,
            "motors": motors,
            "lhu": lhu,
            # Match LeapNode.set_allegro: zeros=False, no angle_safety_clip.
            "curr_pos": lhu.allegro_to_LEAPhand(open_allegro, zeros=False),
        }
        self._norm = 0.0
        logger.info(
            "LeapHand connected on %s curr_lim=%s max_joint_rad=%.2f offsets=%s",
            self.port,
            self.curr_lim,
            self.max_joint_rad,
            {k: round(v, 3) for k, v in self.allegro_open_offsets.items()},
        )
        self.open(settle=True)

    def disconnect(self) -> None:
        if self._node is None:
            return
        try:
            self.open(settle=False)
            client = self._node["client"]
            client.set_torque_enabled(self._node["motors"], False)
            # DynamixelClient may expose disconnect / close.
            if hasattr(client, "disconnect"):
                client.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("LeapHand disconnect: %s", exc)
        finally:
            self._node = None

    def _set_allegro(self, pose: np.ndarray) -> None:
        """Like LeapNode.set_allegro (+ measured open offsets on joints 1/5/6)."""
        if self._node is None:
            raise RuntimeError("LeapHand not connected")
        cmd = np.asarray(pose, dtype=float).copy()
        lhu = self._node["lhu"]
        allegro = apply_allegro_offsets(cmd, self.allegro_open_offsets)
        # Critical: zeros=False and NO angle_safety_clip — clip would cancel
        # ID6's measured open (allegro≈-3.4 → leap≈-0.26).
        leap_pose = lhu.allegro_to_LEAPhand(allegro, zeros=False)
        self._node["client"].write_desired_pos(self._node["motors"], leap_pose)
        self._node["curr_pos"] = leap_pose
        self._allegro_cmd = cmd

    def _ramp_allegro(self, start: np.ndarray, goal: np.ndarray, duration_s: float) -> None:
        """Linear interpolate allegro pose over ``duration_s`` (pre-offset space)."""
        if duration_s <= 0.0:
            self._set_allegro(goal)
            return
        n = max(2, int(round(duration_s * self.ramp_hz)))
        dt = duration_s / float(n)
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        for i in range(1, n + 1):
            t0 = time.monotonic()
            alpha = float(i) / float(n)
            self._set_allegro((1.0 - alpha) * start + alpha * goal)
            sleep = dt - (time.monotonic() - t0)
            if sleep > 0.0:
                time.sleep(sleep)

    def read_open_raw_suggestion(self, motor_ids: tuple[int, ...] = (1, 5, 6)) -> dict[int, int]:
        """Map current leap positions → suggested MEASURED_OPEN_RAW entries."""
        if self._node is None:
            raise RuntimeError("LeapHand not connected")
        act = np.array(self._node["client"].read_pos(), dtype=float)
        return {int(i): leap_rad_to_raw(act[i]) for i in motor_ids if 0 <= i < len(act)}

    def open(self, *, settle: bool = True) -> None:
        if settle and self.open_duration_s > 0.0:
            self._ramp_allegro(self._allegro_cmd, _OPEN_ALLEGRO, self.open_duration_s)
        else:
            self._set_allegro(_OPEN_ALLEGRO)
        self._norm = 0.0
        if settle:
            time.sleep(self.settle_s)
        try:
            act = np.array(self._node["client"].read_pos(), dtype=float)
            sent = self._node["curr_pos"]
            sug = {i: leap_rad_to_raw(act[i]) for i in (1, 5, 6)}
            logger.info(
                "LeapHand open sent leap idx_mcp/mid_mcp/mid_pip=%.2f/%.2f/%.2f "
                "act=%.2f/%.2f/%.2f  suggest_open_raw=%s",
                sent[1],
                sent[5],
                sent[6],
                act[1],
                act[5],
                act[6],
                sug,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LeapHand open read failed: %s", exc)

    def close(self, *, settle: bool = True) -> None:
        allegro = self.close_pose_allegro()
        if settle and self.close_duration_s > 0.0:
            logger.info(
                "LeapHand close ramp %.2fs @ %.0f Hz max_joint_rad=%.2f",
                self.close_duration_s,
                self.ramp_hz,
                self.max_joint_rad,
            )
            self._ramp_allegro(self._allegro_cmd, allegro, self.close_duration_s)
        else:
            self._set_allegro(allegro)
        sent = self._node["curr_pos"]
        logger.info(
            "LeapHand close done leap index=%.2f/%.2f/%.2f middle=%.2f/%.2f/%.2f",
            sent[1],
            sent[2],
            sent[3],
            sent[5],
            sent[6],
            sent[7],
        )
        self._norm = 1.0
        if settle:
            time.sleep(self.settle_s)

    def gripper(self, cmd: str) -> None:
        if cmd == "open":
            self.open()
        elif cmd == "close":
            self.close()
        else:
            raise ValueError(f"Unknown gripper cmd {cmd!r}")

    def read_gripper_width(self) -> float:
        """Pseudo width (m) for ArmExecutor checks: open≈0.08, closed≈0.03."""
        return 0.08 - 0.05 * float(self._norm)

    @property
    def commanded_norm(self) -> float:
        return self._norm


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="LeapHand open/close smoke test")
    p.add_argument("--port", default=None)
    p.add_argument("--curr-lim", type=int, default=None)
    p.add_argument(
        "--max-joint-rad",
        type=float,
        default=None,
        help="Relative allegro curl cap (default 1.2; required <=1.2 with measured calib)",
    )
    p.add_argument("--open", action="store_true")
    p.add_argument("--close", action="store_true")
    p.add_argument("--cycle", action="store_true", help="open → close → open")
    p.add_argument(
        "--suggest-open-raw",
        action="store_true",
        help="Hold open, print MEASURED_OPEN_RAW suggestion from actual leap positions",
    )
    p.add_argument("--calib", type=Path, default=None)
    args = p.parse_args(argv)

    if args.calib is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tabletop_perception.run_xarm_live import _load_calib

        hand = LeapHandEE.from_calib(_load_calib(args.calib), "xarm2")
    else:
        hand = LeapHandEE()
    if args.port:
        hand.port = args.port
    if args.curr_lim is not None:
        hand.curr_lim = int(args.curr_lim)
    if args.max_joint_rad is not None:
        hand.max_joint_rad = float(np.clip(args.max_joint_rad, 0.05, 1.2))

    hand.connect()
    try:
        if args.suggest_open_raw:
            print("open … (keep hand free / open stop)")
            hand.open()
            sug = hand.read_open_raw_suggestion()
            print(f"suggest MEASURED_OPEN_RAW = {sug}")
            print("Tune: ↑ raw → deeper close; ↓ raw → opener / weaker fist (±50..150).")
            return 0
        if args.cycle or (not args.open and not args.close):
            print("open …")
            hand.open()
            print("close …")
            hand.close()
            print("open …")
            hand.open()
        else:
            if args.open:
                print("open …")
                hand.open()
            if args.close:
                print("close …")
                hand.close()
        print("done.")
        return 0
    finally:
        hand.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
