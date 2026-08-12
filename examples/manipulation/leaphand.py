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
# Manual teach open/close poses (torque off → pose by hand → Enter):
python -m manipulation.leaphand --calibrate-poses \\
  --calib ../tabletop_perception/calib/xarm_overhead.json
```
"""

from __future__ import annotations

import argparse
import json
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
# Goal Current register unit (LEAP_Hand_API DEFAULT_CUR_SCALE): mA = raw * 1.34.
CURRENT_SCALE_MA = 1.34
GOAL_CURRENT_ADDRESS = 102


def leap_rad_to_raw(leap_rad: float) -> int:
    return int(round(float(leap_rad) / POSITION_SCALE))


# Allegro offset = (raw - 2048) * POSITION_SCALE; set_allegro adds π.
# LEAP MCP-forward: π ≈ flat open, *higher* = curl closed (NOT lower=open).
# Index/middle MCP (1/5) use stock π (no offset). Only joint 6 keeps a measured
# open raw. Driving MCP to leap≈0 parks them on the wrong side of π → looks shut.
MEASURED_OPEN_RAW = {6: 3772}
MEASURED_ALLEGRO_OFFSETS = {
    motor_id: (open_raw - 2048) * POSITION_SCALE
    for motor_id, open_raw in MEASURED_OPEN_RAW.items()
}
# MCP-forward joints that must reach π for a visually flat finger.
_MCP_FWD_IDS = (1, 5, 9)
# Slow ramp present→π.
_DEFAULT_OPEN_DURATION_S = 2.5
_STALL_CURRENT_MA = 300.0
_STALL_POS_ERR_RAD = 0.5

# Allegro convention: 0 = open, positive = curl closed.
_OPEN_ALLEGRO = np.zeros(16, dtype=float)
_DEFAULT_MAX_JOINT_RAD = 1.2
_LEAP_FLAT_OPEN = float(np.pi)


def build_open_allegro(*, thumb_open_back_rad: float = -0.6) -> np.ndarray:
    """Open pose: fingers flat; thumb MCP-forward negative = park fully back."""
    out = _OPEN_ALLEGRO.copy()
    # 13 = thumb MCP forward; negative allegro opens further toward the back.
    out[13] = float(np.clip(thumb_open_back_rad, -1.2, 0.0))
    out[12] = 0.0
    out[14] = 0.0
    out[15] = 0.0
    return out


def build_close_allegro(
    *,
    max_joint_rad: float = _DEFAULT_MAX_JOINT_RAD,
    thumb_side_scale: float = 0.0,
    thumb_forward_scale: float = 1.2,
) -> np.ndarray:
    """
    Grasp close in Allegro space (before measured open offsets).

    Thumb LEAP ids: 12=MCP side, 13=MCP forward, 14=PIP, 15=DIP.
    Power grasp: flex from the back (13/14/15); keep side (12) at 0.
    """
    c = float(np.clip(max_joint_rad, 0.05, 1.2))
    out = np.zeros(16, dtype=float)
    # index / middle / ring: MCP-fwd, PIP, DIP (leave MCP-side at 0)
    for base in (0, 4, 8):
        out[base + 1] = c
        out[base + 2] = c
        out[base + 3] = c * 0.9
    # thumb: strong MCP-forward opposition (LEAP thumb MCP-fwd max ~2.44)
    side = float(np.clip(thumb_side_scale, 0.0, 1.0))
    fwd = float(np.clip(thumb_forward_scale, 0.0, 2.0))
    thumb_c = float(np.clip(c * fwd, 0.05, 2.2))
    out[12] = c * side
    out[13] = thumb_c
    out[14] = thumb_c * 0.95
    out[15] = thumb_c * 0.85
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


def _parse_leap_pose(value: Any, *, name: str) -> np.ndarray | None:
    """Parse a 16-joint LEAP pose from calib JSON (list of floats)."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 16:
        raise ValueError(f"leaphand.{name} must have 16 joints, got {arr.size}")
    return arr


def _thumb_summary(leap: np.ndarray) -> str:
    """Readable thumb joints 12–15 (side / back / pip / dip)."""
    t = np.asarray(leap, dtype=float).reshape(16)
    return (
        f"thumb12_side={t[12]:.3f} thumb13_back={t[13]:.3f} "
        f"thumb14_pip={t[14]:.3f} thumb15_dip={t[15]:.3f}"
    )


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
        open_duration_s: float = _DEFAULT_OPEN_DURATION_S,
        ramp_hz: float = 30.0,
        api_dir: Path | str = _DEFAULT_API,
        kP: int = 600,
        kI: int = 0,
        kD: int = 200,
        max_joint_rad: float = _DEFAULT_MAX_JOINT_RAD,
        thumb_side_scale: float = 0.0,
        thumb_forward_scale: float = 1.2,
        thumb_open_back_rad: float = -0.6,
        allegro_open_offsets: dict[int, float] | None = None,
        use_measured_calibration: bool = True,
        open_leap: np.ndarray | None = None,
        close_leap: np.ndarray | None = None,
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
        self.thumb_side_scale = float(np.clip(thumb_side_scale, 0.0, 1.0))
        self.thumb_forward_scale = float(np.clip(thumb_forward_scale, 0.0, 2.0))
        self.thumb_open_back_rad = float(np.clip(thumb_open_back_rad, -1.2, 0.0))
        self.use_measured_calibration = bool(use_measured_calibration)
        self.open_leap = (
            np.asarray(open_leap, dtype=float).reshape(16) if open_leap is not None else None
        )
        self.close_leap = (
            np.asarray(close_leap, dtype=float).reshape(16)
            if close_leap is not None
            else None
        )
        self.allegro_open_offsets = (
            dict(allegro_open_offsets)
            if allegro_open_offsets is not None
            else (dict(MEASURED_ALLEGRO_OFFSETS) if self.use_measured_calibration else {})
        )
        # Force stock flat-open (π) on MCP-forward — do not inherit stale raw.
        for mid in _MCP_FWD_IDS:
            self.allegro_open_offsets.pop(int(mid), None)
        self._node: Any | None = None
        self._norm: float = 0.0  # 0 open, 1 closed
        self._allegro_cmd = build_open_allegro(
            thumb_open_back_rad=self.thumb_open_back_rad
        )
        self._leap_cmd = (
            np.asarray(self.open_leap, dtype=float).copy()
            if self.open_leap is not None
            else np.full(16, _LEAP_FLAT_OPEN, dtype=float)
        )
        self._mcp_soft_open_leap: dict[int, float] = {}

    @property
    def uses_manual_poses(self) -> bool:
        return self.open_leap is not None and self.close_leap is not None

    def open_pose_allegro(self) -> np.ndarray:
        return build_open_allegro(thumb_open_back_rad=self.thumb_open_back_rad)

    def close_pose_allegro(self) -> np.ndarray:
        return build_close_allegro(
            max_joint_rad=self.max_joint_rad,
            thumb_side_scale=self.thumb_side_scale,
            thumb_forward_scale=self.thumb_forward_scale,
        )

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
            open_duration_s=float(
                merged.get("open_duration_s", _DEFAULT_OPEN_DURATION_S)
            ),
            ramp_hz=float(merged.get("ramp_hz", 30.0)),
            api_dir=Path(merged.get("api_dir", _DEFAULT_API)),
            kP=int(merged.get("kP", 600)),
            kI=int(merged.get("kI", 0)),
            kD=int(merged.get("kD", 200)),
            max_joint_rad=max_j,
            thumb_side_scale=float(merged.get("thumb_side_scale", 0.0)),
            thumb_forward_scale=float(merged.get("thumb_forward_scale", 1.2)),
            thumb_open_back_rad=float(merged.get("thumb_open_back_rad", -0.6)),
            allegro_open_offsets=(
                _allegro_offsets_from_calib(merged) if use_meas else {}
            ),
            use_measured_calibration=use_meas,
            open_leap=_parse_leap_pose(merged.get("open_leap"), name="open_leap"),
            close_leap=_parse_leap_pose(merged.get("close_leap"), name="close_leap"),
        )

    def connect(self, *, open_on_connect: bool = True) -> None:
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
        # curr_lim is mA. Stock LeapNode writes the number raw (350 → ~469 mA);
        # match quest_leap_real: raw = round(mA / 1.34).
        self._node = {
            "client": client,
            "motors": motors,
            "lhu": lhu,
            "curr_pos": None,
        }
        self._set_goal_current_ma(self.curr_lim)

        if self.uses_manual_poses:
            self._node["curr_pos"] = np.asarray(self.open_leap, dtype=float).copy()
            self._leap_cmd = np.asarray(self.open_leap, dtype=float).copy()
        else:
            open_allegro = apply_allegro_offsets(
                self.open_pose_allegro(), self.allegro_open_offsets
            )
            # Match LeapNode.set_allegro: zeros=False, no angle_safety_clip.
            self._node["curr_pos"] = lhu.allegro_to_LEAPhand(open_allegro, zeros=False)
        self._norm = 0.0
        logger.info(
            "LeapHand connected on %s curr_lim=%s mA max_joint_rad=%.2f "
            "manual_poses=%s offsets=%s (MCP 1/5 open at π≈%.3f)",
            self.port,
            self.curr_lim,
            self.max_joint_rad,
            self.uses_manual_poses,
            {k: round(v, 3) for k, v in self.allegro_open_offsets.items()},
            _LEAP_FLAT_OPEN,
        )
        self._seed_allegro_cmd_from_present()
        if open_on_connect:
            self.open(settle=True)

    def disconnect(self, *, go_open: bool = True) -> None:
        if self._node is None:
            return
        try:
            if go_open:
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

    def _set_goal_current_ma(self, current_ma: float) -> None:
        """Write Goal Current so present-current mA limit ≈ ``current_ma``."""
        if self._node is None:
            raise RuntimeError("LeapHand not connected")
        client = self._node["client"]
        motors = self._node["motors"]
        raw = int(round(float(current_ma) / CURRENT_SCALE_MA))
        client.sync_write(motors, np.ones(len(motors)) * raw, GOAL_CURRENT_ADDRESS, 2)
        # Per-motor verify (GroupSyncWrite has no ACK).
        failed: list[int] = []
        for motor_id in motors:
            comm_result, dxl_error = client.packet_handler.write2ByteTxRx(
                client.port_handler, motor_id, GOAL_CURRENT_ADDRESS, raw
            )
            ok = client.handle_packet_result(
                comm_result, dxl_error, motor_id, context="set_goal_current"
            )
            if not ok:
                failed.append(int(motor_id))
                continue
            value, comm_result, dxl_error = client.packet_handler.read2ByteTxRx(
                client.port_handler, motor_id, GOAL_CURRENT_ADDRESS
            )
            ok = client.handle_packet_result(
                comm_result, dxl_error, motor_id, context="verify_goal_current"
            )
            if not ok or int(value) != raw:
                failed.append(int(motor_id))
        if failed:
            raise RuntimeError(f"Could not verify Goal Current for motors: {failed}")
        logger.info(
            "Verified Goal Current: raw=%s → limit=%.1f mA (curr_lim=%s)",
            raw,
            raw * CURRENT_SCALE_MA,
            current_ma,
        )

    def _seed_allegro_cmd_from_present(self) -> None:
        """Start ramps from actual leap so MCP can travel 0→π smoothly."""
        if self._node is None:
            return
        act = np.array(self._node["client"].read_pos(), dtype=float)
        # Inverse of allegro_to_LEAPhand(..., zeros=False): allegro = leap - π
        allegro = act - _LEAP_FLAT_OPEN
        for mid, off in self.allegro_open_offsets.items():
            if 0 <= int(mid) < len(allegro):
                allegro[int(mid)] -= float(off)
        self._allegro_cmd = allegro.astype(float)
        logger.info(
            "Seed allegro from present MCP leap idx/mid/ring=%.3f/%.3f/%.3f → ramp to π",
            float(act[1]),
            float(act[5]),
            float(act[9]),
        )

    def _adopt_soft_open_for_stalled_mcp(self) -> None:
        """
        If MCP-forward cannot reach π within curr_lim, stop fighting: rebind
        open to present leap so close stays a relative curl from a holdable pose.
        """
        if self._node is None:
            return
        act = np.array(self._node["client"].read_pos(), dtype=float)
        cur = np.array(self._node["client"].read_cur(), dtype=float)
        stalled: list[int] = []
        for mid in _MCP_FWD_IDS:
            err = _LEAP_FLAT_OPEN - float(act[mid])
            if abs(err) < _STALL_POS_ERR_RAD:
                self._mcp_soft_open_leap.pop(int(mid), None)
                self.allegro_open_offsets.pop(int(mid), None)
                continue
            # At/near current ceiling and still far from π → adopt present as open.
            if abs(float(cur[mid])) >= _STALL_CURRENT_MA or abs(err) > 1.0:
                open_leap = float(act[mid])
                self._mcp_soft_open_leap[int(mid)] = open_leap
                self.allegro_open_offsets[int(mid)] = open_leap - _LEAP_FLAT_OPEN
                stalled.append(int(mid))
        if stalled:
            # Hold at present (no more 350mA fight toward unreachable π).
            self._set_allegro(self.open_pose_allegro())
            logger.warning(
                "MCP-forward %s cannot reach π at curr_lim=%smA "
                "(PIP/DIP are already open). Soft-open leap=%s — "
                "finger looks shut because MCP stalled, not because open cmd wrong.",
                stalled,
                self.curr_lim,
                {k: round(v, 3) for k, v in self._mcp_soft_open_leap.items()},
            )

    def set_torque_enabled(self, enabled: bool) -> None:
        if self._node is None:
            raise RuntimeError("LeapHand not connected")
        self._node["client"].set_torque_enabled(self._node["motors"], bool(enabled))
        logger.info("LeapHand torque %s", "ON" if enabled else "OFF")

    def read_leap_pose(self) -> np.ndarray:
        if self._node is None:
            raise RuntimeError("LeapHand not connected")
        return np.array(self._node["client"].read_pos(), dtype=float).reshape(16)

    def _set_leap(self, pose: np.ndarray) -> None:
        """Command LEAP joint angles directly (manual taught poses)."""
        if self._node is None:
            raise RuntimeError("LeapHand not connected")
        leap_pose = np.asarray(pose, dtype=float).reshape(16).copy()
        self._node["client"].write_desired_pos(self._node["motors"], leap_pose)
        self._node["curr_pos"] = leap_pose
        self._leap_cmd = leap_pose
        # Keep allegro mirror approximate for mixed code paths.
        try:
            self._allegro_cmd = self._node["lhu"].LEAPhand_to_allegro(leap_pose, zeros=False)
        except Exception:  # noqa: BLE001
            pass

    def _ramp_leap(self, start: np.ndarray, goal: np.ndarray, duration_s: float) -> None:
        if duration_s <= 0.0:
            self._set_leap(goal)
            return
        n = max(2, int(round(duration_s * self.ramp_hz)))
        dt = duration_s / float(n)
        start = np.asarray(start, dtype=float).reshape(16)
        goal = np.asarray(goal, dtype=float).reshape(16)
        for i in range(1, n + 1):
            t0 = time.monotonic()
            alpha = float(i) / float(n)
            self._set_leap((1.0 - alpha) * start + alpha * goal)
            sleep = dt - (time.monotonic() - t0)
            if sleep > 0.0:
                time.sleep(sleep)

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
        self._leap_cmd = np.asarray(leap_pose, dtype=float).copy()
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
        if self.uses_manual_poses:
            if settle:
                self._leap_cmd = self.read_leap_pose()
            goal = np.asarray(self.open_leap, dtype=float)
            if settle and self.open_duration_s > 0.0:
                logger.info(
                    "LeapHand open (manual pose) ramp %.2fs %s",
                    self.open_duration_s,
                    _thumb_summary(goal),
                )
                self._ramp_leap(self._leap_cmd, goal, self.open_duration_s)
            else:
                self._set_leap(goal)
            self._norm = 0.0
            if settle:
                time.sleep(self.settle_s)
            try:
                act = self.read_leap_pose()
                logger.info(
                    "LeapHand open manual act %s",
                    _thumb_summary(act),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("LeapHand open read failed: %s", exc)
            return

        # Always seed from present so a large 0→π MCP move is ramped, not stepped.
        if settle:
            self._seed_allegro_cmd_from_present()
        open_pose = self.open_pose_allegro()
        if settle and self.open_duration_s > 0.0:
            logger.info(
                "LeapHand open ramp %.2fs toward flat MCP π≈%.3f "
                "thumb_open_back=%.2f",
                self.open_duration_s,
                _LEAP_FLAT_OPEN,
                self.thumb_open_back_rad,
            )
            self._ramp_allegro(self._allegro_cmd, open_pose, self.open_duration_s)
        else:
            self._set_allegro(open_pose)
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
            logger.info(
                "Open check PIP/DIP idx=%.2f/%.2f mid=%.2f/%.2f ring=%.2f/%.2f "
                "(should be ≈π if software open ok)",
                act[2],
                act[3],
                act[6],
                act[7],
                act[10],
                act[11],
            )
            logger.info(
                "Open check MCP-fwd idx/mid/ring=%.2f/%.2f/%.2f (need ≈π to look open)",
                act[1],
                act[5],
                act[9],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LeapHand open read failed: %s", exc)
        if settle:
            self._adopt_soft_open_for_stalled_mcp()

    def close(self, *, settle: bool = True) -> None:
        if self.uses_manual_poses:
            if settle:
                self._leap_cmd = self.read_leap_pose()
            goal = np.asarray(self.close_leap, dtype=float)
            if settle and self.close_duration_s > 0.0:
                logger.info(
                    "LeapHand close (manual pose) ramp %.2fs %s",
                    self.close_duration_s,
                    _thumb_summary(goal),
                )
                self._ramp_leap(self._leap_cmd, goal, self.close_duration_s)
            else:
                self._set_leap(goal)
            self._norm = 1.0
            if settle:
                time.sleep(self.settle_s)
            try:
                act = self.read_leap_pose()
                logger.info("LeapHand close manual act %s", _thumb_summary(act))
            except Exception as exc:  # noqa: BLE001
                logger.warning("LeapHand close read failed: %s", exc)
            return

        allegro = self.close_pose_allegro()
        if settle and self.close_duration_s > 0.0:
            logger.info(
                "LeapHand close ramp %.2fs @ %.0f Hz max_joint_rad=%.2f "
                "thumb_side=%.2f thumb_fwd=%.2f",
                self.close_duration_s,
                self.ramp_hz,
                self.max_joint_rad,
                self.thumb_side_scale,
                self.thumb_forward_scale,
            )
            self._ramp_allegro(self._allegro_cmd, allegro, self.close_duration_s)
        else:
            self._set_allegro(allegro)
        sent = self._node["curr_pos"]
        try:
            act = np.array(self._node["client"].read_pos(), dtype=float)
            logger.info(
                "LeapHand close done leap index=%.2f/%.2f/%.2f middle=%.2f/%.2f/%.2f "
                "act_mcp idx/mid=%.2f/%.2f",
                sent[1],
                sent[2],
                sent[3],
                sent[5],
                sent[6],
                sent[7],
                act[1],
                act[5],
            )
        except Exception as exc:  # noqa: BLE001
            logger.info(
                "LeapHand close done leap index=%.2f/%.2f/%.2f middle=%.2f/%.2f/%.2f "
                "(act read failed: %s)",
                sent[1],
                sent[2],
                sent[3],
                sent[5],
                sent[6],
                sent[7],
                exc,
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

    def diagnose_motors(self, motor_ids: list[int] | None = None) -> None:
        """Read goal/present/current/torque/hw-error for MCP debug."""
        if self._node is None:
            raise RuntimeError("LeapHand not connected")
        ids = motor_ids or [1, 5, 6]
        client = self._node["client"]
        ph = client.port_handler
        pkt = client.packet_handler
        scale = float(client._pos_vel_cur_reader.pos_scale)
        # Dynamixel X control table
        addr_mode, addr_torque, addr_hw = 11, 64, 70
        addr_goal_cur, addr_goal_pos = 102, 116
        print("motor_diag (after commanding current allegro pose):")
        for mid in ids:
            mode, c1, e1 = pkt.read1ByteTxRx(ph, mid, addr_mode)
            torque, c2, e2 = pkt.read1ByteTxRx(ph, mid, addr_torque)
            hw, c3, e3 = pkt.read1ByteTxRx(ph, mid, addr_hw)
            gcur, c4, e4 = pkt.read2ByteTxRx(ph, mid, addr_goal_cur)
            gpos, c5, e5 = pkt.read4ByteTxRx(ph, mid, addr_goal_pos)
            # present pos/cur via bulk reader
            pos = float(client.read_pos()[mid])
            cur = float(client.read_cur()[mid])
            goal_rad = (int(gpos) & 0xFFFFFFFF)
            if goal_rad >= 2**31:
                goal_rad -= 2**32
            goal_rad = float(goal_rad) * scale
            ok = all(
                client.handle_packet_result(c, e, mid, context="diag")
                for c, e in ((c1, e1), (c2, e2), (c3, e3), (c4, e4), (c5, e5))
            )
            print(
                f"  id={mid} ok={ok} mode={mode} torque={torque} hw_err={hw} "
                f"goal_cur_raw={gcur} (≈{gcur * CURRENT_SCALE_MA:.1f}mA) "
                f"goal={goal_rad:.3f} act={pos:.3f} "
                f"err={goal_rad - pos:.3f} cur={cur:.1f}mA"
            )

    def read_gripper_width(self) -> float:
        """Pseudo width (m) for ArmExecutor checks: open≈0.08, closed≈0.03."""
        return 0.08 - 0.05 * float(self._norm)

    @property
    def commanded_norm(self) -> float:
        return self._norm


def _default_calib_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "tabletop_perception"
        / "calib"
        / "xarm_overhead.json"
    )


def _save_manual_poses(calib_path: Path, open_leap: np.ndarray, close_leap: np.ndarray) -> None:
    calib = json.loads(calib_path.read_text())
    block = dict(calib.get("leaphand") or {})
    block["open_leap"] = [round(float(x), 5) for x in np.asarray(open_leap).reshape(16)]
    block["close_leap"] = [round(float(x), 5) for x in np.asarray(close_leap).reshape(16)]
    block["_comment_manual_poses"] = (
        "open_leap/close_leap: taught 16-DOF LEAP poses. When both set, "
        "open()/close() use them directly (skip formula thumb scales)."
    )
    calib["leaphand"] = block
    calib_path.write_text(json.dumps(calib, indent=2) + "\n")
    logger.info("Wrote open_leap/close_leap → %s", calib_path)


def _calibrate_poses_interactive(hand: LeapHandEE, calib_path: Path) -> int:
    print("=" * 60)
    print("LeapHand 手动标定：张开 / 闭合")
    print("流程：力矩关闭 → 你用手掰到目标姿势 → 回车记录")
    print("关节提示：拇指 12=侧向 13=后屈 14/15=PIP/DIP")
    print("=" * 60)
    hand.connect(open_on_connect=False)
    try:
        hand.set_torque_enabled(False)
        input("\n【1/2】掰到「张开」姿势后按 Enter 记录 … ")
        open_leap = hand.read_leap_pose()
        print("OPEN:", _thumb_summary(open_leap))
        print("full=", [round(float(x), 4) for x in open_leap])

        input("\n【2/2】掰到「闭合」姿势后按 Enter 记录 … ")
        close_leap = hand.read_leap_pose()
        print("CLOSE:", _thumb_summary(close_leap))
        print("full=", [round(float(x), 4) for x in close_leap])

        _save_manual_poses(calib_path, open_leap, close_leap)
        hand.open_leap = open_leap.copy()
        hand.close_leap = close_leap.copy()

        ans = input("\n力矩开启并试跑 open→close→open？[y/N] ").strip().lower()
        if ans in {"y", "yes"}:
            hand.set_torque_enabled(True)
            print("open …")
            hand.open()
            print("close …")
            hand.close()
            print("open …")
            hand.open()
            print("试跑完成。若不对可再跑 --calibrate-poses 覆盖。")
        else:
            print("已保存标定，未试跑。之后 open/close 会直接用这两档姿势。")
        return 0
    finally:
        hand.disconnect(go_open=False)


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
    p.add_argument(
        "--diagnose",
        action="store_true",
        help="Command open then print goal/act/torque/hw_err for motors 1,5,6",
    )
    p.add_argument(
        "--calibrate-poses",
        action="store_true",
        help="Teach open/close by hand (torque off) and write open_leap/close_leap to calib",
    )
    p.add_argument("--calib", type=Path, default=None)
    args = p.parse_args(argv)

    calib_path = args.calib if args.calib is not None else _default_calib_path()
    if args.calibrate_poses or args.calib is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tabletop_perception.run_xarm_live import _load_calib

        hand = LeapHandEE.from_calib(_load_calib(calib_path), "xarm2")
    else:
        hand = LeapHandEE()
    if args.port:
        hand.port = args.port
    if args.curr_lim is not None:
        hand.curr_lim = int(args.curr_lim)
    if args.max_joint_rad is not None:
        hand.max_joint_rad = float(np.clip(args.max_joint_rad, 0.05, 1.2))

    if args.calibrate_poses:
        return _calibrate_poses_interactive(hand, calib_path)

    hand.connect()
    try:
        if args.suggest_open_raw:
            print("open … (keep hand free / open stop)")
            hand.open()
            sug = hand.read_open_raw_suggestion()
            print(f"suggest MEASURED_OPEN_RAW = {sug}")
            print("Tune: ↑ raw → deeper close; ↓ raw → opener / weaker fist (±50..150).")
            return 0
        if args.diagnose:
            print("open …")
            hand.open()
            hand.diagnose_motors([1, 5, 6, 2, 3])
            print("close …")
            hand.close()
            hand.diagnose_motors([1, 5, 6, 2, 3])
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
