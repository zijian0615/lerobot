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

import json
import logging
import queue
import socket
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation

from ..robot import Robot
from .config_fanuc import FanucConfig
from .er4ia_fk import controller_tcp_pose, sim_from_controller_deg
from .pose import (
    FANUC_RAW_NAMES,
    controller_degrees,
    decode_fanuc_pose_dict,
    raw_fanuc_pose_dict,
    rotation_matrix_from_wpr,
    wpr_from_rotation_matrix,
    wrap_w_degrees,
)

logger = logging.getLogger(__name__)

# RMIT-009: Initialize rejected (leftover HOLD / RMI_MOVE still selected). Abort+Reset then retry.
_FRC_INITIALIZE_REJECTED = 2556937
# RMI already running / invalid controller state. Abort leftover RMI_MOVE and retry.
_FRC_INVALID_CONTROLLER_STATE = 2556943
# Invalid UFrame/UTool, leftover HOLD (RMIT-027), or Abort sent before Initialize.
_FRC_INVALID_UFRAME_UTOOL = 2556955
# RMIT-029: SequenceID gap / not the next expected id.
_FRC_INVALID_SEQUENCE_ID = 2556957
# Some controllers report this when RMI_MOVE is already initialized.
_FRC_ALREADY_INITIALIZED = 7015


# Names match lerobot-rollout's ".pos" motor filter, in the ACT training order.
JOINT_POS_NAMES = ("J1.pos", "J2.pos", "J3.pos", "J4.pos", "J5.pos", "J6.pos", "gripper.pos")
_JOINT_LIMITS_DEG = {
    "J1": (-170.0, 170.0),
    "J2": (-110.0, 120.0),
    "J3": (-69.0, 205.0),
    "J4": (-190.0, 190.0),
    "J5": (-120.0, 120.0),
    "J6": (-360.0, 360.0),
}


def _limit_step(origin: np.ndarray, target: np.ndarray, max_m: float) -> np.ndarray:
    """Pull target back onto the sphere of radius max_m around origin if it lies outside."""
    delta = target - origin
    distance = float(np.linalg.norm(delta))
    return target if distance <= max_m else origin + delta * (max_m / distance)


def _joints_deg_from_response(resp: dict[str, Any]) -> tuple[float, ...] | None:
    """Extract FANUC J1..J6 degrees from an FRC_ReadJointAngles payload."""
    cand = None
    for key in ("JointAngle", "JointAngles", "Joint"):
        value = resp.get(key)
        if isinstance(value, dict):
            cand = value
            break
    if cand is None:
        return None
    up = {str(k).upper(): v for k, v in cand.items()}
    try:
        return tuple(float(up[f"J{i}"]) for i in range(1, 7))
    except (KeyError, TypeError, ValueError):
        return None


class Fanuc(Robot):
    """FANUC robot over RMI / FRC JSON (cartesian pose + optional gripper I/O)."""

    config_class = FanucConfig
    name = "fanuc"
    STATE_POLL_HZ = 15.0

    def __init__(self, config: FanucConfig) -> None:
        super().__init__(config)
        self.config = config

        self._host = config.host
        self._port = config.port
        self._group = config.group
        self._utool = config.utool
        self._uframe = config.uframe
        self._speed = config.speed
        self._term_type = config.term_type
        self._term_value = config.term_value
        self._gripper_state_port_number = config.gripper_state_port_number

        self._sock: socket.socket | None = None
        self._buf = b""
        self._connected = False
        self._send_lock = threading.Lock()

        self._latest_pose: tuple[float, ...] | None = None
        self._latest_joints: tuple[float, ...] | None = None
        # Joint mode: the controller's active UTOOL / UFRAME (X, Y, Z, W, P, R), read at connect.
        self._utool_data: tuple[float, ...] | None = None
        # Joint mode: the last TCP position (m) actually sent; the next target is limited against it.
        self._joint_target_m: np.ndarray | None = None
        # Joint mode: the last joint target (sim degrees) actually sent; each step is limited against it.
        self._joint_target_deg: np.ndarray | None = None
        # Joint / TCP mode: the last tool orientation actually sent (UF rotation matrix); TCP steps are limited against it.
        self._tcp_target_rot: np.ndarray | None = None
        self._uframe_data: tuple[float, ...] | None = None
        self._latest_t: float | None = None
        self._latest_tick: int | None = None
        self._latest_configuration: dict[str, Any] | None = None
        self._motion_configuration: dict[str, Any] | None = None
        self._latest_gripper_state: int | None = None
        # Last gripper command that reached the controller (j7, 1 = closed). Reported as the gripper state when no
        # state input is configured, so the recorded observation follows the gripper instead of staying 0.
        self._commanded_gripper: float | None = None

        self._pending_futures: dict[int, Future] = {}
        self._pending_lock = threading.Lock()
        self._ack_queue: queue.Queue[tuple[int, int]] = queue.Queue()

        self._recv_thread: threading.Thread | None = None
        self._state_thread: threading.Thread | None = None
        self._poll_enabled = threading.Event()
        self._poll_enabled.set()
        self._uframe_applied = False
        self._last_ack: dict[str, Any] | None = None
        self._gripper_waiting = False
        self._rmi_session = False  # True once FRC_Initialize succeeded on the open connection
        self.reject_streak = 0  # motions refused in a row by the controller (reset by an accepted one / recover)
        self._last_unhandled: dict[str, Any] | None = None
        self._stream_sent_mono: float | None = None
        self._last_stream_j7: float | None = None
        self._stream_log_mono: float = 0.0
        self._stream_log_pose: tuple[float, float, float] | None = None
        self._command_replies: dict[str, dict[str, Any]] = {}

        self.cameras = make_cameras_from_configs(config.cameras)
        self.seq_id = 1

        self._twin_addr: tuple[str, int] | None = None
        self._twin_sock: socket.socket | None = None
        host = (config.twin_udp_host or "").strip()
        if host:
            self._twin_addr = (host, int(config.twin_udp_port))
            self._twin_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # ------------------------------------------------------------------ #
    #  Connection                                                          #
    # ------------------------------------------------------------------ #

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            logger.warning("Already connected - skipping")
            return

        dynamic_port = self._open_tcp_session()

        try:
            self._initialize_rmi()
            self._set_uframe_utool(self._uframe, self._utool)
            self._uframe_applied = True
        except TimeoutError:
            logger.warning(
                "RMI handshake timed out; leftover RMI_MOVE from a killed twin is likely. "
                "Aborting and retrying once."
            )
            self._close_socket()  # sends FRC_Abort only if this connection holds a session
            time.sleep(1.0)
            try:
                dynamic_port = self._open_tcp_session()
                self._initialize_rmi()
                self._set_uframe_utool(self._uframe, self._utool)
                self._uframe_applied = True
            except Exception:
                self._close_socket()
                raise
        except Exception:
            self._close_socket()
            raise
        if self.config.motion == "joint":
            try:
                self._read_tool_and_frame()
            except Exception:
                self._close_socket()
                raise
        self._motion_configuration = {
            "UToolNumber": self._utool,
            "UFrameNumber": self._uframe,
            "Front": 1,
            "Up": 1,
            "Left": 0,
            "Flip": 0,
            "Turn4": 0,
            "Turn5": 0,
            "Turn6": 0,
        }
        self._latest_configuration = dict(self._motion_configuration)
        self._joint_target_m = None
        self._joint_target_deg = None
        self._connected = True

        self._recv_thread = threading.Thread(target=self._recv_loop, name="fanuc-recv", daemon=True)
        self._recv_thread.start()
        self._state_thread = threading.Thread(target=self._state_poll_loop, name="fanuc-state-poll", daemon=True)
        self._state_thread.start()

        deadline = time.time() + 1.0
        while self._latest_pose is None and time.time() < deadline:
            time.sleep(0.01)

        for camera in self.cameras.values():
            camera.connect()

        if calibrate and not self.is_calibrated:
            self.calibrate()
        self.configure()
        logger.info("Fanuc connected to %s:%s with UF=%s, UT=%s", self._host, dynamic_port, self._uframe, self._utool)

    def disconnect(self) -> None:
        if self._connected and self._sock is not None:
            # End RMI_MOVE and wait for the controller to confirm: a session left running blocks the next Initialize.
            try:
                self._command_replies.pop("FRC_Abort", None)
                self._send_json({"Command": "FRC_Abort"})
                logger.info("Fanuc disconnect: FRC_Abort -> %s", self._await_reply("FRC_Abort", 3.0))
            except Exception as exc:  # noqa: BLE001
                logger.warning("FRC_Abort on disconnect failed: %s", exc)
        self._connected = False
        with self._pending_lock:
            for fut in self._pending_futures.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("Disconnected before ACK"))
            self._pending_futures.clear()

        self._close_socket()

        if self._recv_thread is not None:
            self._recv_thread.join(timeout=2.0)
            self._recv_thread = None
        if self._state_thread is not None:
            self._state_thread.join(timeout=2.0)
            self._state_thread = None

        for camera in self.cameras.values():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception as exc:
                logger.warning("Error disconnecting camera: %s", exc)

        if self._twin_sock is not None:
            try:
                self._twin_sock.close()
            except OSError:
                pass
            self._twin_sock = None

        logger.info("Fanuc disconnected")

    @staticmethod
    def _status_next_seq(status: dict[str, Any] | None) -> int | None:
        if not status:
            return None
        for key in ("NextSequenceID", "NextSequenceId", "nextSequenceID", "NextSeqID"):
            raw = status.get(key)
            if raw is not None:
                return int(raw)
        return None

    def _await_reply(self, cmd: str, timeout_s: float = 2.0) -> dict[str, Any] | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            reply = self._command_replies.get(cmd)
            if reply is not None:
                return reply
            time.sleep(0.02)
        return None

    @staticmethod
    def _pendant_problem(status: dict[str, Any] | None) -> str | None:
        """What on the pendant still blocks RMI, from FRC_GetStatus (None: nothing visible)."""
        if not status:
            return None
        problems = []
        if status.get("TPMode") == 1:
            problems.append("teach pendant enabled (switch it OFF)")
        if status.get("SingleStepMode") == 1:
            problems.append("STEP mode on (press STEP so its LED is off)")
        if status.get("ServoReady") == 0:
            problems.append("servos not ready (clear the alarm and press RESET)")
        return "; ".join(problems) or None

    def recover_after_fault(self) -> None:
        """Bring RMI back after a SystemFault / collision / joint-limit alarm, once the pendant is cleared.

        First Abort + Reset + Initialize on the open session, and check that Initialize really succeeded. If it
        did not (sequence desync, stale session), reconnect the RMI session in place: the same as restarting the
        process, but the cameras stay open. Raises with the reason if the pendant still blocks RMI.
        """
        self._require_connected()
        self.pause_state_poll()
        try:
            with self._pending_lock:
                for fut in list(self._pending_futures.values()):
                    if not fut.done():
                        fut.set_result(-1)
                self._pending_futures.clear()
            logger.info("Fanuc recover: FRC_Abort + FRC_Reset + FRC_Initialize")
            self.reject_streak = 0
            self._clear_leftover_rmi()
            self._command_replies.pop("FRC_GetStatus", None)
            self._send_json({"Command": "FRC_GetStatus"})
            problem = self._pendant_problem(self._await_reply("FRC_GetStatus"))
            if problem:
                raise RuntimeError(f"recover: {problem}")
            self._command_replies.pop("FRC_Initialize", None)
            self._send_json({"Command": "FRC_Initialize", "GroupMask": self._group})
            init = self._await_reply("FRC_Initialize", 5.0)  # after a fault the controller can take a few seconds
            init_ok = init is not None and int(init.get("ErrorID", -1)) in (0, _FRC_ALREADY_INITIALIZED)
            if init_ok:
                # Initialize resets the controller SequenceID (RMIT-029 / 2556957 if the PC keeps counting).
                self.seq_id = 1
                self._command_replies.pop("FRC_GetStatus", None)
                self._send_json({"Command": "FRC_GetStatus"})
                status = self._await_reply("FRC_GetStatus")
                nxt = self._status_next_seq(status)
                if nxt is not None:
                    self.seq_id = max(1, nxt)
                logger.info("Fanuc recover SequenceID=%s init=%s status=%s", self.seq_id, init, status)
                self._send_json({"Command": "FRC_SetUFrameUTool", "UFrameNumber": int(self._uframe),
                                 "UToolNumber": int(self._utool), "Group": int(self._group)})
                time.sleep(0.2)
                self._uframe_applied = True
                self._joint_target_m = self._joint_target_deg = self._tcp_target_rot = None
                self._last_ack = None
                return
            logger.warning("Fanuc recover: Initialize gave %s; reconnecting the RMI session", init)
        finally:
            self.resume_state_poll()
        self.reconnect_rmi()

    def reconnect_rmi(self) -> None:
        """Close and reopen the RMI session (what a process restart does), keeping the cameras open."""
        self._connected = False
        with self._pending_lock:
            for fut in self._pending_futures.values():
                if not fut.done():
                    fut.set_result(-1)
            self._pending_futures.clear()
        self._close_socket()
        for th in (self._recv_thread, self._state_thread):
            if th is not None:
                th.join(timeout=2.0)
        self._recv_thread = self._state_thread = None
        time.sleep(1.0)
        self.seq_id = 1
        self.reject_streak = 0
        self._last_ack = None
        self._latest_pose = None
        self._uframe_applied = False
        self._command_replies.clear()
        self._joint_target_m = self._joint_target_deg = self._tcp_target_rot = None
        cameras, self.cameras = self.cameras, {}  # already open and streaming: do not reopen them
        try:
            self.connect()
        finally:
            self.cameras = cameras
        logger.info("Fanuc RMI session reconnected")

    def configure(self) -> None:
        if not self._connected or self._sock is None or self._uframe_applied:
            return
        self._send_json(
            {
                "Command": "FRC_SetUFrameUTool",
                "UFrameNumber": int(self._uframe),
                "UToolNumber": int(self._utool),
                "Group": int(self._group),
            }
        )
        self._uframe_applied = True

    def calibrate(self) -> None:
        return

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------ #
    #  Core interfaces                                                     #
    # ------------------------------------------------------------------ #

    def send_action(self, action: RobotAction) -> RobotAction:
        """Send a motion command and return the action that was sent.

        ``motion="cartesian"`` sends ``FRC_LinearMotion``. ``motion="joint"`` takes
        sim joint degrees named ``J1.pos``..``J6.pos`` (J3 relative to the upper arm),
        maps them through ER-4iA FK onto the controller's UTOOL TCP in the user frame,
        and sends ``FRC_LinearMotion``.
        """
        self._require_connected()
        if self.config.motion == "joint":
            return self._send_joint_action(action)
        if self.config.motion != "cartesian":
            raise ValueError(f"Unknown fanuc motion mode {self.config.motion!r}")
        if action.get("stream"):
            now = time.perf_counter()
            with self._pending_lock:
                inflight = len(self._pending_futures)
            age = now - self._stream_sent_mono if self._stream_sent_mono is not None else 0.0
            # Keep several CNT segments queued so the arm blends through them.
            # A single in-flight move finishes and stops before the next one arrives.
            if inflight >= 5:
                if age < 1.0:
                    return dict(action)
                logger.warning(
                    "Fanuc stream ACK missing (last=%s unhandled=%s); sending the latest target",
                    self._last_ack,
                    self._last_unhandled,
                )
                with self._pending_lock:
                    for fut in self._pending_futures.values():
                        if not fut.done():
                            fut.cancel()
                    self._pending_futures.clear()
            elif inflight >= 1 and age < 0.03:
                return dict(action)
            # Pulse the gripper port only when j7 changes. A port write on every
            # streamed LinearMotion keeps the controller from executing the move.
            if (
                "j7" in action
                and self._last_stream_j7 is not None
                and float(action["j7"]) == self._last_stream_j7
            ):
                action = {key: value for key, value in action.items() if key != "j7"}
        if self._motion_configuration is None:
            self._motion_configuration = self._default_configuration()
        if self._latest_configuration is None:
            self._latest_configuration = dict(self._motion_configuration)

        action = self._apply_gripper_metadata(dict(action))
        x, y, z, w, p, r = self._pose_from_action(action)
        w, p, r = controller_degrees(w), controller_degrees(p), controller_degrees(r)

        configuration = self._sanitize_configuration(self._latest_configuration)
        if "utool" in action:
            configuration["UToolNumber"] = int(action["utool"])
        if "uframe" in action:
            configuration["UFrameNumber"] = int(action["uframe"])
        for src, dst in (
            ("front", "Front"),
            ("up", "Up"),
            ("left", "Left"),
            ("flip", "Flip"),
            ("turn4", "Turn4"),
            ("turn5", "Turn5"),
            ("turn6", "Turn6"),
        ):
            if src in action:
                configuration[dst] = int(action[src])
        self._motion_configuration = dict(configuration)

        seq_id = self.seq_id
        self.seq_id += 1

        packet: dict[str, Any] = {
            "Instruction": "FRC_LinearMotion",
            "SequenceID": seq_id,
            "Configuration": configuration,
            "Position": {
                "X": float(x),
                "Y": float(y),
                "Z": float(z),
                "W": float(w),
                "P": float(p),
                "R": float(r),
                "Ext1": 0.0,
                "Ext2": 0.0,
                "Ext3": 0.0,
            },
            "SpeedType": str(action.get("speed_type", "mmSec")),
            "Speed": int(action.get("speed", self._speed)),
            "TermType": str(action.get("term_type", self._term_type)),
            "TermValue": int(action.get("term_value", self._term_value)),
        }

        lcb_type = action.get("lcb_type", action.get("LCBType"))
        lcb_value = action.get("lcb_value", action.get("LCBValue", 0))
        port_type = action.get("port_type", action.get("PortType"))
        port_number = action.get("port_number", action.get("PortNumber"))
        port_value = action.get("port_value", action.get("PortValue"))
        if (
            lcb_type is not None
            and port_type is not None
            and port_number is not None
            and port_value is not None
        ):
            packet.update(
                {
                    "LCBType": str(lcb_type),
                    "LCBValue": int(lcb_value),
                    "PortType": int(port_type),
                    "PortNumber": int(port_number),
                    "PortValue": str(port_value),
                }
            )

        fut: Future = Future()
        with self._pending_lock:
            self._pending_futures[seq_id] = fut
        self._send_json(packet)
        if "j7" in action and "PortNumber" in packet:
            self._commanded_gripper = float(action["j7"])
        if action.get("stream"):
            now = time.perf_counter()
            self._stream_sent_mono = now
            if "j7" in action:
                self._last_stream_j7 = float(action["j7"])
            moved = self._stream_log_pose is None or max(
                abs(x - self._stream_log_pose[0]),
                abs(y - self._stream_log_pose[1]),
                abs(z - self._stream_log_pose[2]),
            ) >= 5.0
            if moved or now - self._stream_log_mono >= 0.5:
                self._stream_log_mono = now
                self._stream_log_pose = (x, y, z)
                logger.info(
                    "FRC_LinearMotion seq=%s xyz=(%.1f, %.1f, %.1f) wpr=(%.1f, %.1f, %.1f)",
                    seq_id,
                    x,
                    y,
                    z,
                    w,
                    p,
                    r,
                )

        sent = raw_fanuc_pose_dict({"j0": x, "j1": y, "j2": z, "j3": w, "j4": p, "j5": r})
        if "j7" in action:
            sent["j7"] = float(action["j7"])
        for key in ("speed", "term_type", "term_value", "lcb_type", "lcb_value", "port_type", "port_number", "port_value"):
            if key in action:
                sent[key] = action[key]
        sent["sequence_id"] = seq_id
        return sent

    def _joint_values(self, action: RobotAction) -> tuple[list[float], float | None]:
        joints = []
        for index, name in enumerate(JOINT_POS_NAMES[:6]):
            bare = f"J{index + 1}"
            if name in action:
                joints.append(float(action[name]))
            elif bare in action:
                joints.append(float(action[bare]))
            else:
                raise ValueError(f"Joint action is missing {name}")
        if "gripper.pos" in action:
            gripper = float(action["gripper.pos"])
        elif "gripper" in action:
            gripper = float(action["gripper"])
        elif "j7" in action:
            gripper = float(action["j7"])
        else:
            gripper = None
        return joints, gripper

    def _read_tool_and_frame(self) -> None:
        """Read the active UTOOL (and UFRAME unless World) so joint mode targets the real TCP."""

        def frame_values(command: str, number_key: str, number: int) -> tuple[float, ...]:
            self._send_json({"Command": command, number_key: int(number), "Group": int(self._group)})
            resp = self._recv_until(lambda r: r.get("Command") == command)
            if int(resp.get("ErrorID", -1)) != 0:
                raise RuntimeError(f"{command} {number_key}={number} failed: {resp}")
            frame = resp["Frame"]
            return tuple(float(frame[key]) for key in ("X", "Y", "Z", "W", "P", "R"))

        self._utool_data = frame_values("FRC_ReadUToolData", "ToolNumber", self._utool)
        self._uframe_data = (
            frame_values("FRC_ReadUFrameData", "FrameNumber", self._uframe) if self._uframe else None
        )
        logger.info("Joint mode UTOOL %s=%s UFRAME %s=%s", self._utool, self._utool_data, self._uframe, self._uframe_data)

    def _send_joint_action(self, action: RobotAction) -> RobotAction:
        joints, gripper = self._joint_values(action)
        for index, value in enumerate(joints):
            lo, hi = _JOINT_LIMITS_DEG[f"J{index + 1}"]
            if not lo <= value <= hi:
                raise ValueError(f"J{index + 1}={value} deg is outside [{lo}, {hi}]")
        if self._utool_data is None:
            raise RuntimeError("UTOOL data was not read at connect")
        # Limit the joint step against the previous target (or the live joints): a policy can jump between
        # action chunks (J6 109 deg in one step on the real arm), which a FRC_LinearMotion turns into a wrist
        # flick that faults the controller. The whole vector is scaled, so the motion keeps its direction.
        anchor_q = self._joint_target_deg
        if anchor_q is None and self._latest_joints is not None:
            anchor_q = np.array(sim_from_controller_deg(self._latest_joints), dtype=float)
        if anchor_q is not None:
            dq = np.asarray(joints, dtype=float) - anchor_q
            worst = float(np.max(np.abs(dq)))
            if worst > self.config.joint_max_step_deg:
                joints = list(anchor_q + dq * (self.config.joint_max_step_deg / worst))
        rotation, wanted = controller_tcp_pose(np.radians(joints), self._utool_data, self._uframe_data)
        wanted[2] += self.config.joint_z_offset_mm / 1000.0
        if self.config.joint_z_min_mm is not None:
            wanted[2] = max(wanted[2], self.config.joint_z_min_mm / 1000.0)
        if gripper is not None and self._gripper_toggles(gripper) and not self._gripper_waiting:
            self._wait_for_arrival(wanted, lambda: self._send_joint_action({k: v for k, v in action.items() if k not in ("gripper.pos", "gripper", "j7")}))
        position = wanted.copy()
        live = None if self._latest_pose is None else np.array(self._latest_pose[:3], dtype=float) / 1000.0
        # Limit each step against the previous target, not the live pose: the controller then gets one
        # continuous path to blend at joint_tcp_speed_mm_s. Stepping from the live pose made every target
        # trail the last one, so each CNT segment braked and the arm fell 3-9 cm behind the policy.
        anchor = self._joint_target_m if self._joint_target_m is not None else live
        if anchor is not None:
            position = _limit_step(anchor, position, self.config.joint_max_step_mm / 1000.0)
        if live is not None:  # never lead the real arm by more than this
            position = _limit_step(live, position, self.config.joint_max_lead_mm / 1000.0)
        w, p, r = wpr_from_rotation_matrix(rotation.tolist())
        linear: dict[str, Any] = {
            "j0": float(position[0] * 1000.0),
            "j1": float(position[1] * 1000.0),
            "j2": float(position[2] * 1000.0),
            "j3": float(w),
            "j4": float(p),
            "j5": float(r),
            "speed": int(self.config.joint_tcp_speed_mm_s),
            "speed_type": "mmSec",
            "term_type": "CNT",
            "term_value": 50,
            "stream": True,
        }
        if gripper is not None:
            closed = float(gripper) >= 0.5
            last_closed = None if self._last_stream_j7 is None else self._last_stream_j7 >= 0.5
            if closed != last_closed:
                linear["j7"] = 1.0 if closed else 0.0
        logger.info(
            "joint->linear joints=%s gripper=%s xyz_mm=(%.1f, %.1f, %.1f) short_of_policy_mm=%.1f lead_mm=%s",
            [round(v, 2) for v in joints],
            gripper,
            linear["j0"],
            linear["j1"],
            linear["j2"],
            float(np.linalg.norm(wanted - position) * 1000.0),
            None if live is None else round(float(np.linalg.norm(position - live) * 1000.0), 1),
        )
        saved = self.config.motion
        self.config.motion = "cartesian"
        try:
            sent = self.send_action(linear)
        finally:
            self.config.motion = saved
        if "sequence_id" in sent:  # a stream send can be skipped while acks are pending
            self._joint_target_m = position
            self._joint_target_deg = np.asarray(joints, dtype=float)
            self._tcp_target_rot = np.asarray(rotation, dtype=float)
        return sent

    def _gripper_toggles(self, gripper: float) -> bool:
        last = self._last_stream_j7  # None: nothing streamed yet this session, so there is no change to hold back
        return last is not None and (float(gripper) >= 0.5) != (last >= 0.5)

    def _wait_for_arrival(self, wanted_m: np.ndarray, resend: Callable[[], Any]) -> None:
        """Hold the gripper change until the live TCP reaches this row's target (joint_gripper_wait_mm)."""
        tol = self.config.joint_gripper_wait_mm
        if tol is None or self._latest_pose is None:
            return
        self._gripper_waiting = True
        t0 = time.time()
        err = float(np.linalg.norm(np.array(self._latest_pose[:3], dtype=float) / 1000.0 - wanted_m) * 1000.0)
        # Time to travel the remaining distance at the speed cap, plus joint_gripper_wait_s: a far target (a policy
        # that sends one row 250 mm away with the gripper change) is waited for, not closed on in mid-air. Capped,
        # so a target the arm never reaches (clamped by the floor, blocked) cannot hold the trial for long.
        limit = min(self.config.joint_gripper_wait_s + err / max(self.config.joint_tcp_speed_mm_s, 1),
                    self.config.joint_gripper_wait_max_s)
        try:
            while True:
                err = float(np.linalg.norm(np.array(self._latest_pose[:3], dtype=float) / 1000.0 - wanted_m) * 1000.0)
                if err <= tol or time.time() - t0 > limit:
                    break
                if (self._last_ack or {}).get("Communication") == "FRC_SystemFault" or self.reject_streak >= 10:
                    break
                resend()
                time.sleep(0.1)
        finally:
            self._gripper_waiting = False
        logger.info("gripper: waited %.2f s for the arm, %.1f mm from the target", time.time() - t0, err)

    def send_tcp_action(self, xyzwpr_mm_deg, gripper: float | None = None, max_rot_step_deg: float = 5.0) -> RobotAction:
        """Stream a TCP target (UTOOL in the active UFRAME: x, y, z mm, w, p, r deg) under the joint-mode limits.

        The same path as a joint action after its FK: z offset and floor, at most joint_max_step_mm from the previous
        target and joint_max_lead_mm ahead of the live pose, joint_tcp_speed_mm_s; the tool orientation turns at most
        max_rot_step_deg per step. No IK: the controller resolves the arm configuration, as for joint actions.
        """
        if self.config.motion != "joint":
            raise RuntimeError("send_tcp_action needs motion='joint' (the streaming, limited path)")
        x, y, z, w, p, r = (float(v) for v in xyzwpr_mm_deg[:6])
        if not np.all(np.isfinite([x, y, z, w, p, r])):
            raise ValueError("TCP target contains NaN or inf")
        wanted = np.array([x, y, z]) / 1000.0
        wanted[2] += self.config.joint_z_offset_mm / 1000.0
        if self.config.joint_z_min_mm is not None:
            wanted[2] = max(wanted[2], self.config.joint_z_min_mm / 1000.0)
        rot = np.array(rotation_matrix_from_wpr(w, p, r), dtype=float)
        prev = self._tcp_target_rot
        if prev is None and self._latest_pose is not None:
            prev = np.array(rotation_matrix_from_wpr(*self._latest_pose[3:6]), dtype=float)
        if prev is not None:
            rel = prev.T @ rot
            angle = float(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1.0, 1.0)))
            limit = np.radians(max_rot_step_deg)
            if angle > limit:
                axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
                n = float(np.linalg.norm(axis))
                if n < 1e-9:  # ~180 deg: any axis perpendicular to the rotation's fixed one
                    vals, vecs = np.linalg.eigh(rel + rel.T)
                    axis = vecs[:, int(np.argmax(vals))]
                else:
                    axis = axis / n
                k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                part = np.eye(3) + np.sin(limit) * k + (1 - np.cos(limit)) * (k @ k)
                rot = prev @ part
        if gripper is not None and self._gripper_toggles(gripper) and not self._gripper_waiting:
            self._wait_for_arrival(wanted, lambda: self.send_tcp_action(xyzwpr_mm_deg, None, max_rot_step_deg))
        position = wanted.copy()
        live = None if self._latest_pose is None else np.array(self._latest_pose[:3], dtype=float) / 1000.0
        anchor = self._joint_target_m if self._joint_target_m is not None else live
        if anchor is not None:
            position = _limit_step(anchor, position, self.config.joint_max_step_mm / 1000.0)
        if live is not None:
            position = _limit_step(live, position, self.config.joint_max_lead_mm / 1000.0)
        w, p, r = wpr_from_rotation_matrix(rot.tolist())
        linear: dict[str, Any] = {
            "j0": float(position[0] * 1000.0), "j1": float(position[1] * 1000.0), "j2": float(position[2] * 1000.0),
            "j3": float(w), "j4": float(p), "j5": float(r),
            "speed": int(self.config.joint_tcp_speed_mm_s), "speed_type": "mmSec",
            "term_type": "CNT", "term_value": 50, "stream": True,
        }
        if gripper is not None:
            closed = float(gripper) >= 0.5
            last_closed = None if self._last_stream_j7 is None else self._last_stream_j7 >= 0.5
            if closed != last_closed:
                linear["j7"] = 1.0 if closed else 0.0
        logger.info(
            "tcp->linear target=(%.1f, %.1f, %.1f, %.1f, %.1f, %.1f) gripper=%s xyz_mm=(%.1f, %.1f, %.1f) "
            "short_of_policy_mm=%.1f lead_mm=%s",
            x, y, z, float(xyzwpr_mm_deg[3]), float(xyzwpr_mm_deg[4]), float(xyzwpr_mm_deg[5]), gripper,
            linear["j0"], linear["j1"], linear["j2"], float(np.linalg.norm(wanted - position) * 1000.0),
            None if live is None else round(float(np.linalg.norm(position - live) * 1000.0), 1),
        )
        saved = self.config.motion
        self.config.motion = "cartesian"
        try:
            sent = self.send_action(linear)
        finally:
            self.config.motion = saved
        if "sequence_id" in sent:
            self._joint_target_m = position
            self._tcp_target_rot = rot
            self._joint_target_deg = None  # the next joint action steps from the live joints
        return sent

    def reset_episode(self, timeout_s: float = 20.0) -> dict[str, Any] | None:
        """Isaac Sim twin only (config.sim_reset): ask the simulated controller for a fresh scene between episodes.

        The sim drops its motion queue, so pending acks and the streaming state are cleared here too. No-op unless
        `sim_reset` is set, so the real controller never receives the sim-only command.
        """
        if not self.config.sim_reset:
            return None
        self._require_connected()
        with self._pending_lock:
            for fut in self._pending_futures.values():
                if not fut.done():
                    fut.cancel()
            self._pending_futures.clear()
        self._stream_sent_mono = None
        self._last_stream_j7 = None
        self._commanded_gripper = 0.0
        self._joint_target_m = None
        self._joint_target_deg = None
        self._tcp_target_rot = None
        self._command_replies.pop("SIM_Reset", None)
        self._send_json({"Command": "SIM_Reset"})
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            reply = self._command_replies.get("SIM_Reset")
            if reply is not None:
                if int(reply.get("ErrorID", -1)) != 0:
                    raise RuntimeError(f"SIM_Reset failed: {reply}")
                logger.info("Sim scene reset: %s", reply.get("Layout"))
                return reply
            time.sleep(0.02)
        raise TimeoutError("SIM_Reset: no reply from the simulated controller")

    def get_observation(self) -> RobotObservation:
        self._require_connected()
        if self.config.motion != "joint":
            if self._latest_pose is None:
                deadline = time.time() + 1.0
                while self._latest_pose is None and time.time() < deadline:
                    time.sleep(0.01)
            if self._latest_pose is None:
                raise RuntimeError("No cartesian observation received yet.")
        if self._gripper_state_port_number is not None and self._latest_gripper_state is None:
            deadline = time.time() + 1.0
            while self._latest_gripper_state is None and time.time() < deadline:
                time.sleep(0.01)

        if self.config.motion == "joint":
            if self._latest_joints is None:
                deadline = time.time() + 1.0
                while self._latest_joints is None and time.time() < deadline:
                    time.sleep(0.01)
            if self._latest_joints is None:
                raise RuntimeError("No joint observation received yet.")
            # The policy was trained on sim joints: J3 relative to the upper arm.
            joints = sim_from_controller_deg(self._latest_joints)
            obs: RobotObservation = {
                name: float(value) for name, value in zip(JOINT_POS_NAMES[:6], joints, strict=True)
            }
            if self._latest_gripper_state is not None:
                gripper = float(self._latest_gripper_state)
            elif self._commanded_gripper is not None:
                gripper = float(self._commanded_gripper)
            else:
                gripper = 0.0
            obs["gripper.pos"] = gripper
            # Native camera resolution, as recorded. The policy's pre-processor resizes (e.g. to 480x640)
            # with the same op it used in training; resizing here would put a second, different resize in front.
            for cam_name, camera in self.cameras.items():
                obs[cam_name] = camera.read()
            return obs

        x, y, z, w, p, r = self._latest_pose
        if self._latest_gripper_state is not None:
            gripper = float(self._latest_gripper_state)
        elif self._commanded_gripper is not None:
            gripper = float(self._commanded_gripper)
        else:
            gripper = 0.0
        obs: RobotObservation = {
            "j0": float(x),
            "j1": float(y),
            "j2": float(z),
            "j3": wrap_w_degrees(w),
            "j4": float(p),
            "j5": float(r),
            "j7": gripper,
        }

        for cam_name, camera in self.cameras.items():
            obs[cam_name] = camera.read()

        return obs

    def check_ack(self) -> tuple[int | None, int | None]:
        try:
            return self._ack_queue.get_nowait()
        except queue.Empty:
            return None, None

    def pause_state_poll(self) -> None:
        self._poll_enabled.clear()

    def resume_state_poll(self) -> None:
        self._poll_enabled.set()

    def commit_commanded_pose(
        self,
        x: float,
        y: float,
        z: float,
        w: float,
        p: float,
        r: float,
    ) -> None:
        """Keep get_observation() in sync after a blocking LinearMotion ACK."""
        self._latest_pose = (float(x), float(y), float(z), float(w), float(p), float(r))
        self._latest_t = time.perf_counter()

    def wait_for_ack(self, sequence_id: int, timeout_s: float = 60.0) -> int:
        sequence_id = int(sequence_id)
        with self._pending_lock:
            fut = self._pending_futures.get(sequence_id)
        if fut is not None:
            try:
                err_id = fut.result(timeout=timeout_s)
            except TimeoutError as exc:
                raise TimeoutError(
                    f"Timed out waiting for Fanuc ACK seq={sequence_id} last={self._last_unhandled}"
                ) from exc
        else:
            deadline = time.time() + timeout_s
            err_id = None
            while time.time() < deadline:
                seq_id, queued_err = self.check_ack()
                if seq_id is None:
                    time.sleep(0.02)
                    continue
                if int(seq_id) == sequence_id:
                    err_id = queued_err
                    break
            else:
                raise TimeoutError(
                    f"Timed out waiting for Fanuc ACK seq={sequence_id} last={self._last_unhandled}"
                )
        if err_id not in (0, None):
            extra = ""
            if int(err_id) == _FRC_INVALID_SEQUENCE_ID:
                extra = (
                    f" (RMIT-029 invalid SequenceID; controller expected a reset id, "
                    f"PC sent {sequence_id}, next PC seq={self.seq_id})"
                )
            raise RuntimeError(
                f"Fanuc motion ACK ErrorID={err_id} seq={sequence_id} last={self._last_ack}{extra}"
            )
        return int(err_id or 0)

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        names = JOINT_POS_NAMES if self.config.motion == "joint" else FANUC_RAW_NAMES
        feats: dict[str, type | tuple] = {name: float for name in names}
        for cam_name, camera in {**self.config.cameras, **self.cameras}.items():
            height = getattr(camera, "height", None) or 480
            width = getattr(camera, "width", None) or 640
            feats[cam_name] = (int(height), int(width), 3)
        return feats

    @property
    def action_features(self) -> dict[str, type]:
        """Cartesian ``j0``..``j7``, or joint ``J1.pos``..``gripper.pos`` when ``motion="joint"``."""
        names = JOINT_POS_NAMES if self.config.motion == "joint" else FANUC_RAW_NAMES
        return {name: float for name in names}

    # ------------------------------------------------------------------ #
    #  Background loops                                                    #
    # ------------------------------------------------------------------ #

    def _recv_loop(self) -> None:
        while self._connected and self._sock is not None:
            try:
                self._sock.settimeout(1.0)
                resp = self._read_json()
            except TimeoutError:
                continue
            except (ConnectionError, OSError) as exc:
                if self._connected:
                    logger.error("Recv loop socket error: %s", exc)
                    with self._pending_lock:
                        for fut in self._pending_futures.values():
                            if not fut.done():
                                fut.set_exception(exc)
                        self._pending_futures.clear()
                break
            except Exception as exc:
                logger.exception("Recv loop unexpected error: %s", exc)
                break
            try:
                self._dispatch_response(resp)
            except Exception as exc:
                logger.exception("Dispatch error: %s", exc)
        logger.debug("Fanuc recv loop exited")

    def _state_poll_loop(self) -> None:
        interval = 1.0 / self.STATE_POLL_HZ
        while self._connected and self._sock is not None:
            start_t = time.perf_counter()
            if self._poll_enabled.is_set():
                try:
                    if self.config.motion == "joint":
                        self._send_json({"Command": "FRC_ReadJointAngles", "Group": self._group})
                        # The step limit in _send_joint_action needs the live TCP.
                        self._send_json({"Command": "FRC_ReadCartesianPosition", "Group": self._group})
                    else:
                        self._send_json({"Command": "FRC_ReadCartesianPosition", "Group": self._group})
                    if self._gripper_state_port_number is not None:
                        self._send_json({"Command": "FRC_ReadDIN", "PortNumber": int(self._gripper_state_port_number)})
                    if self._twin_sock is not None:
                        self._send_json({"Command": "FRC_ReadJointAngles", "Group": self._group})
                except Exception as exc:
                    if self._connected:
                        logger.error("State poll send error: %s", exc)
                    break
            elapsed = time.perf_counter() - start_t
            time.sleep(max(0.0, interval - elapsed))

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _apply_gripper_metadata(self, action: RobotAction) -> RobotAction:
        if "j7" not in action:
            buttons = action.get("buttons")
            if isinstance(buttons, dict) and "grip" in buttons:
                action["j7"] = float(bool(buttons.get("grip", 0)))

        action.setdefault("speed", self._speed)
        action.setdefault("term_type", self._term_type)
        action.setdefault("term_value", self._term_value)

        if "j7" not in action:
            return action

        discrete_j7 = float(float(action["j7"]) >= 0.5)
        action["j7"] = discrete_j7

        selected_port = None
        if self.config.gripper_open_port_number is not None and self.config.gripper_close_port_number is not None:
            selected_port = (
                self.config.gripper_close_port_number if discrete_j7 else self.config.gripper_open_port_number
            )
        elif self.config.gripper_port_number is not None:
            selected_port = self.config.gripper_port_number

        if (
            self.config.gripper_lcb_type is not None
            and self.config.gripper_port_type is not None
            and selected_port is not None
            and "lcb_type" not in action
            and "LCBType" not in action
        ):
            action.update(
                {
                    "lcb_type": self.config.gripper_lcb_type,
                    "lcb_value": self.config.gripper_lcb_value,
                    "port_type": self.config.gripper_port_type,
                    "port_number": selected_port,
                    "port_value": (
                        self.config.gripper_close_value if discrete_j7 else self.config.gripper_open_value
                    ),
                }
            )
        return action

    def _pose_from_action(self, action: RobotAction) -> tuple[float, float, float, float, float, float]:
        if "j0" in action and "j1" in action:
            decoded = decode_fanuc_pose_dict(action)
            return (
                float(decoded["j0"]),
                float(decoded["j1"]),
                float(decoded["j2"]),
                float(decoded["j3"]),
                float(decoded["j4"]),
                float(decoded["j5"]),
            )
        if "state" in action:
            state = action["state"]
            if len(state) != 6:
                raise ValueError(f"Invalid state format. Expected 6 values, got {len(state)}")
            x, y, z, w, p, r = state
            return float(x), float(y), float(z), float(w), float(p), float(r)
        if "position" in action:
            position = action["position"]
            if isinstance(position, dict):
                rotation = action.get("rotation")
                if not isinstance(rotation, dict):
                    raise ValueError("When 'position' is a dict, 'rotation' must also be a dict")
                return (
                    float(position["x"]),
                    float(position["y"]),
                    float(position["z"]),
                    float(rotation["w"]),
                    float(rotation["p"]),
                    float(rotation["r"]),
                )
            if len(position) == 6:
                x, y, z, w, p, r = position
                return float(x), float(y), float(z), float(w), float(p), float(r)
            if len(position) == 3:
                x, y, z = position
                w, p, r = action["rotation"]
                return float(x), float(y), float(z), float(w), float(p), float(r)
            raise ValueError(f"Invalid position format. Expected 3 or 6 values, got {len(position)}")
        raise ValueError("Action must contain 'j0'-'j5', 'state', or 'position' key")

    def _dispatch_response(self, resp: dict[str, Any]) -> None:
        if resp.get("Command") == "FRC_ReadCartesianPosition":
            self._update_pose_from_response(resp)
            return
        if resp.get("Command") == "FRC_ReadJointAngles":
            self._publish_twin_joints(resp)
            return
        if resp.get("Command") == "FRC_ReadDIN":
            self._update_gripper_state_from_response(resp)
            return
        if resp.get("Communication") == "FRC_SystemFault":
            logger.error("FRC_SystemFault: %s", resp)
            self._last_ack = dict(resp)
            seq_raw = resp.get("SequenceID")
            if seq_raw is not None:
                seq_id = int(seq_raw)
                with self._pending_lock:
                    fut = self._pending_futures.pop(seq_id, None)
                if fut is not None and not fut.done():
                    fut.set_result(-1)
                self._ack_queue.put((seq_id, -1))
            return
        if "SequenceID" in resp:
            seq_id = int(resp["SequenceID"])
            if "ErrorID" in resp and resp["ErrorID"] is not None:
                err_id = int(resp["ErrorID"])
            elif resp.get("Instruction") == "FRC_LinearMotion":
                err_id = 0
            else:
                err_id = -1
            self._last_ack = dict(resp)
            with self._pending_lock:
                fut = self._pending_futures.pop(seq_id, None)
            if fut is not None and not fut.done():
                fut.set_result(err_id)
            self._ack_queue.put((seq_id, err_id))
            if err_id not in (0, None):
                self.reject_streak += 1
                logger.warning("Fanuc motion rejected ErrorID=%s response=%s", err_id, resp)
            elif err_id == 0:
                self.reject_streak = 0
            return
        cmd = resp.get("Command")
        if cmd:
            self._command_replies[str(cmd)] = dict(resp)
        self._last_unhandled = dict(resp)

    def _frc_connect(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(5.0)
            sock.connect((self._host, self._port))
            sock.sendall(b'{"Communication": "FRC_Connect"}\r\n')
            data = json.loads(sock.recv(4096).decode())
        if data.get("ErrorID", -1) != 0:
            raise RuntimeError(f"FRC_Connect failed: {data}")
        return int(data["PortNumber"])

    def _open_tcp_session(self) -> int:
        dynamic_port = self._frc_connect()
        self._buf = b""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            pass
        self._sock.settimeout(5.0)
        self._sock.connect((self._host, dynamic_port))
        return dynamic_port

    def _clear_leftover_rmi(self) -> None:
        """Abort/reset a previous RMI_MOVE session (e.g. twin killed without FRC_Abort).

        Sends best-effort Abort/Reset and does not wait, so leftover acks can be
        skipped later by ``_recv_until`` without stealing the Initialize reply.
        """
        if not self._rmi_session:  # no session of ours to abort: Abort now would poison the next Initialize
            return
        for cmd in ("FRC_Abort", "FRC_Reset"):
            try:
                self._send_json({"Command": cmd})
            except Exception as exc:
                logger.warning("%s during leftover RMI clear failed: %s", cmd, exc)
        time.sleep(0.3)

    def _probe_status(self) -> dict[str, Any] | None:
        try:
            self._send_json({"Command": "FRC_GetStatus"})
            resp = self._recv_until(lambda r: r.get("Command") == "FRC_GetStatus")
        except Exception as exc:  # noqa: BLE001
            logger.warning("FRC_GetStatus failed: %s", exc)
            return None
        logger.info("FRC_GetStatus: %s", resp)
        return dict(resp)

    @staticmethod
    def _initialize_error(resp: dict[str, Any], status: dict[str, Any] | None) -> str:
        return (
            f"FRC_Initialize failed: {resp} status={status}. "
            "RMI needs AUTO, teach pendant DISABLED (enable switch OFF), "
            "fault RESET, and RMI_MOVE not selected. "
            "Do not Abort/run RMI_MOVE on the pendant; Initialize starts it."
        )

    def _try_initialize(self) -> dict[str, Any]:
        self._send_json({"Command": "FRC_Initialize", "GroupMask": self._group})
        resp = self._recv_until(lambda r: r.get("Command") == "FRC_Initialize")
        self._rmi_session = int(resp.get("ErrorID", -1)) in (0, _FRC_ALREADY_INITIALIZED)
        return resp

    def _initialize_rmi(self) -> None:
        # Never send FRC_Abort / FRC_Reset before a session exists: after a SystemFault that poisons every later
        # Initialize (ErrorID 2556955 / GroupMask 16) until RMI is aborted on the pendant. A session left open by a
        # process that died (2556937) is retried on fresh connections instead.
        resp = self._try_initialize()
        for attempt in range(3):
            err = int(resp.get("ErrorID", -1))
            if err in (0, _FRC_ALREADY_INITIALIZED):
                if attempt:
                    logger.info("FRC_Initialize succeeded on a fresh session (attempt %d)", attempt + 1)
                return
            if err not in (_FRC_INITIALIZE_REJECTED, _FRC_INVALID_CONTROLLER_STATE):
                break
            logger.warning("FRC_Initialize ErrorID=%s; reconnecting and retrying in 3 s (%d/3)", err, attempt + 1)
            self._close_socket()
            time.sleep(3.0)
            self._open_tcp_session()
            resp = self._try_initialize()
        if int(resp.get("ErrorID", -1)) in (0, _FRC_ALREADY_INITIALIZED):
            return
        status = self._probe_status()
        raise RuntimeError(
            self._initialize_error(resp, status)
            + " The controller still holds an RMI session: on the pendant press FCTN > ABORT ALL, SELECT another program"
              " (not RMI_MOVE), RESET, then start again."
        )

    def _close_socket(self) -> None:
        if self._sock is None:
            self._buf = b""
            return
        if self._rmi_session:  # Abort only ends a session we own: sent without one it poisons the next Initialize
            try:
                self._send_json({"Command": "FRC_Abort"})
            except Exception:
                pass
        self._rmi_session = False
        try:
            self._send_json({"Communication": "FRC_Disconnect"})
            time.sleep(0.3)  # let the controller take the disconnect before the socket goes away
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass
        self._sock = None
        self._buf = b""

    def _set_uframe_utool(self, uframe: int, utool: int) -> None:
        payload = {
            "Command": "FRC_SetUFrameUTool",
            "UFrameNumber": int(uframe),
            "UToolNumber": int(utool),
            "Group": int(self._group),
        }
        self._send_json(payload)
        resp = self._recv_until(lambda r: r.get("Command") == "FRC_SetUFrameUTool")
        err = int(resp.get("ErrorID", -1))
        if err == 0:
            return
        logger.warning(
            "FRC_SetUFrameUTool failed (ErrorID=%s); clearing leftover RMI and retrying: %s",
            err,
            resp,
        )
        self._clear_leftover_rmi()
        self._send_json({"Command": "FRC_Initialize", "GroupMask": self._group})
        init = self._recv_until(lambda r: r.get("Command") == "FRC_Initialize")
        if int(init.get("ErrorID", -1)) not in (0, _FRC_ALREADY_INITIALIZED):
            raise RuntimeError(f"FRC_Initialize failed during SetUFrameUTool retry: {init}")
        self._send_json(payload)
        resp = self._recv_until(lambda r: r.get("Command") == "FRC_SetUFrameUTool")
        if int(resp.get("ErrorID", -1)) != 0:
            raise RuntimeError(f"FRC_SetUFrameUTool failed: {resp}")

    def _send_json(self, payload: dict[str, Any]) -> None:
        if self._sock is None:
            raise RuntimeError("Fanuc is not connected. Call connect() first.")
        with self._send_lock:
            self._sock.sendall((json.dumps(payload) + "\r\n").encode("utf-8"))

    def _recv_until(self, predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        while True:
            resp = self._read_json()
            self._dispatch_response(resp)
            if predicate(resp):
                return resp

    def _default_configuration(self) -> dict[str, Any]:
        return {
            "UToolNumber": self._utool,
            "UFrameNumber": self._uframe,
            "Front": 1,
            "Up": 1,
            "Left": 0,
            "Flip": 0,
            "Turn4": 0,
            "Turn5": 0,
            "Turn6": 0,
        }

    @staticmethod
    def _cfg_int(value: Any, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str) and value.lstrip("+-").isdigit():
            return int(value)
        return default

    def _sanitize_configuration(self, raw: dict[str, Any] | None) -> dict[str, Any]:
        src = raw or {}
        defaults = self._default_configuration()
        return {key: self._cfg_int(src.get(key), default) for key, default in defaults.items()}

    def _publish_twin_joints(self, resp: dict[str, Any]) -> None:
        if resp.get("ErrorID", -1) not in (0, None):
            logger.warning("FRC_ReadJointAngles failed: %s", resp)
            return
        joints = _joints_deg_from_response(resp)
        if joints is None:
            logger.warning("FRC_ReadJointAngles missing J1..J6: %s", resp)
            return
        self._latest_joints = joints
        if self._twin_sock is None or self._twin_addr is None:
            return
        try:
            self._twin_sock.sendto(
                json.dumps({"joints_deg": [float(x) for x in joints]}).encode(),
                self._twin_addr,
            )
        except OSError as exc:
            logger.debug("Twin UDP publish failed: %s", exc)

    def _update_pose_from_response(self, resp: dict[str, Any]) -> None:
        if resp.get("ErrorID", -1) != 0:
            logger.warning("FRC_ReadCartesianPosition failed: %s", resp)
            return
        position = resp.get("Position") or {}
        config = resp.get("Configuration") or {}
        self._latest_pose = (
            position["X"],
            position["Y"],
            position["Z"],
            position["W"],
            position["P"],
            position["R"],
        )
        self._latest_t = time.perf_counter()
        self._latest_tick = resp.get("TimeTag")
        self._latest_configuration = dict(config)

    def _update_gripper_state_from_response(self, resp: dict[str, Any]) -> None:
        if resp.get("ErrorID", -1) != 0:
            logger.error("FRC_ReadDIN failed: %s", resp)
            return
        try:
            self._latest_gripper_state = int(resp["PortValue"])
        except (KeyError, TypeError, ValueError):
            logger.error("Invalid FRC_ReadDIN response: %s", resp)

    def _read_json(self) -> dict[str, Any]:
        if self._sock is None:
            raise ConnectionError("Connection closed by remote")
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("Connection closed by remote")
            self._buf += chunk
        idx = self._buf.index(b"\n")
        line = self._buf[:idx].rstrip(b"\r")
        self._buf = self._buf[idx + 1 :]
        return json.loads(line)

    def _require_connected(self) -> None:
        if not self._connected or self._sock is None:
            raise RuntimeError("Fanuc is not connected. Call connect() first.")
