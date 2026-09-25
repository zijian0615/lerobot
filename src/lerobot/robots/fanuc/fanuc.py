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

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation

from ..robot import Robot
from .config_fanuc import FanucConfig
from .pose import FANUC_RAW_NAMES, controller_degrees, decode_fanuc_pose_dict, raw_fanuc_pose_dict, wrap_w_degrees

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
            try:
                self._send_json({"Command": "FRC_Abort"})
                self._send_json({"Communication": "FRC_Disconnect"})
            except Exception:
                pass
            self._close_socket()
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

    def recover_after_fault(self) -> None:
        """Abort leftover motion, RESET the controller, re-init RMI after SystemFault.

        After ``FRC_SystemFault`` the socket still accepts ``FRC_LinearMotion``,
        but the arm stays in HOLD until Abort+Reset. Abort is only valid once a
        session already exists — do not send it before the first Initialize.
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
            self._clear_leftover_rmi()
            self._command_replies.pop("FRC_Initialize", None)
            self._send_json({"Command": "FRC_Initialize", "GroupMask": self._group})
            time.sleep(0.4)
            # Initialize resets the controller SequenceID to 1 (RMIT-029 / 2556957
            # if the PC keeps counting from the pre-Abort value).
            self.seq_id = 1
            self._command_replies.pop("FRC_GetStatus", None)
            self._send_json({"Command": "FRC_GetStatus"})
            time.sleep(0.3)
            status = self._command_replies.get("FRC_GetStatus")
            nxt = self._status_next_seq(status)
            if nxt is not None:
                self.seq_id = max(1, nxt)
            logger.info(
                "Fanuc recover SequenceID=%s init=%s status=%s",
                self.seq_id,
                self._command_replies.get("FRC_Initialize"),
                status,
            )
            self._send_json(
                {
                    "Command": "FRC_SetUFrameUTool",
                    "UFrameNumber": int(self._uframe),
                    "UToolNumber": int(self._utool),
                    "Group": int(self._group),
                }
            )
            time.sleep(0.2)
            self._uframe_applied = True
        finally:
            self.resume_state_poll()

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
        """Send a cartesian motion command and return the action that was sent."""
        self._require_connected()
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
        feats: dict[str, type | tuple] = {name: float for name in FANUC_RAW_NAMES}
        for cam_name, camera in {**self.config.cameras, **self.cameras}.items():
            height = getattr(camera, "height", None) or 480
            width = getattr(camera, "width", None) or 640
            feats[cam_name] = (int(height), int(width), 3)
        return feats

    @property
    def action_features(self) -> dict[str, type]:
        """x, y, z (mm), W in [0, 360), P, R (deg) and the gripper (1 = closed); see pose.FANUC_RAW_NAMES."""
        return {name: float for name in FANUC_RAW_NAMES}

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
                logger.warning("Fanuc motion rejected ErrorID=%s response=%s", err_id, resp)
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

    def _initialize_rmi(self) -> None:
        # Abort is only valid after a session exists. Sending it first poisons
        # Initialize (ErrorID 2556955 / GroupMask 16) after a SystemFault.
        self._send_json({"Command": "FRC_Initialize", "GroupMask": self._group})
        resp = self._recv_until(lambda r: r.get("Command") == "FRC_Initialize")
        err = int(resp.get("ErrorID", -1))
        if err in (0, _FRC_ALREADY_INITIALIZED):
            return
        if err not in (
            _FRC_INITIALIZE_REJECTED,
            _FRC_INVALID_CONTROLLER_STATE,
            _FRC_INVALID_UFRAME_UTOOL,
        ):
            raise RuntimeError(self._initialize_error(resp, self._probe_status()))

        logger.warning(
            "FRC_Initialize ErrorID=%s; aborting leftover RMI session and retrying",
            err,
        )
        self._clear_leftover_rmi()
        self._send_json({"Command": "FRC_Initialize", "GroupMask": self._group})
        resp = self._recv_until(lambda r: r.get("Command") == "FRC_Initialize")
        if int(resp.get("ErrorID", -1)) not in (0, _FRC_ALREADY_INITIALIZED):
            raise RuntimeError(self._initialize_error(resp, self._probe_status()))

    def _close_socket(self) -> None:
        if self._sock is None:
            self._buf = b""
            return
        try:
            self._send_json({"Command": "FRC_Abort"})
        except Exception:
            pass
        try:
            self._send_json({"Communication": "FRC_Disconnect"})
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
        if self._twin_sock is None or self._twin_addr is None:
            return
        if resp.get("ErrorID", -1) not in (0, None):
            logger.warning("FRC_ReadJointAngles failed: %s", resp)
            return
        joints = _joints_deg_from_response(resp)
        if joints is None:
            logger.warning("FRC_ReadJointAngles missing J1..J6: %s", resp)
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
