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
from .pose import decode_fanuc_pose_dict, encode_fanuc_pose_dict

logger = logging.getLogger(__name__)

# RMI already running / invalid controller state. Abort leftover RMI_MOVE and retry.
_FRC_INVALID_CONTROLLER_STATE = 2556943
# Some controllers report this when RMI_MOVE is already initialized.
_FRC_ALREADY_INITIALIZED = 7015


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

        self.cameras = make_cameras_from_configs(config.cameras)
        self.seq_id = 1

    # ------------------------------------------------------------------ #
    #  Connection                                                          #
    # ------------------------------------------------------------------ #

    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            logger.warning("Already connected - skipping")
            return

        dynamic_port = self._frc_connect()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            pass
        self._sock.settimeout(5.0)
        self._sock.connect((self._host, dynamic_port))

        try:
            self._initialize_rmi()
            self._set_uframe_utool(self._uframe, self._utool)
            self._uframe_applied = True
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

        logger.info("Fanuc disconnected")

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
        if self._motion_configuration is None:
            self._motion_configuration = self._default_configuration()
        if self._latest_configuration is None:
            self._latest_configuration = dict(self._motion_configuration)

        action = self._apply_gripper_metadata(dict(action))
        x, y, z, w, p, r = self._pose_from_action(action)

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

        sent = encode_fanuc_pose_dict({"j0": x, "j1": y, "j2": z, "j3": w, "j4": p, "j5": r})
        if "j7" in action:
            sent["j7"] = float(action["j7"])
        for key in ("speed", "term_type", "term_value", "lcb_type", "lcb_value", "port_type", "port_number", "port_value"):
            if key in action:
                sent[key] = action[key]
        sent["sequence_id"] = seq_id
        return sent

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
        obs: RobotObservation = {
            "j0": float(x),
            "j1": float(y),
            "j2": float(z),
            "j7": (
                float(self._latest_gripper_state) if self._latest_gripper_state is not None else 0.0
            ),
        }
        obs.update(encode_fanuc_pose_dict({"j3": float(w), "j4": float(p), "j5": float(r)}))
        obs["j3"] = float(w)
        obs["j4"] = float(p)
        obs["j5"] = float(r)

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
            raise RuntimeError(
                f"Fanuc motion ACK ErrorID={err_id} seq={sequence_id} last={self._last_ack}"
            )
        return int(err_id or 0)

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        feats: dict[str, type | tuple] = {
            "j0": float,
            "j1": float,
            "j2": float,
            "j3_sin": float,
            "j3_cos": float,
            "j4_sin": float,
            "j4_cos": float,
            "j5_sin": float,
            "j5_cos": float,
            "j7": float,
        }
        for cam_name, camera in {**self.config.cameras, **self.cameras}.items():
            height = getattr(camera, "height", None) or 480
            width = getattr(camera, "width", None) or 640
            feats[cam_name] = (int(height), int(width), 3)
        return feats

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "j0": float,
            "j1": float,
            "j2": float,
            "j3_sin": float,
            "j3_cos": float,
            "j4_sin": float,
            "j4_cos": float,
            "j5_sin": float,
            "j5_cos": float,
            "j7": float,
        }

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
            return
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

    def _initialize_rmi(self) -> None:
        self._send_json({"Command": "FRC_Initialize", "GroupMask": self._group})
        resp = self._recv_until(lambda r: r.get("Command") == "FRC_Initialize")
        err = int(resp.get("ErrorID", -1))
        if err in (0, _FRC_ALREADY_INITIALIZED):
            return
        if err != _FRC_INVALID_CONTROLLER_STATE:
            raise RuntimeError(f"FRC_Initialize failed: {resp}")

        logger.warning(
            "FRC_Initialize busy (ErrorID=%s); aborting leftover RMI session and retrying",
            err,
        )
        self._send_json({"Command": "FRC_Abort"})
        try:
            self._recv_until(lambda r: r.get("Command") == "FRC_Abort")
        except Exception as exc:
            logger.warning("FRC_Abort during initialize retry failed: %s", exc)
        time.sleep(0.5)
        self._send_json({"Command": "FRC_Initialize", "GroupMask": self._group})
        resp = self._recv_until(lambda r: r.get("Command") == "FRC_Initialize")
        if int(resp.get("ErrorID", -1)) not in (0, _FRC_ALREADY_INITIALIZED):
            raise RuntimeError(f"FRC_Initialize failed after abort: {resp}")

    def _close_socket(self) -> None:
        if self._sock is None:
            self._buf = b""
            return
        try:
            self._send_json({"Command": "FRC_Abort"})
        except Exception:
            pass
        try:
            self._sock.close()
        except Exception:
            pass
        self._sock = None
        self._buf = b""

    def _set_uframe_utool(self, uframe: int, utool: int) -> None:
        self._send_json(
            {
                "Command": "FRC_SetUFrameUTool",
                "UFrameNumber": int(uframe),
                "UToolNumber": int(utool),
                "Group": int(self._group),
            }
        )
        resp = self._recv_until(lambda r: r.get("Command") == "FRC_SetUFrameUTool")
        if resp.get("ErrorID", -1) != 0:
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
