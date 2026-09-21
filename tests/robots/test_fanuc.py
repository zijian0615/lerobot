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
import math
import threading
from unittest.mock import patch

import pytest

from lerobot.robots.fanuc import Fanuc, FanucConfig
from lerobot.robots.fanuc.pose import decode_angle_degrees, decode_fanuc_pose_dict, encode_fanuc_pose_dict
from lerobot.robots.utils import make_robot_from_config


def _line(payload: dict) -> bytes:
    return (json.dumps(payload) + "\r\n").encode()


def _sent_payloads(sock) -> list[dict]:
    payloads = []
    for raw in list(sock.sent):
        for line in raw.decode().splitlines():
            if line.strip():
                payloads.append(json.loads(line))
    return payloads


class FakeSocket:
    def __init__(self, responses: list[bytes] | None = None):
        self.sent: list[bytes] = []
        self._responses = list(responses or [])
        self._lock = threading.Lock()
        self.closed = False
        self.addr = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
        return False

    def connect(self, addr):
        self.addr = addr

    def setsockopt(self, *_args, **_kwargs):
        return None

    def settimeout(self, *_args, **_kwargs):
        return None

    def sendall(self, data: bytes):
        with self._lock:
            self.sent.append(data)

    def recv(self, _n: int) -> bytes:
        with self._lock:
            if self._responses:
                return self._responses.pop(0)
        raise TimeoutError()

    def close(self):
        self.closed = True


class SocketFactory:
    def __init__(self):
        self.connect_sock = FakeSocket(
            [_line({"Communication": "FRC_Connect", "ErrorID": 0, "PortNumber": 16002})]
        )
        self.session_sock = FakeSocket(
            [
                _line({"Command": "FRC_Initialize", "ErrorID": 0}),
                _line({"Command": "FRC_SetUFrameUTool", "ErrorID": 0}),
                _line(
                    {
                        "Command": "FRC_ReadCartesianPosition",
                        "ErrorID": 0,
                        "TimeTag": 1,
                        "Position": {
                            "X": 100.0,
                            "Y": 200.0,
                            "Z": 300.0,
                            "W": 10.0,
                            "P": 20.0,
                            "R": 30.0,
                        },
                        "Configuration": {"UFrameNumber": 0, "UToolNumber": 1},
                    }
                ),
            ]
        )
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return self.connect_sock
        return self.session_sock


@pytest.fixture
def fanuc_config(tmp_path):
    return FanucConfig(id="test_fanuc", calibration_dir=tmp_path)


def test_config_registration_and_factory(fanuc_config):
    assert fanuc_config.type == "fanuc"
    robot = make_robot_from_config(fanuc_config)
    assert isinstance(robot, Fanuc)
    assert robot.name == "fanuc"


def test_features_available_before_connect(fanuc_config):
    robot = Fanuc(fanuc_config)
    assert robot.observation_features["j0"] is float
    assert robot.observation_features["j3_sin"] is float
    assert robot.observation_features["j7"] is float
    assert robot.action_features["j5_cos"] is float
    assert robot.is_calibrated is True
    robot.calibrate()
    robot.configure()


def test_connect_retries_after_busy_initialize(fanuc_config):
    factory = SocketFactory()
    factory.session_sock._responses = [
        _line({"Command": "FRC_Initialize", "ErrorID": 2556943, "GroupMask": 1}),
        _line({"Command": "FRC_Abort", "ErrorID": 0}),
        _line({"Command": "FRC_Initialize", "ErrorID": 0}),
        _line({"Command": "FRC_SetUFrameUTool", "ErrorID": 0}),
        _line(
            {
                "Command": "FRC_ReadCartesianPosition",
                "ErrorID": 0,
                "TimeTag": 1,
                "Position": {"X": 1.0, "Y": 2.0, "Z": 3.0, "W": 0.0, "P": 0.0, "R": 0.0},
                "Configuration": {"UFrameNumber": 0, "UToolNumber": 1},
            }
        ),
    ]
    robot = Fanuc(fanuc_config)
    with (
        patch("lerobot.robots.fanuc.fanuc.socket.socket", factory),
        patch("lerobot.robots.fanuc.fanuc.time.sleep", return_value=None),
    ):
        robot.connect()
        try:
            commands = [p.get("Command") for p in _sent_payloads(factory.session_sock)]
            assert commands.count("FRC_Initialize") >= 2
            assert "FRC_Abort" in commands
            assert robot.is_connected
        finally:
            robot.disconnect()


def test_connect_handshake(fanuc_config):
    factory = SocketFactory()
    robot = Fanuc(fanuc_config)
    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        try:
            assert robot.is_connected
            connect_payloads = _sent_payloads(factory.connect_sock)
            assert connect_payloads[0]["Communication"] == "FRC_Connect"
            assert factory.connect_sock.addr == (fanuc_config.host, fanuc_config.port)
            assert factory.session_sock.addr == (fanuc_config.host, 16002)

            session_payloads = _sent_payloads(factory.session_sock)
            commands = [p.get("Command") for p in session_payloads]
            assert "FRC_Initialize" in commands
            assert "FRC_SetUFrameUTool" in commands
        finally:
            robot.disconnect()
    assert robot.is_connected is False


def test_send_action_decodes_orientation_and_injects_gripper(fanuc_config):
    factory = SocketFactory()
    robot = Fanuc(fanuc_config)
    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        try:
            sent = robot.send_action(
                {
                    "j0": 1.0,
                    "j1": 2.0,
                    "j2": 3.0,
                    "j3_sin": 1.0,
                    "j3_cos": 0.0,
                    "j4_sin": 0.0,
                    "j4_cos": 1.0,
                    "j5_sin": -1.0,
                    "j5_cos": 0.0,
                    "j7": 1.0,
                }
            )
            assert sent["j0"] == 1.0
            assert sent["j7"] == 1.0
            assert sent["lcb_type"] == "TA"
            assert sent["port_number"] == fanuc_config.gripper_close_port_number
            assert sent["port_value"] == fanuc_config.gripper_close_value

            motion = next(
                p for p in _sent_payloads(factory.session_sock) if p.get("Instruction") == "FRC_LinearMotion"
            )
            assert motion["Position"]["X"] == 1.0
            assert math.isclose(motion["Position"]["W"], 90.0, abs_tol=1e-6)
            assert math.isclose(motion["Position"]["P"], 0.0, abs_tol=1e-6)
            assert math.isclose(motion["Position"]["R"], -90.0, abs_tol=1e-6)
            assert motion["LCBType"] == "TA"
            assert motion["PortNumber"] == fanuc_config.gripper_close_port_number
        finally:
            robot.disconnect()


def test_wait_for_ack_after_motion(fanuc_config):
    factory = SocketFactory()
    robot = Fanuc(fanuc_config)
    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        try:
            robot.pause_state_poll()
            sent = robot.send_action({"j0": 1.0, "j1": 2.0, "j2": 3.0, "j3": 0.0, "j4": 0.0, "j5": 0.0})
            seq = int(sent["sequence_id"])
            factory.session_sock._responses.append(
                _line({"Instruction": "FRC_LinearMotion", "SequenceID": seq, "ErrorID": 0})
            )
            assert robot.wait_for_ack(seq, timeout_s=2.0) == 0
        finally:
            robot.disconnect()


def test_send_action_uses_latest_configuration(fanuc_config):
    factory = SocketFactory()
    robot = Fanuc(fanuc_config)
    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        try:
            robot._latest_configuration = {
                "UToolNumber": 1,
                "UFrameNumber": 0,
                "Front": 1,
                "Up": 1,
                "Left": 0,
                "Flip": 1,
                "Turn4": 0,
                "Turn5": 0,
                "Turn6": 1,
                "TimeTag": 99,
                "Extra": "drop-me",
            }
            robot.send_action({"j0": 1.0, "j1": 2.0, "j2": 3.0, "j3": 180.0, "j4": 0.0, "j5": 0.0})
            motion = next(
                p for p in _sent_payloads(factory.session_sock) if p.get("Instruction") == "FRC_LinearMotion"
            )
            assert motion["Configuration"]["Flip"] == 1
            assert motion["Configuration"]["Turn6"] == 1
        finally:
            robot.disconnect()


def test_send_action_open_gripper_from_buttons(fanuc_config):
    factory = SocketFactory()
    robot = Fanuc(fanuc_config)
    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        try:
            sent = robot.send_action(
                {
                    "j0": 0.0,
                    "j1": 0.0,
                    "j2": 0.0,
                    "j3": 0.0,
                    "j4": 0.0,
                    "j5": 0.0,
                    "buttons": {"grip": 0},
                }
            )
            assert sent["j7"] == 0.0
            assert sent["port_number"] == fanuc_config.gripper_open_port_number
        finally:
            robot.disconnect()


def test_get_observation_encodes_pose(fanuc_config):
    factory = SocketFactory()
    robot = Fanuc(fanuc_config)
    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        try:
            obs = robot.get_observation()
            assert obs["j0"] == 100.0
            assert obs["j1"] == 200.0
            assert obs["j2"] == 300.0
            assert obs["j3"] == 10.0
            assert obs["j4"] == 20.0
            assert obs["j5"] == 30.0
            assert math.isclose(obs["j3_sin"], math.sin(math.radians(10.0)), abs_tol=1e-6)
            assert math.isclose(obs["j3_cos"], math.cos(math.radians(10.0)), abs_tol=1e-6)
            assert obs["j7"] == 0.0
        finally:
            robot.disconnect()


def test_send_action_requires_connection(fanuc_config):
    robot = Fanuc(fanuc_config)
    with pytest.raises(RuntimeError, match="not connected"):
        robot.send_action({"j0": 0, "j1": 0, "j2": 0, "j3": 0, "j4": 0, "j5": 0})


def test_pose_encode_decode_roundtrip():
    raw = {"j0": 10.0, "j1": 20.0, "j2": 30.0, "j3": 90.0, "j4": 0.0, "j5": -45.0}
    encoded = encode_fanuc_pose_dict(raw)
    decoded = decode_fanuc_pose_dict(encoded)
    assert decoded["j0"] == 10.0
    assert math.isclose(decoded["j3"], 90.0, abs_tol=1e-6)
    assert math.isclose(decoded["j5"], -45.0, abs_tol=1e-6)
    assert math.isclose(decode_angle_degrees(0.0, 0.0), 0.0)
