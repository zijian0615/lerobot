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

import numpy as np
import pytest

from lerobot.robots.fanuc import Fanuc, FanucConfig
from lerobot.robots.fanuc.er4ia_fk import (
    controller_from_sim_deg,
    controller_tcp_pose,
    sim_from_controller_deg,
)
from lerobot.robots.fanuc.pose import (
    FANUC_RAW_NAMES,
    decode_angle_degrees,
    decode_fanuc_pose_dict,
    encode_fanuc_pose_dict,
    raw_fanuc_pose_dict,
    rotation_matrix_from_wpr,
)
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
    def __init__(self, w: float = 10.0, extra: tuple[dict, ...] = ()):
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
                            "W": w,
                            "P": 20.0,
                            "R": 30.0,
                        },
                        "Configuration": {"UFrameNumber": 0, "UToolNumber": 1},
                    }
                ),
                *(_line(payload) for payload in extra),
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
    assert robot.observation_features["j3"] is float
    assert robot.observation_features["j7"] is float
    assert robot.action_features["j5"] is float
    assert robot.is_calibrated is True
    robot.calibrate()
    robot.configure()


@pytest.mark.parametrize("init_error", [2556937, 2556943, 2556955])
def test_connect_retries_after_busy_initialize(fanuc_config, init_error):
    factory = SocketFactory()
    factory.session_sock._responses = [
        _line({"Command": "FRC_Initialize", "ErrorID": init_error, "GroupMask": 1}),
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
            assert math.isclose(sent["j3"], 90.0, abs_tol=1e-6)
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


def test_stream_send_waits_for_inflight_ack(fanuc_config):
    factory = SocketFactory()
    robot = Fanuc(fanuc_config)
    pose = {"j0": 1.0, "j1": 2.0, "j2": 3.0, "j3": 0.0, "j4": 0.0, "j5": 0.0, "stream": True}
    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        try:
            robot.pause_state_poll()
            first = robot.send_action(dict(pose))
            second = robot.send_action(dict(pose))
            motions = [p for p in _sent_payloads(factory.session_sock) if p.get("Instruction") == "FRC_LinearMotion"]
            assert len(motions) == 1
            assert "sequence_id" in first
            assert "sequence_id" not in second

            seq = int(first["sequence_id"])
            factory.session_sock._responses.append(
                _line({"Instruction": "FRC_LinearMotion", "SequenceID": seq, "ErrorID": 0})
            )
            assert robot.wait_for_ack(seq, timeout_s=2.0) == 0
            robot.send_action({**pose, "j0": 20.0})
            motions = [p for p in _sent_payloads(factory.session_sock) if p.get("Instruction") == "FRC_LinearMotion"]
            assert len(motions) == 2
        finally:
            robot.disconnect()


def test_get_observation_is_raw_pose(fanuc_config):
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
            assert obs["j7"] == 0.0
            assert not any(key.endswith(("_sin", "_cos")) for key in obs)
        finally:
            robot.disconnect()


def test_features_are_seven_raw_values(fanuc_config):
    robot = Fanuc(fanuc_config)
    assert tuple(robot.action_features) == FANUC_RAW_NAMES
    assert tuple(k for k in robot.observation_features if k.startswith("j")) == FANUC_RAW_NAMES


def test_w_is_continuous_across_180(fanuc_config):
    """Tool-down W crosses +-180 deg: recorded in [0, 360), sent back to the controller in (-180, 180]."""
    factory = SocketFactory(w=-179.0)
    robot = Fanuc(fanuc_config)
    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        try:
            assert math.isclose(robot.get_observation()["j3"], 181.0)
            sent = robot.send_action({"j0": 1.0, "j1": 2.0, "j2": 3.0, "j3": 190.0, "j4": 0.0, "j5": 90.0})
            motion = next(
                p for p in _sent_payloads(factory.session_sock) if p.get("Instruction") == "FRC_LinearMotion"
            )
            assert math.isclose(motion["Position"]["W"], -170.0)
            assert math.isclose(sent["j3"], 190.0)
        finally:
            robot.disconnect()


def test_gripper_state_follows_the_last_command_without_a_state_input(fanuc_config):
    factory = SocketFactory()
    robot = Fanuc(fanuc_config)
    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        try:
            assert robot.get_observation()["j7"] == 0.0
            robot.send_action({"j0": 1.0, "j1": 2.0, "j2": 3.0, "j3": 180.0, "j4": 0.0, "j5": 0.0, "j7": 1.0})
            assert robot.get_observation()["j7"] == 1.0
        finally:
            robot.disconnect()


def test_gripper_state_input_is_polled_when_configured(tmp_path):
    config = FanucConfig(id="test_fanuc", calibration_dir=tmp_path, gripper_state_port_number=7)
    factory = SocketFactory(extra=({"Command": "FRC_ReadDIN", "ErrorID": 0, "PortNumber": 7, "PortValue": 1},))
    robot = Fanuc(config)
    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        try:
            assert robot.get_observation()["j7"] == 1.0
            reads = [p for p in _sent_payloads(factory.session_sock) if p.get("Command") == "FRC_ReadDIN"]
            assert reads and reads[0]["PortNumber"] == 7
        finally:
            robot.disconnect()


def test_reset_episode_is_a_no_op_on_the_real_robot(fanuc_config):
    factory = SocketFactory()
    robot = Fanuc(fanuc_config)
    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        try:
            assert robot.reset_episode() is None
            assert not any(p.get("Command") == "SIM_Reset" for p in _sent_payloads(factory.session_sock))
        finally:
            robot.disconnect()


def test_reset_episode_asks_the_sim_and_clears_pending_moves(tmp_path):
    config = FanucConfig(id="test_fanuc", calibration_dir=tmp_path, sim_reset=True)
    factory = SocketFactory()
    robot = Fanuc(config)

    def answer_reset():  # like the sim: reply only after the request arrives
        for _ in range(200):
            if any(p.get("Command") == "SIM_Reset" for p in _sent_payloads(factory.session_sock)):
                factory.session_sock._responses.append(
                    _line({"Command": "SIM_Reset", "ErrorID": 0, "Episode": 1, "Layout": {}})
                )
                return
            threading.Event().wait(0.01)

    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        threading.Thread(target=answer_reset, daemon=True).start()
        try:
            robot.send_action({"j0": 1.0, "j1": 2.0, "j2": 3.0, "j3": 180.0, "j4": 0.0, "j5": 0.0, "j7": 1.0, "stream": True})
            reply = robot.reset_episode(timeout_s=2.0)
            assert reply["Episode"] == 1
            assert any(p.get("Command") == "SIM_Reset" for p in _sent_payloads(factory.session_sock))
            assert not robot._pending_futures
            assert robot.get_observation()["j7"] == 0.0
        finally:
            robot.disconnect()


def test_raw_pose_dict_accepts_old_sin_cos_encoding():
    old = encode_fanuc_pose_dict({"j0": 1.0, "j1": 2.0, "j2": 3.0, "j3": -179.5, "j4": 5.0, "j5": -90.0})
    raw = raw_fanuc_pose_dict({**old, "j7": 1.0})
    assert set(raw) == set(FANUC_RAW_NAMES)
    assert math.isclose(raw["j3"], 180.5, abs_tol=1e-6)
    assert math.isclose(raw["j5"], -90.0, abs_tol=1e-6)


def test_send_action_requires_connection(fanuc_config):
    robot = Fanuc(fanuc_config)
    with pytest.raises(RuntimeError, match="not connected"):
        robot.send_action({"j0": 0, "j1": 0, "j2": 0, "j3": 0, "j4": 0, "j5": 0})


def test_set_uframe_retries_after_leftover_session(fanuc_config):
    factory = SocketFactory()
    factory.session_sock._responses = [
        _line({"Command": "FRC_Initialize", "ErrorID": 0}),
        _line({"Command": "FRC_SetUFrameUTool", "ErrorID": 2556955, "Group": 164}),
        _line({"Command": "FRC_Initialize", "ErrorID": 0}),
        _line({"Command": "FRC_SetUFrameUTool", "ErrorID": 0, "Group": 1}),
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
            assert commands.count("FRC_SetUFrameUTool") >= 2
            assert "FRC_Abort" in commands
            assert robot.is_connected
        finally:
            robot.disconnect()


def test_joint_angles_publish_to_twin_udp(fanuc_config):
    robot = Fanuc(fanuc_config)
    captured: list[tuple[dict, tuple]] = []

    class _Udp:
        def sendto(self, data, addr):
            captured.append((json.loads(data.decode()), addr))

        def close(self):
            pass

    robot._twin_sock = _Udp()
    robot._twin_addr = ("127.0.0.1", 5005)
    robot._dispatch_response(
        {
            "Command": "FRC_ReadJointAngles",
            "ErrorID": 0,
            "JointAngle": {"J1": 10.0, "J2": 20.0, "J3": 30.0, "J4": 0.0, "J5": -10.0, "J6": 5.0},
        }
    )
    assert captured == [
        ({"joints_deg": [10.0, 20.0, 30.0, 0.0, -10.0, 5.0]}, ("127.0.0.1", 5005))
    ]
    robot._twin_sock = None


def test_recover_after_fault_aborts_resets_and_reinitializes(fanuc_config):
    factory = SocketFactory()
    robot = Fanuc(fanuc_config)
    with (
        patch("lerobot.robots.fanuc.fanuc.socket.socket", factory),
        patch("lerobot.robots.fanuc.fanuc.time.sleep", return_value=None),
    ):
        robot.connect()
        try:
            factory.session_sock.sent.clear()
            robot.seq_id = 56
            robot.recover_after_fault()
            commands = [p.get("Command") for p in _sent_payloads(factory.session_sock)]
            assert "FRC_Abort" in commands
            assert "FRC_Reset" in commands
            assert "FRC_Initialize" in commands
            assert "FRC_GetStatus" in commands
            assert "FRC_SetUFrameUTool" in commands
            assert robot.seq_id == 1
        finally:
            robot.disconnect()


def test_status_next_seq_keys():
    assert Fanuc._status_next_seq({"NextSequenceID": 4}) == 4
    assert Fanuc._status_next_seq({"NextSequenceId": 2}) == 2
    assert Fanuc._status_next_seq({}) is None
    assert Fanuc._status_next_seq(None) is None


def test_telegrip_fanuc_does_not_require_urdf():
    import numpy as np

    from lerobot.teleoperators.telegrip.config_telegrip import ControllerSide, TelegripConfig
    from lerobot.teleoperators.telegrip.telegrip_processor import make_telegrip_processors

    robot = Fanuc(FanucConfig())
    teleop, _, _ = make_telegrip_processors(
        robot, TelegripConfig(controller_side=ControllerSide.RIGHT, urdf_path="")
    )
    obs = {
        "j0": 100.0,
        "j1": -20.0,
        "j2": -200.0,
        "j3": 178.2,
        "j4": -1.6,
        "j5": -2.2,
        "j7": 0.0,
    }
    idle = teleop(({"vr.right.enabled": False, "vr.right.gripper_closed": True}, obs))
    assert idle == {}

    held = {
        "vr.right.enabled": True,
        "vr.right.target_delta": np.zeros(3),
        "vr.right.wrist_roll_deg": 0.0,
        "vr.right.wrist_flex_deg": 0.0,
        "vr.right.gripper_closed": False,
        "vr.right.reset_origin": True,
    }
    action = teleop((held, obs))
    assert math.isclose(action["j0"], 100.0, abs_tol=1e-4)
    assert action["j7"] == 1.0
    assert action["stream"] is True

    moved = dict(held)
    moved["vr.right.reset_origin"] = False
    # telegrip target_delta is [-vr_x, vr_z, vr_y]: 2 cm forward (vr_z = -0.02) -> user-frame +Y
    moved["vr.right.target_delta"] = np.array([0.0, -0.02, 0.0])
    action = teleop((moved, obs))
    assert math.isclose(action["j0"], 100.0, abs_tol=1e-3)
    assert math.isclose(action["j1"], 0.0, abs_tol=1e-3)
    # 2 cm right (vr_x = +0.02) -> user-frame +X
    moved["vr.right.target_delta"] = np.array([-0.02, 0.0, 0.0])
    action = teleop((moved, obs))
    assert math.isclose(action["j0"], 120.0, abs_tol=1e-3)
    assert math.isclose(action["j1"], -20.0, abs_tol=1e-3)


def test_telegrip_fanuc_maps_controller_rotation():
    import numpy as np

    from lerobot.teleoperators.telegrip.config_telegrip import ControllerSide, TelegripConfig
    from lerobot.teleoperators.telegrip.telegrip_processor import make_telegrip_processors
    from lerobot.utils.rotation import Rotation

    robot = Fanuc(FanucConfig())
    teleop, _, _ = make_telegrip_processors(
        robot, TelegripConfig(controller_side=ControllerSide.RIGHT, urdf_path="")
    )
    obs = {"j0": 100.0, "j1": -20.0, "j2": -200.0, "j3": 0.0, "j4": 0.0, "j5": 0.0, "j7": 0.0}
    held = {
        "vr.right.enabled": True,
        "vr.right.target_delta": np.zeros(3),
        "vr.right.wrist_quat": Rotation.from_rotvec(np.array([0.0, math.pi / 2.0, 0.0])).as_quat(),
        "vr.right.gripper_closed": False,
        "vr.right.reset_origin": True,
    }
    action = teleop((held, obs))
    decoded = decode_fanuc_pose_dict(action)
    assert math.isclose(decoded["j3"], 0.0, abs_tol=1e-3)
    assert math.isclose(decoded["j4"], 0.0, abs_tol=1e-3)
    assert math.isclose(decoded["j5"], 90.0, abs_tol=1e-3)


def test_wpr_matrix_roundtrip():
    from lerobot.robots.fanuc.pose import rotation_matrix_from_wpr, wpr_from_rotation_matrix

    for wpr in ((178.2, -1.6, -2.2), (90.0, 0.0, -45.0), (0.0, 30.0, 10.0)):
        back = wpr_from_rotation_matrix(rotation_matrix_from_wpr(*wpr))
        for got, expected in zip(back, wpr, strict=True):
            assert math.isclose(got, expected, abs_tol=1e-6)


def test_phone_fanuc_pipeline_maps_quest_delta():
    import numpy as np
    from scipy.spatial.transform import Rotation

    from lerobot.robots.fanuc.phone_processor import make_phone_fanuc_processors
    from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS

    robot = Fanuc(FanucConfig())
    teleop, _, _ = make_phone_fanuc_processors(robot, PhoneConfig(phone_os=PhoneOS.ANDROID))
    obs = {
        "j0": 100.0,
        "j1": -20.0,
        "j2": -200.0,
        "j3": 178.2,
        "j4": -1.6,
        "j5": -2.2,
        "j7": 0.0,
    }
    held = {
        "phone.enabled": True,
        "phone.pos": np.zeros(3),
        "phone.rot": Rotation.identity(),
        "phone.raw_inputs": {},
    }
    action = teleop((held, obs))
    assert math.isclose(action["j0"], 100.0, abs_tol=1e-4)
    assert math.isclose(action["j1"], -20.0, abs_tol=1e-4)
    decoded = decode_fanuc_pose_dict(action)
    assert math.isclose(decoded["j3"], 178.2, abs_tol=1e-3)
    assert action["j7"] == 0.0

    moved = dict(held)
    moved["phone.pos"] = np.array([0.0, 0.02, 0.0])
    action = teleop((moved, obs))
    assert math.isclose(action["j0"], 115.0, abs_tol=1e-3)

    closing = dict(held)
    closing["phone.raw_inputs"] = {"reservedButtonA": 1.0}
    for _ in range(20):
        action = teleop((closing, obs))
    assert action["j7"] == 1.0
    assert action["stream"] is True


def test_pose_encode_decode_roundtrip():
    raw = {"j0": 10.0, "j1": 20.0, "j2": 30.0, "j3": 90.0, "j4": 0.0, "j5": -45.0}
    encoded = encode_fanuc_pose_dict(raw)
    decoded = decode_fanuc_pose_dict(encoded)
    assert decoded["j0"] == 10.0
    assert math.isclose(decoded["j3"], 90.0, abs_tol=1e-6)
    assert math.isclose(decoded["j5"], -45.0, abs_tol=1e-6)
    assert math.isclose(decode_angle_degrees(0.0, 0.0), 0.0)


# Real ER-4iA at rest, UF0/UT1, 2026-10-07 (fanuc_er4ia_sim/check_j3_convention.py).
_REAL_JOINTS = {"J1": 15.9, "J2": -17.13, "J3": 0.85, "J4": 13.84, "J5": -85.03, "J6": -79.87}
_REAL_XYZWPR = (252.529, -0.6921, -10.23, -176.722, -14.652, -63.888)
_UTOOL_REPLY = {
    "Command": "FRC_ReadUToolData",
    "ErrorID": 0,
    "ToolNumber": 1,
    "Group": 1,
    "Frame": {"X": 0.0, "Y": 0.0, "Z": 223.0, "W": 0.0, "P": 0.0, "R": 0.0},
}


def _joint_robot(fanuc_config):
    fanuc_config.motion = "joint"
    fanuc_config.twin_udp_host = ""
    return SocketFactory(extra=(_UTOOL_REPLY,)), Fanuc(fanuc_config)


def test_controller_tcp_pose_matches_the_real_arm():
    joints = sim_from_controller_deg([_REAL_JOINTS[f"J{i}"] for i in range(1, 7)])
    rotation, position = controller_tcp_pose(np.radians(joints), (0.0, 0.0, 223.0, 0.0, 0.0, 0.0))
    assert np.allclose(position * 1000.0, _REAL_XYZWPR[:3], atol=0.5)
    measured = np.asarray(rotation_matrix_from_wpr(*_REAL_XYZWPR[3:]))
    assert np.allclose(rotation, measured, atol=1e-3)
    assert np.allclose(controller_from_sim_deg(joints), list(_REAL_JOINTS.values()))


def test_joint_observation_reports_sim_j3(fanuc_config):
    factory, robot = _joint_robot(fanuc_config)
    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        try:
            robot.pause_state_poll()
            robot._dispatch_response({"Command": "FRC_ReadJointAngles", "ErrorID": 0, "JointAngle": _REAL_JOINTS})
            obs = robot.get_observation()
            assert obs["J2.pos"] == -17.13
            assert math.isclose(obs["J3.pos"], 0.85 - 17.13)
            assert obs["J6.pos"] == -79.87
        finally:
            robot.disconnect()


def test_joint_motion_targets_the_controller_tcp(fanuc_config):
    factory, robot = _joint_robot(fanuc_config)
    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        try:
            robot.pause_state_poll()
            assert tuple(robot.action_features)[0] == "J1.pos"
            assert robot._utool_data == (0.0, 0.0, 223.0, 0.0, 0.0, 0.0)
            assert any(
                p.get("Command") == "FRC_ReadUToolData" and p["ToolNumber"] == fanuc_config.utool
                for p in _sent_payloads(factory.session_sock)
            )
            robot.commit_commanded_pose(*_REAL_XYZWPR)
            joints = sim_from_controller_deg([_REAL_JOINTS[f"J{i}"] for i in range(1, 7)])
            action = {f"J{i + 1}.pos": value for i, value in enumerate(joints)}
            robot.send_action({**action, "gripper.pos": 1.0})
            motion = next(
                p for p in _sent_payloads(factory.session_sock) if p.get("Instruction") == "FRC_LinearMotion"
            )
            position = motion["Position"]
            assert motion["SpeedType"] == "mmSec"
            assert motion["Speed"] == fanuc_config.joint_tcp_speed_mm_s
            assert np.allclose([position[k] for k in "XYZ"], _REAL_XYZWPR[:3], atol=0.5)
            assert np.allclose(
                rotation_matrix_from_wpr(position["W"], position["P"], position["R"]),
                rotation_matrix_from_wpr(*_REAL_XYZWPR[3:]),
                atol=1e-3,
            )
            assert motion["PortNumber"] == fanuc_config.gripper_close_port_number
        finally:
            robot.disconnect()


def test_joint_motion_limits_one_step(fanuc_config):
    factory, robot = _joint_robot(fanuc_config)
    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        try:
            robot.pause_state_poll()
            robot.commit_commanded_pose(*_REAL_XYZWPR)
            joints = sim_from_controller_deg([_REAL_JOINTS[f"J{i}"] for i in range(1, 7)])
            joints[0] += 20.0
            robot.send_action({f"J{i + 1}.pos": value for i, value in enumerate(joints)})
            motion = next(
                p for p in _sent_payloads(factory.session_sock) if p.get("Instruction") == "FRC_LinearMotion"
            )
            step = np.linalg.norm(np.subtract([motion["Position"][k] for k in "XYZ"], _REAL_XYZWPR[:3]))
            _, target = controller_tcp_pose(np.radians(joints), robot._utool_data)
            full = np.linalg.norm(target * 1000.0 - np.asarray(_REAL_XYZWPR[:3]))
            assert full > 50.0
            assert np.isclose(step, fanuc_config.joint_max_step_mm, atol=0.01)
        finally:
            robot.disconnect()


def test_joint_observation_keeps_native_camera_resolution(fanuc_config):
    class _Cam:
        width, height = 1920, 1080
        is_connected = True

        def connect(self):
            pass

        def disconnect(self):
            pass

        def read(self):
            return np.zeros((1080, 1920, 3), np.uint8)

    factory, robot = _joint_robot(fanuc_config)
    robot.cameras = {"table": _Cam()}
    assert robot.observation_features["table"] == (1080, 1920, 3)
    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        try:
            robot.pause_state_poll()
            robot._dispatch_response({"Command": "FRC_ReadJointAngles", "ErrorID": 0, "JointAngle": _REAL_JOINTS})
            assert robot.get_observation()["table"].shape == (1080, 1920, 3)
        finally:
            robot.disconnect()


def test_joint_motion_steps_from_the_previous_target(fanuc_config):
    """While the arm has not moved yet, successive targets still advance one step each, up to the lead cap."""
    factory, robot = _joint_robot(fanuc_config)
    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        try:
            robot.pause_state_poll()
            robot.commit_commanded_pose(*_REAL_XYZWPR)
            joints = sim_from_controller_deg([_REAL_JOINTS[f"J{i}"] for i in range(1, 7)])
            joints[0] += 20.0
            live = np.asarray(_REAL_XYZWPR[:3])
            leads = []
            for _ in range(8):
                robot._stream_sent_mono = None  # pretend the acks came back, so every send goes out
                with robot._pending_lock:
                    robot._pending_futures.clear()
                robot.send_action({f"J{i + 1}.pos": value for i, value in enumerate(joints)})
                motion = [p for p in _sent_payloads(factory.session_sock) if p.get("Instruction") == "FRC_LinearMotion"][-1]
                leads.append(float(np.linalg.norm(np.subtract([motion["Position"][k] for k in "XYZ"], live))))
            step = fanuc_config.joint_max_step_mm
            assert np.allclose(leads[:5], [step * (k + 1) for k in range(5)], atol=0.05)
            assert np.isclose(max(leads), fanuc_config.joint_max_lead_mm, atol=0.05)
        finally:
            robot.disconnect()


def test_joint_motion_limits_the_joint_step(fanuc_config):
    """A 109 deg J6 jump between chunks becomes joint_max_step_deg per step, all joints scaled together."""
    factory, robot = _joint_robot(fanuc_config)
    with patch("lerobot.robots.fanuc.fanuc.socket.socket", factory):
        robot.connect()
        try:
            robot.pause_state_poll()
            robot._dispatch_response({"Command": "FRC_ReadJointAngles", "ErrorID": 0, "JointAngle": _REAL_JOINTS})
            robot.commit_commanded_pose(*_REAL_XYZWPR)
            start = np.array(sim_from_controller_deg([_REAL_JOINTS[f"J{i}"] for i in range(1, 7)]))
            goal = start.copy()
            goal[5] += 109.0
            goal[1] += 10.0
            robot.send_action({f"J{i + 1}.pos": float(v) for i, v in enumerate(goal)})
            sent = robot._joint_target_deg - start
            assert np.isclose(np.max(np.abs(sent)), fanuc_config.joint_max_step_deg)
            assert np.isclose(sent[1] / sent[5], 10.0 / 109.0)
        finally:
            robot.disconnect()
