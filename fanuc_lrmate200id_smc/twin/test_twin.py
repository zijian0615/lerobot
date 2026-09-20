"""Tests for the digital twin. Run:
    uv run --no-project --with mujoco --with numpy --with pytest pytest twin/test_twin.py -q
The RMI tests run against mock_rmi_server.py, i.e. they check OUR protocol assumptions and the pipeline,
not the real controller.
"""
import os
import sys
import time

import mujoco
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import twin  # noqa: E402
from joint_map import flange_world_mm, fanuc_to_model, model_to_fanuc  # noqa: E402
from mock_rmi_server import MockRmiController  # noqa: E402
from sources import JointStatePublisher, RmiError, RmiJointSource, UdpJointSource, parse_joint_response  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(twin.SCENE)


def wait_for(cond, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def cart_fn(model, mode):
    d = mujoco.MjData(model)

    def f(joints_deg):
        d.qpos[:6] = fanuc_to_model(joints_deg, mode)
        mujoco.mj_forward(model, d)
        return tuple(flange_world_mm(model, d)) + (0.0, 0.0, 0.0)
    return f


def test_joint_map_roundtrip():
    rng = np.random.default_rng(1)
    for mode in ("coupled", "direct"):
        j = rng.uniform(-100, 100, 6)
        assert np.allclose(model_to_fanuc(fanuc_to_model(j, mode), mode), j)
    with pytest.raises(ValueError):
        fanuc_to_model([0] * 6, "nope")


def test_coupled_mode_keeps_forearm_level_when_only_j2_moves(model):
    """FANUC convention: with J3 = 0 the forearm stays horizontal whatever J2 is (that is what 'coupled' encodes)."""
    d = mujoco.MjData(model)
    for j2 in (0, 30, 60):
        d.qpos[:6] = fanuc_to_model([0, j2, 0, 0, 0, 0], "coupled")
        mujoco.mj_forward(model, d)
        approach = d.site("tcp").xmat.reshape(3, 3)[:, 2]          # tool Z = flange normal
        assert np.allclose(approach, [1, 0, 0], atol=1e-6)


@pytest.mark.parametrize("resp,ok", [
    ({"Command": "FRC_ReadJointAngles", "ErrorID": 0, "JointAngle": {"J1": 1, "J2": 2, "J3": 3, "J4": 4, "J5": 5, "J6": 6, "J7": 0}}, True),
    ({"ErrorID": 0, "JointAngles": {"j1": 1, "j2": 2, "j3": 3, "j4": 4, "j5": 5, "j6": 6}}, True),
    ({"ErrorID": 0, "J1": 1, "J2": 2, "J3": 3, "J4": 4, "J5": 5, "J6": 6}, True),
    ({"ErrorID": 2556957}, False),
    ({"ErrorID": 0, "JointAngle": {"J1": 1}}, False),
])
def test_parse_joint_response(resp, ok):
    if ok:
        assert parse_joint_response(resp) == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    else:
        with pytest.raises(RmiError):
            parse_joint_response(resp)


def test_rmi_end_to_end_tracks_mock_controller_and_is_read_only(model):
    ctrl = MockRmiController().start()
    try:
        data = twin.main(["--source", "rmi", "--host", "127.0.0.1", "--port", str(ctrl.port),
                          "--headless", "--duration", "2.5", "--rate", "50"])
        assert ctrl.n_reads > 40                                    # ~50 Hz polling for 2.5 s
        q_expected = fanuc_to_model(ctrl.last_joints, "coupled")
        assert np.abs(data.qpos[:6] - q_expected).max() < 0.1       # within motion during ~2 polls
        assert ctrl.forbidden == []                                 # never a motion instruction, never FRC_Abort
    finally:
        ctrl.stop()


def test_reconnects_after_controller_restart():
    ctrl = MockRmiController().start()
    port = ctrl.port
    src = RmiJointSource("127.0.0.1", port, rate_hz=50, timeout=1.0).start()
    try:
        assert wait_for(lambda: src.count > 5)
        ctrl.stop()
        assert wait_for(lambda: src.status.startswith("error"), 5.0)
        n = src.count
        ctrl2 = MockRmiController(port=port).start()
        try:
            assert wait_for(lambda: src.status == "connected" and src.count > n + 5, 10.0)
        finally:
            ctrl2.stop()
    finally:
        src.close()


def test_second_rmi_client_is_refused_and_first_keeps_working():
    ctrl = MockRmiController(single_session=True).start()
    a = RmiJointSource("127.0.0.1", ctrl.port, rate_hz=50, timeout=1.0).start()
    try:
        assert wait_for(lambda: a.count > 5)
        b = RmiJointSource("127.0.0.1", ctrl.port, rate_hz=50, timeout=1.0).start()
        try:
            assert wait_for(lambda: b.status.startswith("error"), 5.0)
            assert b.latest() is None
            n = a.count
            assert wait_for(lambda: a.count > n + 5)
        finally:
            b.close()
    finally:
        a.close()
        ctrl.stop()


def test_udp_source_receives_published_joints():
    src = UdpJointSource(port=0, bind="127.0.0.1").start()
    try:
        JointStatePublisher("127.0.0.1", src.port).publish([1, 2, 3, 4, 5, 6])
        assert wait_for(lambda: src.latest() is not None)
        assert src.latest().joints_deg == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    finally:
        src.close()


@pytest.mark.parametrize("mode", ["coupled", "direct"])
def test_check_cartesian_identifies_the_j3_convention(model, capsys, mode):
    ctrl = MockRmiController(j3_mode=mode, cartesian_fn=cart_fn(model, mode)).start()
    try:
        twin.main(["--source", "rmi", "--host", "127.0.0.1", "--port", str(ctrl.port), "--check-cartesian"])
        assert f"best match: --j3-mode {mode}" in capsys.readouterr().out
    finally:
        ctrl.stop()
