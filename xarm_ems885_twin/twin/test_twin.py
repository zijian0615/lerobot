"""Tests for the EMS885 xArm twin (no robot, no Isaac Sim, no pxr needed).

    uv run pytest xarm_ems885_twin/twin/test_twin.py -q
"""

from __future__ import annotations

import json
import math
import re
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_usd  # noqa: E402
import omni_twin  # noqa: E402
from sources import Publisher, UdpSource  # noqa: E402
from xarm_model import Chain, arm_bodies, build_spec, gripper_pos_to_drive  # noqa: E402

TWIN = Path(__file__).resolve().parent.parent
CONFIG = json.loads((TWIN / "twin_config.json").read_text())

# Robot1 (192.168.1.204) on 2026-09-24, tcp_offset 0: reported joints (rad) and `arm.position` (mm).
CONTROLLER_SAMPLES = [
    ([-0.20194, -0.44836, -0.8929, 0.0, 1.34126, -0.20194], (344.880676, -70.607613, 447.113068)),
    ([-0.16828, -0.44685, -0.8934, -0.15272, 1.34873, -0.03999], (344.881439, -70.608078, 447.112976)),
    ([-1.343094, 0.605518, -1.684376, -2e-06, 1.078851, -1.343094], (140.052444, -604.401917, 280.003723)),
]


@pytest.fixture(scope="module")
def chain():
    return Chain(build_spec(CONFIG))


@pytest.mark.parametrize("q,pos_mm", CONTROLLER_SAMPLES)
def test_fk_matches_controller(chain, q, pos_mm):
    vals = chain.joint_values({"xarm": (q, 0.0)})
    assert np.allclose(chain.flange_in_base_mm("xarm", vals), pos_mm, atol=0.5)


def test_second_arm_uses_its_own_base(chain):
    q = CONTROLLER_SAMPLES[0][0]
    vals = chain.joint_values({"xarm": (q, 0.0), "xarm2": (q, 0.0)})
    # same joints -> same pose in each arm's own base frame, shifted by the base offset in the world
    assert np.allclose(chain.flange_in_base_mm("xarm2", vals), CONTROLLER_SAMPLES[0][1], atol=0.5)
    d = chain.site("xarm2/flange", vals) - chain.site("xarm/flange", vals)
    assert np.allclose(d, CONFIG["arms"]["xarm2"]["base_xyz"], atol=1e-9)


def test_gripper_mapping_and_mimic(chain):
    assert gripper_pos_to_drive(850) == 0.0
    assert math.isclose(gripper_pos_to_drive(0), 0.85)
    assert gripper_pos_to_drive(-10) == 0.85 and gripper_pos_to_drive(900) == 0.0
    q = CONTROLLER_SAMPLES[0][0]

    def finger_gap(drive):
        f = chain.forward(chain.joint_values({"xarm": (q, drive)}))
        return np.linalg.norm(f["xarm_left_finger"][1] - f["xarm_right_finger"][1])

    assert finger_gap(0.85) < finger_gap(0.0) - 0.05          # closing brings the fingers together
    vals = chain.joint_values({"xarm": (q, 0.4)})
    gripper_joints = [n for n in chain.joints if n.startswith("xarm_") and ("knuckle" in n or "finger" in n or "drive" in n)]
    assert len(gripper_joints) == 6 and all(math.isclose(vals[n], 0.4) for n in gripper_joints)


def test_leaphand_arm_has_no_gripper_joints():
    names = {b["name"] for b in arm_bodies("xarm2", ee="leaphand")}
    assert "xarm2_leaphand" in names and not any("knuckle" in n for n in names)


def _usda_prim_paths(text):
    """Prim paths from the def/class nesting of a .usda text (enough to check what build_usd writes)."""
    paths, stack = set(), []
    pending = None
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r'def (\w+ )?"([^"]+)"', s)
        if m:
            pending = m.group(2)
            continue
        if s == "{":
            stack.append(pending)
            if pending is not None:
                paths.add("/" + "/".join(p for p in stack if p is not None))
            pending = None
        elif s == "}":
            stack.pop()
    assert not stack, "unbalanced braces"
    return paths


def test_stage_matches_chain(tmp_path):
    out = tmp_path / "t.usda"
    spec = build_usd.build(TWIN / "twin_config.json", out, tmp_path / "t.twin.json", tmp_path / "t_scan.jpg",
                           with_scan=False)
    text = out.read_text()
    paths = _usda_prim_paths(text)
    for b in spec["bodies"]:
        assert b["path"] in paths, b["path"]
    for group in ("status", "trail"):
        for p in spec["prims"][group].values():
            assert p in paths
    for p in spec["prims"]["cameras"].values():
        assert p in paths
    n_joint_ops = text.count("quatf xformOp:orient:joint")
    assert n_joint_ops == sum(1 for b in spec["bodies"] if b["joint"])
    # the chain JSON on disk round-trips into the same FK
    ch = Chain.load(tmp_path / "t.twin.json")
    vals = ch.joint_values({"xarm": (CONTROLLER_SAMPLES[2][0], 0.0)})
    assert np.allclose(ch.flange_in_base_mm("xarm", vals), CONTROLLER_SAMPLES[2][1], atol=0.5)


def test_built_stage_is_current():
    """A locally built xarm_ems885.twin.json must describe the current model (rebuild with build_usd.py if this fails)."""
    path = TWIN / "xarm_ems885.twin.json"
    if not path.exists():
        pytest.skip("stage not built on this machine")
    on_disk = json.loads(path.read_text())
    fresh = build_spec(CONFIG)
    assert [b["name"] for b in on_disk["bodies"]] == [b["name"] for b in fresh["bodies"]]
    for a, b in zip(on_disk["bodies"], fresh["bodies"]):
        assert np.allclose(a["pos"], b["pos"]) and np.allclose(a["quat"], b["quat"])


def test_scan_alignment_is_rigid_and_level():
    align = json.loads((TWIN / CONFIG["scan"]["alignment"]).read_text())
    T = np.asarray(align["T_world_scan"])
    R = T[:3, :3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(R), 1.0)
    assert align["metrics"]["tape_chamfer_mm"] < 5.0
    from glb import load_glb

    mesh = load_glb(TWIN / CONFIG["scan"]["glb"])
    z = mesh.positions @ R[2] + T[2, 3]
    table = z[np.abs(z) < 0.01]
    assert len(table) > 0.3 * len(z) and abs(np.median(table)) < 0.003     # the table lands on z = 0


def test_udp_roundtrip():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    src = UdpSource(port, bind="127.0.0.1").start()
    pub = Publisher("127.0.0.1", port)
    try:
        q = [0.1, -0.2, -0.3, 0.0, 0.5, 0.6]
        for _ in range(50):
            pub.publish("xarm", q, 425.0)
            time.sleep(0.02)
            got = src.latest().get("xarm")
            if got:
                break
        assert got is not None and np.allclose(got.q, q) and math.isclose(got.drive(), 0.425)
    finally:
        pub.close()
        src.close()


@pytest.fixture(scope="module")
def stage_args(tmp_path_factory):
    d = tmp_path_factory.mktemp("stage")
    build_usd.build(TWIN / "twin_config.json", d / "s.usda", d / "s.twin.json", d / "s.jpg", with_scan=False)
    return ["--stage", str(d / "s.usda"), "--chain", str(d / "s.twin.json")]


def test_twin_loop_demo(stage_args):
    w = omni_twin.main(stage_args + ["--source", "demo", "--no-render", "--duration", "1.0", "--trail", "20"])
    assert set(w.status) == {"xarm", "xarm2"} and all(v == "live" for v in w.status.values())
    assert w.joint_values["xarm_drive_joint"] == w.joint_values["xarm_right_finger_joint"]
    assert all(len(t) >= 2 for t in w.trails.values())


def test_twin_loop_marks_missing_arm(stage_args):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    w = omni_twin.main(stage_args + ["--source", "udp", "--udp-port", str(port), "--no-render", "--duration", "0.5"])
    assert w.status == {"xarm": "none", "xarm2": "none"}
