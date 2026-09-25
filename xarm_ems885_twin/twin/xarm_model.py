"""xArm6 (+ xArm Gripper / LeapHand outline) kinematic model for the EMS885 tabletop twin. numpy only.

Numbers come from UFACTORY's xarm_ros2 `xarm_description` (commit 62936f7e): `config/kinematics/default/
xarm6_default_kinematics.yaml` for the arm and `urdf/gripper/xarm_gripper.urdf.xacro` for the gripper. Every arm joint turns
about its own +Z, so the model equals the controller's: FK of the reported joints reproduces `arm.position` to 0.0 mm
(tcp_offset 0, checked on Robot1 on 2026-09-24; see test_twin.py).

A chain is a list of bodies, parents first. Each body: name, parent, pos (m), quat (w, x, y, z), joint (or None) and geoms.
Joint: {name, type "hinge", axis, range, mimic: {joint, multiplier} | None}. Mimic joints (the gripper linkage) take their
value from the joint they follow, so the twin only needs J1..J6 and one gripper value per arm.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# (xyz m, rpy rad) of joint i in its parent link; axis +Z for all six.
XARM6_JOINTS = (
    ((0.0, 0.0, 0.267), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-1.5708, 0.0, 0.0)),
    ((0.0535, -0.2845, 0.0), (0.0, 0.0, 0.0)),
    ((0.0775, 0.3425, 0.0), (-1.5708, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (1.5708, 0.0, 0.0)),
    ((0.076, 0.097, 0.0), (-1.5708, 0.0, 0.0)),
)
XARM6_LIMITS = (
    (-2 * math.pi, 2 * math.pi),
    (-2.059, 2.0944),
    (-3.927, 0.19198),
    (-2 * math.pi, 2 * math.pi),
    (-1.69297, math.pi),
    (-2 * math.pi, 2 * math.pi),
)
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))

# xArm Gripper: drive_joint 0 = open ... 0.85 rad = closed. The SDK reports the gripper position 850 (open) ... 0 (closed).
GRIPPER_DRIVE_MAX = 0.85
GRIPPER_POS_OPEN = 850.0
GRIPPER_TCP_Z = 0.172  # link_tcp above the gripper base (xarm_gripper.urdf.xacro joint_tcp)
# (link, parent, xyz, axis, follows drive?) — rpy is zero for every gripper joint.
GRIPPER_LINKS = (
    ("left_outer_knuckle", "gripper_base", (0.0, 0.035, 0.059098), (1, 0, 0)),
    ("left_finger", "left_outer_knuckle", (0.0, 0.035465, 0.042039), (-1, 0, 0)),
    ("left_inner_knuckle", "gripper_base", (0.0, 0.02, 0.074098), (1, 0, 0)),
    ("right_outer_knuckle", "gripper_base", (0.0, -0.035, 0.059098), (-1, 0, 0)),
    ("right_finger", "right_outer_knuckle", (0.0, -0.035465, 0.042039), (1, 0, 0)),
    ("right_inner_knuckle", "gripper_base", (0.0, -0.02, 0.074098), (-1, 0, 0)),
)

# LeapHand on Robot2: not modelled in CAD here; a palm block plus four finger blocks along the flange +Z so the twin shows
# where the hand is. Sizes are rough (LEAP Hand palm ~ 0.1 x 0.09 m, fingers ~ 0.1 m).
LEAPHAND_BOXES = (
    ("mount", (0.0, 0.0, 0.015), (0.08, 0.08, 0.03)),
    ("palm", (0.0, 0.0, 0.075), (0.10, 0.09, 0.09)),
    ("finger_1", (0.035, 0.03, 0.17), (0.02, 0.02, 0.10)),
    ("finger_2", (0.0, 0.03, 0.17), (0.02, 0.02, 0.10)),
    ("finger_3", (-0.035, 0.03, 0.17), (0.02, 0.02, 0.10)),
    ("thumb", (0.0, -0.045, 0.13), (0.02, 0.02, 0.07)),
)


def rpy_to_quat(rpy):
    """URDF fixed-axis roll-pitch-yaw -> quaternion (w, x, y, z)."""
    r, p, y = (0.5 * float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def axis_angle_quat(axis, angle):
    a = np.asarray(axis, dtype=float)
    return np.concatenate([[math.cos(angle / 2)], a / np.linalg.norm(a) * math.sin(angle / 2)])


def gripper_pos_to_drive(pos):
    """SDK gripper position (850 open ... 0 closed) -> drive_joint angle (0 open ... 0.85 rad closed)."""
    frac = 1.0 - float(pos) / GRIPPER_POS_OPEN
    return float(np.clip(frac, 0.0, 1.0) * GRIPPER_DRIVE_MAX)


def _mesh(file):
    return {"type": "mesh", "file": file, "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]}


def arm_bodies(arm, base_xyz=(0.0, 0.0, 0.0), base_yaw=0.0, ee="xarm_gripper"):
    """Body list for one arm. Names are prefixed with `<arm>_` so two arms can share one stage."""
    p = f"{arm}_"
    bodies = [{
        "name": p + "link_base", "parent": None, "pos": [float(v) for v in base_xyz],
        "quat": rpy_to_quat((0.0, 0.0, base_yaw)), "joint": None,
        "geoms": [_mesh("xarm6/link_base.stl")], "material": "xarm_white",
    }]
    parent = p + "link_base"
    for i, ((xyz, rpy), rng) in enumerate(zip(XARM6_JOINTS, XARM6_LIMITS), start=1):
        name = p + f"link{i}"
        bodies.append({
            "name": name, "parent": parent, "pos": list(xyz), "quat": rpy_to_quat(rpy),
            "joint": {"name": p + f"joint{i}", "type": "hinge", "axis": [0.0, 0.0, 1.0], "range": list(rng), "mimic": None},
            "geoms": [_mesh(f"xarm6/link{i}.stl")], "material": "xarm_white",
        })
        parent = name
    if ee == "xarm_gripper":
        bodies.append({
            "name": p + "gripper_base", "parent": parent, "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
            "joint": None, "geoms": [_mesh("xarm_gripper/base_link.stl")], "material": "gripper_dark",
        })
        drive = p + "drive_joint"
        for link, par, xyz, axis in GRIPPER_LINKS:
            follows = None if link == "left_outer_knuckle" else {"joint": drive, "multiplier": 1.0}
            bodies.append({
                "name": p + link, "parent": p + par, "pos": list(xyz), "quat": [1.0, 0.0, 0.0, 0.0],
                "joint": {"name": drive if follows is None else p + link + "_joint", "type": "hinge",
                          "axis": [float(a) for a in axis], "range": [0.0, GRIPPER_DRIVE_MAX], "mimic": follows},
                "geoms": [_mesh(f"xarm_gripper/{link}.stl")], "material": "gripper_dark",
            })
    elif ee == "leaphand":
        bodies.append({
            "name": p + "leaphand", "parent": parent, "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0],
            "joint": None, "material": "leaphand_black",
            "geoms": [{"type": "box", "name": n, "pos": list(c), "size": list(s), "quat": [1.0, 0.0, 0.0, 0.0]}
                      for n, c, s in LEAPHAND_BOXES],
        })
    elif ee not in (None, "none"):
        raise ValueError(f"unknown end effector {ee!r} (xarm_gripper | leaphand | none)")
    for b in bodies:
        b["arm"] = arm
    return bodies


def tcp_site(arm, ee):
    """Frame the controller reports (tcp_offset 0 = flange) plus the gripper tip for the trail."""
    p = f"{arm}_"
    sites = {"flange": {"body": p + "link6", "pos": [0.0, 0.0, 0.0]}}
    if ee == "xarm_gripper":
        sites["tcp"] = {"body": p + "gripper_base", "pos": [0.0, 0.0, GRIPPER_TCP_Z]}
    elif ee == "leaphand":
        sites["tcp"] = {"body": p + "leaphand", "pos": [0.0, 0.0, 0.22]}
    else:
        sites["tcp"] = sites["flange"]
    return {f"{arm}/{k}": v for k, v in sites.items()}


class Chain:
    """Forward kinematics over the body list (numpy only; used by omni_twin, the tests and the overlay renders)."""

    def __init__(self, spec):
        self.spec = spec
        self.bodies = spec["bodies"]
        self.sites = spec["sites"]
        self.arms = spec["arms"]
        self.joints = {b["joint"]["name"]: b for b in self.bodies if b["joint"]}

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def joint_values(self, arm_state):
        """{arm: (q[6] rad, gripper drive rad)} -> {joint name: value}, mimic joints included."""
        vals = {}
        for arm, (q, drive) in arm_state.items():
            for name, v in zip(ARM_JOINTS, q):
                vals[f"{arm}_{name}"] = float(v)
            vals[f"{arm}_drive_joint"] = float(drive)
        for name, b in self.joints.items():
            m = b["joint"]["mimic"]
            if m is not None:
                vals[name] = vals.get(m["joint"], 0.0) * float(m["multiplier"])
        return vals

    def forward(self, joint_values):
        """{body name: (R 3x3, p 3)} in the world frame. Missing joint values count as 0."""
        out = {}
        for b in self.bodies:
            R0, p0 = out[b["parent"]] if b["parent"] else (np.eye(3), np.zeros(3))
            R, p = R0 @ quat_to_mat(b["quat"]), R0 @ np.asarray(b["pos"], dtype=float) + p0
            j = b["joint"]
            if j:
                R = R @ quat_to_mat(axis_angle_quat(j["axis"], joint_values.get(j["name"], 0.0)))
            out[b["name"]] = (R, p)
        return out

    def site(self, name, joint_values, frames=None):
        s = self.sites[name]
        R, p = (frames or self.forward(joint_values))[s["body"]]
        return R @ np.asarray(s["pos"], dtype=float) + p

    def flange_in_base_mm(self, arm, joint_values):
        """Flange position in the arm's own base frame [mm] — the controller's `position` when tcp_offset is 0."""
        frames = self.forward(joint_values)
        Rb, pb = frames[f"{arm}_link_base"]
        return Rb.T @ (self.site(f"{arm}/flange", joint_values, frames) - pb) * 1000.0

    def limit_violations(self, arm, q, tol_deg=0.5):
        out = []
        for i, name in enumerate(ARM_JOINTS):
            lo, hi = np.degrees(self.joints[f"{arm}_{name}"]["joint"]["range"])
            v = math.degrees(q[i])
            if v < lo - tol_deg or v > hi + tol_deg:
                out.append((i + 1, round(v, 1), round(lo, 1), round(hi, 1)))
        return out


def build_spec(config):
    """Chain spec (the .twin.json) for every arm in twin_config.json."""
    bodies, sites, arms = [], {}, {}
    for arm, cfg in config["arms"].items():
        ee = cfg.get("ee", "xarm_gripper")
        bodies += arm_bodies(arm, cfg.get("base_xyz", (0, 0, 0)), math.radians(cfg.get("base_yaw_deg", 0.0)), ee)
        sites.update(tcp_site(arm, ee))
        arms[arm] = {"ee": ee, "ip": cfg.get("ip"), "home_joints": cfg.get("home_joints")}
    return {"version": 1, "root": "/World/robots", "bodies": bodies, "sites": sites, "arms": arms}
