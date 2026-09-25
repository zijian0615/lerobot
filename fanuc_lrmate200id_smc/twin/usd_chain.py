"""Kinematic chain of the USD twin model (numpy only: no mujoco, no pxr), loaded from the JSON that build_usd.py writes.

The chain mirrors the MuJoCo body tree, so joint values use the same conventions as joint_map.py (model space, rad / m).
"""
import json

import numpy as np

from joint_map import BASE_HEIGHT_M

ARM_JOINTS = tuple(f"joint_{i}" for i in range(1, 7))
FINGER_JOINTS = ("finger_l", "finger_r")


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def axis_angle_quat(axis, angle):
    a = np.asarray(axis, dtype=float)
    return np.concatenate([[np.cos(angle / 2)], a / np.linalg.norm(a) * np.sin(angle / 2)])


class Chain:
    def __init__(self, spec):
        self.root = spec["root"]
        self.bodies = spec["bodies"]              # parents come before children
        self.sites = spec["sites"]
        self.joints = {b["joint"]["name"]: b for b in self.bodies if b["joint"]}

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def joint_values(self, q_arm, gripper_m):
        """{joint name: value} for J1..J6 (rad, model space) and both fingers (m per side)."""
        vals = dict(zip(ARM_JOINTS, (float(v) for v in q_arm[:6])))
        vals.update({n: float(gripper_m) for n in FINGER_JOINTS})
        return vals

    def limit_violations(self, q_arm, tol_deg=0.5):
        """[(joint number, angle_deg, lo_deg, hi_deg)], same format as joint_map.limit_violations."""
        out = []
        for i, name in enumerate(ARM_JOINTS):
            lo, hi = np.degrees(self.joints[name]["joint"]["range"])
            v = np.degrees(q_arm[i])
            if v < lo - tol_deg or v > hi + tol_deg:
                out.append((i + 1, float(v), float(lo), float(hi)))
        return out

    def forward(self, joint_values):
        """{body name: (R 3x3, p 3)} in the stage root frame."""
        out = {}
        for b in self.bodies:
            R0, p0 = out[b["parent"]] if b["parent"] else (np.eye(3), np.zeros(3))
            R, p = R0 @ quat_to_mat(b["quat"]), R0 @ np.asarray(b["pos"]) + p0
            j = b["joint"]
            if j:
                v = joint_values[j["name"]]
                if j["type"] == "hinge":
                    R = R @ quat_to_mat(axis_angle_quat(j["axis"], v))
                else:
                    p = p + R @ (np.asarray(j["axis"]) * v)
            out[b["name"]] = (R, p)
        return out

    def site_frame(self, name, joint_values):
        """Site rotation and position in the stage root frame. The site has no extra rotation."""
        s = self.sites[name]
        rotation, origin = self.forward(joint_values)[s["body"]]
        return rotation, rotation @ np.asarray(s["pos"], dtype=float) + origin

    def site(self, name, joint_values):
        """Site position in the stage root frame [m]."""
        return self.site_frame(name, joint_values)[1]

    def arm_limits(self):
        """Lower and upper bounds of J1..J6 in model radians."""
        bounds = [self.joints[name]["joint"]["range"] for name in ARM_JOINTS]
        return np.array([row[0] for row in bounds]), np.array([row[1] for row in bounds])

    def flange_world_mm(self, joint_values):
        """Same definition as joint_map.flange_world_mm (FANUC world origin = J1/J2 axes intersection)."""
        return (self.site("flange", joint_values) - np.array([0.0, 0.0, BASE_HEIGHT_M])) * 1000.0
