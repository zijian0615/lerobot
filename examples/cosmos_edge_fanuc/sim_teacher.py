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

"""Scripted FANUC pick-and-place trajectories in the MuJoCo twin.

Task (task_scene.json): put the black cube in, or lay the pen on, the blue or orange bin. Each episode scatters the
four scanned objects (scanner/objects/<name>/<name>.usda) on the table, picks one item and one bin at random, and
solves a top-down grasp: the fingers close across the item (across the pen's long axis), carry it above the bin, and
release it (the cube drops in onto the floor, the pen is set down across the rim). Differential IK keeps the tool
pointing down with a set yaw. The item is parented to the TCP while the gripper is closed; no contact physics.
Frames hold pendant joints, the binary gripper, the finger opening and every object's pose (asset origin + yaw).
This does not talk to the robot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

try:
    import mujoco
except ImportError:  # constants are imported by the dataset exporter, which has no mujoco
    mujoco = None  # type: ignore[assignment]
import numpy as np

_TWIN = Path(__file__).resolve().parents[2] / "fanuc_lrmate200id_smc"
sys.path.insert(0, str(_TWIN / "twin"))
from joint_map import FINGER_OPEN_M, model_to_fanuc  # noqa: E402

SCENE_XML = _TWIN / "scene.xml"
HERE = Path(__file__).resolve().parent
TASK_PATH = HERE / "task_scene.json"
OBJECTS_DIR = HERE / "scanner" / "objects"
FPS = 30
SPEED_M_S = 0.16
TABLE = (-0.45, 0.45, -0.35, 0.35)
TCP_TOL_M = 0.008
HOLD_FRAMES = 5
GRIP_FRAMES = 6
APPROACH_ABOVE_M = 0.08   # approach / lift height above the grasp point
CARRY_Z_M = 0.14          # TCP height while moving between pick and bin (clears the ~6 cm bins with the pen)
DROP_ABOVE_RIM_M = 0.035  # cube: TCP this far above the rim when the fingers open
LAY_ABOVE_RIM_M = 0.004   # pen: its underside this far above the rim when the fingers open
CLOSED_GAP_M = 0.002      # pad gap at zero finger travel
GRAVITY = 9.81


class IkError(RuntimeError):
    pass


def load_task(path: Path = TASK_PATH) -> dict:
    task = json.loads(path.read_text(encoding="utf-8"))
    assets = {}
    for name in [*task["items"], *task["bins"]]:
        info = json.loads((OBJECTS_DIR / name / f"{name}.json").read_text(encoding="utf-8"))
        info["usd"] = str(OBJECTS_DIR / name / f"{name}.usda")
        assets[name] = info
    task["assets"] = assets
    return task


def _item_geometry(info: dict) -> tuple[float, float]:
    """Grasp height above the table and the width the fingers close on."""
    c = info["collider"]
    if c["type"] == "capsule":
        return float(c["radius"]), 2 * float(c["radius"])
    return float(c["size"][2]) / 2, float(min(c["size"][0], c["size"][1]))


def _bin_geometry(info: dict) -> tuple[float, float]:
    """Rim and inner floor height of a container."""
    c = info["collider"]
    return float(c["outer"][2]), float(c["floor"])


def _load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(SCENE_XML))


def _ready_q(model: mujoco.MjModel) -> np.ndarray:
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "ready")
    mujoco.mj_resetDataKeyframe(model, data, key)
    return np.array(data.qpos[:6], dtype=float)


def _tcp_frame(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data.qpos[:6] = q
    mujoco.mj_forward(model, data)
    return data.site("tcp").xpos.copy(), data.site("tcp").xmat.reshape(3, 3).copy()


def ik_to(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, target: np.ndarray,
          tool_x: np.ndarray | None = None) -> np.ndarray:
    """Damped least squares onto a TCP position. Tool +Z stays world -Z; tool +X follows `tool_x` when given."""
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    lo, hi = model.jnt_range[:6, 0], model.jnt_range[:6, 1]
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    desired_z = np.array([0.0, 0.0, -1.0])
    q = np.array(q, dtype=float)
    pos = target
    for _ in range(120):
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)
        pos = data.site_xpos[site].copy()
        rot = data.site_xmat[site].reshape(3, 3)
        rot_err = np.cross(rot[:, 2], desired_z)
        if tool_x is not None:
            rot_err = rot_err + np.cross(rot[:, 0], tool_x)
        mujoco.mj_jacSite(model, data, jacp, jacr, site)
        err = np.concatenate([target - pos, rot_err])
        yaw_ok = tool_x is None or float(rot[:, 0] @ tool_x) > 0.999
        if np.linalg.norm(err[:3]) < 0.001 and float(rot[:, 2] @ desired_z) > 0.998 and yaw_ok:
            return q
        jac = np.vstack([jacp[:, :6], jacr[:, :6]])
        step = jac.T @ np.linalg.solve(jac @ jac.T + (0.05**2) * np.eye(6), err)
        q = np.clip(q + 0.6 * step, lo, hi)
    if np.linalg.norm(target - pos) > TCP_TOL_M:
        raise IkError(f"TCP missed {target.tolist()} by {np.linalg.norm(target - pos):.4f} m")
    return q


def _grasp_axis(current_x: np.ndarray, yaw: float, symmetry: int) -> np.ndarray:
    """Horizontal tool +X for an item at `yaw`: along the item's x axis modulo its symmetry, nearest the current one."""
    best = None
    for k in range(symmetry):
        a = yaw + 2 * math.pi * k / symmetry
        d = np.array([math.cos(a), math.sin(a), 0.0])
        if best is None or d @ current_x > best @ current_x:
            best = d
    return best


def sample_layout(task: dict, rng: np.random.Generator, tries: int = 2000) -> dict:
    """Non-overlapping positions and yaws for all objects inside the workspace."""
    ws, radii = task["workspace"], task["footprint_radius_m"]
    names = [*task["bins"], *task["items"]]  # big ones first
    placed: dict[str, dict] = {}
    for name in names:
        for _ in range(tries):
            x, y = rng.uniform(*ws["x"]), rng.uniform(*ws["y"])
            r = math.hypot(x, y)
            if not ws["radius"][0] <= r <= ws["radius"][1]:
                continue
            if all(math.hypot(x - p["xy"][0], y - p["xy"][1]) >= radii[name] + radii[o] + task["clearance_m"]
                   for o, p in placed.items()):
                placed[name] = {"xy": [x, y], "yaw": float(rng.uniform(-math.pi, math.pi))}
                break
        else:
            raise IkError(f"no room for {name}")
    return placed


class _Episode:
    def __init__(self, model, task, layout, item, target):
        self.model, self.data = model, mujoco.MjData(model)
        self.task, self.layout, self.item, self.target = task, layout, item, target
        self.frames: list[dict] = []
        self.poses = {n: {"pos": [p["xy"][0], p["xy"][1], 0.0], "yaw": p["yaw"]} for n, p in layout.items()}
        self.grasp_h, self.width = _item_geometry(task["assets"][item])
        self.gripper, self.finger = 0.0, FINGER_OPEN_M
        self.attached: tuple[np.ndarray, float] | None = None  # item offset from the TCP and yaw offset from the tool

    def _frame(self, q):
        tcp, rot = _tcp_frame(self.model, self.data, q)
        if self.attached is not None:
            offset, dyaw = self.attached
            tool_yaw = math.atan2(rot[1, 0], rot[0, 0])
            self.poses[self.item] = {"pos": (tcp + offset).tolist(), "yaw": tool_yaw + dyaw}
        self.frames.append({
            "joints_deg": [float(v) for v in model_to_fanuc(q, "coupled")],
            "gripper": float(self.gripper),
            "finger_m": float(self.finger),
            "objects": [{"name": n, "usd": self.task["assets"][n]["usd"], "pos": [float(v) for v in p["pos"]],
                         "yaw": float(p["yaw"])} for n, p in self.poses.items()],
        })

    def hold(self, q, n=HOLD_FRAMES):
        for _ in range(n):
            self._frame(q)
        return q

    def move(self, q, target, tool_x):
        start, _ = _tcp_frame(self.model, self.data, q)
        steps = max(1, int(np.ceil(np.linalg.norm(target - start) / SPEED_M_S * FPS)))
        for i in range(1, steps + 1):
            q = ik_to(self.model, self.data, q, start + (target - start) * (i / steps), tool_x)
            self._frame(q)
        return q

    def grip(self, q, close: bool):
        closed = max(0.0, (self.width - CLOSED_GAP_M) / 2)
        a, b = (FINGER_OPEN_M, closed) if close else (closed, FINGER_OPEN_M)
        for i in range(1, GRIP_FRAMES + 1):
            s = i / GRIP_FRAMES
            self.gripper = s if close else 1.0 - s
            self.finger = a + (b - a) * s
            if close and i == GRIP_FRAMES // 2:  # fingers meet the item
                tcp, rot = _tcp_frame(self.model, self.data, q)
                pose = self.poses[self.item]
                self.attached = (np.asarray(pose["pos"]) - tcp, pose["yaw"] - math.atan2(rot[1, 0], rot[0, 0]))
            if not close and i == 1:
                self.attached = None
                self._settle_start = len(self.frames)
            self._frame(q)
        return q

    def settle(self, rest_z: float):
        """After release: the item falls onto rest_z (bin floor or rim), in the frames recorded after the opening."""
        pose = self.poses[self.item]
        z0 = pose["pos"][2]
        for k, frame in enumerate(self.frames[self._settle_start:]):
            z = max(rest_z, z0 - 0.5 * GRAVITY * (k / FPS) ** 2)
            obj = next(o for o in frame["objects"] if o["name"] == self.item)
            obj["pos"][2] = z
        pose["pos"][2] = rest_z

    def run(self) -> dict:
        model = self.model
        q = _ready_q(model)
        home, rot = _tcp_frame(model, self.data, q)
        item_pose, bin_pose = self.layout[self.item], self.layout[self.target]
        symmetry = 2 if self.task["assets"][self.item]["collider"]["type"] == "capsule" else 4
        tool_x = _grasp_axis(rot[:, 0], item_pose["yaw"], symmetry)
        ix, iy = item_pose["xy"]
        bx, by = bin_pose["xy"]
        rim, floor = _bin_geometry(self.task["assets"][self.target])
        lay = self.task["items"][self.item]["place"] == "lay"

        q = self.hold(q)
        q = self.move(q, np.array([ix, iy, self.grasp_h + APPROACH_ABOVE_M]), tool_x)
        q = self.move(q, np.array([ix, iy, self.grasp_h]), tool_x)
        q = self.hold(q)
        q = self.grip(q, close=True)
        q = self.move(q, np.array([ix, iy, CARRY_Z_M]), tool_x)
        q = self.move(q, np.array([bx, by, CARRY_Z_M]), tool_x)
        release_z = rim + self.grasp_h + LAY_ABOVE_RIM_M if lay else rim + DROP_ABOVE_RIM_M
        q = self.move(q, np.array([bx, by, release_z]), tool_x)
        q = self.hold(q)
        q = self.grip(q, close=False)
        q = self.hold(q, HOLD_FRAMES + 4)
        self.settle(rim if lay else floor)
        q = self.move(q, np.array([bx, by, CARRY_Z_M]), None)
        q = self.move(q, home, None)
        self.hold(q)
        phrase = self.task["items"][self.item]["verb"].format(item=self.task["items"][self.item]["phrase"],
                                                              bin=self.task["bins"][self.target]["phrase"])
        return {"task": phrase, "fps": FPS, "table": list(TABLE), "item": self.item, "bin": self.target,
                "layout": self.layout, "frames": self.frames}


def solve_episode(model: mujoco.MjModel, task: dict, rng: np.random.Generator) -> dict:
    layout = sample_layout(task, rng)
    item = str(rng.choice(sorted(task["items"])))
    target = str(rng.choice(sorted(task["bins"])))
    return _Episode(model, task, layout, item, target).run()


def generate_episodes(n: int, seed: int, max_attempts: int | None = None) -> list[dict]:
    model = _load_model()
    task = load_task()
    rng = np.random.default_rng(seed)
    episodes = []
    attempts = 0
    limit = max_attempts or n * 8
    while len(episodes) < n and attempts < limit:
        attempts += 1
        try:
            episodes.append(solve_episode(model, task, rng))
        except IkError:
            continue
    if len(episodes) < n:
        raise RuntimeError(f"only solved {len(episodes)} / {n} episodes in {attempts} attempts")
    return episodes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write scripted FANUC sim trajectories (no images)")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    episodes = generate_episodes(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(episodes))
    lengths = [len(ep["frames"]) for ep in episodes]
    print(f"wrote {len(episodes)} episodes, frames {min(lengths)}..{max(lengths)}, total {sum(lengths)}")
    for ep in episodes:
        print(f"  {ep['task']}: {len(ep['frames'])} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
