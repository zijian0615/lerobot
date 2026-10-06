"""Roll the ACT policy in this cell and report whether a screw ends in the bin.

    OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=... python eval_act_sim.py --seed 20
"""
import argparse
import os
import pickle
import socket
import sys

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--steps", type=int, default=80)
args = parser.parse_args()

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import cv2
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
import omni.usd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell
import fk
import ik
import scene_config as C

ik.set_tool_axis(C.TOOL_AXIS)
ik.set_spread_axis(C.FINGER_SPREAD_AXIS)


def scatter(seed: int):
    rng = np.random.default_rng(seed)
    bx, by = C.BIN_XY
    half = max(C.BIN_SIZE) / 2 + 0.04
    placed = []
    for _ in range(len(C.SCREWS)):
        for _try in range(200):
            x = float(rng.uniform(0.32, 0.50))
            y = float(rng.uniform(-0.20, 0.08))
            if abs(x - bx) < half and abs(y - by) < half:
                continue
            if any((x - px) ** 2 + (y - py) ** 2 < 0.035 ** 2 for px, py, _ in placed):
                continue
            placed.append((x, y, float(rng.uniform(-180.0, 180.0))))
            break
        else:
            raise RuntimeError("could not place screws")
    return placed


C.SCREWS = scatter(args.seed)

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage)
cams = []
for name, path, cfg in (
    ("table", cell.TABLE_CAM_PATH, C.TABLE_CAM),
    ("wrist", cell.WRIST_CAM_PATH, C.WRIST_CAM),
    ("side", cell.SIDE_CAM_PATH, C.SIDE_CAM),
):
    cam = Camera(prim_path=path, resolution=cfg["resolution"])
    cams.append((name, cam))
cell.enable_ccd(world)
world.reset()
for _, cam in cams:
    cam.initialize()
for _ in range(40):
    world.step(render=True)

arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
arm.initialize()
names = list(arm.dof_names)
ARM_IDX = [names.index(f"J{i}") for i in range(1, 7)]
IDX = {n: names.index(n) for n in names}
LIM = np.deg2rad(np.array([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)]))
TOOL_OFF = tuple(np.array(C.TOOL_AXIS) * cell.tip_offset())
HOME = np.array([0.30, 0.0, 0.24])
q = np.array([np.deg2rad(C.HOME_POSE_DEG[f"J{i}"]) for i in range(1, 7)])
q, _ = ik.solve(q, HOME, (0.0, 0.0, -1.0), LIM, tool_offset=TOOL_OFF)


def command(q6, jaw):
    full = arm.get_joint_positions().copy()
    for k, i in enumerate(ARM_IDX):
        full[i] = q6[k]
    for jn, v in cell.jaw_targets(jaw).items():
        full[IDX[jn]] = v
    arm.apply_action(ArticulationAction(joint_positions=full))


for _ in range(60):
    command(q, C.JAW_OPEN)
    world.step(render=False)

sock = socket.create_connection(("127.0.0.1", args.port))


def rpc(msg):
    payload = pickle.dumps(msg)
    sock.sendall(len(payload).to_bytes(8, "big") + payload)
    header = sock.recv(8)
    n = int.from_bytes(header, "big")
    buf = b""
    while len(buf) < n:
        buf += sock.recv(n - len(buf))
    return pickle.loads(buf)


rpc({"op": "reset"})
dt = world.get_physics_dt()
sub = max(1, int(round(0.1 / dt)))
gripper = 0.0
for step in range(args.steps):
    images = {}
    for name, cam in cams:
        rgb = cam.get_rgba()[:, :, :3].astype(np.uint8)
        images[name] = np.asarray(Image.fromarray(rgb).resize((224, 224), Image.BILINEAR))
    jp = arm.get_joint_positions()
    state = np.asarray([float(np.degrees(jp[i])) for i in ARM_IDX] + [gripper], np.float32)
    reply = rpc({
        "op": "act",
        "images": images,
        "state": state,
        "task": "pick one countersunk screw and drop it in the yellow bin",
    })
    action = np.asarray(reply["action"], np.float32).reshape(-1)
    q = np.deg2rad(action[:6])
    gripper = 1.0 if action[6] >= 0.5 else 0.0
    jaw = C.JAW_CLOSED if gripper else C.JAW_OPEN
    command(q, jaw)
    for _ in range(sub):
        world.step(render=True)

bx, by = C.BIN_XY
bw, bd = C.BIN_SIZE
in_bin = []
for i in range(len(C.SCREWS)):
    pos = SingleXFormPrim(f"/World/Screw{i}").get_world_pose()[0]
    inside = abs(pos[0] - bx) < bw / 2 + 0.02 and abs(pos[1] - by) < bd / 2 + 0.02
    in_bin.append(bool(inside))
    print(f"[eval] seed {args.seed} Screw{i} xy=({pos[0]:+.3f},{pos[1]:+.3f}) "
          f"{'IN BIN' if inside else 'out'}", flush=True)
print(f"[eval] seed {args.seed} success {int(any(in_bin))} count {sum(in_bin)}", flush=True)
sock.close()
sim_app.close()
