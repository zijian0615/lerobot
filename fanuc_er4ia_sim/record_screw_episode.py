"""Record one scripted screw pick in this cell, with all three real cameras.

Writes PNGs and frames.json. Pack them with pack_lerobot_episode.py.
Does not move the real robot.

    OMNI_KIT_ACCEPT_EULA=YES \\
      LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \\
      ~/isaacsim-venv/bin/python record_screw_episode.py --out data/screw_ep0
"""
import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="data/screw_ep0")
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--screw", type=int, default=0, help="which screw this episode picks")
parser.add_argument("--screw-xy", nargs=2, type=float, metavar=("X", "Y"),
                    help="place the picked screw at this table XY (metres, robot frame)")
parser.add_argument("--screw-yaw", type=float, default=None,
                    help="head-to-tip yaw of the picked screw, degrees")
parser.add_argument("--random-layout", action="store_true",
                    help="scatter every screw on the mat before the episode")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
args.out = os.path.abspath(args.out)

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import cv2
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
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

# cell.build reads C.SCREWS, so the layout has to be replaced before that.
rng = np.random.default_rng(args.seed)
layout = [list(p) for p in C.SCREWS]
if args.random_layout:
    # Stay on the printed target, clear of the bin footprint.
    bx, by = C.BIN_XY
    half = max(C.BIN_SIZE) / 2 + 0.04
    placed = []
    for _ in range(len(layout)):
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
            raise RuntimeError("could not place a screw clear of the bin and the others")
    layout = placed
if args.screw_xy is not None or args.screw_yaw is not None:
    x, y, yaw = layout[args.screw]
    if args.screw_xy is not None:
        x, y = args.screw_xy
    if args.screw_yaw is not None:
        yaw = args.screw_yaw
    layout[args.screw] = (x, y, yaw)
C.SCREWS = [tuple(p) for p in layout]

sx, sy, yaw = C.SCREWS[args.screw]
yaw_r = np.radians(yaw)
plan = {
    "bin_xy": list(C.BIN_XY),
    "bin_release_z": C.BIN_RELEASE_Z,
    "picks": [dict(
        label="screw",
        xy=[sx + np.cos(yaw_r) * C.SCREW_LENGTH / 2,
            sy + np.sin(yaw_r) * C.SCREW_LENGTH / 2],
        spread_dir=[-np.sin(yaw_r), np.cos(yaw_r), 0.0],
        jaw_half=C.SCREW_SHANK_D / 2,
    )],
}

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage)

CAMERAS = (
    ("table", cell.TABLE_CAM_PATH, C.TABLE_CAM),
    ("wrist", cell.WRIST_CAM_PATH, C.WRIST_CAM),
    ("side", cell.SIDE_CAM_PATH, C.SIDE_CAM),
)
cams = []
for name, path, cfg in CAMERAS:
    cam = Camera(prim_path=path, resolution=cfg["resolution"])
    cams.append((name, cam, cfg["resolution"]))

cell.enable_ccd(world)
world.reset()
for _, cam, _ in cams:
    cam.initialize()
for _ in range(60):
    world.step(render=True)

arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
arm.initialize()
names = list(arm.dof_names)
ARM_IDX = [names.index(f"J{i}") for i in range(1, 7)]
IDX = {n: names.index(n) for n in names}
LIM = np.deg2rad(np.array([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)]))

TOOL_OFF = tuple(np.array(C.TOOL_AXIS) * cell.tip_offset())
DOWN = (0.0, 0.0, -1.0)
OPEN = C.JAW_OPEN
HOVER = 0.13
PICK_Z = C.SURFACE_DZ + C.TIP_CLEARANCE
HOME = np.array([0.30, 0.0, 0.24])


def command(q6, jaw):
    full = arm.get_joint_positions().copy()
    for k, i in enumerate(ARM_IDX):
        full[i] = q6[k]
    for jn, v in cell.jaw_targets(jaw).items():
        full[IDX[jn]] = v
    arm.apply_action(ArticulationAction(joint_positions=full))


def ease(t):
    return t * t * (3 - 2 * t)


def slerp_about_z(a, b, t):
    ang = np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])
    c, s_ = np.cos(ang * t), np.sin(ang * t)
    return [a[0] * c - a[1] * s_, a[0] * s_ + a[1] * c, 0.0]


p = plan["picks"][0]
pick = np.array([p["xy"][0], p["xy"][1], PICK_Z])
spread = list(p["spread_dir"])
bin_xy = np.array(plan["bin_xy"], float)
side = np.cross(DOWN, np.array(spread, float))
light = np.array(p["xy"]) + side[:2] * C.WRISTCAM_LATERAL
light_flip = np.array(p["xy"]) - side[:2] * C.WRISTCAM_LATERAL
if np.linalg.norm(light_flip - bin_xy) > np.linalg.norm(light - bin_xy):
    spread = [-v for v in spread]
drop = np.array([bin_xy[0], bin_xy[1], plan["bin_release_z"]])
up_p = pick + [0, 0, HOVER]
up_b = drop + [0, 0, 0.06]
shut = C.JAW_CLOSED
segments = [
    ("approach", HOME, up_p, 1.1, OPEN, OPEN, spread),
    ("descend", up_p, pick, 0.7, OPEN, OPEN, spread),
    ("close", pick, pick, 0.5, OPEN, shut, spread),
    ("grip", pick, pick, 0.3, shut, shut, spread),
    ("lift", pick, up_p, 0.6, shut, shut, spread),
    ("carry", up_p, up_b, 1.2, shut, shut, spread),
    ("lower", up_b, drop, 0.5, shut, shut, spread),
    ("release", drop, drop, 0.4, shut, OPEN, spread),
    ("retreat", drop, up_b, 0.5, OPEN, OPEN, None),
]

for name, _, _ in cams:
    os.makedirs(os.path.join(args.out, name), exist_ok=True)

q = np.array([np.deg2rad(C.HOME_POSE_DEG[f"J{i}"]) for i in range(1, 7)])
q, ok = ik.solve(q, HOME, DOWN, LIM, tool_offset=TOOL_OFF)
print(f"[record] home IK ok={ok}", flush=True)
for _ in range(80):
    command(q, OPEN)
    world.step(render=False)

frames = []
frame = 0
for name, p0, p1, dur, g0, g1, spread_dir in segments:
    n = max(1, int(round(dur * args.fps)))
    spread0 = None
    if spread_dir is not None and name == "approach":
        R0, _ = fk.tool_pose(q, TOOL_OFF)
        s0 = R0 @ np.array(C.FINGER_SPREAD_AXIS, float)
        s0[2] = 0.0
        if np.linalg.norm(s0) > 1e-6:
            spread0 = list(s0 / np.linalg.norm(s0))
    for i in range(n):
        t = ease((i + 1) / n)
        target = np.array(p0) + (np.array(p1) - np.array(p0)) * t
        sp = spread_dir if spread0 is None else slerp_about_z(spread0, spread_dir, t)
        q, ok = ik.solve(q, target, DOWN, LIM, tool_offset=TOOL_OFF, spread_dir=sp)
        jaw = g0 + (g1 - g0) * t
        command(q, jaw)
        world.step(render=True)
        jp = arm.get_joint_positions()
        joints = [float(np.degrees(jp[i])) for i in ARM_IDX]
        gap = 2 * C.JAW_OPEN - abs(float(jp[IDX["jaw_l"]])) - abs(float(jp[IDX["jaw_r"]]))
        gripper = 1.0 if jaw < (OPEN + shut) / 2 else 0.0
        images = {}
        for cam_name, cam, _res in cams:
            rgb = cam.get_rgba()[:, :, :3]
            path = os.path.join(args.out, cam_name, f"{frame:06d}.png")
            cv2.imwrite(path, cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
            images[cam_name] = [cam_name, f"{frame:06d}.png"]
        frames.append({
            "segment": name,
            "joints_deg": joints,
            "gripper": gripper,
            "jaw_gap_m": gap,
            "ik_ok": bool(ok),
            "images": {k: v[1] for k, v in (
                ("table", images["table"]),
                ("wrist", images["wrist"]),
                ("side", images["side"]),
            )},
        })
        frame += 1
        if frame % 10 == 0:
            print(f"[record] {name} frame {frame}", flush=True)

meta = {
    "fps": args.fps,
    "task": "pick one countersunk screw and drop it in the yellow bin",
    "screw_index": args.screw,
    "screw_xy_yaw": [sx, sy, yaw],
    "screw_layout": [list(p) for p in C.SCREWS],
    "cameras": {
        "table": {"role": "overhead", "resolution": list(C.TABLE_CAM["resolution"])},
        "wrist": {"role": "wrist", "resolution": list(C.WRIST_CAM["resolution"])},
        "side": {"role": "fixed side", "resolution": list(C.SIDE_CAM["resolution"])},
    },
    "frames": frames,
}
with open(os.path.join(args.out, "frames.json"), "w") as f:
    json.dump(meta, f)
print(f"[record] {frame} frames -> {args.out}", flush=True)
sim_app.close()
