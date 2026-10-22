"""Render the three training cameras with the arm at given sim joint angles and the single nominal screw.

    OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \\
      ~/isaacsim-venv/bin/python render_pose.py --pose data/vis_check/pose.json --out data/vis_check

--pose may also be a list of {"tag", "sim_joints_deg"}; each is written to <out>/sim_<tag>/.
"""
import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--pose", required=True, help="json with sim_joints_deg")
parser.add_argument("--out", required=True)
parser.add_argument("--screw", type=int, default=0)
parser.add_argument("--gripper", type=float, default=0.0, help="1 = closed")
parser.add_argument("--mat", default=None, help="mat texture, relative to fanuc_er4ia_sim (default scene_config)")
parser.add_argument("--screw-pose", type=float, nargs=3, default=None, metavar=("X", "Y", "YAW"),
                    help="the screw's head xy (m) and head->tip yaw (deg) instead of SCREWS[--screw]")
args = parser.parse_args()
args.pose, args.out = os.path.abspath(args.pose), os.path.abspath(args.out)

from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": True})

import cv2
import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell
import scene_config as C

C.SCREWS = [tuple(args.screw_pose) if args.screw_pose else tuple(C.SCREWS[args.screw])]
if args.mat:
    C.MAT_TEXTURE = args.mat
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage)
cams = {name: Camera(prim_path=path, resolution=cfg["resolution"])
        for name, path, cfg in (("table", cell.TABLE_CAM_PATH, C.TABLE_CAM),
                                ("side", cell.SIDE_CAM_PATH, C.SIDE_CAM),
                                ("wrist", cell.WRIST_CAM_PATH, C.WRIST_CAM))}
cell.enable_ccd(world)
world.reset()
for cam in cams.values():
    cam.initialize()
    cam.add_distance_to_image_plane_to_frame()
arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
arm.initialize()
names = list(arm.dof_names)
spec = json.load(open(args.pose))
poses = [(s["tag"], s["sim_joints_deg"]) for s in spec] if isinstance(spec, list) else [(None, spec["sim_joints_deg"])]
for k, (tag, joints) in enumerate(poses):
    q = arm.get_joint_positions()
    for i, deg in enumerate(joints):
        q[names.index(f"J{i + 1}")] = np.deg2rad(deg)
    for jn, v in cell.jaw_targets(C.JAW_CLOSED if args.gripper >= 0.5 else C.JAW_OPEN).items():
        q[names.index(jn)] = v
    arm.set_joint_positions(q)
    for _ in range(90 if k == 0 else 20):  # materials compile on the first frames
        arm.apply_action(ArticulationAction(joint_positions=q))
        world.step(render=True)
    out = args.out if tag is None else os.path.join(args.out, f"sim_{tag}")
    os.makedirs(out, exist_ok=True)
    for name, cam in cams.items():
        rgb = cam.get_rgba()[:, :, :3].astype(np.uint8)
        cv2.imwrite(os.path.join(out, f"sim_{name}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        # Pixels with no geometry behind them (the sky dome) come back as inf / very far.
        np.save(os.path.join(out, f"sim_{name}_depth.npy"), np.asarray(cam.get_depth(), np.float32))
        print(f"[render] {tag or ''} sim_{name}.png {rgb.shape}", flush=True)
sim_app.close()
