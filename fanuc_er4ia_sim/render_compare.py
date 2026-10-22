"""Render the sim cell at the joint angles of calib_capture.py samples and set it beside the real frames.

For each sample: real | sim | 50/50 blend, one row per camera, written to <dir>/compare_<n>.jpg.
Screw layout and props differ between the cells; judge the robot, the table edges and the mat.

    OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \\
      ~/isaacsim-venv/bin/python render_compare.py --samples 0 10 13
"""
import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cam_calib"))
parser.add_argument("--samples", type=int, nargs="+", default=[0])
args = parser.parse_args()
args.dir = os.path.abspath(args.dir)

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

samples = json.load(open(os.path.join(args.dir, "samples.json")))

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage)
cams = {}
for name, path, cfg in (("table", cell.TABLE_CAM_PATH, C.TABLE_CAM),
                        ("side", cell.SIDE_CAM_PATH, C.SIDE_CAM),
                        ("wrist", cell.WRIST_CAM_PATH, C.WRIST_CAM)):
    cams[name] = Camera(prim_path=path, resolution=cfg["resolution"])
cell.enable_ccd(world)
world.reset()
for cam in cams.values():
    cam.initialize()

arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
arm.initialize()
names = list(arm.dof_names)


def set_pose(joints_deg):
    # FRC_ReadJointAngles J3 is ground-referenced; the sim chain wants J3 + J2.
    sim = list(joints_deg)
    sim[2] = sim[2] + sim[1]
    q = arm.get_joint_positions()
    for i, deg in enumerate(sim):
        q[names.index(f"J{i + 1}")] = np.deg2rad(deg)
    for jn, v in cell.jaw_targets(C.JAW_OPEN).items():
        q[names.index(jn)] = v
    arm.set_joint_positions(q)
    arm.apply_action(ArticulationAction(joint_positions=q))


first = True
for n in args.samples:
    s = samples[n]
    set_pose(s["joints_deg"])
    for _ in range(90 if first else 20):  # MDL materials compile on the first frames
        world.step(render=True)
    first = False
    rows = []
    for name, cam in cams.items():
        if name not in s["images"]:  # side-only samples (calib_capture.py --cams side)
            continue
        sim = cv2.cvtColor(cam.get_rgba()[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.dir, f"{n:02d}_{name}_sim.png"), sim)
        real = cv2.imread(os.path.join(args.dir, s["images"][name]["file"]))
        real = cv2.resize(real, (sim.shape[1], sim.shape[0]))
        blend = cv2.addWeighted(real, 0.5, sim, 0.5, 0)
        row = np.hstack([real, sim, blend])
        rows.append(cv2.resize(row, (1920, int(1920 * row.shape[0] / row.shape[1]))))
        for x, label in ((10, "real"), (650, "sim"), (1290, "blend")):
            cv2.putText(rows[-1], f"{name} {label}", (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    out = os.path.join(args.dir, f"compare_{n:02d}.jpg")
    cv2.imwrite(out, np.vstack(rows))
    print(f"[compare] sample {n} J={np.round(s['joints_deg'], 1).tolist()} -> {out}", flush=True)

sim_app.close()
