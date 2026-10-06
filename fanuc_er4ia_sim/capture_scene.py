"""
Stage 1 of the perception pipeline (runs under Isaac's interpreter).

Builds the cell, renders the world camera, and writes out everything the later
stages need: the image, the camera's exact intrinsics/extrinsics, and the
ground-truth object poses so the VLM's guesses can actually be scored.

    $ISAAC/python.sh capture_scene.py --out run/
"""
import argparse, json, os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="run")
parser.add_argument("--camera", default="table",
                    choices=["table", "side", "closeup", "wrist"])
args = parser.parse_args()
# resolve before Isaac starts: Kit changes the working directory
args.out = os.path.abspath(args.out)

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import numpy as np
import cv2
from pxr import UsdGeom, Gf
import omni.usd
from isaacsim.core.api import World

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_config as C, cell

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage)

_CAMS = {"table":   (cell.TABLE_CAM_PATH,   C.TABLE_CAM),
         "side":    (cell.SIDE_CAM_PATH,    C.SIDE_CAM),
         "closeup": (cell.CLOSEUP_CAM_PATH, C.CLOSEUP_CAM),
         "wrist":   (cell.WRIST_CAM_PATH,   C.WRIST_CAM)}
cam_path, cfg = _CAMS[args.camera]

from isaacsim.sensors.camera import Camera
cam = Camera(prim_path=cam_path, resolution=cfg["resolution"])
cell.enable_ccd(world)
world.reset()
cam.initialize()

# park the arm out of the overhead camera's view before the snapshot
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
arm.initialize()
names = list(arm.dof_names)
q = arm.get_joint_positions()
for j, deg in C.CAPTURE_POSE_DEG.items():
    q[names.index(j)] = np.deg2rad(deg)
for jn, v in cell.jaw_targets(C.JAW_OPEN).items():
    q[names.index(jn)] = v
arm.set_joint_positions(q)
arm.apply_action(ArticulationAction(joint_positions=q))

# MDL materials need frames to compile before capture
for _ in range(90):
    world.step(render=True)

os.makedirs(args.out, exist_ok=True)
rgb = cam.get_rgba()[:, :, :3].astype(np.uint8)
img_path = os.path.join(args.out, "world_camera.png")
cv2.imwrite(img_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# ---- camera model, read back from the stage rather than recomputed
prim = stage.GetPrimAtPath(cam_path)
gcam = UsdGeom.Camera(prim)
xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
W, H = cfg["resolution"]
focal = gcam.GetFocalLengthAttr().Get()
h_ap = gcam.GetHorizontalApertureAttr().Get()
v_ap = gcam.GetVerticalApertureAttr().Get()

cam_json = {
    "camera": args.camera,
    "resolution": [W, H],
    "focal_length_mm": float(focal),
    "horizontal_aperture_mm": float(h_ap),
    "vertical_aperture_mm": float(v_ap),
    # row-major 4x4, USD convention (row-vector times matrix)
    "camera_to_world": [[float(xf[r][c]) for c in range(4)] for r in range(4)],
    # the work surface is the mat, 2 mm proud of the table top -- objects rest
    # on it, so this is the plane detections must be projected onto
    "table_top_z": float(C.TABLE_TOP_Z + C.SURFACE_DZ),
    "robot_base_xy": [0.0, 0.0],
}
with open(os.path.join(args.out, "camera.json"), "w") as f:
    json.dump(cam_json, f, indent=2)

# ---- ground truth, for scoring the VLM later
gt = {"objects": [], "bin": {"xy": list(C.BIN_XY), "size": list(C.BIN_SIZE)}}
for i, y in enumerate(C.CUBE_ROW_Y):
    colour = C.CUBE_MATS[i % len(C.CUBE_MATS)].replace("plastic_", "")
    gt["objects"].append({"name": f"Cube{i}", "category": "cube", "colour": colour,
                          "xyz": [C.CUBE_ROW_X, y, C.CUBE_EDGE / 2],
                          "size": C.CUBE_EDGE})
# screws: the settled pose, reported at mid-length (where the grasp goes);
# the configured (x, y) is the head, and physics settles them slightly
from isaacsim.core.prims import SingleXFormPrim
for i, (x, y, yaw) in enumerate(C.SCREWS):
    pos, quat = SingleXFormPrim(f"/World/Screw{i}").get_world_pose()
    w, qx, qy, qz = [float(v) for v in quat]
    # local +X is head -> tip after the lie-down rotation; rotate it by the pose
    R = np.array([[1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*w), 2*(qx*qz + qy*w)],
                  [2*(qx*qy + qz*w), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*w)],
                  [2*(qx*qz - qy*w), 2*(qy*qz + qx*w), 1 - 2*(qx*qx + qy*qy)]])
    tip_dir = R @ np.array([0.0, 0.0, -1.0])       # revolve axis runs to -Z
    mid = np.array(pos) + tip_dir * C.SCREW_LENGTH / 2
    gt["objects"].append({"name": f"Screw{i}", "category": "screw", "colour": "black",
                          "xyz": [float(mid[0]), float(mid[1]),
                                  float(mid[2] - C.TABLE_TOP_Z - C.SURFACE_DZ)],
                          "size": C.SCREW_LENGTH,
                          "yaw_deg": float(np.degrees(np.arctan2(tip_dir[1], tip_dir[0])))})
with open(os.path.join(args.out, "ground_truth.json"), "w") as f:
    json.dump(gt, f, indent=2)

print(f"[capture] image      {img_path}  ({W}x{H})")
print(f"[capture] camera     {args.out}/camera.json")
print(f"[capture] gt objects {len(gt['objects'])}")
sim_app.close()
