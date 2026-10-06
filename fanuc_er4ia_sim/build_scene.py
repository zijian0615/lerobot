"""
Build the FANUC ER-4iA grasping cell in Isaac Sim 6.0.

    /isaacsim/python.sh ~/fanuc_er4ia_sim/build_scene.py            # interactive
    /isaacsim/python.sh ~/fanuc_er4ia_sim/build_scene.py --headless --save out.usd

Scene construction lives in cell.py, dimensions in scene_config.py.
"""
import argparse, os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--save", default="", help="write the composed stage to this .usd")
parser.add_argument("--shoot", default="", help="render one frame from each camera into this dir")
parser.add_argument("--pathtrace", action="store_true",
                    help="render with path tracing instead of RTX real-time")
parser.add_argument("--spp", type=int, default=64)
args = parser.parse_args()
# Kit changes the process working directory, so a relative output path silently
# writes somewhere the caller will never find. Resolve before Isaac starts.
if args.shoot:
    args.shoot = os.path.abspath(args.shoot)
if args.save:
    args.save = os.path.abspath(args.save)

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": args.headless})

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_config as C, cell, render_settings

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage)
if args.pathtrace:
    render_settings.enable_path_tracing(args.spp)

cell.enable_ccd(world)
world.reset()

arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
arm.initialize()
names = list(arm.dof_names)
print(f"[scene] articulation DOFs: {names}")
q = arm.get_joint_positions()
for j, deg in C.HOME_POSE_DEG.items():
    if j in names:
        q[names.index(j)] = np.deg2rad(deg)
arm.set_joint_positions(q)

# MDL materials need frames to compile before capture
for _ in range(90):
    world.step(render=not args.headless)

if args.shoot:
    os.makedirs(args.shoot, exist_ok=True)
    import omni.replicator.core as rep
    for path, cfg, tag in ((cell.TABLE_CAM_PATH, C.TABLE_CAM, "tablecam"),
                           (cell.SIDE_CAM_PATH, C.SIDE_CAM, "sidecam"),
                           (cell.CLOSEUP_CAM_PATH, C.CLOSEUP_CAM, "closeup"),
                           (cell.WRIST_CAM_PATH, C.WRIST_CAM, "wristcam")):
        if not stage.GetPrimAtPath(path).IsValid():
            continue
        rp = rep.create.render_product(path, cfg["resolution"])
        w = rep.WriterRegistry.get("BasicWriter")
        w.initialize(output_dir=os.path.join(args.shoot, tag), rgb=True)
        w.attach([rp])
    rep.orchestrator.step()
    print(f"[scene] renders written to {args.shoot}")

if args.save:
    stage.Export(args.save)
    print(f"[scene] stage exported to {args.save}")

if not args.headless:
    while sim_app.is_running():
        world.step(render=True)
sim_app.close()
