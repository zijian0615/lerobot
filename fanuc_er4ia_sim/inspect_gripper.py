"""Render the end effector on its own, for checking the gripper geometry."""
import os, sys
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import numpy as np
from pxr import UsdGeom, UsdLux, Gf
import omni.usd
from isaacsim.core.api import World

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_config as C, cell, render_settings

# resolve before Isaac starts: Kit changes the working directory
out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "gripper_shots")
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
cell.init_materials(stage)
cell.add_lighting(stage)   # same rig as the cell, incl. the studio environment

# build the gripper alone, on a bare Xform standing in for the flange
UsdGeom.Xform.Define(stage, cell.FLANGE)
cell.attach_gripper(stage, jaw=C.JAW_OPEN)

L = C.TOOL_LENGTH
mid = L * 0.5
# each view carries its own up vector: the "front" view looks along -Z, so a
# (0,0,1) up would be parallel to the view direction and degenerate
views = {
    # broadside: sees the whole stack, flange to fingertip
    "side":  ((mid, -L * 1.9, L * 0.30), (mid, 0, 0), 35.0, (0, 0, 1)),
    # looking along the jaw-separation axis, so both fingers and the gap show
    "front": ((mid, 0.0, L * 1.9),       (mid, 0, 0), 35.0, (1, 0, 0)),
    # close on the scallop pocket and hooked tips
    "tip":   ((L * 0.86, -L * 0.42, L * 0.16), (L * 0.86, 0, 0), 50.0, (0, 0, 1)),
}
for name, (eye, tgt, foc, up) in views.items():
    cam = UsdGeom.Camera.Define(stage, f"/World/Cam_{name}")
    v = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(*up))
    UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(v.GetInverse())
    cam.CreateFocalLengthAttr(foc)
    cam.CreateHorizontalApertureAttr(24.0)
    cam.CreateVerticalApertureAttr(24.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.005, 10.0))

if os.environ.get("PATHTRACE"):
    render_settings.enable_path_tracing(int(os.environ.get("SPP", 64)))
world.reset()
# MDL materials need a few frames to compile; capturing sooner renders white
for _ in range(90):
    world.step(render=True)

os.makedirs(out, exist_ok=True)
import omni.replicator.core as rep
for name in views:
    rp = rep.create.render_product(f"/World/Cam_{name}", (900, 900))
    w = rep.WriterRegistry.get("BasicWriter")
    w.initialize(output_dir=os.path.join(out, name), rgb=True)
    w.attach([rp])
rep.orchestrator.step()
print(f"[inspect] wrote {out}")
sim_app.close()
