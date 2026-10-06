"""Sanity check: is gravity on, do objects rest on the table, do they fall?"""
import os, sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import UsdGeom, Gf
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_config as C, cell

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage)
cell.enable_ccd(world)
world.reset()

g = world.get_physics_context().get_gravity()
print(f"[phys] gravity = {g}")

names = [f"/World/Cube{i}" for i in range(len(C.CUBE_ROW_Y))] + \
        [f"/World/Screw{i}" for i in range(len(C.SCREWS))]
prims = {n: SingleXFormPrim(n) for n in names}

def z_of(n):
    return float(prims[n].get_world_pose()[0][2]) - C.TABLE_TOP_Z

print("\n--- settle on the table (120 steps) ---")
start = {n: z_of(n) for n in names}
for _ in range(120):
    world.step(render=False)
print(f"{'object':10s} {'start z':>9s} {'settled z':>10s} {'expected':>9s}")
ok = True
for n in names:
    # the prim origin sits on the object's axis: cube centre, or the centre of
    # a countersunk screw's head face, which rests on its rim
    exp = (C.CUBE_EDGE / 2 if "Cube" in n else C.SCREW_HEAD_D / 2) + C.SURFACE_DZ
    z = z_of(n)
    good = abs(z - exp) < 0.004
    ok &= good
    print(f"{n.split('/')[-1]:10s} {start[n]*1000:8.1f}mm {z*1000:9.1f}mm "
          f"{exp*1000:8.1f}mm {'' if good else '  <-- OFF'}")
print("settle:", "PASS" if ok else "FAIL")

probe = names[0]
rest = (C.CUBE_EDGE / 2 if "Cube" in probe else C.SCREW_HEAD_D / 2) + C.SURFACE_DZ
print(f"\n--- drop test: lift {probe.split('/')[-1]} by 150 mm and release ---")
c0 = prims[probe]
pos, quat = c0.get_world_pose()
c0.set_world_pose(position=np.array([pos[0], pos[1], pos[2] + 0.15]))
world.step(render=False)
print(f"  released at {(z_of(probe))*1000:.1f} mm")
zs = []
for i in range(90):
    world.step(render=False)
    if i % 15 == 0:
        zs.append(z_of(probe) * 1000)
print("  z during fall (mm):", " ".join(f"{v:.0f}" for v in zs))
final = z_of(probe) * 1000
print(f"  landed at {final:.1f} mm (expect ~{rest*1000:.0f} mm)")
print("drop:", "PASS" if abs(final - rest * 1000) < 4 else "FAIL")
app.close()
