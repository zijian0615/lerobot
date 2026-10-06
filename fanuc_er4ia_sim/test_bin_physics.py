"""
Rigorous check of the collection bin's collider.

    $ISAAC/python.sh test_bin_physics.py

Verifies three things the visual mesh cannot tell you:
  1. the authored collider matches the bin's visual outer dimensions
  2. objects dropped in land on the floor and stay inside the walls
  3. objects stack rather than interpenetrating or falling through
"""
import os, sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from pxr import UsdGeom, Usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_config as C, cell

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage)
cell.enable_ccd(world)
world.reset()

Z0 = C.TABLE_TOP_Z + C.SURFACE_DZ
bx, by = C.BIN_XY
ow, od = C.BIN_OUTER
fails = []

# ---- 1. collider vs visual mesh -------------------------------------------
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
vis = stage.GetPrimAtPath("/World/Bin")
col = stage.GetPrimAtPath("/World/BinCollider")
if vis.IsValid() and col.IsValid():
    vb = cache.ComputeWorldBound(vis).ComputeAlignedRange()
    cb = cache.ComputeWorldBound(col).ComputeAlignedRange()
    vsz = np.array([vb.GetMax()[i] - vb.GetMin()[i] for i in range(3)])
    csz = np.array([cb.GetMax()[i] - cb.GetMin()[i] for i in range(3)])
    print(f"[bin] visual mesh   {np.round(vsz*1000,1)} mm")
    print(f"[bin] collider      {np.round(csz*1000,1)} mm")
    d = np.abs(vsz - csz) * 1000
    ok = (d[:2] < 6).all() and d[2] < 12
    print(f"[bin] footprint/height agreement within {np.round(d,1)} mm -> "
          f"{'OK' if ok else 'MISMATCH'}")
    if not ok:
        fails.append("collider does not match the visual mesh")

# ---- 2 & 3. drop a stack ---------------------------------------------------
names = [f"/World/Cube{i}" for i in range(len(C.CUBE_ROW_Y))] + \
        [f"/World/Screw{i}" for i in range(len(C.SCREWS))]
prims = [SingleXFormPrim(n) for n in names]
sizes = [C.CUBE_EDGE] * len(C.CUBE_ROW_Y) + [C.SCREW_HEAD_D] * len(C.SCREWS)

print(f"\n[bin] dropping {len(prims)} objects in from 120 mm above the rim")
for i, p in enumerate(prims):
    jitter = np.array([(i % 3 - 1) * 0.012, (i // 3 - 1) * 0.012, 0.0])
    p.set_world_pose(position=np.array([bx, by, Z0 + C.BIN_WALL_H + 0.12]) + jitter)
    for _ in range(90):
        world.step(render=False)
for _ in range(300):
    world.step(render=False)

print(f"{'object':9s} {'x':>8s} {'y':>8s} {'z(mm)':>8s}  verdict")
inside = 0
pos = []
for n, p, sz in zip(names, prims, sizes):
    w = np.array(p.get_world_pose()[0])
    z = (w[2] - Z0) * 1000
    in_xy = abs(w[0] - bx) < ow / 2 + 0.006 and abs(w[1] - by) < od / 2 + 0.006
    fell = z < -2.0
    v = "FELL THROUGH" if fell else ("in bin" if in_xy and z > 0 else "outside")
    if fell:
        fails.append(f"{n} fell through the floor")
    if in_xy and not fell:
        inside += 1
        pos.append((w, sz))
    print(f"{n.split('/')[-1]:9s} {w[0]:8.3f} {w[1]:8.3f} {z:8.1f}  {v}")

print(f"\n[bin] {inside}/{len(prims)} contained")
if inside < len(prims):
    fails.append(f"only {inside}/{len(prims)} objects stayed in the bin")

# interpenetration: no two centres closer than 55% of their mean size
bad = 0
for i in range(len(pos)):
    for j in range(i + 1, len(pos)):
        (a, sa), (b, sb) = pos[i], pos[j]
        d = np.linalg.norm(a - b)
        if d < 0.55 * (sa + sb) / 2:
            bad += 1
print(f"[bin] overlapping pairs: {bad} -> {'OK' if bad == 0 else 'INTERPENETRATION'}")
if bad:
    fails.append(f"{bad} overlapping pairs")

# stacking: with 8 objects in a 36 mm pocket something must sit above the floor
zs = sorted((w[2] - Z0) * 1000 for w, _ in pos)
print(f"[bin] z spread {zs[0]:.1f} .. {zs[-1]:.1f} mm "
      f"(rim at {C.BIN_WALL_H*1000:.0f} mm)")
if len(zs) > 3 and zs[-1] - zs[0] < 5:
    fails.append("objects are not stacking (all at the same height)")

# ---- 4. lateral tunnelling -------------------------------------------------
# Dropping objects in only exercises the floor. A fast object could tunnel
# straight through a wall between solver steps.
print("\n[bin] firing an object at the wall at 3 m/s")
from isaacsim.core.prims import SingleRigidPrim
tgt = SingleRigidPrim(names[0])
tgt.set_world_pose(position=np.array([bx - 0.008, by, Z0 + C.BIN_FLOOR_T + 0.012]))
tgt.set_linear_velocity(np.array([3.0, 0.0, 0.0]))
for _ in range(240):
    world.step(render=False)
w = np.array(tgt.get_world_pose()[0])
esc = abs(w[0] - bx) > ow / 2 + 0.02 or abs(w[1] - by) > od / 2 + 0.02
print(f"[bin] ended at ({w[0]:.3f},{w[1]:.3f}) z={(w[2]-Z0)*1000:.1f} mm -> "
      f"{'ESCAPED' if esc else 'contained'}")
if esc:
    fails.append("object tunnelled through a bin wall at 3 m/s")

print("\n[bin] RESULT:", "PASS" if not fails else "FAIL")
for f in fails:
    print("   -", f)
app.close()
