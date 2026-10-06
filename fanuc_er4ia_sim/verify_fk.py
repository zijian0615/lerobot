"""Check fk.py against the simulator's own link poses at random configurations."""
import os, sys
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_config as C, cell, fk

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage)
world.reset()

arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia"); arm.initialize()
names = list(arm.dof_names)
flange = SingleXFormPrim(cell.FLANGE)
base_z = C.TABLE_TOP_Z

# The articulation also carries the two prismatic jaw joints, so index the six
# arm joints by name rather than assuming dof order is J1..J6.
ARM = [names.index(f"J{i}") for i in range(1, 7)]
lim = np.deg2rad(np.array([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)]))
rng = np.random.default_rng(0)
worst_p = worst_R = 0.0
for k in range(8):
    q = rng.uniform(lim[:, 0] * 0.6, lim[:, 1] * 0.6)
    full = arm.get_joint_positions().copy()
    for j, i in enumerate(ARM):
        full[i] = q[j]
    arm.set_joint_positions(full)
    world.step(render=False)
    p_sim, quat_sim = flange.get_world_pose()          # (w,x,y,z)
    p_sim = np.array(p_sim) - np.array([0, 0, base_z])  # into robot base frame
    w, x, y, z = quat_sim
    R_sim = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
    R_fk, p_fk = fk.fk(q)
    ep = np.linalg.norm(p_fk - p_sim)
    eR = np.degrees(np.arccos(np.clip((np.trace(R_fk @ R_sim.T) - 1) / 2, -1, 1)))
    worst_p = max(worst_p, ep); worst_R = max(worst_R, eR)
    print(f"  cfg {k}: pos err {ep*1000:7.3f} mm   rot err {eR:7.3f} deg")

print(f"\nworst: {worst_p*1000:.3f} mm, {worst_R:.3f} deg")
# Tolerance is sub-millimetre, not exact: set_joint_positions is followed by a
# physics step, so the arm sags a little under gravity before we read it back.
print("FK VERIFIED" if (worst_p < 5e-4 and worst_R < 0.1) else "FK MISMATCH")

# where does the tool axis point at the zero pose?
R0, p0 = fk.fk(np.zeros(6))
print(f"\nzero pose: flange at {np.round(p0,4)}, "
      f"tool axis {C.TOOL_AXIS} -> world {np.round(R0 @ np.array(C.TOOL_AXIS), 3)}")
sim_app.close()
