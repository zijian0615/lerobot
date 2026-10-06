"""
Stage 4: execute the perception-driven plan (runs under Isaac's interpreter).

    $ISAAC/python.sh execute_plan.py --run run/ --camera table --out run/frames

Reads run/plan.json, drives the arm to each planned pick with IK, and drops each
object into the bin. The object being carried is chosen as the nearest cube prim
to the planned grasp point at execution time -- the controller never consults
ground truth, it only acts on what perception reported.
"""
import argparse, json, os, sys

parser = argparse.ArgumentParser()
parser.add_argument("--run", default="run")
parser.add_argument("--out", default=None)
parser.add_argument("--camera", default="table",
                    choices=["table", "side", "wrist", "closeup"])
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--inspect-bin", action="store_true",
                    help="after the run, pose the arm so the WRIST camera looks "
                         "into the bin and save run/bin_view.png")
parser.add_argument("--demo", action="store_true",
                    help="ignore plan.json and pick everything from its known "
                         "position -- a scripted demo with no perception in the loop")
args = parser.parse_args()
# Kit changes the process working directory, so relative paths must be resolved
# before Isaac starts or output lands somewhere unfindable.
args.run = os.path.abspath(args.run)
out_dir = os.path.abspath(args.out) if args.out else os.path.join(args.run, "frames")

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import numpy as np
import cv2
from pxr import UsdGeom, Gf
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_config as C, cell, fk, ik

ik.set_tool_axis(C.TOOL_AXIS)
ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
if args.demo:
    import scene_config as _C
    plan = {"bin_xy": list(_C.BIN_XY), "bin_release_z": _C.BIN_RELEASE_Z,
            "picks": [dict(label=f"{c.replace('plastic_','')} cube",
                           category="cube",
                           xy=[_C.CUBE_ROW_X, y],
                           jaw_half=_C.CUBE_EDGE / 2)
                      for y, c in zip(_C.CUBE_ROW_Y, _C.CUBE_MATS)]
                     + [dict(label="screw", category="screw",
                             # mid-shank, jaws closing across the screw axis
                             xy=[x + np.cos(np.radians(yaw)) * _C.SCREW_LENGTH / 2,
                                 y + np.sin(np.radians(yaw)) * _C.SCREW_LENGTH / 2],
                             spread_dir=[-np.sin(np.radians(yaw)),
                                         np.cos(np.radians(yaw)), 0.0],
                             jaw_half=_C.SCREW_SHANK_D / 2)
                        for (x, y, yaw) in _C.SCREWS]}
else:
    plan = json.load(open(os.path.join(args.run, "plan.json")))

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage)

_CAMS = {"table":   (cell.TABLE_CAM_PATH,   C.TABLE_CAM),
         "side":    (cell.SIDE_CAM_PATH,    C.SIDE_CAM),
         "wrist":   (cell.WRIST_CAM_PATH,   C.WRIST_CAM),
         "closeup": (cell.CLOSEUP_CAM_PATH, C.CLOSEUP_CAM)}
cam_path, cfg = _CAMS[args.camera]
from isaacsim.sensors.camera import Camera
cam = Camera(prim_path=cam_path, resolution=cfg["resolution"])
# render products must exist before world.reset(), so create the wrist camera
# now even though it is only used at the very end
wrist_cam = None
if args.inspect_bin and cam_path != cell.WRIST_CAM_PATH:
    wrist_cam = Camera(prim_path=cell.WRIST_CAM_PATH,
                       resolution=C.WRIST_CAM["resolution"])

cell.enable_ccd(world)
world.reset()
cam.initialize()
if wrist_cam is not None:
    wrist_cam.initialize()
for _ in range(90):
    world.step(render=True)

arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia"); arm.initialize()
names = list(arm.dof_names)
ARM_IDX = [names.index(f"J{i}") for i in range(1, 7)]
IDX = {n: names.index(n) for n in names}
LIM = np.deg2rad(np.array([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)]))
from isaacsim.core.utils.types import ArticulationAction

# Grasps are now physical: the fingertips are the reference, not the scallop
# pocket. The pocket sits 20 mm up the finger, so commanding it to an object's
# centre height drives the tips through the table -- a hard collision now that
# the fingers carry colliders.
TOOL_OFF = tuple(np.array(C.TOOL_AXIS) * cell.tip_offset())
DOWN = (0.0, 0.0, -1.0)
OPEN = C.JAW_OPEN
# A pneumatic gripper does not close to a computed width -- it commands full
# closure and stalls against the part. Modelling it that way matters: a 1.5 mm
# squeeze on a 5 mm half-width screw left the drive barely loaded and the screw
# slipped, while full closure holds it easily.
HOVER = 0.13
PICK_Z = C.SURFACE_DZ + C.TIP_CLEARANCE
# Cylindrical objects are cradled in the scallop pocket rather than pinched
# between the flat faces, so the tip is commanded low enough to bring the pocket
# level with the object's axis.
POCKET_ABOVE_TIP = (1.0 - C.SCALLOP_AT) * C.FINGER_LENGTH
HOME = np.array([0.30, 0.0, 0.24])
SUBSTEPS = 2              # physics steps per rendered frame


def command(q6, jaw):
    full = arm.get_joint_positions().copy()
    for k, i in enumerate(ARM_IDX):
        full[i] = q6[k]
    for jn, v in cell.jaw_targets(jaw).items():
        full[IDX[jn]] = v
    arm.apply_action(ArticulationAction(joint_positions=full))


def ease(t):
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ segments
bin_xy = plan["bin_xy"]


def bin_slot(k, n):
    """Spread releases across the bin floor instead of stacking them all at the
    centre, where only the last one would be visible."""
    if n <= 1:
        return np.array(bin_xy, float)
    w, d = C.BIN_SIZE
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    r, c = divmod(k, cols)
    fx = (c + 0.5) / cols - 0.5
    fy = (r + 0.5) / rows - 0.5
    return np.array([bin_xy[0] + fx * w * 0.86, bin_xy[1] + fy * d * 0.86])


SEGMENTS = []
cursor = HOME
n_picks = len(plan["picks"])
for k, p in enumerate(plan["picks"]):
    shut = C.JAW_CLOSED
    # The pocket sits ~14 mm above the tip, too high to cradle anything lying on
    # the table, so every pick here is a flat-face pinch near the fingertips.
    pick_z = PICK_Z
    pick = np.array([p["xy"][0], p["xy"][1], pick_z])
    spread = p.get("spread_dir")
    if spread is not None:
        # Parallel jaws are symmetric, so +d and -d are the same grasp -- but
        # the ring light hangs WRISTCAM_LATERAL off the tool, perpendicular to
        # the jaws, and at pick height it clears the bin rim by only a few mm.
        # Take the sign that swings it away from the bin.
        s = np.array(spread, float)
        side = np.cross(DOWN, s)        # flange side axis in world, tool down
        light = np.array(p["xy"]) + side[:2] * C.WRISTCAM_LATERAL
        light_flip = np.array(p["xy"]) - side[:2] * C.WRISTCAM_LATERAL
        if (np.linalg.norm(light_flip - bin_xy) > np.linalg.norm(light - bin_xy)):
            spread = [-v for v in spread]
    slot = bin_slot(k, n_picks)
    drop = np.array([slot[0], slot[1], plan["bin_release_z"]])
    up_p = pick + [0, 0, HOVER]
    up_b = drop + [0, 0, 0.06]
    # roll is pinned while approaching and holding the object, then released so
    # the wrist can take the shortest path to the bin
    SEGMENTS += [
        (f"approach{k}", cursor, up_p, 1.1, OPEN, OPEN, spread),
        (f"descend{k}",  up_p,   pick, 0.7, OPEN, OPEN, spread),
        (f"close{k}",    pick,   pick, 0.6, OPEN, shut, spread),
        # Dwell with the jaws commanded shut before lifting, as a pneumatic
        # cell waits on its grip-confirm sensor. Without it the jaws are still
        # 1.5-5 mm from the part when the lift starts; whichever part they had
        # furthest to travel on gets left behind.
        (f"grip{k}",     pick,   pick, 0.4, shut, shut, spread),
        (f"lift{k}",     pick,   up_p, 0.7, shut, shut, spread),
        (f"carry{k}",    up_p,   up_b, 1.4, shut, shut, spread),
        (f"lower{k}",    up_b,   drop, 0.6, shut, shut, spread),
        (f"release{k}",  drop,   drop, 0.6, shut, OPEN, spread),
        (f"settle{k}",   drop,   drop, 0.4, OPEN, OPEN, None),
        (f"retreat{k}",  drop,   up_b, 0.6, OPEN, OPEN, None),
    ]
    cursor = up_b
SEGMENTS.append(("home", cursor, HOME, 1.3, OPEN, OPEN, None))

os.makedirs(out_dir, exist_ok=True)
for f in os.listdir(out_dir):
    if f.endswith(".png"):
        os.remove(out_dir + "/" + f)

q = np.array([np.deg2rad(C.HOME_POSE_DEG[f"J{i}"]) for i in range(1, 7)])
q, ok = ik.solve(q, HOME, DOWN, LIM, tool_offset=TOOL_OFF)
print(f"[exec] fingertip {cell.tip_offset()*1000:.0f} mm from flange; home IK ok={ok}")

# let the arm drive to the start pose before recording
for _ in range(120):
    command(q, OPEN)
    world.step(render=False)

frame = fails = 0
fail_by_seg = {}
def _slerp_about_z(a, b, t):
    """Rotate horizontal unit vector a toward b about world Z by fraction t."""
    ang = np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])
    c, s_ = np.cos(ang * t), np.sin(ang * t)
    return [a[0] * c - a[1] * s_, a[0] * s_ + a[1] * c, 0.0]


for name, p0, p1, dur, g0, g1, spread in SEGMENTS:
    n = max(1, int(round(dur * args.fps)))
    # On approach, turn the jaws toward their target direction over the move
    # rather than asking the IK for the whole wrist roll (up to 180 deg) at once.
    spread0 = None
    if spread is not None and name.startswith("approach"):
        R0, _ = fk.tool_pose(q, TOOL_OFF)
        s0 = R0 @ np.array(C.FINGER_SPREAD_AXIS, float)
        s0[2] = 0.0
        if np.linalg.norm(s0) > 1e-6:
            spread0 = list(s0 / np.linalg.norm(s0))
    for i in range(n):
        t = ease((i + 1) / n)
        target = np.array(p0) + (np.array(p1) - np.array(p0)) * t
        sp = spread if spread0 is None else _slerp_about_z(spread0, spread, t)
        q, ok = ik.solve(q, target, DOWN, LIM, tool_offset=TOOL_OFF,
                         spread_dir=sp)
        fails += (not ok)
        if not ok:
            fail_by_seg[name] = fail_by_seg.get(name, 0) + 1
        command(q, g0 + (g1 - g0) * t)
        for _ in range(SUBSTEPS):
            world.step(render=True)
        if frame % args.stride == 0:
            rgb = cam.get_rgba()[:, :, :3]
            cv2.imwrite(os.path.join(out_dir, f"f{frame:05d}.png"),
                        cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
        frame += 1
    if name.startswith("grip"):
        # jaw gap after closing: fully shut means the jaws missed the part
        jp = arm.get_joint_positions()
        gap = 2 * C.JAW_OPEN - abs(jp[IDX["jaw_l"]]) - abs(jp[IDX["jaw_r"]])
        msg = ""
        if os.environ.get("EXEC_DEBUG"):
            from isaacsim.core.prims import SingleXFormPrim as _X
            _, ptcp = fk.tool_pose(arm.get_joint_positions()[ARM_IDX], TOOL_OFF)
            best = None
            for i in range(len(C.SCREWS)):
                pos, qt = _X(f"/World/Screw{i}").get_world_pose()
                w, qx, qy, qz = [float(v) for v in qt]
                Rq = np.array([[1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*w), 2*(qx*qz+qy*w)],
                               [2*(qx*qy+qz*w), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*w)],
                               [2*(qx*qz-qy*w), 2*(qy*qz+qx*w), 1-2*(qx*qx+qy*qy)]])
                ax_ = Rq @ np.array([0, 0, -1.0])
                rel = ptcp + np.array([0, 0, C.TABLE_TOP_Z]) - np.array(pos)
                along = float(rel @ ax_); off = float(np.linalg.norm(rel - along * ax_))
                if best is None or off < best[1]:
                    best = (i, off, along, float(np.degrees(np.arcsin(abs(ax_[2])))))
            msg = (f"  Screw{best[0]}: tcp {best[2]*1000:.1f} mm from head face, "
                   f"{best[1]*1000:.1f} mm off axis, axis tilt {best[3]:.1f} deg")
        print(f"[exec] {name}: jaw gap {gap*1000:.1f} mm "
              f"({'EMPTY' if gap < 2 * C.JAW_CLOSED + 0.0005 else 'holding'}){msg}")

# report where everything actually ended up -- physically, not by construction
bx, by = C.BIN_XY
bw, bd = C.BIN_SIZE
from isaacsim.core.prims import SingleXFormPrim
in_bin = 0
for tag, n in (("Cube", len(C.CUBE_ROW_Y)), ("Screw", len(C.SCREWS))):
    for i in range(n):
        pos = SingleXFormPrim(f"/World/{tag}{i}").get_world_pose()[0]
        inside = (abs(pos[0] - bx) < bw / 2 + 0.02 and
                  abs(pos[1] - by) < bd / 2 + 0.02)
        in_bin += bool(inside)
        print(f"[exec] {tag}{i}: xy=({pos[0]:+.3f},{pos[1]:+.3f}) "
              f"z={(pos[2]-C.TABLE_TOP_Z)*1000:5.1f}mm  {'IN BIN' if inside else 'on table'}")
print(f"[exec] {in_bin} object(s) ended up in the bin")
if fail_by_seg:
    print("[exec] IK non-convergence by segment (frames): "
          + ", ".join(f"{k}={v}" for k, v in fail_by_seg.items()))


def _rot_between(a, b):
    """Minimal rotation taking unit vector a onto unit vector b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


if args.inspect_bin:
    # Pose the WRIST camera over the bin. Solving for the camera pose directly
    # and back-computing the tool is far better behaved than nudging the TCP and
    # watching where the ray lands -- that walked the TCP outside the robot's
    # reach, so the arm never arrived and the picture was of the wrong place.
    eye_l = np.array(C.WRIST_CAM["eye"], float)
    fwd_l = np.array(C.WRIST_CAM["look_at"], float) - eye_l
    fwd_l /= np.linalg.norm(fwd_l)
    tool_v = np.array(TOOL_OFF, float)

    def _axis_rot(axis, ang):
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)

    # The SC130EF2 looks straight down the tool from 85 mm off its axis, so the
    # natural inspection pose is the tool near vertical with the camera over the
    # bin. Search a little tilt, the tilt azimuth, and wrist roll (which swings
    # the camera round the tool) -- a few hundred IK solves, not the ~200k the
    # sideways-camera search needed.
    bin_pt = np.array([C.BIN_XY[0], C.BIN_XY[1],
                       C.TABLE_TOP_Z + C.SURFACE_DZ + C.BIN_WALL_H])
    hfov = 2 * np.arctan((C.WRIST_CAM["horiz_aperture"] / 2)
                         / C.WRIST_CAM["focal_length"])
    tool_axis = np.array(C.TOOL_AXIS, float)
    spread_axis = np.array(C.FINGER_SPREAD_AXIS, float)

    shot, best = None, -1.0
    aim_off = np.zeros(3)
    for standoff in (0.12, 0.16, 0.20):
      for tilt_deg in (0, 10, 20):
        for az_deg in ((0,) if tilt_deg == 0 else range(0, 360, 45)):
            az = np.deg2rad(az_deg)
            approach = _axis_rot(np.array([np.cos(az), np.sin(az), 0.0]),
                                 np.deg2rad(tilt_deg)) @ np.array([0.0, 0.0, -1.0])
            R_base = _rot_between(tool_axis, approach)
            for roll_deg in range(0, 360, 30):
                R_des = _axis_rot(approach, np.deg2rad(roll_deg)) @ R_base
                fwd_w = R_des @ fwd_l
                if fwd_w[2] > -0.72:
                    continue
                t = standoff / (-fwd_w[2])
                tcp = (bin_pt + aim_off) - t * fwd_w - R_des @ (eye_l - tool_v)
                tcp_robot = tcp - np.array([0.0, 0.0, C.TABLE_TOP_Z])
                if tcp_robot[2] < 0.04:
                    continue
                if np.linalg.norm(tcp_robot
                                  - np.array([0, 0, C.SHOULDER_HEIGHT])) > 0.50:
                    continue
                q_try, _ = ik.solve(q, tcp_robot, approach, LIM,
                                    tool_offset=TOOL_OFF,
                                    spread_dir=R_des @ spread_axis)
                R_act, p_act = fk.tool_pose(q_try, TOOL_OFF)
                if np.linalg.norm(p_act - tcp_robot) > 3e-3:
                    continue
                tip_w = p_act + np.array([0.0, 0.0, C.TABLE_TOP_Z])
                cam_w = tip_w + R_act @ (eye_l - tool_v)
                axis_w = R_act @ fwd_l
                v_tip = tip_w - cam_w
                v_bin = bin_pt - cam_w
                if min(np.linalg.norm(v_tip), np.linalg.norm(v_bin)) < 1e-6:
                    continue
                v_tip /= np.linalg.norm(v_tip)
                v_bin /= np.linalg.norm(v_bin)
                if np.degrees(np.arccos(np.clip(np.dot(v_bin, axis_w), -1, 1))) \
                        > np.degrees(hfov) * 0.32:
                    continue
                depress = np.degrees(np.arcsin(-axis_w[2]))
                sep = np.degrees(np.arccos(np.clip(np.dot(v_tip, v_bin), -1, 1)))
                # the bin must be clear of the gripper first and foremost, then
                # viewed steeply, then close enough to fill some of the frame
                score = 3.0 * sep + 0.5 * depress - 40.0 * standoff
                if score > best:
                    best, shot = score, (q_try, standoff, sep, depress)

    if shot is None:
        print("[exec] bin inspection: no reachable camera pose found")
    else:
        q_look, standoff, sep, depress = shot
        print(f"[exec] bin-view: camera {standoff*1000:.0f} mm above the rim, "
              f"looking down {depress:.0f} deg, bin {sep:.0f} deg clear of the "
              f"gripper")
        # keep recording so the move to the inspection pose appears in the video
        q_from = arm.get_joint_positions()[ARM_IDX].copy()
        n_move = int(round(1.6 * args.fps))
        for i in range(n_move + int(round(0.8 * args.fps))):
            t = ease(min(1.0, (i + 1) / n_move))
            command(q_from + (q_look - q_from) * t, OPEN)
            for _ in range(SUBSTEPS):
                world.step(render=True)
            if frame % args.stride == 0:
                rgb = cam.get_rgba()[:, :, :3]
                cv2.imwrite(os.path.join(out_dir, f"f{frame:05d}.png"),
                            cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
            frame += 1
        for _ in range(90):
            command(q_look, OPEN)
            world.step(render=True)
        src = wrist_cam if wrist_cam is not None else cam
        rgb = src.get_rgba()[:, :, :3]
        bv = os.path.join(args.run, "bin_view.png")
        cv2.imwrite(bv, cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(f"[exec] wrist view of the bin -> {bv}")

print(f"[exec] {frame} frames -> {out_dir}  (IK failures: {fails})")
sim_app.close()
