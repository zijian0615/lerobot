"""Record one scripted screw pick in this cell, with all three real cameras.

Writes PNGs and frames.json. Pack them with pack_lerobot_episode.py.
Does not move the real robot.

    OMNI_KIT_ACCEPT_EULA=YES \\
      LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \\
      ~/isaacsim-venv/bin/python record_screw_episode.py --out data/screw_ep0
"""
import argparse
from pathlib import Path
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="data/screw_ep0")
parser.add_argument("--fps", type=int, default=10)
parser.add_argument("--max-speed", type=float, default=50.0,
                    help="fingertip speed cap, mm/s; match FanucConfig.joint_tcp_speed_mm_s on the real arm")
parser.add_argument("--screw", type=int, default=0, help="which screw this episode picks")
parser.add_argument("--all-screws", action="store_true",
                    help="pick every screw in the layout, nearest first")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--start", type=int, default=0, help="first episode index to record")
parser.add_argument("--screw-xy", nargs=2, type=float, metavar=("X", "Y"),
                    help="place the picked screw at this table XY (metres, robot frame)")
parser.add_argument("--screw-yaw", type=float, default=None,
                    help="head-to-tip yaw of the picked screw, degrees")
parser.add_argument("--random-layout", action="store_true",
                    help="scatter every screw on the mat before the episode")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--jitter-xy", type=float, default=0.0,
                    help="with --single: move the screw uniformly within +/- this many metres per episode")
parser.add_argument("--jitter-yaw", type=float, default=0.0,
                    help="with --single: turn the screw uniformly within +/- this many degrees per episode")
parser.add_argument("--bg-plates", default=None,
                    help="dir with plate_table.png / plate_side.png: put the real background behind those views")
parser.add_argument("--mat", default=None, help="mat texture relative to fanuc_er4ia_sim, e.g. assets/mat_print_real.png")
parser.add_argument("--tone", default=None, help="per-camera tone json from fit_cam_tone.py (mat pixels only)")
parser.add_argument("--bg-gain-jitter", type=float, default=0.10,
                    help="per-episode brightness scale of the plates, uniform in 1 +/- this")
parser.add_argument("--random-xy", type=float, nargs=4, default=None, metavar=("X0", "X1", "Y0", "Y1"),
                    help="with --single: place the screw uniformly in this box (m, robot frame) every episode, "
                         "clear of the bin and reachable, yaw uniform in +/- --random-yaw (bin stays put)")
parser.add_argument("--random-yaw", type=float, default=180.0)
parser.add_argument("--single", action="store_true",
                    help="spawn only the picked screw (--screw, at --screw-xy/--screw-yaw if given)")
args = parser.parse_args()
args.out = os.path.abspath(args.out)
if args.bg_plates:
    args.bg_plates = os.path.abspath(args.bg_plates)

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

def scatter_layout(seed: int):
    rng = np.random.default_rng(seed)
    bx, by = C.BIN_XY
    half = max(C.BIN_SIZE) / 2 + 0.04
    placed = []
    for _ in range(len(C.SCREWS)):
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
    return placed


# cell.build reads C.SCREWS. Episode 0 is placed before the stage exists;
# later episodes teleport the same prims.
episode_seeds = [args.seed + i for i in range(args.start, args.episodes)]
if args.random_layout:
    C.SCREWS = [tuple(p) for p in scatter_layout(episode_seeds[0])]
elif args.screw_xy is not None or args.screw_yaw is not None:
    layout = [list(p) for p in C.SCREWS]
    x, y, yaw = layout[args.screw]
    if args.screw_xy is not None:
        x, y = args.screw_xy
    if args.screw_yaw is not None:
        yaw = args.screw_yaw
    layout[args.screw] = (x, y, yaw)
    C.SCREWS = [tuple(p) for p in layout]
if args.single:
    if args.random_layout or args.all_screws:
        parser.error("--single picks one fixed screw; drop --random-layout / --all-screws")
    C.SCREWS = [tuple(C.SCREWS[args.screw])]
    args.screw = 0
NOMINAL = [tuple(p) for p in C.SCREWS]


def random_layout(seed: int):
    """--random-xy: one screw anywhere in the box, clear of the bin, with a reachable fingertip-down pick."""
    rng = np.random.default_rng(30_000 + seed)
    x0, x1, y0, y1 = args.random_xy
    bx, by = C.BIN_XY
    keep = C.BIN_OUTER[0] / 2 + 0.04 + C.SCREW_LENGTH / 2  # bin half-width, a finger's clearance, half a screw
    q0 = np.array([np.deg2rad(C.HOME_POSE_DEG[f"J{i}"]) for i in range(1, 7)])
    for _ in range(500):
        x, y = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
        if abs(x - bx) < keep and abs(y - by) < keep:
            continue
        _, ok = ik.solve(q0, np.array([x, y, PICK_Z]), DOWN, LIM, tool_offset=TOOL_OFF)
        if ok:
            return [(x, y, float(rng.uniform(-args.random_yaw, args.random_yaw)))]
    raise RuntimeError(f"seed {seed}: no reachable screw position clear of the bin in {args.random_xy}")


def jittered_layout(seed: int):
    """The nominal single screw, nudged by --jitter-xy / --jitter-yaw. Seed 0 stays nominal."""
    if args.single and args.random_xy is not None:
        return random_layout(seed)
    if seed == 0 or not args.single:
        return list(NOMINAL)
    rng = np.random.default_rng(10_000 + seed)
    x, y, yaw = NOMINAL[0]
    return [(x + float(rng.uniform(-args.jitter_xy, args.jitter_xy)),
             y + float(rng.uniform(-args.jitter_xy, args.jitter_xy)),
             yaw + float(rng.uniform(-args.jitter_yaw, args.jitter_yaw)))]

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
if args.mat:
    C.MAT_TEXTURE = args.mat
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

import bg_composite
from scipy.spatial.transform import Rotation as _Rot

TONE = bg_composite.load_tone(os.path.abspath(args.tone)) if args.tone else {}
CAM_CFG = {"table": C.TABLE_CAM, "side": C.SIDE_CAM, "wrist": C.WRIST_CAM}


def to_real(name, cam, rgb, gain=1.0):
    """Tone (mat pixels only) and real background, as bg_composite.render_to_real; needs the depth annotator."""
    if not (TONE or name in PLATES):
        return rgb
    depth = cam.get_depth()
    T = None
    if name not in bg_composite.FIXED:
        pos, quat = cam.get_world_pose(camera_axes="ros")  # x right, y down, z forward = OpenCV
        T = np.eye(4)
        T[:3, :3] = _Rot.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        T[:3, 3] = pos
    cfg = CAM_CFG[name]
    fx = cfg["focal_length"] / cfg["horiz_aperture"] * cfg["resolution"][0]
    return bg_composite.render_to_real(name, rgb, depth, PLATES.get(name), TONE.get(name), gain, T, fx)

PLATES = bg_composite.load_plates(Path(args.bg_plates)) if args.bg_plates else {}
if args.bg_plates and set(PLATES) != set(bg_composite.FIXED):
    raise SystemExit(f"--bg-plates {args.bg_plates} needs plate_table.png and plate_side.png")

cell.enable_ccd(world)
world.reset()
for name, cam, _ in cams:
    cam.initialize()
    if name in PLATES or TONE:
        cam.add_distance_to_image_plane_to_frame()
for _ in range(60):
    world.step(render=True)

from isaacsim.core.prims import SingleRigidPrim

arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
arm.initialize()
screw_bodies = []
for i in range(len(C.SCREWS)):
    body = SingleRigidPrim(prim_path=f"/World/Screw{i}", name=f"screw{i}")
    body.initialize()
    screw_bodies.append(body)
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


shut = C.JAW_CLOSED
bin_xy = np.array(C.BIN_XY, float)
drop = np.array([bin_xy[0], bin_xy[1], C.BIN_RELEASE_Z])
up_b = drop + np.array([0.0, 0.0, 0.06])


def pick_order(layout):
    """Nearest screw first, measured from the home XY, then from the last screw."""
    pending = list(enumerate(layout))
    if not args.all_screws:
        return [item for item in pending if item[0] == args.screw]
    cursor = HOME[:2]
    ordered = []
    while pending:
        index, pose = min(
            pending,
            key=lambda item: (item[1][0] - cursor[0]) ** 2 + (item[1][1] - cursor[1]) ** 2,
        )
        pending.remove((index, pose))
        ordered.append((index, pose))
        cursor = np.array(pose[:2])
    return ordered


def segments_for(head_xy, yaw, start):
    yaw_r = np.radians(yaw)
    center = np.array([
        head_xy[0] + np.cos(yaw_r) * C.SCREW_LENGTH / 2,
        head_xy[1] + np.sin(yaw_r) * C.SCREW_LENGTH / 2,
    ])
    spread = [-np.sin(yaw_r), np.cos(yaw_r), 0.0]
    side = np.cross(DOWN, np.array(spread, float))
    light = center + side[:2] * C.WRISTCAM_LATERAL
    light_flip = center - side[:2] * C.WRISTCAM_LATERAL
    if np.linalg.norm(light_flip - bin_xy) > np.linalg.norm(light - bin_xy):
        spread = [-v for v in spread]
    pick = np.array([center[0], center[1], PICK_Z])
    up_p = pick + np.array([0.0, 0.0, HOVER])
    # The screw (30 mm) is longer than the bin's pocket side, so it only drops in lying along a diagonal:
    # turn it onto the nearer diagonal during the carry (the screw axis is perpendicular to the spread).
    axis = yaw_r % np.pi
    diag = min((np.pi / 4, 3 * np.pi / 4), key=lambda d: min(abs(axis - d), np.pi - abs(axis - d)))
    drop_spread = [-np.sin(diag), np.cos(diag), 0.0]
    if np.dot(drop_spread, spread) < 0:
        drop_spread = [-v for v in drop_spread]
    return [
        ("approach", np.array(start), up_p, 1.1, OPEN, OPEN, spread),
        ("descend", up_p, pick, 0.7, OPEN, OPEN, spread),
        ("close", pick, pick, 0.5, OPEN, shut, spread),
        ("grip", pick, pick, 0.3, shut, shut, spread),
        ("lift", pick, up_p, 0.6, shut, shut, spread),
        ("carry", up_p, up_b, 1.2, shut, shut, drop_spread),
        ("lower", up_b, drop, 0.5, shut, shut, drop_spread),
        ("release", drop, drop, 0.4, shut, OPEN, drop_spread),
        ("retreat", drop, up_b, 0.5, OPEN, OPEN, None),
    ]


def _quat_wxyz(yaw_deg):
    """Match add_screws: yaw about Z, then lay the screw down with Ry(-90).

    q = q_z(yaw) * q_y(-90) in closed form. (The rotation-matrix route this replaced fell back to the
    identity whenever cos(yaw) < 0, so screws with |yaw| > 90 deg spawned standing up and fell anywhere.)
    """
    h = np.radians(yaw_deg) / 2
    c = np.sqrt(0.5)
    return np.array([c * np.cos(h), c * np.sin(h), -c * np.cos(h), c * np.sin(h)], dtype=float)


def place_screws(layout):
    """Move the already-spawned screws onto a new layout."""
    z = C.TABLE_TOP_Z + C.SURFACE_DZ + C.SCREW_HEAD_D / 2 + 0.0005
    for i, (x, y, yaw) in enumerate(layout):
        body = screw_bodies[i]
        body.set_world_pose(
            position=np.array([x, y, z], dtype=float),
            orientation=_quat_wxyz(yaw),
        )
        body.set_linear_velocity(np.zeros(3))
        body.set_angular_velocity(np.zeros(3))
    for _ in range(5):
        world.step(render=False)


def record_episode(out_dir, layout, seed=0):
    bg_gain = float(np.random.default_rng(20_000 + seed).uniform(1 - args.bg_gain_jitter, 1 + args.bg_gain_jitter))
    C.SCREWS = [tuple(p) for p in layout]
    place_screws(layout)
    for name, _, _ in cams:
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    q = np.array([np.deg2rad(C.HOME_POSE_DEG[f"J{i}"]) for i in range(1, 7)])
    q, ok = ik.solve(q, HOME, DOWN, LIM, tool_offset=TOOL_OFF)
    print(f"[record] {out_dir} home IK ok={ok} screws={len(layout)}", flush=True)
    for _ in range(40):
        command(q, OPEN)
        world.step(render=False)
    # The renderer lags the stage by several frames, more while a large texture streams in: render the
    # settled home pose until the views stop changing (as eval_act_sim.py), or frame 0 shows a stale view
    # (the sky on episode 0, the previous episode's last pose after that).
    prev = None
    for k in range(60):
        command(q, OPEN)
        world.step(render=True)
        view = np.concatenate([cam.get_rgba()[::8, ::8, :3].astype(np.float32).ravel() for _, cam, _ in cams])
        if k >= 10 and prev is not None and np.abs(view - prev).mean() < 0.5:
            break
        prev = view

    segments = []
    order = []
    start = HOME
    for index, (x, y, yaw) in pick_order(layout):
        segments.extend(segments_for((x, y), yaw, start))
        order.append({"index": index, "xy_yaw": [x, y, yaw]})
        start = up_b
    segments.append(("home", up_b if order else HOME, HOME, 1.0, OPEN, OPEN, None))

    frames = []
    frame = 0
    for name, p0, p1, dur, g0, g1, spread_dir in segments:
        # The ease peaks at 1.5x the mean speed. Stretch moves so the peak stays
        # under --max-speed; the real arm cannot follow anything faster.
        dist = float(np.linalg.norm(np.array(p1) - np.array(p0)))
        dur = max(dur, 1.5 * dist * 1000.0 / args.max_speed)
        n = max(1, int(round(dur * args.fps)))
        spread0 = None
        if spread_dir is not None and name in ("approach", "carry"):  # turn smoothly over the segment
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
            for cam_name, cam, _res in cams:
                rgb = cam.get_rgba()[:, :, :3].astype(np.uint8)
                rgb = to_real(cam_name, cam, rgb, gain=bg_gain)
                path = os.path.join(out_dir, cam_name, f"{frame:06d}.png")
                cv2.imwrite(path, cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))
            frames.append({
                "segment": name,
                "joints_deg": joints,
                "gripper": gripper,
                "jaw_gap_m": gap,
                "ik_ok": bool(ok),
            })
            frame += 1
            if frame % 20 == 0:
                print(f"[record] {os.path.basename(out_dir)} {name} frame {frame}", flush=True)

    task = (
        "pick up all the countersunk screws and drop them in the yellow bin"
        if args.all_screws
        else "pick one countersunk screw and drop it in the yellow bin"
    )
    # Where every screw came to rest, and whether it is in the bin pocket.
    for _ in range(30):
        world.step(render=False)
    bx, by = C.BIN_XY
    final = []
    for body in screw_bodies:
        pos = body.get_world_pose()[0]
        inside = abs(pos[0] - bx) < C.BIN_SIZE[0] / 2 and abs(pos[1] - by) < C.BIN_SIZE[1] / 2
        final.append({"xyz": [float(v) for v in pos], "in_bin": bool(inside)})
    print(f"[record] {os.path.basename(out_dir)} in bin: {sum(f['in_bin'] for f in final)}/{len(final)}", flush=True)
    meta = {
        "background": {"plates": args.bg_plates, "gain": bg_gain} if PLATES else None,
        "mat": C.MAT_TEXTURE,
        "tone": args.tone,
        "final_screws": final,
        "fps": args.fps,
        "max_speed_mm_s": args.max_speed,
        "task": task,
        "pick_order": order,
        "screw_layout": [list(p) for p in layout],
        "cameras": {
            "table": {"role": "overhead", "resolution": list(C.TABLE_CAM["resolution"])},
            "wrist": {"role": "wrist", "resolution": list(C.WRIST_CAM["resolution"])},
            "side": {"role": "fixed side", "resolution": list(C.SIDE_CAM["resolution"])},
        },
        "frames": frames,
    }
    with open(os.path.join(out_dir, "frames.json"), "w") as f:
        json.dump(meta, f)
    print(f"[record] {frame} frames -> {out_dir}", flush=True)


for seed in episode_seeds:
    episode = seed - args.seed
    layout = scatter_layout(seed) if args.random_layout else jittered_layout(seed)
    if args.episodes == 1 and args.start == 0:
        out_dir = args.out
    else:
        out_dir = os.path.join(args.out, f"episode_{episode:02d}")
    if os.path.isfile(os.path.join(out_dir, "frames.json")):
        print(f"[record] skip {out_dir}", flush=True)
        continue
    record_episode(out_dir, layout, seed)

sim_app.close()
