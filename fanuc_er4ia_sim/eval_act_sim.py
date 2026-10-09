"""Roll the ACT policy in this cell and report whether the screw ends in the bin.

The scene matches record_screw_episode.py --single: one screw at SCREWS[--screw], nudged per seed
by --jitter-xy / --jitter-yaw with the recorder's own generator (training used seeds 0-49, so the
default seeds are unseen). Frames go to act_policy_server.py at full camera resolution; the server
resizes them the way the policy's preprocessor did in training.

    python act_policy_server.py --ckpt zijian2022/act_fanuc_er4ia_screw_single &
    OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \\
      ~/isaacsim-venv/bin/python eval_act_sim.py --seeds 100 101 102 --video-dir data/eval_single
"""
import argparse
import json
import os
import pickle
import socket
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 110)))
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--steps", type=int, default=450, help="policy steps; training episodes are ~390")
parser.add_argument("--substeps", type=int, default=6,
                    help="physics steps per policy step: 6 x 1/60 s = the real 0.1 s. With 1 (what the recorder "
                         "took per frame) the arm cannot reach each target and the lag compounds")
parser.add_argument("--max-speed", type=float, default=50.0,
                    help="fingertip speed cap, mm/s, as the real driver applies it")
parser.add_argument("--screw", type=int, default=0)
parser.add_argument("--jitter-xy", type=float, default=0.01)
parser.add_argument("--jitter-yaw", type=float, default=15.0)
parser.add_argument("--video-dir", default=None, help="save table | side | wrist tiles every --frame-every steps")
parser.add_argument("--frame-every", type=int, default=5)
parser.add_argument("--log-every", type=int, default=25, help="print tip / gripper / action every N steps")
parser.add_argument("--out", default=None, help="json file for the per-seed results")
parser.add_argument("--bg-plates", default=None, help="composite the real background into table/side, as the recorder")
parser.add_argument("--mat", default=None, help="mat texture relative to fanuc_er4ia_sim, as the recorder")
parser.add_argument("--tone", default=None, help="per-camera tone json, as the recorder")
parser.add_argument("--legacy-look", action="store_true",
                    help="the asset's own lime arm and the old lemon bin, as in datasets recorded before 2026-10-08")
args = parser.parse_args()
if args.video_dir:
    args.video_dir = os.path.abspath(args.video_dir)
if args.out:
    args.out = os.path.abspath(args.out)
if args.bg_plates:
    args.bg_plates = os.path.abspath(args.bg_plates)

from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": True})

import cv2
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cell
import fk
import ik
import scene_config as C

ik.set_tool_axis(C.TOOL_AXIS)
ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
TASK = "pick one countersunk screw and drop it in the yellow bin"
NOMINAL = tuple(C.SCREWS[args.screw])


def layout_for(seed):
    """Same generator as record_screw_episode.jittered_layout."""
    if seed == 0:
        return NOMINAL
    rng = np.random.default_rng(10_000 + seed)
    x, y, yaw = NOMINAL
    return (x + float(rng.uniform(-args.jitter_xy, args.jitter_xy)),
            y + float(rng.uniform(-args.jitter_xy, args.jitter_xy)),
            yaw + float(rng.uniform(-args.jitter_yaw, args.jitter_yaw)))


def quat_wxyz(yaw_deg):
    """Match add_screws: yaw about Z, then lay the screw down with Ry(-90).

    q = q_z(yaw) * q_y(-90) in closed form. (The rotation-matrix route this replaced fell back to the
    identity whenever cos(yaw) < 0, so screws with |yaw| > 90 deg spawned standing up and fell anywhere.)
    """
    h = np.radians(yaw_deg) / 2
    c = np.sqrt(0.5)
    return np.array([c * np.cos(h), c * np.sin(h), -c * np.cos(h), c * np.sin(h)], dtype=float)


C.SCREWS = [NOMINAL]
world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
if args.mat:
    C.MAT_TEXTURE = args.mat
if args.legacy_look:
    C.ROBOT_YELLOW = None
    cell._MATERIAL_SPEC["bin_yellow"] = ((0.820, 0.560, 0.055), 0.00, 0.62)
cell.build(stage)
cams = []
for name, path, cfg in (("table", cell.TABLE_CAM_PATH, C.TABLE_CAM),
                        ("wrist", cell.WRIST_CAM_PATH, C.WRIST_CAM),
                        ("side", cell.SIDE_CAM_PATH, C.SIDE_CAM)):
    cams.append((name, Camera(prim_path=path, resolution=cfg["resolution"])))
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
from pathlib import Path

PLATES = bg_composite.load_plates(Path(args.bg_plates)) if args.bg_plates else {}
cell.enable_ccd(world)
world.reset()
for name, cam in cams:
    cam.initialize()
    if name in PLATES or TONE:
        cam.add_distance_to_image_plane_to_frame()
for _ in range(60):
    world.step(render=True)

arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
arm.initialize()
screw = SingleRigidPrim(prim_path="/World/Screw0", name="screw0")
screw.initialize()
names = list(arm.dof_names)
ARM_IDX = [names.index(f"J{i}") for i in range(1, 7)]
IDX = {n: names.index(n) for n in names}
LIM = np.deg2rad(np.array([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)]))
TOOL_OFF = tuple(np.array(C.TOOL_AXIS) * cell.tip_offset())
HOME = np.array([0.30, 0.0, 0.24])
q_home = np.array([np.deg2rad(C.HOME_POSE_DEG[f"J{i}"]) for i in range(1, 7)])
q_home, _ = ik.solve(q_home, HOME, (0.0, 0.0, -1.0), LIM, tool_offset=TOOL_OFF)


def command(q6, jaw):
    full = arm.get_joint_positions().copy()
    for k, i in enumerate(ARM_IDX):
        full[i] = q6[k]
    for jn, v in cell.jaw_targets(jaw).items():
        full[IDX[jn]] = v
    arm.apply_action(ArticulationAction(joint_positions=full))


def screw_in_bin():
    pos = screw.get_world_pose()[0]
    bx, by = C.BIN_XY
    return (abs(pos[0] - bx) < C.BIN_SIZE[0] / 2 and abs(pos[1] - by) < C.BIN_SIZE[1] / 2), pos


def reset_scene(layout):
    full = arm.get_joint_positions().copy()
    for k, i in enumerate(ARM_IDX):
        full[i] = q_home[k]
    for jn, v in cell.jaw_targets(C.JAW_OPEN).items():
        full[IDX[jn]] = v
    arm.set_joint_positions(full)
    arm.set_joint_velocities(np.zeros_like(full))
    x, y, yaw = layout
    z = C.TABLE_TOP_Z + C.SURFACE_DZ + C.SCREW_HEAD_D / 2 + 0.0005
    screw.set_world_pose(position=np.array([x, y, z]), orientation=quat_wxyz(yaw))
    screw.set_linear_velocity(np.zeros(3))
    screw.set_angular_velocity(np.zeros(3))
    for _ in range(40):  # like the recorder: settle at home, jaws open
        command(q_home, C.JAW_OPEN)
        world.step(render=False)
    # The renderer lags the stage by several frames. Without this the first observation still shows the
    # previous seed's last pose (arm at the bin, screw inside), and the first 100-step chunk follows it.
    prev = None
    for k in range(60):
        command(q_home, C.JAW_OPEN)
        world.step(render=True)
        frame = np.concatenate([cam.get_rgba()[::8, ::8, :3].astype(np.float32).ravel() for _, cam in cams])
        if k >= 10 and prev is not None and np.abs(frame - prev).mean() < 0.5:
            break
        prev = frame


sock = socket.create_connection(("127.0.0.1", args.port))


def rpc(msg):
    payload = pickle.dumps(msg)
    sock.sendall(len(payload).to_bytes(8, "big") + payload)
    header = b""
    while len(header) < 8:
        header += sock.recv(8 - len(header))
    n = int.from_bytes(header, "big")
    buf = b""
    while len(buf) < n:
        buf += sock.recv(n - len(buf))
    return pickle.loads(buf)


results = []
for seed in args.seeds:
    layout = layout_for(seed)
    reset_scene(layout)
    rpc({"op": "reset"})
    gripper = 0.0
    frame_dir = os.path.join(args.video_dir, f"seed_{seed:03d}") if args.video_dir else None
    if frame_dir:
        os.makedirs(frame_dir, exist_ok=True)
    first_success = None
    for step in range(args.steps):
        images = {name: cam.get_rgba()[:, :, :3].astype(np.uint8) for name, cam in cams}
        for name, cam in cams:
            images[name] = to_real(name, cam, images[name])
        if frame_dir and step % args.frame_every == 0:
            tile = np.hstack([cv2.resize(images[k], (640, 360)) for k in ("table", "side", "wrist")])
            cv2.imwrite(os.path.join(frame_dir, f"{step:04d}.jpg"), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
        jp = arm.get_joint_positions()
        state = np.asarray([float(np.degrees(jp[i])) for i in ARM_IDX] + [gripper], np.float32)
        action = np.asarray(rpc({"op": "act", "images": images, "state": state, "task": TASK})["action"],
                            np.float32).reshape(-1)
        q = np.deg2rad(action[:6])
        # Like Fanuc joint mode: one step moves the fingertip at most max_speed / 10 Hz.
        q_now = np.array([jp[i] for i in ARM_IDX])
        _, tip_now = fk.tool_pose(q_now, TOOL_OFF)
        _, tip_goal = fk.tool_pose(q, TOOL_OFF)
        reach = float(np.linalg.norm(tip_goal - tip_now))
        step_m = args.max_speed / 1000.0 * 0.1
        if reach > step_m:
            q = q_now + (q - q_now) * (step_m / reach)
        if step % args.log_every == 0:
            print(f"[eval]   step {step:3d} tip ({tip_now[0]:+.3f},{tip_now[1]:+.3f},{tip_now[2]:+.3f}) "
                  f"goal ({tip_goal[0]:+.3f},{tip_goal[1]:+.3f},{tip_goal[2]:+.3f}) grip_out {action[6]:.2f} "
                  f"state {np.round(state, 1).tolist()} act {np.round(action, 1).tolist()}", flush=True)
        gripper = 1.0 if action[6] >= 0.5 else 0.0
        command(q, C.JAW_CLOSED if gripper else C.JAW_OPEN)
        for _ in range(args.substeps):
            world.step(render=True)
        if first_success is None and step % 5 == 4 and screw_in_bin()[0]:
            first_success = step + 1
    for _ in range(30):
        world.step(render=False)
    inside, pos = screw_in_bin()
    results.append({"seed": seed, "layout": list(layout), "success": bool(inside),
                    "first_in_bin_step": first_success, "screw_xyz": [float(v) for v in pos]})
    print(f"[eval] seed {seed} screw ({layout[0]:.3f},{layout[1]:.3f},{layout[2]:.1f}) "
          f"-> {'IN BIN' if inside else 'out'} at ({pos[0]:+.3f},{pos[1]:+.3f},{pos[2] - C.TABLE_TOP_Z:+.3f})"
          f"{'' if first_success is None else f', first in bin at step {first_success}'}", flush=True)

n_ok = sum(r["success"] for r in results)
print(f"[eval] success {n_ok}/{len(results)}", flush=True)
if args.out:
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
sock.close()
sim_app.close()
