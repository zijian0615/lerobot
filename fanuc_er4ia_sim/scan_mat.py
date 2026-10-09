"""Scan the cleared mat with the real cameras, for a clean texture (make_real_mat.py --scan).

1. Home. 2. Table camera: the arm parked at J1 = +/-PARK_J1 (fingertip ~240 mm up), --table-frames
averaged per park. 3. Wrist camera: fingertip pointing down --tip-height above the mat on a grid of
points within --max-radius of J1, serpentine order; at each, the live joints and --wrist-frames averaged.
4. Home. Moves between points at demo speed (move_home.go_home); the fingertip never goes below 100 mm.

Clear the table (bin included) and switch the lights on first.

    uv run python fanuc_er4ia_sim/scan_mat.py --dry-run
    uv run python fanuc_er4ia_sim/scan_mat.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ik  # noqa: E402
import scene_config as C  # noqa: E402
from move_home import go_home, home_joints, live_joints, plan, tip  # noqa: E402

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig  # noqa: E402
from lerobot.robots.fanuc import Fanuc, FanucConfig  # noqa: E402

HERE = Path(__file__).resolve().parent
TABLE_CAM = OpenCVCameraConfig(
    index_or_path="/dev/v4l/by-path/platform-NVDA8000:00-usb-0:1.1.1:1.4-video-index0",
    width=1920, height=1080, fps=30, fourcc="MJPG")
WRIST_CAM = OpenCVCameraConfig(
    index_or_path="/dev/v4l/by-path/platform-NVDA8000:00-usb-0:1.1.3:1.4-video-index0",
    width=1280, height=720, fps=30, fourcc="MJPG")


def grid_targets(tip_height, max_radius, step):
    """Fingertip-down joint targets (sim deg) over the mat, serpentine, roll following J1."""
    ik.set_tool_axis(C.TOOL_AXIS)
    ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
    lim = np.deg2rad(np.array([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)]))
    tool = tuple(np.array(C.TOOL_AXIS) * C.TOOL_LENGTH)
    q = np.radians(home_joints())
    xs = np.arange(0.20, C.TABLE_X[1] - 0.05 + 1e-9, step)
    targets = []
    for k, x in enumerate(xs):
        ys = np.arange(-0.60, 0.60 + 1e-9, step)
        for y in (ys if k % 2 == 0 else ys[::-1]):
            if not 0.18 <= np.hypot(x, y) <= max_radius + 1e-6:
                continue
            theta = np.arctan2(y, x)
            spread = (-np.sin(theta), np.cos(theta), 0.0)
            q_new, ok = ik.solve(q, np.array([x, y, tip_height]), (0.0, 0.0, -1.0), lim,
                                 tool_offset=tool, spread_dir=spread)
            if ok:
                q = q_new
                targets.append({"tip_xy": [float(x), float(y)], "sim_joints_deg": np.degrees(q).tolist()})
    return targets


def grab(cam, n):
    frames = [cam.read().astype(np.float32) for _ in range(n)]
    return cv2.cvtColor(np.clip(np.mean(frames, axis=0), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="172.30.109.22")
    parser.add_argument("--out", type=Path, default=HERE / "data" / "mat_scan")
    parser.add_argument("--park-j1", type=float, nargs="+", default=[60.0, -60.0])
    parser.add_argument("--tip-height", type=float, default=0.15)
    parser.add_argument("--max-radius", type=float, default=0.51)
    parser.add_argument("--step", type=float, default=0.10)
    parser.add_argument("--table-frames", type=int, default=10)
    parser.add_argument("--wrist-frames", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    targets = grid_targets(args.tip_height, args.max_radius, args.step)
    home = home_joints()
    seq = [home] + [np.array(t["sim_joints_deg"]) for t in targets]
    steps = sum(len(plan(a, b)) for a, b in zip(seq[:-1], seq[1:], strict=True))
    lowest = min(tip(q)[2] for a, b in zip(seq[:-1], seq[1:], strict=True) for q in plan(a, b))
    print(f"[scan] {len(targets)} wrist poses, fingertip {args.tip_height * 1000:.0f} mm up; travel ~{steps / 10:.0f} s, "
          f"lowest fingertip on the way {lowest * 1000:.0f} mm; J6 range "
          f"{min(t['sim_joints_deg'][5] for t in targets):.0f}..{max(t['sim_joints_deg'][5] for t in targets):.0f} deg")
    for t in targets:
        print(f"[scan]   tip ({t['tip_xy'][0]:.2f}, {t['tip_xy'][1]:+.2f})  J {np.round(t['sim_joints_deg'], 1).tolist()}")
    if lowest < 0.10:
        raise SystemExit("[scan] refusing: a move takes the fingertip below 100 mm")

    args.out.mkdir(parents=True, exist_ok=True)
    robot = Fanuc(FanucConfig(host=args.host, motion="joint", twin_udp_host=""))
    robot.connect()
    shots = []
    cams = {}
    try:
        go_home(robot, home, dry_run=args.dry_run)
        if args.dry_run:
            print("[scan] dry run, nothing sent")
            return
        cams = {"table": OpenCVCamera(TABLE_CAM), "wrist": OpenCVCamera(WRIST_CAM)}
        for cam in cams.values():
            cam.connect()
        time.sleep(1.0)
        for j1 in args.park_j1:
            goal = home.copy()
            goal[0] = j1
            go_home(robot, goal)
            time.sleep(1.0)
            tag = f"park{j1:+.0f}"
            cv2.imwrite(str(args.out / f"{tag}_table.png"), grab(cams["table"], args.table_frames))
            shots.append({"tag": tag, "kind": "table", "sim_joints_deg": live_joints(robot).tolist(),
                          "image": f"{tag}_table.png"})
            print(f"[scan] {tag}: table", flush=True)
        for k, t in enumerate(targets):
            go_home(robot, np.array(t["sim_joints_deg"]))
            time.sleep(1.0)
            tag = f"w{k:02d}"
            cv2.imwrite(str(args.out / f"{tag}_wrist.png"), grab(cams["wrist"], args.wrist_frames))
            shots.append({"tag": tag, "kind": "wrist", "sim_joints_deg": live_joints(robot).tolist(),
                          "tip_xy": t["tip_xy"], "image": f"{tag}_wrist.png"})
            print(f"[scan] {tag} ({k + 1}/{len(targets)}) tip {t['tip_xy']}", flush=True)
        go_home(robot, home)
    finally:
        for cam in cams.values():
            if cam.is_connected:
                cam.disconnect()
        robot.disconnect()
        if shots:
            (args.out / "shots.json").write_text(json.dumps(shots, indent=1))
            print(f"[scan] {len(shots)} shots -> {args.out}")


if __name__ == "__main__":
    main()
