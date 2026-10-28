"""Photograph the real background for the fixed cameras (table, side) with the arm parked aside.

Goes home, swings J1 to each --j1 angle (everything else at the home pose, fingertip ~240 mm up),
photographs table and side with the rollout camera settings, records the joints, and returns home.
make_bg_plates.py then merges the shots into one arm-free plate per camera.

    uv run python fanuc_er4ia_sim/capture_bg_plates.py --dry-run
    uv run python fanuc_er4ia_sim/capture_bg_plates.py
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
from move_home import go_home, home_joints, live_joints  # noqa: E402

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig  # noqa: E402
from lerobot.robots.fanuc import Fanuc, FanucConfig  # noqa: E402

HERE = Path(__file__).resolve().parent
CAMERAS = {
    "table": OpenCVCameraConfig(
        index_or_path="/dev/v4l/by-path/platform-NVDA8000:00-usb-0:1.2.4.2:1.4-video-index0",
        width=1920, height=1080, fps=30, fourcc="MJPG"),
    "side": OpenCVCameraConfig(
        index_or_path="/dev/v4l/by-id/usb-SHENZHEN_AONI_ELECTRONIC_CO.__LTD_Full_HD_webcam_20200729001-video-index0",
        width=1920, height=1080, fps=30, fourcc="MJPG", exposure=115),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="172.30.109.22")
    parser.add_argument("--j1", type=float, nargs="+", default=[60.0, -60.0])
    parser.add_argument("--out", type=Path, default=HERE / "data" / "bg_plates")
    parser.add_argument("--frames", type=int, default=8, help="frames averaged per shot (sensor noise)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    home = home_joints()
    robot = Fanuc(FanucConfig(host=args.host, motion="joint", twin_udp_host=""))
    robot.connect()
    shots = []
    try:
        go_home(robot, home, dry_run=args.dry_run)
        for j1 in args.j1:
            goal = home.copy()
            goal[0] = j1
            go_home(robot, goal, dry_run=args.dry_run)
            if args.dry_run:
                continue
            time.sleep(1.0)
            joints = live_joints(robot)
            shot = {"j1": j1, "sim_joints_deg": joints.tolist(), "images": {}}
            for name, cfg in CAMERAS.items():
                cam = OpenCVCamera(cfg)
                cam.connect()
                time.sleep(1.0)
                frames = [cam.read().astype(np.float32) for _ in range(args.frames)]
                cam.disconnect()
                img = np.clip(np.mean(frames, axis=0), 0, 255).astype(np.uint8)
                path = args.out / f"j1_{j1:+.0f}_{name}.png"
                cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                shot["images"][name] = path.name
                print(f"[plates] J1 {j1:+.0f}: {name} -> {path.name}", flush=True)
            json.dump({"sim_joints_deg": joints.tolist()}, open(args.out / f"j1_{j1:+.0f}_pose.json", "w"))
            shots.append(shot)
        go_home(robot, home, dry_run=args.dry_run)
    finally:
        robot.disconnect()
        if shots:
            (args.out / "shots.json").write_text(json.dumps(shots, indent=1))


if __name__ == "__main__":
    main()
