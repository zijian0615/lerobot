"""Replay a recorded sim demo on the real ER-4iA, through the same joint-mode driver a policy uses.

First drives to the training start pose (move_home.go_home), then sends the episode's actions
(each frame's next joint state + gripper, i.e. the training labels) at 10 Hz. Every tick logs the
target and the live joints / fingertip, so tracking lag and the grasp height can be read afterwards.

Put the screw where the episode had it (frames.json "screw_layout": x, y of the head, yaw head->tip)
before running. Needs AUTO mode with the pendant disabled.

    uv run python fanuc_er4ia_sim/replay_episode.py --dry-run
    uv run python fanuc_er4ia_sim/replay_episode.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from move_home import NAMES, go_home, live_joints, tip  # noqa: E402

import scene_config as C  # noqa: E402

from lerobot.robots.fanuc import Fanuc, FanucConfig  # noqa: E402

HERE = Path(__file__).resolve().parent
# Never closer to the work surface than this. The surface is C.SURFACE_DZ in the base frame (measured
# 5.1 mm below the base plane); demos grasp TIP_CLEARANCE (2.5 mm) above it.
MIN_TIP_CLEARANCE_M = 0.001


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="172.30.109.22")
    parser.add_argument("--episode", type=Path, default=HERE / "data" / "screw_v4" / "episode_00" / "frames.json")
    parser.add_argument("--out", type=Path, default=HERE / "data" / "replay")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hold-s", type=float, default=3.0, help="keep sending the last action this long")
    parser.add_argument("--fps", type=float, default=None,
                        help="send rate; default the episode's. Lower it when the arm cannot keep the demo pace")
    args = parser.parse_args()

    meta = json.loads(args.episode.read_text())
    frames = meta["frames"]
    targets = [(np.array(f["joints_deg"], float), float(f["gripper"])) for f in frames[1:]]
    lowest = min(tip(q)[2] for q, _ in targets)
    closes = [i for i in range(1, len(targets)) if targets[i][1] >= 0.5 > targets[i - 1][1]]
    fps = args.fps or meta["fps"]
    print(f"[replay] {args.episode.parent.name}: {len(targets)} actions at {fps} Hz (~{len(targets) / fps:.0f} s), "
          f"screw layout {meta.get('screw_layout', meta.get('object_layout'))}")
    print(f"[replay] lowest fingertip target {(lowest - C.SURFACE_DZ) * 1000:.1f} mm above the work surface "
          f"({lowest * 1000:.1f} mm in the base frame); gripper closes at step "
          f"{closes} with the tip at {[np.round(tip(targets[i][0]) * 1000, 1).tolist() for i in closes]} mm")
    if lowest - C.SURFACE_DZ < MIN_TIP_CLEARANCE_M:
        raise SystemExit(f"[replay] refusing: a target takes the fingertip within {MIN_TIP_CLEARANCE_M * 1000:.0f} mm "
                         "of the work surface")

    robot = Fanuc(FanucConfig(host=args.host, motion="joint", twin_udp_host=""))
    robot.connect()
    log = []
    try:
        go_home(robot, np.array(frames[0]["joints_deg"], float), dry_run=args.dry_run)
        if args.dry_run:
            print("[replay] dry run, nothing sent")
            return
        dt = 1.0 / (args.fps or meta["fps"])
        hold = [targets[-1]] * int(round(args.hold_s / dt))
        t0 = t_next = time.perf_counter()
        for step, (q, grip) in enumerate(targets + hold):
            robot.send_action({**{n: float(v) for n, v in zip(NAMES, q, strict=True)}, "gripper.pos": grip})
            t_next += dt
            time.sleep(max(0.0, t_next - time.perf_counter()))
            now = live_joints(robot)
            lag = float(np.linalg.norm(tip(now) - tip(q)) * 1000)
            log.append({"t": time.perf_counter() - t0, "step": step, "target_deg": q.tolist(), "gripper": grip,
                        "live_deg": now.tolist(), "target_tip_mm": (tip(q) * 1000).tolist(),
                        "live_tip_mm": (tip(now) * 1000).tolist(), "lag_mm": lag})
            if step % 25 == 0 or step in closes:
                print(f"[replay] step {step:3d} {frames[min(step + 1, len(frames) - 1)]['segment']:<8} grip {grip:.0f} "
                      f"tip target {np.round(tip(q) * 1000, 0).tolist()} live {np.round(tip(now) * 1000, 0).tolist()} "
                      f"lag {lag:.1f} mm", flush=True)
    finally:
        robot.disconnect()
        if log:
            args.out.mkdir(parents=True, exist_ok=True)
            path = args.out / f"replay_{time.strftime('%Y%m%d_%H%M%S')}.json"
            path.write_text(json.dumps({"episode": str(args.episode), "screw_layout": meta.get('screw_layout', meta.get('object_layout')), "log": log}))
            lags = np.array([e["lag_mm"] for e in log[: len(targets)]])
            at_close = [log[i]["live_tip_mm"] for i in closes if i < len(log)]
            print(f"[replay] tracking lag median {np.median(lags):.1f} mm, p90 {np.percentile(lags, 90):.1f}, "
                  f"max {lags.max():.1f}; live tip when the gripper closed {np.round(at_close, 1).tolist()} mm -> {path}")


if __name__ == "__main__":
    main()
