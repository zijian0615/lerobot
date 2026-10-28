"""Move the real ER-4iA to the training start pose through the joint-mode driver.

The target is frame 0 of the recorded episodes (sim joint convention). The path is a straight
line in joint space, cut so every 10 Hz step stays within what the demos do: fingertip <= 5 mm,
J1-J5 <= 1 deg, J6 <= 3.45 deg. Each waypoint goes through Fanuc.send_action exactly as a policy
action would (FK -> FRC_LinearMotion, joint_tcp_speed_mm_s, joint_max_step_mm).

Needs AUTO mode with the pendant disabled (RMI). Gripper is commanded open.

    uv run python fanuc_er4ia_sim/move_home.py --dry-run
    uv run python fanuc_er4ia_sim/move_home.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lerobot.robots.fanuc import Fanuc, FanucConfig
from lerobot.robots.fanuc.er4ia_fk import TOOL_TIP_M, sim_from_controller_deg, tool_pose

HERE = Path(__file__).resolve().parent
STEP_TIP_M = 0.005
STEP_JOINT_DEG = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 3.45])
MIN_TIP_Z_M = 0.100  # above the table / base mounting face
NAMES = [f"J{i}.pos" for i in range(1, 7)]


def tip(q_deg):
    return tool_pose(np.radians(q_deg), TOOL_TIP_M)[1]


def plan(start, goal):
    n = int(np.ceil(max(np.max(np.abs(goal - start) / STEP_JOINT_DEG),
                        np.linalg.norm(tip(goal) - tip(start)) / STEP_TIP_M, 1.0)))
    while True:  # joint-space lines bend in Cartesian space; refine until every step fits
        path = [start + (goal - start) * k / n for k in range(1, n + 1)]
        steps = np.linalg.norm(np.diff([tip(start)] + [tip(q) for q in path], axis=0), axis=1)
        if steps.max() <= STEP_TIP_M:
            return path
        n = int(np.ceil(n * steps.max() / STEP_TIP_M)) + 1


def home_joints(episode: Path = HERE / "data" / "screw_v4" / "episode_00" / "frames.json") -> np.ndarray:
    """Training start pose (sim joint convention): frame 0 of a recorded episode."""
    return np.array(json.loads(Path(episode).read_text())["frames"][0]["joints_deg"], float)


def live_joints(robot: Fanuc) -> np.ndarray:
    # The controller's joint stream, not get_observation(): homing must not depend on the cameras.
    if robot._latest_joints is not None:
        return np.array(sim_from_controller_deg(robot._latest_joints), float)
    obs = robot.get_observation()
    return np.array([obs[n] for n in NAMES], float)


def go_home(robot: Fanuc, goal: np.ndarray | None = None, dry_run: bool = False, timeout_s: float = 60.0) -> None:
    """Drive a connected joint-mode Fanuc to the training start pose at demo speed. Gripper open."""
    goal = home_joints() if goal is None else goal
    start = live_joints(robot)
    path = plan(start, goal)
    lowest = min(tip(q)[2] for q in path)
    print(f"[home] start (sim deg) {np.round(start, 2).tolist()} tip {np.round(tip(start) * 1000, 1).tolist()} mm")
    print(f"[home] goal  (sim deg) {np.round(goal, 2).tolist()} tip {np.round(tip(goal) * 1000, 1).tolist()} mm")
    print(f"[home] {len(path)} steps at 10 Hz (~{len(path) / 10:.1f} s), lowest fingertip {lowest * 1000:.0f} mm "
          f"above the table, speed {robot.config.joint_tcp_speed_mm_s} mm/s")
    floor = min(MIN_TIP_Z_M, tip(start)[2]) - 0.005  # starting low (after a fault) is fine; going lower is not
    if lowest < floor:
        raise SystemExit(f"[home] refusing: the path takes the fingertip below {floor * 1000:.0f} mm")
    if dry_run:
        print("[home] dry run, nothing sent")
        return
    t_start = t_next = time.perf_counter()
    for q in path + [goal] * 1000:
        robot.send_action({**{n: float(v) for n, v in zip(NAMES, q, strict=True)}, "gripper.pos": 0.0})
        t_next += 0.1
        time.sleep(max(0.0, t_next - time.perf_counter()))
        now = live_joints(robot)
        if getattr(robot, "reject_streak", 0) >= 10:
            raise SystemExit(f"[home] the controller refused {robot.reject_streak} motions in a row "
                             f"(last {robot._last_ack}): clear the pendant (RESET, STEP off), then recover")
        if q is goal and np.max(np.abs(now - goal)) < 0.3 and np.linalg.norm(tip(now) - tip(goal)) < 0.002:
            break
        if time.perf_counter() - t_start > len(path) / 10 + timeout_s:
            raise SystemExit(f"[home] timeout, last joints {np.round(now, 2).tolist()}")
    print(f"[home] reached: joints {np.round(now, 2).tolist()}, tip {np.round(tip(now) * 1000, 1).tolist()} mm, "
          f"error {np.max(np.abs(now - goal)):.2f} deg / {np.linalg.norm(tip(now) - tip(goal)) * 1000:.1f} mm")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="172.30.109.22")
    parser.add_argument("--episode", type=Path, default=HERE / "data" / "screw_v4" / "episode_00" / "frames.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    args = parser.parse_args()

    robot = Fanuc(FanucConfig(host=args.host, motion="joint", twin_udp_host=""))
    robot.connect()
    try:
        go_home(robot, home_joints(args.episode), args.dry_run, args.timeout_s)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
