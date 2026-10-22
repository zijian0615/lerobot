"""Example: the shape of a sorting program (reference only -- not deployed).

Shows how to read the cameras, back-project the top camera's depth, move the
arm in small steps, pick a part lying on its side and drop it over a bin.
Detecting the parts, telling their types apart and finding the bins are left
as TODOs: this is a starting point, not a solution.
"""
import numpy as np

# which bin each part type goes to
TARGET_BIN = {
    "battery": "BATTERY",
    "steel_pin": "METAL",
    "plastic_sleeve": "PLASTIC",
}

DOWN = [0.0, 0.0, -1.0]
HOME_Q = np.array([0.0, 0.611, -0.436, 0.0, -1.396, 0.0])     # start pose (TASK.md)
PARK_Q = np.array([2.793, 0.611, -0.436, 0.0, -1.396, 0.0])   # behind the base (TASK.md)


def tip(robot):
    return np.asarray(robot.get_obs()["arms"]["A"]["ee_position"], float)


def move(robot, target, spread=None, gripper=None, step=0.008):
    """Straight line to target, tool pointing down, <= `step` m per control step."""
    p0, p1 = tip(robot), np.asarray(target, float)
    n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / step)))
    for k in range(1, n + 1):
        act = {"ee_position": (p0 + (p1 - p0) * k / n).tolist(), "approach": DOWN}
        if spread is not None:
            act["spread"] = list(spread)
        if gripper is not None:
            act["gripper"] = gripper
        robot.send_act(act)
    robot.wait(0.3)


def joint_move(robot, q1, steps=90):
    """Interpolated move in joint space (for the big swings to and from PARK_Q:
    a straight fingertip line between them would pass through the base)."""
    q0 = np.asarray(robot.get_obs()["arms"]["A"]["joint_state"], float)
    for k in range(1, steps + 1):
        robot.send_act({"joint_positions": (q0 + (np.asarray(q1) - q0) * k / steps).tolist()})
    robot.wait(0.5)


def park(robot):
    """Out of the top camera's view."""
    joint_move(robot, PARK_Q)


def back_project(cam, u, v):
    """Robot-frame point seen at pixel (u, v) of the top camera."""
    d = float(cam["depth"][int(v), int(u)])
    K, T = cam["K"], cam["T"]
    x = (u - K[0, 2]) * d / K[0, 0]
    y = (v - K[1, 2]) * d / K[1, 1]
    return (T @ np.array([x, y, d, 1.0]))[:3]


def detect_parts(top):
    """TODO: find the parts in the tray: centre (robot frame) and axis angle."""
    raise NotImplementedError


def find_bins(top):
    """TODO: read the signs: {"BATTERY": centre, "METAL": centre, "PLASTIC": centre}."""
    raise NotImplementedError


def classify(robot, part):
    """TODO: decide the part's type ("battery", "steel_pin", "plastic_sleeve").
    docs/parts_sheet.md describes how the parts differ."""
    raise NotImplementedError


def pick(robot, centre, axis_angle):
    spread = [-np.sin(axis_angle), np.cos(axis_angle), 0.0]   # jaws across the axis
    grasp = np.asarray(centre, float) - [0.0, 0.0, 0.003]
    move(robot, grasp + [0, 0, 0.10], spread, gripper=0.0)
    move(robot, grasp, spread, step=0.003)
    for _ in range(15):
        robot.send_act({"gripper": 1.0})
    move(robot, grasp + [0, 0, 0.12], spread)
    return robot.get_obs()["arms"]["A"]["gripper_state"]["gap_m"] > 0.004


def drop(robot, bin_centre):
    above = np.array([bin_centre[0], bin_centre[1], 0.15])
    move(robot, above)
    move(robot, above - [0, 0, 0.04])
    for _ in range(15):
        robot.send_act({"gripper": 0.0})
    move(robot, above)


def main(robot):
    park(robot)
    top = robot.get_obs(images=True)["images"]["top"]
    bins = find_bins(top)
    for part in detect_parts(top):
        kind = classify(robot, part)
        joint_move(robot, HOME_Q)                     # back over the work area
        if pick(robot, part["centre"], part["angle"]):
            drop(robot, bins[TARGET_BIN[kind]])
        park(robot)
