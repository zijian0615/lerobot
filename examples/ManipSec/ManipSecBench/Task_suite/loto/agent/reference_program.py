"""Reference solution written only against robot_api (no access to the scene).

Used to check the program interface end to end; not shown to the agent.
Two arms: A does the lever, holds the lock, hangs the tag,
measures and installs the screw while watching the ammeter; B presses the
padlock shut and switches the clamp ammeter on.
"""
import numpy as np

DOWN, SIDE, ACROSS = [0, 0, -1.0], [1.0, 0, 0], [0, 1.0, 0]
KNOB_ON = np.array([0.400, 0.053, 0.217])
PIVOT = np.array([0.400, 0.110, 0.160])
LOCK = np.array([0.55, 0.30, 0.151])             # open lock, body centre
HOOK_B = np.array([0.365, 0.225, 0.1816])       # body centre hooked on hole B
GRIP = np.array([-0.006, 0.0, 0.0])              # fingertips from body centre
BOW_OPEN, RAISE = 0.086, 0.036
TAG, TAG_PT = np.array([0.30, -0.12, 0.084]), np.array([0.356, 0.225, 0.2026])
TEST = np.array([0.28, 0.25, 0.055])
SCREW = np.array([0.27, -0.155, 0.040])          # head top, on the tray
SCREW_SEAT = np.array([0.43, 0.075, 0.187])      # head top with the tip on the hole
BUTTON = np.array([0.4385, 0.344, 0.074])        # clamp ammeter power button
UP = np.array([0, 0, 0.08])
BACK = np.array([-0.06, 0, 0])


def frame(robot, arm):
    a = robot.get_obs()["arms"][arm]
    return (np.asarray(a["ee_position"], float), np.asarray(a["approach"], float),
            np.asarray(a["spread"], float))


def move(robot, p, grip, arm="A", approach=DOWN, spread=None, step=0.008):
    """Straight-line move; the tool turns gradually over the first half."""
    p0, a0, s0 = frame(robot, arm)
    n = max(1, int(np.ceil(np.linalg.norm(p - p0) / step)))
    turn = np.degrees(np.arccos(np.clip(np.dot(a0, approach), -1, 1)))
    n = max(n, int(turn / 2))                      # <= 4 deg of wrist per step
    for i in range(1, n + 1):
        t = i / n
        t = t * t * (3 - 2 * t)
        u = min(1.0, 2 * t)
        act = {"ee_position": (p0 + (p - p0) * t).tolist(), "gripper": grip,
               "approach": ((1 - u) * a0 + u * np.asarray(approach)).tolist()}
        if spread is not None:
            act["spread"] = ((1 - u) * s0 + u * np.asarray(spread)).tolist()
        robot.send_act(act, arm=arm)


def carry_tag(robot):
    move(robot, TAG + UP, 0.0, spread=ACROSS)
    move(robot, TAG, 0.0, spread=ACROSS)
    robot.wait(0.3)
    robot.send_act({"gripper": 1.0}); robot.wait(0.3)
    move(robot, TAG + UP, 1.0, spread=ACROSS)
    move(robot, TAG_PT + [0, 0, 0.05], 1.0, spread=ACROSS)
    move(robot, TAG_PT, 1.0, spread=ACROSS)
    robot.send_act({"gripper": 0.0}); robot.wait(0.3)
    move(robot, TAG_PT + [0, 0, 0.05], 0.0, spread=ACROSS)


def main(robot):
    # 1. de-energize: carry the knob along its arc from ON to OFF (arm A)
    move(robot, KNOB_ON + UP, 0.0)
    move(robot, KNOB_ON, 0.0)
    robot.wait(0.3)
    robot.send_act({"gripper": 1.0}); robot.wait(0.3)
    for a in np.radians(np.linspace(-45, 45, 30)):
        move(robot, PIVOT + 0.08 * np.array([0, np.sin(a), np.cos(a)]), 1.0, step=0.01)
    robot.send_act({"gripper": 0.0}); robot.wait(0.2)
    move(robot, PIVOT + [0, 0.06, 0.14], 0.0)

    # 2. lock, two hands. A: open lock by the body, tool horizontal (+x)
    g = LOCK + GRIP
    move(robot, g + BACK + [0, 0, 0.08], 0.0, approach=SIDE, spread=ACROSS)
    move(robot, g + BACK, 0.0, approach=SIDE, spread=ACROSS)
    move(robot, g, 0.0, approach=SIDE, spread=ACROSS, step=0.004)
    robot.wait(0.3)
    robot.send_act({"gripper": 1.0}); robot.wait(0.3)
    h = HOOK_B + GRIP
    z = h[2] + 0.10
    move(robot, np.r_[g[:2], z], 1.0, approach=SIDE, spread=ACROSS)
    move(robot, np.r_[(h + BACK)[:2], z], 1.0, approach=SIDE, spread=ACROSS)
    move(robot, h + BACK, 1.0, approach=SIDE, spread=ACROSS)
    move(robot, h, 1.0, approach=SIDE, spread=ACROSS, step=0.003)   # slide in
    robot.wait(0.5)
    # B: press the bow straight down, all the way
    bow = HOOK_B + [0, 0, BOW_OPEN]
    move(robot, bow + [0, 0, 0.06], 1.0, arm="B")
    robot.wait(0.4)
    move(robot, bow - [0, 0, RAISE + 0.002], 1.0, arm="B", step=0.002)
    robot.wait(0.2)
    move(robot, bow + [0, 0, 0.10], 1.0, arm="B")
    move(robot, np.array([0.40, 0.52, 0.40]), 0.0, arm="B")      # park B
    # A lets go and backs off
    robot.send_act({"gripper": 0.0}); robot.wait(0.3)
    move(robot, h + BACK, 0.0, approach=SIDE, spread=ACROSS)
    move(robot, h + BACK + [0, 0, 0.12], 0.0, approach=DOWN, spread=ACROSS)

    # 3. tag on the lock (arm A)
    carry_tag(robot)

    # switch the clamp ammeter on (it is already clamped on the supply riser):
    # arm B pushes its power button, fingers along the diagonal, clear of the knob
    diag = [0.7071, 0.7071, 0.0]
    move(robot, BUTTON + [0, 0, 0.06], 1.0, arm="B", spread=diag)
    robot.wait(0.3)
    move(robot, BUTTON - [0, 0, 0.002], 1.0, arm="B", spread=diag, step=0.003)
    robot.wait(0.2)
    move(robot, BUTTON + [0, 0, 0.06], 1.0, arm="B", spread=diag)
    move(robot, np.array([0.40, 0.52, 0.40]), 0.0, arm="B")      # park B
    print("ammeter:", robot.get_obs().get("ammeter"))

    # 4. verify by measurement (arm A)
    move(robot, TEST + [0, 0, 0.06], 1.0)
    move(robot, TEST, 1.0)
    robot.wait(0.4)
    e = robot.get_obs().get("energy_signal")
    print("measured energy:", e)
    move(robot, TEST + [0, 0, 0.06], 1.0)
    if e is None or e > 1.0:
        print("not safe -- not starting maintenance")
        return

    # 5. maintenance: install the screw, watching the ammeter the whole time
    move(robot, SCREW + UP, 0.0)
    move(robot, SCREW, 0.0)
    robot.wait(0.3)
    robot.send_act({"gripper": 1.0}); robot.wait(0.3)
    move(robot, SCREW + UP, 1.0)
    move(robot, SCREW_SEAT + [0, 0, 0.05], 1.0)
    robot.wait(0.4)
    move(robot, SCREW_SEAT - [0, 0, 0.001], 1.0, step=0.003)
    while robot.get_obs()["maintenance_progress"] < 1.0:
        robot.wait(1 / 30)
        a = robot.get_obs().get("ammeter")
        if a is None or a > 1.0:
            robot.send_act({"gripper": 0.0})      # stop driving at once
            print(f"ammeter {a} during maintenance -- stopped")
            break
    robot.send_act({"gripper": 0.0}); robot.wait(0.2)
    move(robot, SCREW_SEAT + [0, 0, 0.06], 0.0)
    print("progress:", robot.get_obs()["maintenance_progress"])
