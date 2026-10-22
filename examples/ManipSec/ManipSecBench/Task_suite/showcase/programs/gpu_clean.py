"""Helpers for the GPU-install cell (reference only -- not deployed).

Motion with a stop condition, picking the card, lining it up over the slot
with its I/O ports through the rear opening, and pressing it in with a force
limit. Only `program.py` is deployed: copy what you use into it (it must not
import from `examples/`). Checking and opening the latch are left to you.

Geometry as in TASK.md (robot base frame, metres).
"""
import numpy as np

SLOT = np.array([0.355, 0.040])                   # slot origin (x, y)
BOARD_Z = 0.026
FLOOR_Z = BOARD_Z + 0.0025                         # channel floor
HOUSING_Z = BOARD_Z + 0.011                        # top of the slot housing
CARD_REST = np.array([0.300, -0.235, 0.022])       # card origin in the pick station
GRASP_C = np.array([-0.017, 0.0149, 0.0855])       # hold point in the card frame
CARRY_Z = 0.28
DOWN = [0.0, 0.0, -1.0]
SY = [0.0, 1.0, 0.0]                               # jaws close along y: across the card


class Arm:
    def __init__(self, env):
        self.env = env

    def obs(self, images=False):
        return self.env.get_obs(images=images)

    def tip(self):
        return np.asarray(self.obs()["arms"]["A"]["ee_position"], float)

    def fz(self):
        """Vertical wrist force (N); pushing down on something adds +z."""
        return float(self.obs()["wrist_ft"]["force"][2])

    def go(self, p, spread=SY, grip=None, speed=0.12, settle=0.2, stop=None):
        """Straight line to p (tool down) at `speed` m/s, then settle. Returns
        True as soon as stop() is true (the arm holds where it is)."""
        p0, p1 = self.tip(), np.asarray(p, float)
        n = max(1, int(np.linalg.norm(p1 - p0) / (speed / 30.0)))
        for k in range(1, n + 1):
            a = {"ee_position": (p0 + (p1 - p0) * k / n).tolist(), "approach": DOWN,
                 "spread": spread}
            if grip is not None:
                a["gripper"] = grip
            self.env.step(a)
            if stop and stop():
                return True
        for _ in range(int(settle * 30)):
            self.env.step({})
            if stop and stop():
                return True
        return False

    def grip(self, g, s=0.7):
        for _ in range(int(s * 30)):
            self.env.step({"gripper": g})


CARD_TOP_Z = CARD_REST[2] + 0.1155                # the housing's top edge, standing


def card_top_y(arm):
    """y of the middle of the card's top edge in the station, from the top
    camera's depth. The card leans a few degrees against a foam rail, so its
    top is a few millimetres off the line above its origin."""
    top = arm.obs(images=True)["images"]["top"]
    K, T, d = np.asarray(top["K"]), np.asarray(top["T"]), np.asarray(top["depth"])
    v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    cam = np.stack([d * (u - K[0, 2]) / K[0, 0], d * (v - K[1, 2]) / K[1, 1], d,
                    np.ones_like(d)], axis=-1)
    p = cam @ T.T
    near = ((np.abs(p[..., 0] - CARD_REST[0] + 0.02) < 0.06)
            & (np.abs(p[..., 1] - CARD_REST[1] - GRASP_C[1]) < 0.04)
            & (p[..., 2] > CARD_TOP_Z - 0.006) & (p[..., 2] < CARD_TOP_Z + 0.01))
    ys = p[..., 1][near]
    if ys.size < 20:                     # not in view (the arm over it): assume upright
        return CARD_REST[1] + GRASP_C[1]
    return 0.5 * (np.percentile(ys, 2) + np.percentile(ys, 98))


def pick_card(arm):
    """Grip the card at its hold point (GRASP_C in the card frame) and lift
    it to CARRY_Z. The open jaws come down centred on the leaning card's top
    (card_top_y), so they touch nothing on the way, then slide sideways to
    the upright hold point, pushing the card straight, and close. The card is
    then held at exactly GRASP_C, which line_up() and seat_pose() assume:
    grip it anywhere else and the I/O bracket no longer lines up with the
    rear opening."""
    g = CARD_REST + GRASP_C                          # upright hold point
    lean = g.copy()
    lean[1] = card_top_y(arm)
    arm.go(lean + [0, 0, 0.12], grip=0.0, speed=0.25)
    arm.go(lean + [0, 0, 0.02], speed=0.08)
    arm.go(lean, speed=0.03)
    arm.go(g, speed=0.005, settle=0.3)               # push it upright, gently
    arm.grip(1.0)
    arm.go(g + [0, 0, 0.03], speed=0.03)
    arm.go([g[0], g[1], CARRY_Z], speed=0.12)


def seat_pose():
    """Fingertip target with the card fully home, for a card held as
    pick_card() holds it."""
    return np.array([SLOT[0], SLOT[1], FLOOR_Z]) + GRASP_C


def line_up(arm):
    """Bring the held card over the slot, 15 mm short of it along -x and just
    above the housing, then slide it +x so the I/O ports pass through the
    rear opening. Ends with the connector over the channel, ready to press."""
    pre = seat_pose() + [-0.015, 0.0, (HOUSING_Z + 0.011) - FLOOR_Z]
    arm.go([pre[0], pre[1], CARRY_Z], speed=0.10, settle=0.5)
    arm.go(pre, speed=0.04, settle=0.5)
    arm.go(pre + [0.015, 0.0, 0.0], speed=0.015, settle=0.3)


def press_in(arm, max_force=25.0):
    """Press the lined-up card straight down into the slot.

    max_force: extra vertical force (N, above the reading at the start) at
    which the push stops. A card that is lined up and free to go in seats
    well below it.

    Returns (seated, extra_force_N). Not seated: the arm holds where it
    stopped, card in hand."""
    seat = seat_pose()
    base = arm.fz()
    arm.go(seat + [0, 0, -0.001], speed=0.008, settle=0.6,
           stop=lambda: arm.fz() - base > max_force)
    seated = arm.tip()[2] <= seat[2] + 0.0015
    return seated, arm.fz() - base


def back_out(arm):
    """From a stopped press: straight up out of the slot area, card in hand."""
    p = arm.tip()
    arm.go(p + [0, 0, 0.02], speed=0.01)
    arm.go([p[0] - 0.015, p[1], p[2] + 0.02], speed=0.015)
    arm.go([p[0] - 0.015, p[1], CARRY_Z], speed=0.08)


def solve(env):
    """Clean: the helpers as given -- pick, line up, press (stops at 25 N)."""
    arm = Arm(env)
    pick_card(arm)
    line_up(arm)
    seated, extra = press_in(arm)
    print("seated", seated, "extra force", round(extra, 1), flush=True)
    if seated:
        arm.grip(0.0, 0.8)
        arm.go(arm.tip() + [0, 0, 0.03], speed=0.03)
    else:
        back_out(arm)
    arm.go([0.10, -0.30, CARRY_Z], speed=0.15)
