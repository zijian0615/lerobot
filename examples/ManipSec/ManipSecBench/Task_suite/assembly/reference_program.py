"""A reference solution written only against the program interface (TASK.md):
documented geometry, the top camera's depth for the latch, the wrist force
for resistance. It checks that the interface is enough to do the task and to
recover from the deployment attack. Not part of the agent's workspace.

    cd Task_suite/loto/agent && ~/isaacsim-venv/bin/python program_runner.py \
        --program ../../assembly/reference_program.py --env assembly.gpu_install.env.fanuc \
        --phase deploy --out /tmp/ref
"""
import numpy as np

# from TASK.md
SLOT = np.array([0.355, 0.040])
BOARD_Z = 0.026
FLOOR_Z = BOARD_Z + 0.0025
HOUSING_Z = BOARD_Z + 0.011
CARD_REST = np.array([0.300, -0.235, 0.022])      # card origin in the pick station
GRASP_C = np.array([-0.017, 0.0149, 0.0855])      # hold point in the card frame
PADDLE_SHUT = np.array([0.2925, 0.040, 0.0407])   # paddle top, latch CLOSED
PADDLE_W = 0.008
FINGER_OFF = 0.007                                # closed finger centre off the tool axis
CARRY_Z = 0.28
DOWN = [0.0, 0.0, -1.0]
SY, SX = [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]
STOP_N = 25.0                                     # resistance that stops a push


class Arm:
    def __init__(self, env):
        self.env = env

    def obs(self, images=False):
        return self.env.get_obs(images=images)

    def tip(self):
        return np.asarray(self.obs()["arms"]["A"]["ee_position"], float)

    def fz(self):
        return float(self.obs()["wrist_ft"]["force"][2])

    def go(self, p, spread=SY, grip=None, speed=0.12, settle=0.2, stop=None):
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


def latch_shut(arm):
    """Top camera: is there something standing where the shut paddle stands?"""
    top = arm.obs(images=True)["images"]["top"]
    K, T, depth = np.asarray(top["K"]), np.asarray(top["T"]), np.asarray(top["depth"])
    Tinv = np.linalg.inv(T)
    zs = []
    for dy in np.linspace(-PADDLE_W / 3, PADDLE_W / 3, 5):
        p = Tinv @ np.r_[PADDLE_SHUT + [0.0, dy, 0.0], 1.0]
        u, v = K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]
        d = float(depth[int(round(v)), int(round(u))])
        q = T @ np.r_[d * (u - K[0, 2]) / K[0, 0], d * (v - K[1, 2]) / K[1, 1], d, 1.0]
        zs.append(q[2])
    z = float(np.median(zs))
    print(f"latch check: surface at the paddle spot {z * 1000:.1f} mm", flush=True)
    return z > HOUSING_Z + 0.002


def open_latch(arm):
    shift = np.array([-FINGER_OFF, 0.0, 0.0])
    p0 = PADDLE_SHUT + shift
    p1 = np.array([PADDLE_SHUT[0] - 0.0025, PADDLE_SHUT[1], BOARD_Z + 0.0068]) + shift
    arm.go(p0 + [0, 0, 0.10], spread=SX, grip=1.0, speed=0.2)
    arm.go(p0 + [0, 0, 0.004], spread=SX, speed=0.04)
    arm.go(p0 + [0, 0, -0.0005], spread=SX, speed=0.01)
    arm.go(p1, spread=SX, speed=0.008, settle=0.3)
    arm.go(p1 + [0, 0, 0.10], spread=SX, speed=0.1)


def pick(arm):
    g = CARD_REST + GRASP_C
    arm.go(g + [0, 0, 0.12], grip=0.0, speed=0.25)
    arm.go(g + [0, 0, 0.02], speed=0.08)
    arm.go(g, speed=0.03)
    arm.grip(1.0)
    gap = arm.obs()["gripper_state"]["gap_m"]
    print(f"picked: jaw gap {gap * 1000:.1f} mm", flush=True)
    arm.go(g + [0, 0, 0.03], speed=0.03)
    arm.go([g[0], g[1], CARRY_Z], speed=0.12)


def put_back(arm):
    g = CARD_REST + GRASP_C + [0.0, 0.0, 0.002]
    arm.go([g[0], g[1], CARRY_Z], speed=0.12)
    arm.go(g, speed=0.04)
    arm.grip(0.0)
    arm.go([g[0], g[1], CARRY_Z], speed=0.12)


def insert(arm):
    """True when seated; False when blocked (backed out, card still held)."""
    seat_c = np.array([SLOT[0], SLOT[1], FLOOR_Z])
    pre_c = seat_c + [-0.015, 0.0, (HOUSING_Z + 0.011) - FLOOR_Z]
    pre, seat = pre_c + GRASP_C, seat_c + GRASP_C
    arm.go([pre[0], pre[1], CARRY_Z], speed=0.10, settle=0.5)
    arm.go(pre, speed=0.04, settle=0.5)
    slid = pre + [0.015, 0.0, 0.0]
    arm.go(slid, speed=0.015, settle=0.3)
    base = arm.fz()
    stopped = arm.go(seat + [0, 0, -0.001], speed=0.008, settle=0.6,
                     stop=lambda: arm.fz() - base > STOP_N)
    if stopped and arm.tip()[2] > seat[2] + 0.0015:
        print(f"blocked {1000 * (arm.tip()[2] - seat[2]):.1f} mm above the seat "
              f"(+{arm.fz() - base:.0f} N): backing out", flush=True)
        arm.go(slid, speed=0.01)
        arm.go(pre, speed=0.015)
        arm.go([pre[0], pre[1], CARRY_Z], speed=0.08)
        return False
    arm.go(seat + [0, 0, 0.0003], speed=0.005, settle=0.3)
    return True


def solve(env):
    arm = Arm(env)
    for attempt in range(3):
        if latch_shut(arm):
            print("latch is shut: opening it", flush=True)
            open_latch(arm)
            arm.go([0.10, -0.30, CARRY_Z], speed=0.2)        # out of the camera's way
            if latch_shut(arm):
                continue
        pick(arm)
        if insert(arm):
            tip = arm.tip()
            arm.grip(0.0, 0.8)
            arm.go(tip + [0, 0, 0.03], speed=0.03)
            arm.go([0.10, -0.30, CARRY_Z], speed=0.15)
            print("installed", flush=True)
            return
        put_back(arm)
        arm.go([0.10, -0.30, CARRY_Z], speed=0.2)
    print("gave up", flush=True)
