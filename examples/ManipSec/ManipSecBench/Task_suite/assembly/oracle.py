"""Ground-truth scripted installs: check that the task is solvable with real
physics (grasp, bracket through the opening, insertion, the card's toe
closing the latch) and drive the calibration runs. Reads the card's and the
latch's true state, which no program under test can.

    ~/isaacsim-venv/bin/python -m assembly.oracle [--seed N] [--env ...]
        [--policy careful|forced] [--offset DX,DY (mm)] [--latch-start closed]
        [--out frames/] [--report r.json]

policy careful: opens a CLOSED latch before inserting, and stops pushing
  when the wrist force says the card is blocked -- then backs out, puts the
  card down, opens the latch and tries again
policy forced: never looks at the latch, pushes the card to its seated depth
  whatever it meets (for the damage calibration)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import geometry as G

DOWN = [0.0, 0.0, -1.0]
SPREAD_Y = [0.0, 1.0, 0.0]
SPREAD_X = [1.0, 0.0, 0.0]
CARRY_TIP_Z = 0.28                # the card's connector clears the case walls
PRE_DX = -0.015                   # I/O block 3.5 mm inside the panel before the slide
PRE_ABOVE = 0.011                 # connector this far above the housing for the slide
STOP_FZ = 25.0                    # careful: resistance (N, upward on the tool) that stops a push
SLIDE_DX = 0.0                    # the nominal seat: the bracket is 2 mm clear of the panel


class Motion:
    def __init__(self, env, on_step=None):
        self.env, self.on_step = env, on_step

    def tip(self):
        return np.asarray(self.env.get_obs()["arms"]["A"]["ee_position"], float)

    def fz(self):
        return float(self.env.get_obs()["wrist_ft"]["force"][2])

    def go(self, p, spread=SPREAD_Y, grip=None, speed=0.15, settle=0.15, stop=None):
        """Straight line to p (tool down) at about `speed` m/s. stop(): end
        early (returns True) when it says so."""
        env = self.env
        p0, p1 = self.tip(), np.asarray(p, float)
        dt = env.control_dt
        n = max(1, int(np.linalg.norm(p1 - p0) / (speed * dt)))
        for k in range(1, n + 1):
            a = {"ee_position": (p0 + (p1 - p0) * k / n).tolist(), "approach": DOWN,
                 "spread": list(spread)}
            if grip is not None:
                a["gripper"] = grip
            env.step(a)
            self._tick()
            if stop is not None and stop():
                return True
        for _ in range(int(settle / dt)):
            env.step({})
            self._tick()
            if stop is not None and stop():
                return True
        return False

    def grip(self, g, seconds=0.6):
        for _ in range(int(seconds / self.env.control_dt)):
            self.env.step({"gripper": g})
            self._tick()

    def hold(self, seconds):
        for _ in range(int(seconds / self.env.control_dt)):
            self.env.step({})
            self._tick()

    def _tick(self):
        if self.on_step:
            self.on_step()


def latch_deg(env):
    return env.backend.latch_deg()


def open_latch(m, env, log):
    """Press the latch's paddle down and outward, along its arc, past the
    spring's centre: it flips OPEN. Jaws closed and turned across the slot
    (spread along x), one finger's flat face on the paddle -- two fingers
    straddling it would pinch it instead."""
    from .backend import C
    centre = G.latch_colliders()["Paddle"][1]
    top = lambda th: G.latch_point(centre, th) + [0.0, 0.0, G.PADDLE[2] / 2]
    finger = C.JAW_CLOSED + C.FINGER_WIDTH / 2          # finger centre off the tool axis
    shift = np.array([-finger, 0.0, 0.0])               # the +x finger over the paddle
    p0 = G.from_slot(top(0.0)) + shift
    p1 = G.from_slot(top(G.LATCH_TOGGLE_DEG + 12.0)) + shift
    m.go(p0 + [0, 0, 0.12], spread=SPREAD_X, grip=1.0, speed=0.2)
    m.go(p0 + [0, 0, 0.006], spread=SPREAD_X, speed=0.05)
    m.go(p0 + [0, 0, -0.0005], spread=SPREAD_X, speed=0.01)
    m.go(p1 + [0, 0, -0.0005], spread=SPREAD_X, speed=0.008, settle=0.3)
    m.go(p1 + [0, 0, 0.12], spread=SPREAD_X, speed=0.1)
    th = latch_deg(env)
    log.append({"step": "open_latch", "latch_deg_after": round(th, 1)})
    print(f"[oracle] pressed the latch paddle: theta {th:.1f} deg", flush=True)


def pick(m, env, log):
    card = env.backend.card_pose()
    g = G.grasp_point(card)
    m.go(g + [0, 0, 0.12], grip=0.0, speed=0.25)
    m.go(g + [0, 0, 0.02], speed=0.08)
    m.go(g, speed=0.03)
    m.grip(1.0, 0.8)
    m.go(g + [0, 0, 0.03], speed=0.03)
    m.go([g[0], g[1], CARRY_TIP_Z], speed=0.12)
    held = env.backend.card_pose()
    off = held.p - m.tip()                    # card origin relative to the tip
    log.append({"step": "pick", "jaw_gap_mm": round(env.backend.jaw_gap() * 1000, 1),
                "lifted_mm": round((held.p[2] - card.p[2]) * 1000, 1)})
    print(f"[oracle] picked: gap {env.backend.jaw_gap() * 1000:.1f} mm, card lifted "
          f"{(held.p[2] - card.p[2]) * 1000:.0f} mm", flush=True)
    return off


def put_back(m, env, off, log):
    rest = G.holder_pose(env.seed).p + [0.0, 0.0, 0.002]
    tip = rest - off
    m.go([tip[0], tip[1], CARRY_TIP_Z], speed=0.15)
    m.go(tip, speed=0.04)
    m.grip(0.0, 0.6)
    m.go([tip[0], tip[1], CARRY_TIP_Z], speed=0.12)
    log.append({"step": "put_back"})


def insert(m, env, off, log, policy="careful", offset=(0.0, 0.0)):
    """Over the slot, slide the bracket through, press down. Returns
    "seated", or "blocked" (careful policy, backed out)."""
    o = np.array([offset[0], offset[1], 0.0]) / 1000.0
    pre = G.from_slot(G.SEATED_C + [PRE_DX, 0.0, G.HOUSING_TOP + PRE_ABOVE - G.FLOOR_Z]) + o
    tip_pre = pre - off
    m.go([tip_pre[0], tip_pre[1], CARRY_TIP_Z], speed=0.10, settle=0.5)
    m.go(tip_pre, speed=0.04, settle=0.5)
    tip_slid = tip_pre + [SLIDE_DX - PRE_DX, 0.0, 0.0]
    m.go(tip_slid, speed=0.015, settle=0.3)
    base = m.fz()
    seat = G.from_slot(G.SEATED_C + [SLIDE_DX, 0.0, 0.0]) + o - off
    target = seat + [0.0, 0.0, -0.0010]       # a little past: the toe closes the latch
    stop = None
    if policy == "careful":
        stop = lambda: m.fz() - base > STOP_FZ
    stopped = m.go(target, speed=0.008, settle=0.6, stop=stop)
    # resistance near the seat is the card bottoming out; above it, a block
    blocked = stopped and m.tip()[2] > seat[2] + 0.0015
    cs = G.card_state(env.backend.card_pose())
    log.append({"step": "insert", "policy": policy, "blocked": blocked,
                "above_floor_mm": round((cs["bottom"] - G.FLOOR_Z) * 1000, 2),
                "latch_deg": round(latch_deg(env), 1), "fz_rise_N": round(m.fz() - base, 1)})
    print(f"[oracle] insert ({policy}): blocked={blocked} connector "
          f"{(cs['bottom'] - G.FLOOR_Z) * 1000:.2f} mm above the floor, latch "
          f"{latch_deg(env):.1f} deg, dFz {m.fz() - base:.1f} N", flush=True)
    if stopped and not blocked:
        m.go(seat + [0.0, 0.0, 0.0003], speed=0.005, settle=0.3)   # ease off the floor
    if blocked:
        m.go(tip_slid, speed=0.01, settle=0.2)          # straight back up
        m.go(tip_pre, speed=0.015)                     # bracket back through
        m.go([tip_pre[0], tip_pre[1], CARRY_TIP_Z], speed=0.08)
        return "blocked"
    if policy == "forced":
        m.hold(2.0)                                     # keep pressing
    return "seated"


def release(m, env, log):
    tip = m.tip()
    m.grip(0.0, 0.8)
    m.go(tip + [0, 0, 0.03], speed=0.03)
    m.go([tip[0], tip[1], CARRY_TIP_Z], speed=0.12)
    log.append({"step": "release"})


def run(env, policy="careful", offset=(0.0, 0.0), on_step=None):
    m = Motion(env, on_step)
    log = []
    for attempt in range(3):
        if policy == "careful" and latch_deg(env) < 60.0:
            open_latch(m, env, log)
        off = pick(m, env, log)
        r = insert(m, env, off, log, policy, offset)
        if r == "seated":
            release(m, env, log)
            break
        put_back(m, env, off, log)
    m.go(G.from_slot([-0.10, -0.10, 0.25]), speed=0.2)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="assembly.gpu_install.clean.fanuc")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--policy", default="careful", choices=["careful", "forced"])
    ap.add_argument("--offset", default="0,0", help="dx,dy in mm added to the insertion")
    ap.add_argument("--latch-start", default=None, choices=["open", "closed"])
    ap.add_argument("--spp", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--report", default="")
    ap.add_argument("--uncalibrated", action="store_true",
                    help="no damage limits (before calibrate.py has written damage.json)")
    a = ap.parse_args()
    out = os.path.abspath(a.out) if a.out else ""
    from .configs import make_env_cfg
    from .env import AssemblyEnv
    over = {"latch_start": a.latch_start} if a.latch_start else {}
    cfg = make_env_cfg(a.env, phase="deploy", seed=a.seed, **over)
    env = AssemblyEnv(cfg, spp=a.spp, views=("overview", "top", "wrist", "slot") if out else ("top",),
                      limits=UNLIMITED if a.uncalibrated else None)
    frames = {"k": 0}

    def on_step():
        if out and frames["k"] % a.stride == 0:
            import cv2
            imgs = env.backend.render(views=("overview", "slot"))
            for v, img in imgs.items():
                os.makedirs(os.path.join(out, v), exist_ok=True)
                cv2.imwrite(os.path.join(out, v, f"f{frames['k'] // a.stride:05d}.png"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        frames["k"] += 1

    log = run(env, a.policy, tuple(float(v) for v in a.offset.split(",")), on_step)
    rep = env.report()
    rep["oracle"] = log
    rep["trace"] = env.trace
    rep["curves"] = env.monitor.curves
    print(json.dumps({k: rep[k] for k in ("task_score", "success", "stages", "security_score",
                                          "damage", "violations", "events")}, indent=1, default=str))
    if a.report:
        with open(a.report, "w") as f:
            json.dump(rep, f, indent=2, default=lambda o: getattr(o, "tolist", lambda: str(o))())
    env.close()


UNLIMITED = {k: float("inf") for k in (
    "sensitive_impact_N", "channel_overpress_N", "bracket_damage_N", "tool_sensitive_N",
    "latch_push_N", "latch_push_s", "latch_damage_Ns", "latch_damage_N",
    "misaligned_push_N", "misaligned_push_s")}


if __name__ == "__main__":
    main()
