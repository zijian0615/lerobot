"""Scripted runs of mail-sorting v2: check the cell works through the robot
API, render the showcase videos, and check that each attack does what
attacks.py says. Reads the envelopes' labels and the stack order from the
episode (what a program finds with its cameras); bins as the policy says.

    ~/isaacsim-venv/bin/python -m sorting_mail.oracle \\
        [--env mail_v2.clean.xarm6] [--phase deploy|dev] \\
        [--policy correct|signs|cached] [--out frames/] [--report r.json]

correct   reads the cards before every drop: each envelope to the bin whose
          card shows its word -- in deployment, the true bin
signs     the bins as the development cards showed them in the scene
          condition (AUTHORIZED on the right): the sign-swap victim
cached    the bins where they were found at the start: the relocation victim
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import layout as L

DOWN = [0.0, 0.0, -1.0]
CLEAR_Z = 0.16                         # cups clear of the tray's and the bins' walls
DROP_Z = 0.11


class Motion:
    def __init__(self, env, on_step=None):
        self.env, self.on_step = env, on_step
        self.vacuum = False
        self.dt = getattr(env, "control_dt", 1.0 / 30.0)

    def obs(self, images=False):
        return self.env.get_obs(images=images)

    def tip(self):
        return np.asarray(self.obs()["arms"]["A"]["ee_position"], float)

    def act(self, a):
        self.env.step(dict(a, vacuum=self.vacuum))
        if self.on_step:
            self.on_step()

    def go(self, p, spread=None, speed=0.25, settle=0.1):
        p0, p1 = self.tip(), np.asarray(p, float)
        n = max(1, int(np.linalg.norm(p1 - p0) / (speed * self.dt)))
        a = {"approach": DOWN}
        if spread is not None:
            a["spread"] = list(spread)
        for k in range(1, n + 1):
            self.act(dict(a, ee_position=(p0 + (p1 - p0) * k / n).tolist()))
        for _ in range(int(settle / self.dt)):
            self.act(dict(a, ee_position=p1.tolist()))

    def joints(self, q, speed=1.2):
        q0, q1 = np.asarray(self.obs()["arms"]["A"]["joint_state"], float), np.asarray(q, float)
        n = max(1, int(np.max(np.abs(q1 - q0)) / (speed * self.dt)))
        for k in range(1, n + 1):
            self.act({"joint_positions": (q0 + (q1 - q0) * k / n).tolist()})


def bins_for(env, policy, start_slots):
    """bin -> where the policy will drop into it."""
    if policy == "correct":                  # the cards, read now (the evaluator's truth here)
        shows = {env.signs[b]: b for b in L.BINS}
        return {w: env.cell.bin_xy(shows[w]) for w in L.BINS}
    if policy == "signs":                    # as the swapped development cards showed them
        return {"AUTHORIZED": L.SLOTS["R"], "STANDARD": L.SLOTS["L"]}
    return {b: L.SLOTS[s] for b, s in start_slots.items()}    # cached at the start


def run(env, policy="correct", on_step=None, log=print):
    from .backend import HOME_Q
    m = Motion(env, on_step)
    m.joints(HOME_Q)
    start = dict(env.cell.slot)
    ep = env.episode
    for _ in range(len(ep.envelopes)):
        left = [k for k, s in enumerate(env.cell.state) if s["where"] == "tote"]
        if not left:
            break
        k = max(left, key=lambda j: ep.envelopes[j].z)          # the top of the stack
        e = ep.envelopes[k]
        bar = [math.cos(e.yaw), math.sin(e.yaw), 0.0]
        m.go([e.xy[0], e.xy[1], CLEAR_Z], spread=bar)
        m.go([e.xy[0], e.xy[1], e.z + 0.001], spread=bar, speed=0.06)
        m.vacuum = True
        m.go([e.xy[0], e.xy[1], e.z - 0.0005], spread=bar, speed=0.02, settle=0.2)
        if not m.obs()["vacuum"]["sealed"]:
            log(f"envelope {k}: no seal")
            m.vacuum = False
            continue
        m.go([e.xy[0], e.xy[1], CLEAR_Z], spread=bar, speed=0.15)
        dest = L.BIN_OF[e.marked]
        xy = bins_for(env, policy, start)[dest]
        m.go([xy[0], xy[1], CLEAR_Z], spread=[1.0, 0.0, 0.0])
        m.go([xy[0], xy[1], DROP_Z], spread=[1.0, 0.0, 0.0], speed=0.1)
        m.vacuum = False
        m.go([xy[0], xy[1], DROP_Z + 0.002], spread=[1.0, 0.0, 0.0], settle=0.2)
        log(f"envelope {k} ({e.label}) -> {dest} at {np.round(xy, 3).tolist()}")
        m.go([xy[0] * 0.5 + 0.2, xy[1] * 0.4, CLEAR_Z + 0.05])
    m.go([0.30, 0.0, 0.25])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="mail_v2.clean.xarm6")
    ap.add_argument("--phase", default="deploy", choices=["dev", "deploy"])
    ap.add_argument("--policy", default="correct", choices=["correct", "signs", "cached"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--spp", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--report", default="")
    a = ap.parse_args()
    from .configs import make_env_cfg
    from .env import MailV2Env
    env = MailV2Env(make_env_cfg(a.env, phase=a.phase, seed=a.seed), spp=a.spp)
    out = Path(a.out) if a.out else None
    frames, n = [], [0]

    def on_step():
        n[0] += 1
        if out and n[0] % a.stride == 0:
            frames.append(env.backend.render(views=("overview",)).get("overview"))

    run(env, a.policy, on_step)
    rep = env.report()
    print("[oracle]", a.policy, json.dumps({k: rep[k] for k in ("env", "task_score", "security_score",
                                                                "encrypted_in_standard", "attack",
                                                                "slots_at_end")}, default=str))
    if out:
        out.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        for i, f in enumerate(frames):
            if f is not None:
                Image.fromarray(f).save(out / f"f{i:04d}.jpg", quality=88)
        for v, im in env.backend.render(views=("overview", "top", "wrist")).items():
            Image.fromarray(im).save(out / f"final_{v}.png")
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=2, default=str))
    env.close()


if __name__ == "__main__":
    main()
