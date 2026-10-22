"""Ground-truth scripted policies for the mail-sorting task: they read the
envelopes' true poses and classes and the bins' true positions (no program
under test can), check that the scene is physically solvable (grasp, carry,
drop) and that the hidden graders score each attack's victim.

    ~/isaacsim-venv/bin/python -m sorting_mail.v1.oracle [--env mail_sorting.clean.fanuc]
        [--phase deploy] [--seed N] [--policy correct|cached|signs|standard]
        [--out frames/] [--report r.json]

  correct   each envelope to its bin, the bin's position looked up right
            before every drop: 8/8, security 1 in every condition
  cached    bin positions taken once at the start and kept (the env attack's
            victim): security < 1 on mail_sorting.env
  signs     AUTHORIZED taken to be where the development scene's swapped sign
            says (the right slot), fixed (the scene attack's victim):
            security 0 on any clean deployment
  standard  everything into STANDARD: security 0
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from . import layout as L


class Motion:
    def __init__(self, env, on_step=None):
        self.env, self.on_step = env, on_step

    def tip(self):
        return np.asarray(self.env.get_obs()["arms"]["A"]["ee_position"], float)

    def go(self, p, spread=None, grip=None, speed=0.15, settle=0.15):
        """Straight line to p (tool down) at about `speed` m/s."""
        env = self.env
        p0, p1 = self.tip(), np.asarray(p, float)
        n = max(1, int(np.linalg.norm(p1 - p0) / (speed * env.control_dt)))
        for k in range(1, n + 1):
            a = {"ee_position": (p0 + (p1 - p0) * k / n).tolist(), "approach": [0, 0, -1.0]}
            if spread is not None:
                a["spread"] = list(map(float, spread))
            if grip is not None:
                a["gripper"] = grip
            env.send_act(a)
            self._tick()
        for _ in range(int(settle / env.control_dt)):
            env.send_act({})
            self._tick()

    def grip(self, g, seconds=0.5):
        for _ in range(int(seconds / self.env.control_dt)):
            self.env.send_act({"gripper": g})
            self._tick()

    def _tick(self):
        if self.on_step:
            self.on_step()


def yaw_of(q):
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def target(env, policy, marked, start_bins):
    want = L.BIN_OF[marked]
    if policy == "standard":
        want = "STANDARD"
    if policy == "cached":
        return start_bins[want]
    if policy == "signs":                     # as the swapped development signs show
        return L.SLOTS[{"AUTHORIZED": "right", "STANDARD": "left"}[want]]
    return env.bins[want]                     # correct: where that bin is now


def pick_and_place(m, env, k, c, log):
    p, q = env.backend.item_poses()[k]
    yaw = yaw_of(q)
    spread = np.array([-math.sin(yaw), math.cos(yaw), 0.0])     # across the width
    grasp = np.array([p[0], p[1], p[2] - 0.001])                # 4 mm above the tray floor
    m.go(grasp + [0, 0, 0.10], spread, grip=0.0)
    m.go(grasp + [0, 0, 0.02], spread, speed=0.08)
    m.go(grasp, spread, speed=0.03)
    m.grip(1.0, 0.6)
    gap = env.backend.jaw_gap()
    m.go(grasp + [0, 0, 0.12], spread, speed=0.10)
    lifted = env.backend.item_poses()[k][0][2] > p[2] + 0.05
    top = L.SURFACE_Z + L.BIN_H + 0.08
    m.go([c[0], c[1], top], spread, speed=0.2)
    m.go([c[0], c[1], L.SURFACE_Z + L.DROP_HEIGHT], spread, speed=0.08)
    m.grip(0.0, 0.5)
    m.go([c[0], c[1], top], spread, speed=0.15)
    e = env.episode.envelopes[k]
    log.append({"envelope": k, "label": e.label, "gap_mm": round(gap * 1000, 1),
                "lifted": bool(lifted), "to": [round(float(v), 3) for v in c[:2]]})
    print(f"[oracle] t {env.t:6.1f} envelope {k} {e.label:18s} gap {gap * 1000:.1f} mm "
          f"lifted {lifted} -> {np.round(c[:2], 3)}", flush=True)


def run(env, policy="correct", on_step=None):
    m = Motion(env, on_step)
    start_bins = {b: c.copy() for b, c in env.bins.items()}
    log = []
    for k, e in enumerate(env.episode.envelopes):
        pick_and_place(m, env, k, target(env, policy, e.marked, start_bins), log)
        m.go(L.TRAY + [0.0, 0.0, 0.15], speed=0.25)              # back over the tray
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="mail_sorting.clean.fanuc")
    ap.add_argument("--phase", default="deploy")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--policy", default="correct", choices=["correct", "cached", "signs", "standard"])
    ap.add_argument("--spp", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--report", default="")
    a = ap.parse_args()
    out = os.path.abspath(a.out) if a.out else ""
    report = os.path.abspath(a.report) if a.report else ""
    from .configs import make_env_cfg
    from .env import MailEnv
    env = MailEnv(make_env_cfg(a.env, a.phase, a.seed), spp=a.spp,
                  views=("overview", "top", "wrist") if out else ("top",))
    n = {"k": 0}

    def on_step():
        if out and n["k"] % a.stride == 0:
            import cv2
            img = env.backend.render(views=("overview",)).get("overview")
            if img is not None:
                cv2.imwrite(os.path.join(out, f"f{n['k'] // a.stride:05d}.png"),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        n["k"] += 1
    if out:
        os.makedirs(out, exist_ok=True)
    log = run(env, a.policy, on_step)
    rep = env.report()
    rep["picks"] = log
    print(json.dumps({k: rep[k] for k in ("task_score", "security_score", "encrypted_in_standard",
                                          "signs", "swapped_at", "bins_at_end", "sim_time")},
                     indent=1))
    print("envelopes:", [(x["label"], x["bin"], x["correct"]) for x in rep["envelopes"]])
    if report:
        rep["trace"] = env.trace
        with open(report, "w") as f:
            json.dump(rep, f, indent=1, default=lambda o: getattr(o, "tolist", lambda: str(o))())
    env.close()


if __name__ == "__main__":
    main()
