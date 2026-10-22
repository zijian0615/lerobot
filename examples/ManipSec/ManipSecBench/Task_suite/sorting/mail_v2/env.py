"""The mail-sorting v2 environment: the robot API the program sees, on the
Isaac backend, with the evaluator (mechanics.py) fed every physics step.

Program side (robot_api.Robot, over a socket; `solve(env)` gets it):
  send_act(action) / step(action)   one control step, 1/30 s (4 physics steps)
      action: {"ee_position": [x,y,z], "approach": [...], "spread": [...],
               "vacuum": bool, "steps": n}  or  {"joint_positions": [6], ...}
      ee_position is the middle of the two cups' lips; approach the
      direction the cups face (default straight down); spread the direction
      of the cup bar (horizontal; default: as it is)
  wait(seconds)                     hold the targets (and the vacuum)
  get_obs(images=False)             arm, vacuum (on, sealed), sim time;
                                    images=True adds the top RGB-D and the
                                    wrist RGB
  Never: the envelopes' labels or poses, the bins' identities, scores.
"""

from __future__ import annotations

import numpy as np

from assembly.instrument import kinematics as K
from . import attacks
from . import layout as L
from .backend import HOME_Q, MailV2Backend
from .configs import EnvCfg
from .mechanics import MailCell

STEPS_PER_ACT = 4
TRACE_EVERY_S = 0.1
DOWN = (0.0, 0.0, -1.0)
BAR = (0.0, 1.0, 0.0)                  # the cup bar in the flange frame
PARK_Q = HOME_Q.copy()
PARK_Q[0] = np.pi                      # J1 at 180 deg: the arm behind its base, out of the top view


def tool_pose(q):
    tip, axis = K.tip(q, L.TOOL)
    return tip, axis, K._side(q, BAR)


class MailV2Env:
    def __init__(self, cfg: EnvCfg, spp=1, views=("overview", "top", "wrist"), headless=True):
        self.cfg = cfg
        self.episode = L.sample(cfg.seed)
        self.signs = attacks.signs(cfg)
        self.backend = MailV2Backend(self.episode, self.signs, headless=headless, spp=spp, views=views)
        self.backend.finalize()
        self.control_dt = self.backend.dt * STEPS_PER_ACT
        be = self.backend
        q = be.art.get_joint_positions()
        q[be.arm_idx] = PARK_Q
        be.art.set_joint_positions(q)
        for _ in range(int(be.settle_s / be.dt)):
            be.command(PARK_Q)
            be.world.step(render=False)
        be.t = 0.0
        self.q_cmd = be.joint_positions()
        self.ik_ok = True
        self.vacuum_cmd = False
        self.cell = MailCell(self.episode, cfg.bin_swap)
        self._slots = dict(self.cell.slot)
        self._shown = [None] * len(self.episode.envelopes)
        self.trace, self._next_trace = [], 0.0
        self._physics_step(0.0)

    # ---- robot API ------------------------------------------------------------------
    def send_act(self, action: dict, arm: str = "A") -> None:
        if arm != "A":
            raise ValueError("this cell has one arm, 'A'")
        if "A" in action and isinstance(action["A"], dict):
            action = action["A"]
        if "joint_positions" in action:
            q = np.clip(np.asarray(action["joint_positions"], float), K.LIMITS[:, 0], K.LIMITS[:, 1])
            self.ik_ok = True
        elif "ee_position" in action:
            spread = action.get("spread")
            ref = tool_pose(self.q_cmd)[2] if spread is None else spread
            q, ok = K.ik(self.q_cmd, action["ee_position"], action.get("approach", DOWN), tool=L.TOOL,
                         side_local=BAR, side_ref=ref, symmetric=True,
                         w_roll=0.3 if spread is not None else 0.05)
            self.ik_ok = bool(ok)
            if not ok:
                q = self.q_cmd
        else:
            q = self.q_cmd
        self.q_cmd = np.asarray(q, float)
        if "vacuum" in action:
            self.vacuum_cmd = bool(action["vacuum"])
        n = int(np.clip(action.get("steps", STEPS_PER_ACT), 1, 240))
        self.backend.step(self.q_cmd, n, self._physics_step)

    step = send_act

    def wait(self, seconds: float) -> None:
        for _ in range(max(1, int(round(float(seconds) / self.control_dt)))):
            self.send_act({})

    def get_obs(self, images: bool = False) -> dict:
        be = self.backend
        q = be.joint_positions()
        tip, axis, bar = tool_pose(q)
        vac = dict(self.cell.vacuum)
        obs = {"arms": {"A": {"ee_position": tip, "approach": axis, "spread": bar,
                              "joint_state": q, "vacuum": vac}},
               "ee_position": tip, "vacuum": vac, "sim_time": be.t, "ik_ok": self.ik_ok}
        if images:
            views = [v for v in ("top", "wrist") if v in be.cams]
            imgs = be.render(views=tuple(views), depth=True)
            out = {}
            for v in views:
                if v in imgs:
                    K_, T = be.camera_model(v)
                    out[v] = {"rgb": imgs[v], "K": K_, "T": T}
            if "top" in out and "top_depth" in imgs:
                out["top"]["depth"] = imgs["top_depth"]
            obs["images"] = out
        return obs

    # ---- physics ----------------------------------------------------------------------
    def _physics_step(self, t):
        be, c = self.backend, self.cell
        tip, axis, bar = tool_pose(be.joint_positions())
        c.update(t, tip, axis, bar, self.vacuum_cmd)
        if c.slot != self._slots:
            self._slots = dict(c.slot)
            for b in L.BINS:
                be.set_bin(b, c.bin_xy(b))
        for k, e in enumerate(self.episode.envelopes):
            pose = (round(float(e.xy[0]), 5), round(float(e.xy[1]), 5), round(e.yaw, 4), round(e.z, 5))
            if pose != self._shown[k]:
                self._shown[k] = pose
                be.set_envelope(k, e.xy, e.yaw, e.z)
        if t >= self._next_trace:
            self._next_trace = t + TRACE_EVERY_S
            self.trace.append({"t": round(t, 3), "tip": [round(float(v), 4) for v in tip],
                               "vacuum": c.vacuum["on"], "sealed": c.vacuum["sealed"], "held": c.held,
                               "slots": dict(c.slot),
                               "where": [s["where"] + (f":{s['bin']}" if s["bin"] else "") for s in c.state],
                               "leaked": sorted(c.leaked)})

    # ---- grading ----------------------------------------------------------------------
    def report(self) -> dict:
        for _ in range(int(0.5 / self.control_dt)):   # the vacuum as commanded: held stays held
            self.send_act({})
        rep = self.cell.report()
        rep.update(env=self.cfg.name, seed=self.cfg.seed, signs=self.signs,
                   sim_time=round(self.backend.t, 2), episode=self.episode.to_dict())
        self.last_report = rep
        return rep

    def close(self):
        self.backend.close()
