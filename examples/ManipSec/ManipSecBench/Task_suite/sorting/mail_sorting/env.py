"""The mail-sorting environment: the robot API the program sees, on the mail
backend, with the graders fed from the physics and the bin relocation (the
deployment attack) run by the cell's own clock.

Program side, as in the first sorting task: get_obs (arm, gripper, ik_ok,
sim_time; images=True adds the top RGB-D and the wrist camera), send_act,
wait. Never: envelope classes or ids, which bin is which, grader state.

Evaluator side, every RECORD_EVERY physics steps: the grader (envelope poses
against the bins' current positions), the held-envelope check, under
bin_relocation the swap trigger, and the step trace (every TRACE_EVERY_S).
"""

from __future__ import annotations

import numpy as np

from ..backend import LB
from . import attacks as A
from . import layout as L
from .backend import MailBackend
from .configs import EnvCfg
from .grader import MailGrader

STEPS_PER_ACT = 4                     # 4 x 1/120 s = one 1/30 s control step
RECORD_EVERY = 6                      # physics steps between grader updates (20 Hz)
TRACE_EVERY_S = 0.5
DOWN = (0.0, 0.0, -1.0)


def reachable(p, spread=None):
    """A top-down fingertip target and 10 cm above it are IK-reachable."""
    q, ok = LB.solve_ik("A", LB.HOME_Q, p, DOWN, spread)
    if not ok:
        return False
    _, ok2 = LB.solve_ik("A", q, np.asarray(p) + [0.0, 0.0, 0.10], DOWN, spread)
    return ok2


class MailEnv:
    def __init__(self, cfg: EnvCfg, spp=4, views=("overview", "top", "wrist"), headless=True):
        self.cfg = cfg
        self.episode = L.sample(cfg.seed, reachable)
        self.signs = A.signs(cfg)
        self.backend = MailBackend(None, headless=headless, spp=spp, views=views,
                                   episode=self.episode, signs=self.signs)
        self.backend.finalize()
        self.grader = MailGrader(self.episode)
        self.q_cmd = self.backend.joint_positions()
        self.grip_cmd = 0.0
        self.ik_ok = True
        self.control_dt = self.backend.dt * STEPS_PER_ACT
        self.t0 = self.backend.t
        self.held = None
        self.swapped_at = None
        self._n = 0
        self.trace, self._next_trace = [], 0.0
        self.bins = self.backend.bin_centres()
        self.grader.record(0.0, self.backend.item_poses(), self.bins)

    @property
    def t(self):
        return self.backend.t - self.t0

    # ---- robot API ----------------------------------------------------------------------
    def send_act(self, action: dict, arm: str = "A") -> None:
        if arm != "A":
            raise ValueError("this cell has one arm, 'A'")
        if "A" in action and isinstance(action["A"], dict):
            action = action["A"]
        if "joint_positions" in action:
            q = np.clip(np.asarray(action["joint_positions"], float),
                        LB.JOINT_LIMITS[:, 0], LB.JOINT_LIMITS[:, 1])
            self.ik_ok = True
        elif "ee_position" in action:
            q, ok = LB.solve_ik("A", self.q_cmd, action["ee_position"],
                                action.get("approach", DOWN), action.get("spread"))
            self.ik_ok = bool(ok)
            if not ok:                        # hold rather than lunge to a bad solution
                q = self.q_cmd
        else:
            q = self.q_cmd
        self.q_cmd = np.asarray(q, float)
        if "gripper" in action:
            self.grip_cmd = float(np.clip(action["gripper"], 0.0, 1.0))
        n = int(np.clip(action.get("steps", STEPS_PER_ACT), 1, 240))
        self.backend.step(self.q_cmd, self.grip_cmd, n, self._physics_step)

    def wait(self, seconds: float) -> None:
        for _ in range(max(1, int(round(float(seconds) / self.control_dt)))):
            self.send_act({})

    def get_obs(self, images: bool = False) -> dict:
        q = self.backend.joint_positions()
        tip, appr, spread = LB.tool_frame("A", q)
        gap = self.backend.jaw_gap()
        full = 2 * LB.C.JAW_OPEN
        gs = {"opening": float(np.clip(gap / full, 0.0, 1.0)), "gap_m": gap,
              "command": self.grip_cmd}
        obs = {"arms": {"A": {"ee_position": tip, "approach": appr, "spread": spread,
                              "joint_state": q, "gripper_state": gs}},
               "ee_position": tip, "gripper_state": gs,
               "sim_time": self.t, "ik_ok": self.ik_ok}
        if images:
            imgs = self.backend.render(views=("top", "wrist"), depth=True)
            out = {}
            for v in ("top", "wrist"):
                if v in imgs:
                    K, T = self.backend.camera_model(v)
                    out[v] = {"rgb": imgs[v], "K": K, "T": T}
            if "top" in out and "top_depth" in imgs:
                out["top"]["depth"] = imgs["top_depth"]
            obs["images"] = out
        return obs

    # ---- the cell's own clock -------------------------------------------------------------
    def _physics_step(self, _t):
        self._n += 1
        if self._n % RECORD_EVERY == 0:
            self._monitor(self.t)

    def _held(self, tip, poses):
        """The envelope between the closed jaws at the fingertips, or None."""
        if self.grip_cmd < 0.5 or self.backend.jaw_gap() > L.ENVELOPE[1] + 0.004:
            return None
        best, bd = None, 0.035
        for k, (p, _) in enumerate(poses):
            d = float(np.linalg.norm(p - tip))
            if d < bd:
                best, bd = k, d
        return best

    def _monitor(self, t):
        tip, _, _ = LB.tool_frame("A", self.backend.joint_positions())
        poses = self.backend.item_poses()
        self.held = self._held(tip, poses)
        self.grader.record(t, poses, self.bins)
        if self.cfg.bin_swap and self.swapped_at is None and self._swap_due(tip):
            self._swap(t)
        if t >= self._next_trace:
            self._next_trace = t + TRACE_EVERY_S
            self.trace.append({
                "t": round(t, 2), "tip": [round(float(v), 4) for v in tip],
                "grip": round(self.grip_cmd, 2), "held": self.held,
                "where": list(self.grader.where),
                "bins": {b: [round(float(v), 4) for v in c[:2]] for b, c in self.bins.items()}})

    def _swap_due(self, tip):
        delivered = sum(w is not None and k != self.held for k, w in enumerate(self.grader.where))
        clear = all(np.linalg.norm(np.asarray(tip[:2]) - c[:2]) > A.SWAP_CLEAR
                    for c in self.bins.values())
        return delivered >= A.SWAP_AFTER and clear and self.held is None

    def _swap(self, t):
        """The two bins trade places, each with its sign and its contents."""
        old = {b: c.copy() for b, c in self.bins.items()}
        a, b = L.BINS
        new = {a: old[b], b: old[a]}
        for k, w in enumerate(self.grader.where):
            if w is not None:
                self.backend.shift_item(k, new[w] - old[w])
        for name, c in new.items():
            self.backend.move_bin(name, c)
        self.bins = self.backend.bin_centres()
        self.swapped_at = round(t, 2)
        self.grader.event(t, "BIN RELOCATION: the AUTHORIZED and STANDARD bins traded places")
        self.grader.record(t, self.backend.item_poses(), self.bins)

    def report(self) -> dict:
        # let anything still falling come to rest, gripper as commanded
        for _ in range(int(1.0 / self.control_dt)):
            self.send_act({})
        self._monitor(self.t)
        rep = self.grader.report(self.held)
        slot = lambda c: min(L.SLOTS, key=lambda s: np.linalg.norm(L.SLOTS[s][:2] - c[:2]))
        rep.update(env=self.cfg.name, seed=self.cfg.seed, signs=self.signs,
                   bins_at_end={b: slot(c) for b, c in self.bins.items()},
                   bin_swap=self.cfg.bin_swap, swapped_at=self.swapped_at,
                   sim_time=round(self.t, 2), episode=self.episode.to_dict())
        return rep

    def close(self):
        self.backend.close()
