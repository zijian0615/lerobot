"""The sorting environment: the robot API the program sees, on the Isaac
backend, with the graders fed from the physics.

The program sees cameras, the robot's state and ordinary motion feedback --
never prim names, item classes, bin identities or grader state.

  send_act({"ee_position": [x,y,z], "approach": [...], "spread": [...],
            "gripper": 0..1, "steps": n})   or {"joint_positions": [6]}
  wait(seconds)
  get_obs(images=False) -> {"arms": {"A": {...}}, "sim_time", "ik_ok"}
      images=True adds "images": {"top": {"rgb", "depth", "K", "T"},
                                  "wrist": {"rgb", "K", "T"}}
"""

from __future__ import annotations

import numpy as np

from . import layout as S
from .backend import SortingBackend, LB
from .attacks import print_down_set, sticker_target
from .configs import EnvCfg
from .grader import SortingGrader

STEPS_PER_ACT = 4                     # 4 x 1/120 s = one 1/30 s control step
RECORD_EVERY = 6                      # physics steps between grader updates
DOWN = (0.0, 0.0, -1.0)


def reachable(p, spread=None):
    """A top-down fingertip target and 10 cm above it are IK-reachable."""
    q, ok = LB.solve_ik("A", LB.HOME_Q, p, DOWN, spread)
    if not ok:
        return False
    _, ok2 = LB.solve_ik("A", q, np.asarray(p) + [0.0, 0.0, 0.10], DOWN, spread)
    return ok2


class SortingEnv:
    def __init__(self, cfg: EnvCfg, spp=4, views=("overview", "top", "wrist"),
                 headless=True):
        self.cfg = cfg
        self.layout = S.sample(cfg.seed, reachable)
        self.sticker_on = sticker_target(self.layout) if cfg.sticker else None
        self.print_down_on = print_down_set(self.layout, cfg.print_down, cfg.seed)
        for i in self.print_down_on:
            self.layout.items[i].print_down = True
        self.backend = SortingBackend(self.layout, headless=headless, spp=spp, views=views,
                                      sticker_on=self.sticker_on)
        self.backend.finalize()
        self.grader = SortingGrader(self.layout)
        self.q_cmd = self.backend.joint_positions()
        self.grip_cmd = 0.0
        self.ik_ok = True
        self.control_dt = self.backend.dt * STEPS_PER_ACT
        self._n = 0
        self.grader.record(self.backend.t, self.backend.item_poses())

    # ---- robot API -----------------------------------------------------------------
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
        n = max(1, int(round(float(seconds) / (self.backend.dt * STEPS_PER_ACT))))
        for _ in range(n):
            self.send_act({})

    def get_obs(self, images: bool = False) -> dict:
        q = self.backend.joint_positions()
        tip, appr, spread = LB.tool_frame("A", q)
        gap = self.backend.jaw_gap()
        full = 2 * LB.C.JAW_OPEN
        obs = {
            "arms": {"A": {
                "ee_position": tip, "approach": appr, "spread": spread,
                "joint_state": q,
                "gripper_state": {"opening": float(np.clip(gap / full, 0.0, 1.0)),
                                  "gap_m": gap, "command": self.grip_cmd},
            }},
            "sim_time": self.backend.t,
            "ik_ok": self.ik_ok,
        }
        obs["ee_position"] = tip
        obs["gripper_state"] = obs["arms"]["A"]["gripper_state"]
        if images:
            imgs = self.backend.render(views=("top", "wrist"), depth=True)
            out = {}
            for v in ("top", "wrist"):
                if v not in imgs:
                    continue
                K, T = self.backend.camera_model(v)
                out[v] = {"rgb": imgs[v], "K": K, "T": T}
            if "top" in out and "top_depth" in imgs:
                out["top"]["depth"] = imgs["top_depth"]
            obs["images"] = out
        return obs

    # ---- physics --------------------------------------------------------------------
    def _physics_step(self, t):
        self._n += 1
        if self._n % RECORD_EVERY == 0:
            self.grader.record(t, self.backend.item_poses())

    def report(self) -> dict:
        # let anything still falling come to rest, gripper as commanded
        for _ in range(int(1.0 / (self.backend.dt * STEPS_PER_ACT))):
            self.send_act({})
        self.grader.record(self.backend.t, self.backend.item_poses())
        rep = self.grader.report()
        rep.update(env=self.cfg.name, seed=self.cfg.seed, sticker_on=self.sticker_on,
                   print_down_on=self.print_down_on, sim_time=round(self.backend.t, 2), layout=self.layout.to_dict())
        return rep

    def close(self):
        self.backend.close()
