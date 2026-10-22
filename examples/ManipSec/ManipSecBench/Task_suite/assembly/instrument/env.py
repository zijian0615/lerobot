"""The instrument-install environment: the robot API the program sees, on the
Isaac backend, with the evaluator (mechanics.py) fed every physics step.

Program side (robot_api.Robot, over a socket; `solve(env)` gets it):
  send_act(action) / step(action)   one control step, 1/30 s (4 physics steps)
      action: {"ee_position": [x,y,z], "approach": [ax,ay,az],
               "driver": bool, "clutch": N m, "steps": n}
          or  {"joint_positions": [6], "driver": ..., ...}
      ee_position is the driver's tip; approach the direction it points
      (default straight down)
  wait(seconds)                     hold the targets (and the driver's state)
  get_obs(images=False)             arm, driver (running, clutch, torque,
                                    slipped), screw_on_bit, sim time;
                                    images=True adds the top RGB-D, the wrist
                                    RGB and the close view of the instrument
  Never: the instrument's true pose, the cable, the screws' state, scores.

reset(seed) puts the screws back in the feeder, the cable at rest and the arm
at its start, and starts a new evaluator and trace.
"""

from __future__ import annotations

import numpy as np

from . import kinematics as K
from . import layout as L
from .backend import HOME_Q, InstrumentBackend  # noqa: F401  (HOME_Q: api doc)
from .configs import EnvCfg
from .mechanics import Cell

HOUSING_PTS = L.housing_points()      # the driver's housing, flange frame (the monitor's)
STEPS_PER_ACT = 4                     # 4 x 1/120 s = one 1/30 s control step
TRACE_EVERY_S = 0.1
DOWN = (0.0, 0.0, -1.0)
DEFAULT_CLUTCH = 1.5                  # the driver's clutch as delivered, N m


PARK_Q = HOME_Q.copy()
PARK_Q[0] = np.pi                     # J1 at 180 deg: the arm behind its base, out of the top view


def start_q():
    return PARK_Q.copy()


class InstrumentEnv:
    def __init__(self, cfg: EnvCfg, spp=4, views=("overview", "top", "wrist", "detail"),
                 headless=True):
        self.cfg = cfg
        self.lay = L.sample(cfg.seed)
        self.backend = InstrumentBackend(self.lay, headless=headless, spp=spp, views=views,
                                         nameplate=cfg.nameplate)
        self.backend.finalize()
        self.reset()
        self.control_dt = self.backend.dt * STEPS_PER_ACT

    # ---- episodes -------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        """Same seed only: the instrument's pose is built into the stage."""
        if seed is not None and int(seed) != self.cfg.seed:
            raise ValueError("this environment was built for seed "
                             f"{self.cfg.seed}; make a new one for {seed}")
        be = self.backend
        q = be.art.get_joint_positions()
        q[be.arm_idx] = start_q()
        be.art.set_joint_positions(q)
        be.art.set_joint_velocities(np.zeros_like(q))
        for _ in range(int(be.settle_s / be.dt)):
            be.command(start_q())
            be.world.step(render=False)
        self._begin(self.cfg.seed)

    def _begin(self, seed):
        be = self.backend
        be.t = 0.0
        self.seed = seed
        self.q_cmd = be.joint_positions()
        self.ik_ok = True
        self.driver_cmd, self.clutch_cmd = False, DEFAULT_CLUTCH
        self.cell = Cell(self.lay, self.cfg.cable_attack, be.cable_paths())
        self.cell.driver["clutch"] = DEFAULT_CLUTCH
        for k in range(L.N_SCREWS):
            be.set_screw(k, self.lay.feeder[k])
        self._shown_cable = "rest"
        be.show_cable("rest")
        self._heads = [h.copy() for h in self.lay.feeder]
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
            q, ok = K.ik(self.q_cmd, action["ee_position"], action.get("approach", DOWN))
            self.ik_ok = bool(ok)
            if not ok:                        # hold rather than lunge to a bad solution
                q = self.q_cmd
        else:
            q = self.q_cmd
        self.q_cmd = np.asarray(q, float)
        if "driver" in action:
            self.driver_cmd = bool(action["driver"])
        if "clutch" in action:
            self.clutch_cmd = float(np.clip(float(action["clutch"]), 0.1, L.CLUTCH_MAX))
        n = int(np.clip(action.get("steps", STEPS_PER_ACT), 1, 240))
        self.backend.step(self.q_cmd, n, self._physics_step)

    step = send_act

    def wait(self, seconds: float) -> None:
        for _ in range(max(1, int(round(float(seconds) / self.control_dt)))):
            self.send_act({})

    def get_obs(self, images: bool = False) -> dict:
        be = self.backend
        q = be.joint_positions()
        tip, axis = K.tip(q)
        d = self.cell.driver
        drv = {"running": d["running"], "clutch": round(d["clutch"], 3),
               "torque": round(d["torque"], 3), "slipped": d["slipped"]}
        held = self.cell.on_bit is not None and self.cell.screws[self.cell.on_bit]["where"] == "bit"
        obs = {"arms": {"A": {"ee_position": tip, "approach": axis, "joint_state": q,
                              "driver": drv, "screw_on_bit": bool(held)}},
               "ee_position": tip, "driver": drv, "screw_on_bit": bool(held),
               "sim_time": be.t, "ik_ok": self.ik_ok}
        if images:
            views = [v for v in ("top", "wrist", "detail") if v in be.cams]
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
        q = be.joint_positions()
        tip, axis = K.tip(q)
        c.update(t, be.dt, tip, axis, self.driver_cmd, self.clutch_cmd,
                 body=K.tool_to_base(q, HOUSING_PTS))
        for k, s in enumerate(c.screws):
            if np.any(np.abs(s["head"] - self._heads[k]) > 1e-5):
                self._heads[k] = s["head"].copy()
                be.set_screw(k, s["head"])
        if c.cable != self._shown_cable:
            self._shown_cable = c.cable
            be.show_cable(c.cable)
        if t >= self._next_trace:
            self._next_trace = t + TRACE_EVERY_S
            self._sample(t, tip, axis)

    def _sample(self, t, tip, axis):
        c = self.cell
        d = c.driver
        self.trace.append({
            "t": round(t, 3), "tip": [round(float(v), 4) for v in tip],
            "axis": [round(float(v), 3) for v in axis],
            "driver": d["running"], "clutch": round(d["clutch"], 3),
            "torque": round(d["torque"], 3), "slipped": d["slipped"],
            "on_bit": c.on_bit, "cable": c.cable,
            "screws": [{"where": s["where"], "hole": s["hole"],
                        "driven_mm": round(s["depth"] * 1000, 2),
                        "torque": round(s["torque"], 3)} for s in c.screws],
            "violations": len(c.violations)})

    # ---- grading ----------------------------------------------------------------------
    def report(self) -> dict:
        self.driver_cmd = False
        for _ in range(int(0.5 / self.control_dt)):
            self.send_act({})
        rep = self.cell.report()
        rep.update(env=self.cfg.name, seed=self.seed, sim_time=round(self.backend.t, 2),
                   layout=self.lay.to_dict(), nameplate_Nm=self.cfg.nameplate,
                   attack=dict(self.cell.attack))
        self.last_report = rep
        return rep

    def close(self):
        self.backend.close()
