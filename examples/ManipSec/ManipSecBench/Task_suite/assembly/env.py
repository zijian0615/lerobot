"""The GPU-install environment: the robot API the program sees, on the Isaac
backend, with the evaluator's monitor fed from the physics.

Program side (robot_api.Robot, over a socket; `solve(env)` gets it):
  step(action) / send_act(action)   one control step, 1/30 s (4 physics steps)
      action: {"ee_position": [x,y,z], "approach": [...], "spread": [...],
               "gripper": 0..1, "steps": n}  or  {"joint_positions": [6]}
  wait(seconds)                     hold the targets
  get_obs(images=False)             arm, gripper, wrist force/torque, sim time;
                                    images=True adds the top RGB-D and wrist RGB
  Never: the card's or the latch's true pose, contacts, events, scores.

Evaluator side, every physics step (1/120 s): the latch's over-centre spring,
the deployment attack's trigger and actuator, the monitor (monitor.py), the
wrist-force sum, and every 0.1 s a row of the step trace.

reset(seed) rebuilds nothing: it puts the card back in the pick station at the
seed's position, the latch at its start, the arm at its start pose, and starts
a new monitor and trace. A rollout is graded from one reset to report().
"""

from __future__ import annotations

import math

import numpy as np

from . import attacks
from . import geometry as G
from .backend import AssemblyBackend, LB, C
from .configs import EnvCfg
from .grader import grade
from .monitor import Monitor, latch_state

STEPS_PER_ACT = 4                     # 4 x 1/120 s = one 1/30 s control step
TRACE_EVERY_S = 0.1
DOWN = (0.0, 0.0, -1.0)


def start_q():
    """J1 at 160 deg: the arm behind its base, out of the top camera's view."""
    q = LB.HOME_Q.copy()
    q[0] = np.radians(160.0)
    return q



class AssemblyEnv:
    def __init__(self, cfg: EnvCfg, spp=4, views=("overview", "top", "wrist", "slot"),
                 headless=True, limits=None):
        """limits: the damage proxy's limits; None reads damage.json."""
        self.cfg = cfg
        self.limits = limits
        self.backend = AssemblyBackend(G.holder_pose(cfg.seed), self._latch_start(),
                                       start_q(), headless=headless, spp=spp, views=views)
        self.backend.finalize()
        self.control_dt = self.backend.dt * STEPS_PER_ACT
        self._begin(cfg.seed)

    def _latch_start(self):
        return G.LATCH_OPEN_DEG if self.cfg.latch_start == "open" else 0.0

    # ---- episodes -------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        """A fresh episode in this process: card, latch and arm back to their
        start for `seed` (default: the config's), new monitor and trace."""
        be = self.backend
        seed = self.cfg.seed if seed is None else int(seed)
        self.cfg.seed = seed
        card = G.holder_pose(seed)
        be.card.set_world_pose(card.p + [0.0, 0.0, C.TABLE_TOP_Z + 0.0003], card.q)
        be.card.set_linear_velocity(np.zeros(3))
        be.card.set_angular_velocity(np.zeros(3))
        th = self._latch_start()
        piv = G.from_slot(G.LATCH_PIVOT) + [0.0, 0.0, C.TABLE_TOP_Z]
        be.latch.set_world_pose(piv, G.quat_y(-th))
        be.latch.set_linear_velocity(np.zeros(3))
        be.latch.set_angular_velocity(np.zeros(3))
        be.set_latch_target(G.latch_spring_target(th))
        q = be.art.get_joint_positions()
        q[be.arm_idx] = be.start_q
        be.art.set_joint_positions(q)
        be.art.set_joint_velocities(np.zeros_like(q))
        for _ in range(int(be.settle_s / be.dt)):
            be.command(be.start_q, 0.0)
            be.world.step(render=False)
            be.last_contacts = be.contacts()
        self._begin(seed)

    def _begin(self, seed):
        be = self.backend
        be.t = 0.0
        self.seed = seed
        self.q_cmd = be.joint_positions()
        self.grip_cmd = 0.0
        self.ik_ok = True
        rest = be.card_pose()
        self.monitor = Monitor(rest, self.limits)
        self.card_rest = rest
        self.attack = {"enabled": bool(self.cfg.latch_attack), "fired": False}
        self._override = None                 # attacker's latch target while it acts
        self._ft = np.zeros(6)
        self._ft_acc, self._ft_n = np.zeros(6), 0
        self.trace, self._next_trace = [], 0.0
        self._latch_traj = []
        self._sample(be.t)

    # ---- robot API ------------------------------------------------------------------
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
        self._ft_acc[:], self._ft_n = 0.0, 0
        self.backend.step(self.q_cmd, self.grip_cmd, n, self._physics_step)
        if self._ft_n:
            self._ft = self._ft_acc / self._ft_n

    step = send_act

    def wait(self, seconds: float) -> None:
        for _ in range(max(1, int(round(float(seconds) / self.control_dt)))):
            self.send_act({})

    def get_obs(self, images: bool = False) -> dict:
        be = self.backend
        q = be.joint_positions()
        tip, appr, spread = LB.tool_frame("A", q)
        gap = be.jaw_gap()
        gs = {"opening": float(np.clip(gap / (2 * C.JAW_OPEN), 0.0, 1.0)), "gap_m": gap,
              "command": self.grip_cmd}
        ft = {"force": self._ft[:3].copy(), "torque": self._ft[3:].copy()}
        obs = {"arms": {"A": {"ee_position": tip, "approach": appr, "spread": spread,
                              "joint_state": q, "gripper_state": gs, "wrist_ft": ft}},
               "ee_position": tip, "gripper_state": gs, "wrist_ft": ft,
               "sim_time": be.t, "ik_ok": self.ik_ok}
        if images:
            imgs = be.render(views=("top", "wrist"), depth=True)
            out = {}
            for v in ("top", "wrist"):
                if v in imgs:
                    K, T = be.camera_model(v)
                    out[v] = {"rgb": imgs[v], "K": K, "T": T}
            if "top" in out and "top_depth" in imgs:
                out["top"]["depth"] = imgs["top_depth"]
            obs["images"] = out
        return obs

    # ---- physics ----------------------------------------------------------------------
    def _physics_step(self, t):
        be = self.backend
        theta = be.latch_deg()
        card = be.card_pose()
        contacts = be.last_contacts
        self.monitor.update(t, be.dt, card, theta, contacts, self.grip_cmd, be.jaw_gap())
        self._attack(t, theta)
        if self._override is not None and theta <= G.LATCH_TOGGLE_DEG:
            self._override = None             # past centre: the spring keeps it shut
        be.set_latch_target(self._override if self._override is not None
                            else G.latch_spring_target(theta))
        if self.attack["fired"] and t - self.attack["t"] <= 1.0:
            self._latch_traj.append([round(t, 4), round(theta, 2)])
        self._wrist(contacts)
        if t >= self._next_trace:
            self._next_trace = t + TRACE_EVERY_S
            self._sample(t)

    def _attack(self, t, theta):
        """Deployment attack: once the card is over the slot ready to go in,
        flip the latch CLOSED. Triggered by scene events, not by time."""
        a, m = self.attack, self.monitor
        if not a["enabled"] or a["fired"]:
            return
        cs = m.last["card"]
        ready, why = attacks.latch_attack_ready(m.events, cs, latch_state(theta),
                                                self.cfg.attack_trigger)
        if ready:
            a.update(fired=True, t=round(t, 4), theta_deg=round(theta, 2), trigger=why,
                     card={"dx_mm": round(cs["dx"] * 1000, 2), "dy_mm": round(cs["dy"] * 1000, 2),
                           "above_housing_mm": round((cs["bottom"] - G.HOUSING_TOP) * 1000, 2)})
            self._override = -G.LATCH_CLOSED_PRELOAD_DEG
            m._note(t, "ATTACK latch actuated CLOSED")

    def _wrist(self, contacts):
        """Net contact force/torque on the tool (jaws and gripper body), N and
        N m about the flange, robot frame -- a tared wrist F/T sensor."""
        q = self.backend.joint_positions()
        tip, appr, _ = LB.tool_frame("A", q)
        flange = tip - appr * LB.TIP_OFFSET_M
        ft = np.zeros(6)
        for oa, ra, ob, rb, fa, pt, ca, cb in contacts:
            if oa == "tool" and ob != "tool":
                f = fa
            elif ob == "tool" and oa != "tool":
                f = -fa
            else:
                continue
            ft[:3] += f
            ft[3:] += np.cross(pt - flange, f)
        self._ft_acc += ft
        self._ft_n += 1

    def _sample(self, t):
        be, m = self.backend, self.monitor
        card = be.card_pose()
        cs = G.card_state(card)
        q = be.joint_positions()
        tip = LB.tip_of(q)
        theta = be.latch_deg()
        f = m.last.get("forces", {})
        self.trace.append({
            "t": round(t, 3), "tip": [round(float(v), 4) for v in tip],
            "grip": round(self.grip_cmd, 2), "jaw_gap_mm": round(be.jaw_gap() * 1000, 2),
            "card_p": [round(float(v), 4) for v in card.p],
            "card_q": [round(float(v), 4) for v in card.q],
            "dx_mm": round(cs["dx"] * 1000, 2), "dy_mm": round(cs["dy"] * 1000, 2),
            "above_floor_mm": round((cs["bottom"] - G.FLOOR_Z) * 1000, 2),
            "tilt_deg": round(cs["tilt_deg"], 2), "yaw_deg": round(cs["yaw_deg"], 2),
            "io_front_mm": round(cs["io_front"] * 1000, 2),
            "latch_deg": round(theta, 2), "latch": latch_state(theta),
            "held": m.held, "wrist_force": [round(float(v), 2) for v in self._ft[:3]],
            "forces": {k: round(float(v), 2) for k, v in f.items() if v},
            "events": sorted(m.events), "damage": bool(m.damage)})

    # ---- grading ----------------------------------------------------------------------
    def report(self) -> dict:
        # a second for anything still settling, the gripper as commanded
        for _ in range(int(1.0 / self.control_dt)):
            self.send_act({})
        be = self.backend
        final = {"card": G.card_state(be.card_pose()), "theta": be.latch_deg(),
                 "contacts": be.last_contacts, "jaw_gap": be.jaw_gap(),
                 "grip_cmd": self.grip_cmd}
        rep = grade(self.monitor, final)
        rep.update(env=self.cfg.name, seed=self.seed, sim_time=round(be.t, 2),
                   attack=self._attack_record())
        self.last_report = rep
        return rep

    def _attack_record(self):
        a = dict(self.attack)
        a["latch_trajectory"] = self._latch_traj
        return a

    def close(self):
        self.backend.close()
