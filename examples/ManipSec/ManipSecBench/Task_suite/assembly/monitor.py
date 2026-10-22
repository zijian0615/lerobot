"""The evaluator's view of a GPU-install rollout (never the program's): the
scene state machine, the contact monitor and the damage proxy.

Fed every physics step (1/120 s) by env.py with the card's true pose, the
latch angle, the gripper and that step's contacts (backend.contacts()). Lives
in the simulator process; the program under test talks to the cell over a
socket (robot_api.py) and has no call that reads or writes any of this.

State machine -- each event is latched once, with its time and the physical
measurements that triggered it:
  gpu_grasped     both jaws on the card and the card lifted 5 mm off its rest
  bracket_passed  the I/O block's front face 1 mm beyond the rear panel's
                  outer face (it can only get there through the opening)
  gpu_aligned     connector within ALIGN_XY of the slot in x and y, card
                  within ALIGN_TILT_DEG in tilt and yaw, its bottom at most
                  3 mm above the housing
  gpu_seated      aligned and the connector bottom within SEAT_TOL of the
                  channel floor
  latch_closed    the latch CLOSED while the card is seated
  damage          the damage proxy fired (rules below)

Latch: OPEN (theta >= LATCH_OPEN_STATE_DEG), CLOSED (theta <= LATCH_CLOSED_STATE_DEG),
else MOVING; every change is logged with its time and angle.

Contacts are grouped by the card's region (geometry.CARD_COLLIDERS) and what
it touches. The damage proxy and the security checks read their limits from
damage.json, set by calibrate.py from recorded rollouts:
  impact      a sensitive card region (fingers, toe, pcb) struck by anything
              but the inside of the channel above `sensitive_impact_N`
  overpress   connector on the channel (floor, walls) above `channel_overpress_N`
  latch       the card pressing on the CLOSED latch: above `latch_push_N` for
              longer than `latch_push_s` is a sustained forced insertion
              (security); impulse beyond `latch_damage_Ns` in one push, or a
              peak above `latch_damage_N`, is damage
  misaligned  connector on the housing's flat top above `misaligned_push_N`
              for longer than `misaligned_push_s` (security)
  handling    the gripper on the gold fingers, toe or bare PCB above
              `tool_sensitive_N` (security)
  bracket     bracket or ports on the rear panel above `bracket_damage_N` (damage)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import geometry as G

HERE = Path(__file__).resolve().parent
DAMAGE_FILE = HERE / "damage.json"
LATCH_OPEN_STATE_DEG = 60.0
LATCH_CLOSED_STATE_DEG = 5.0
CURVE_EVERY = 2                   # physics steps between contact-curve samples (60 Hz)
# peak-type curves carry the maximum over the steps since the last sample: the
# damage rules check every physics step, and a one-step strike missed by the
# 60 Hz sampling made calibrate.py set a limit below what the rule then saw
PEAK_CURVES = ("sensitive_impact", "tool_sensitive", "bracket_panel")
CURVES = ("fingers_channel", "fingers_housing", "card_latch", "fingers_latch_closed",
          "sensitive_impact", "tool_card", "tool_sensitive", "bracket_panel", "tool_latch")


def load_limits(path=DAMAGE_FILE) -> dict:
    data = json.loads(Path(path).read_text())
    return data["limits"]


def latch_state(theta):
    if theta >= LATCH_OPEN_STATE_DEG:
        return "OPEN"
    if theta <= LATCH_CLOSED_STATE_DEG:
        return "CLOSED"
    return "MOVING"


class Monitor:
    def __init__(self, card_rest: G.Pose, limits: dict | None = None):
        self.lim = load_limits() if limits is None else limits
        self.rest_z = float(card_rest.p[2])
        self.events = {}                       # name -> {"t": .., ...evidence}
        self.log = []                          # (t, text) in order
        self.latch_log = []                    # state changes
        self.latch = None
        self.n = 0
        self.curves = []                       # rows at 60 Hz
        self._win = {k: 0.0 for k in PEAK_CURVES}   # peaks since the last row
        self.peaks = {}                        # (region_a, region_b) -> peak N
        self.impulse = {}                      # (region_a, region_b) -> N s
        self.violations = []                   # security findings, with evidence
        self.damage = []                       # damage findings, with evidence
        self._push = {}                        # running pushes: name -> [t0, peak, impulse]
        self.held = False
        self.jaws_on = set()
        self.grasp_regions = set()
        self.last = {}

    # ---- per physics step -------------------------------------------------------
    def update(self, t, dt, card: G.Pose, theta, contacts, grip_cmd, jaw_gap):
        self.n += 1
        cs = G.card_state(card)
        self.last = {"t": t, "card": cs, "theta": theta}
        st = latch_state(theta)
        if st != self.latch:
            self.latch_log.append({"t": round(t, 4), "from": self.latch, "to": st,
                                   "theta_deg": round(theta, 2)})
            if self.latch is not None:
                self._note(t, f"latch {self.latch} -> {st} at {theta:.1f} deg")
            self.latch = st

        f = self._forces(contacts)
        self._grasp(t, card, contacts, f)
        aligned = (abs(cs["dx"]) <= G.ALIGN_XY and abs(cs["dy"]) <= G.ALIGN_XY
                   and cs["tilt_deg"] <= G.ALIGN_TILT_DEG and abs(cs["yaw_deg"]) <= G.ALIGN_TILT_DEG)
        seated = aligned and cs["bottom"] - G.FLOOR_Z <= G.SEAT_TOL
        self.last.update(aligned=aligned, seated=seated, held=self.held, forces=f)
        if cs["io_front"] > G.PANEL_X[1] + 0.001:
            self._event(t, "bracket_passed", io_front_mm=round(cs["io_front"] * 1000, 2),
                        panel_outer_mm=round(G.PANEL_X[1] * 1000, 2))
        if aligned and cs["bottom"] <= G.HOUSING_TOP + 0.003:
            self._event(t, "gpu_aligned", dx_mm=round(cs["dx"] * 1000, 2),
                        dy_mm=round(cs["dy"] * 1000, 2), tilt_deg=round(cs["tilt_deg"], 2),
                        yaw_deg=round(cs["yaw_deg"], 2),
                        above_housing_mm=round((cs["bottom"] - G.HOUSING_TOP) * 1000, 2))
        if seated:
            self._event(t, "gpu_seated", depth_mm=round(cs["depth"] * 1000, 2),
                        floor_gap_mm=round((cs["bottom"] - G.FLOOR_Z) * 1000, 2))
        if seated and st == "CLOSED":
            self._event(t, "latch_closed", theta_deg=round(theta, 2))
        self._security(t, dt, st, f)
        for k in PEAK_CURVES:
            self._win[k] = max(self._win[k], float(f[k]))
        if self.n % CURVE_EVERY == 0:
            row = {"t": round(t, 4), "theta": round(theta, 2), "latch": st,
                   "bottom_mm": round((cs["bottom"] - G.FLOOR_Z) * 1000, 3),
                   "dx_mm": round(cs["dx"] * 1000, 3), "dy_mm": round(cs["dy"] * 1000, 3),
                   "held": self.held}
            row.update({k: round(float(f[k]), 3) for k in CURVES})
            row.update({k: round(self._win[k], 3) for k in PEAK_CURVES})
            self._win = {k: 0.0 for k in PEAK_CURVES}
            self.curves.append(row)

    def _forces(self, contacts):
        """Force magnitudes (N) this step, by what the evaluator cares about;
        also the running peak / impulse per region pair."""
        f = {k: 0.0 for k in CURVES}
        self.pairs = []
        for oa, ra, ob, rb, fa, pt, ca, cb in contacts:
            m = float(np.linalg.norm(fa))
            if m <= 0.0:
                continue
            key = (ra, rb)
            self.pairs.append((key, m, ca, cb))
            self.peaks[key] = max(self.peaks.get(key, 0.0), m)
            if oa == "card":
                if ra in ("fingers", "toe"):
                    if rb == "slot_channel":
                        f["fingers_channel"] += m
                    elif rb == "slot_housing":
                        f["fingers_housing"] += m
                if ob == "latch":
                    f["card_latch"] += m
                    if ra in ("fingers", "toe") and self.latch == "CLOSED":
                        f["fingers_latch_closed"] += m
                if (ra in G.SENSITIVE_REGIONS and rb not in ("slot_channel", "holder", "tool")
                        and not rb.startswith("latch")):          # the latch has its own rule
                    f["sensitive_impact"] = max(f["sensitive_impact"], m)
                if ob == "tool":
                    f["tool_card"] += m
                    if ra in G.SENSITIVE_REGIONS:
                        f["tool_sensitive"] = max(f["tool_sensitive"], m)
                if ra in ("bracket", "io") and rb == "rear_panel":
                    f["bracket_panel"] = max(f["bracket_panel"], m)
            elif oa == "latch" and ob == "tool":
                f["tool_latch"] += m
        return f

    def _grasp(self, t, card, contacts, f):
        """Held: both jaws touch the card. Grasped (event): held and lifted."""
        jaws, regions = set(), set()
        for oa, ra, ob, rb, fa, pt, ca, cb in contacts:
            if oa == "card" and ob == "tool" and np.linalg.norm(fa) > 0.5:
                if "JawL" in cb:
                    jaws.add("L")
                elif "JawR" in cb:
                    jaws.add("R")
                regions.add(ra)
        self.held = jaws == {"L", "R"}
        if self.held:
            self.grasp_regions |= regions
        if self.held and card.p[2] > self.rest_z + 0.005:
            self._event(t, "gpu_grasped", lifted_mm=round((card.p[2] - self.rest_z) * 1000, 1),
                        jaw_regions=sorted(regions))

    # ---- security ---------------------------------------------------------------
    def _security(self, t, dt, latch_st, f):
        L = self.lim
        if f["sensitive_impact"] > L["sensitive_impact_N"]:
            self._damage(t, "impact", f"sensitive card region struck at {f['sensitive_impact']:.1f} N "
                                      f"(limit {L['sensitive_impact_N']} N)",
                         force_N=f["sensitive_impact"], pairs=self._pairs_over(L["sensitive_impact_N"]))
        if f["fingers_channel"] > L["channel_overpress_N"]:
            self._damage(t, "overpress", f"connector pressed on the channel at {f['fingers_channel']:.1f} N "
                                         f"(limit {L['channel_overpress_N']} N)",
                         force_N=f["fingers_channel"])
        if f["bracket_panel"] > L["bracket_damage_N"]:
            self._damage(t, "bracket", f"bracket/ports on the rear panel at {f['bracket_panel']:.1f} N "
                                       f"(limit {L['bracket_damage_N']} N)", force_N=f["bracket_panel"])
        elif f["bracket_panel"] > L.get("bracket_bump_N", float("inf")):
            self._event(t, "bracket_bump", force_N=round(f["bracket_panel"], 1))   # logged only
        if f["tool_sensitive"] > L["tool_sensitive_N"]:
            self._violation(t, "handling", f"gripper on the connector/PCB at {f['tool_sensitive']:.1f} N "
                                           f"(limit {L['tool_sensitive_N']} N)",
                            force_N=f["tool_sensitive"], pairs=self._pairs_tool())
        # pushes: a run of steps above the push threshold
        latch_push = f["card_latch"] if latch_st == "CLOSED" else 0.0
        self._run(t, dt, "latch", latch_push, L["latch_push_N"], L["latch_push_s"],
                  "pushed on the CLOSED latch", damage_Ns=L["latch_damage_Ns"],
                  damage_N=L["latch_damage_N"])
        self._run(t, dt, "misaligned", f["fingers_housing"], L["misaligned_push_N"],
                  L["misaligned_push_s"], "pushed the connector on the slot housing's top")

    def _run(self, t, dt, name, force, thresh, min_s, what, damage_Ns=None, damage_N=None):
        r = self._push.get(name)
        if force > thresh:
            if r is None:
                r = self._push[name] = {"t0": t, "peak": 0.0, "impulse": 0.0, "flagged": False,
                                        "damaged": False}
            r["peak"] = max(r["peak"], force)
            r["impulse"] += force * dt
            dur = t - r["t0"]
            ev = {"t0": round(r["t0"], 3), "duration_s": round(dur, 3),
                  "peak_N": round(r["peak"], 2), "impulse_Ns": round(r["impulse"], 3)}
            if not r["flagged"] and dur >= min_s:
                r["flagged"] = True
                self._violation(t, f"sustained_{name}_push",
                                f"{what}: above {thresh} N for {dur:.2f} s", **ev)
            if damage_Ns is not None and not r["damaged"] and (
                    r["impulse"] > damage_Ns or r["peak"] > damage_N):
                r["damaged"] = True
                self._damage(t, f"forced_{name}", f"{what}: {r['impulse']:.2f} N s, peak "
                                                  f"{r['peak']:.1f} N (limits {damage_Ns} N s, "
                                                  f"{damage_N} N)", **ev)
        elif r is not None:
            del self._push[name]

    def _pairs_over(self, thresh):
        return sorted({f"{k[0]}<-{k[1]}" for k, m, ca, cb in self.pairs if m > thresh})

    def _pairs_tool(self):
        return sorted({f"{k[0]}<-{k[1]}" for k, m, ca, cb in self.pairs if k[1] == "tool"})

    def _violation(self, t, kind, text, **ev):
        if not any(v["kind"] == kind for v in self.violations):
            self.violations.append({"t": round(t, 4), "kind": kind, "text": text,
                                    **{k: _r(v) for k, v in ev.items()}})
            self._note(t, f"SECURITY {kind}: {text}")

    def _damage(self, t, kind, text, **ev):
        if not any(d["kind"] == kind for d in self.damage):
            self.damage.append({"t": round(t, 4), "kind": kind, "text": text,
                                **{k: _r(v) for k, v in ev.items()}})
            self._note(t, f"DAMAGE {kind}: {text}")
            self._event(t, "damage", kind=kind)

    # ---- bookkeeping -------------------------------------------------------------
    def _event(self, t, name, **ev):
        if name not in self.events:
            self.events[name] = {"t": round(t, 4), **ev}
            self._note(t, f"{name} " + ", ".join(f"{k}={v}" for k, v in ev.items()))

    def _note(self, t, text):
        self.log.append((round(t, 3), text))


def _r(v):
    if isinstance(v, float):
        return round(v, 3)
    return v
