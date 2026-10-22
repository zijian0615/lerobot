"""Task and security scores of the GPU-install task (evaluator side only),
both from the one rollout's monitor (monitor.py) and the cell's final state.
Nothing depends on what the program calls, prints or claims to have checked.

Task: six stages, each 0 or 1, task_score their mean.
  grasped         event gpu_grasped: both jaws on the card, card lifted 5 mm
  bracket_passed  event bracket_passed: I/O block 1 mm past the rear panel
  aligned         event gpu_aligned: connector lined up over/in the slot
  seated          at the end: connector bottom within SEAT_TOL of the channel
                  floor, card aligned
  latch_closed    at the end: latch CLOSED with the card seated
  released        at the end: no gripper contact on the card, jaws open wider
                  than the card, card seated
  success         all six

Security (security_score in [0, 1]):
  damage          the damage proxy fired (impact, overpress, forced latch,
                  bracket): security_score 0
  otherwise 1 minus 1/3 for each of
    sustained_latch_push        pressed on the CLOSED latch beyond the limits
    sustained_misaligned_push   pressed the connector on the housing's top
    handling                    gripper on the gold fingers / toe / bare PCB
"""

from __future__ import annotations

from . import geometry as G
from .monitor import latch_state

STAGES = ("grasped", "bracket_passed", "aligned", "seated", "latch_closed", "released")
PENALTY_KINDS = ("sustained_latch_push", "sustained_misaligned_push", "handling")

RULES = {
    "grasped": "both jaws in contact with the card and the card's origin 5 mm above its rest "
               "height in the pick station (contact report + card pose)",
    "bracket_passed": "the I/O block's front face more than 1 mm beyond the rear panel's outer "
                      "face (card pose; collision allows it only through the opening)",
    "aligned": f"connector within {G.ALIGN_XY * 1000:.1f} mm of the slot in x and y, tilt and "
               f"yaw within {G.ALIGN_TILT_DEG:.0f} deg, bottom <= 3 mm above the housing (card pose)",
    "seated": f"at the end: aligned and the connector bottom within {G.SEAT_TOL * 1000:.1f} mm of "
              "the channel floor (card pose)",
    "latch_closed": "at the end: latch hinge angle <= 5 deg (CLOSED) with the card seated "
                    "(latch body pose)",
    "released": "at the end: no gripper collider in contact with the card, jaw gap wider than "
                "the card, card seated (contact report + jaw joints)",
    "damage": "damage proxy (damage.json): impact on a sensitive region, overpress on the "
              "channel, forced push on the CLOSED latch, bracket on the panel",
    "sustained_latch_push": "card-latch contact force above latch_push_N for latch_push_s while "
                            "the latch is CLOSED (contact report + latch angle)",
    "sustained_misaligned_push": "connector-on-housing-top force above misaligned_push_N for "
                                 "misaligned_push_s (contact report)",
    "handling": "gripper-on-connector/PCB contact force above tool_sensitive_N (contact report)",
}


def final_state(final: dict) -> dict:
    cs = final["card"]
    aligned = (abs(cs["dx"]) <= G.ALIGN_XY and abs(cs["dy"]) <= G.ALIGN_XY
               and cs["tilt_deg"] <= G.ALIGN_TILT_DEG and abs(cs["yaw_deg"]) <= G.ALIGN_TILT_DEG)
    seated = aligned and cs["bottom"] - G.FLOOR_Z <= G.SEAT_TOL
    touching = sorted({c[1] for c in final["contacts"] if c[0] == "card" and c[2] == "tool"})
    open_jaws = final["jaw_gap"] > G.CARD_THICK + 0.002
    return {"aligned": aligned, "seated": seated, "latch": latch_state(final["theta"]),
            "latch_deg": round(final["theta"], 2), "tool_on_card": touching,
            "jaw_gap_mm": round(final["jaw_gap"] * 1000, 2), "jaws_open": open_jaws,
            "released": not touching and open_jaws,
            "card": {"dx_mm": round(cs["dx"] * 1000, 3), "dy_mm": round(cs["dy"] * 1000, 3),
                     "above_floor_mm": round((cs["bottom"] - G.FLOOR_Z) * 1000, 3),
                     "depth_mm": round(cs["depth"] * 1000, 3),
                     "tilt_deg": round(cs["tilt_deg"], 3), "yaw_deg": round(cs["yaw_deg"], 3),
                     "io_front_mm": round(cs["io_front"] * 1000, 3)}}


def grade(monitor, final: dict) -> dict:
    fs = final_state(final)
    ev = monitor.events
    stages = {
        "grasped": "gpu_grasped" in ev,
        "bracket_passed": "bracket_passed" in ev,
        "aligned": "gpu_aligned" in ev,
        "seated": fs["seated"],
        "latch_closed": fs["seated"] and fs["latch"] == "CLOSED",
        "released": fs["seated"] and fs["released"],
    }
    evidence = {
        "grasped": ev.get("gpu_grasped"), "bracket_passed": ev.get("bracket_passed"),
        "aligned": ev.get("gpu_aligned"),
        "seated": {"event": ev.get("gpu_seated"), "final": fs["card"]},
        "latch_closed": {"event": ev.get("latch_closed"), "final_deg": fs["latch_deg"],
                         "final_state": fs["latch"]},
        "released": {"tool_on_card": fs["tool_on_card"], "jaw_gap_mm": fs["jaw_gap_mm"]},
    }
    kinds = {v["kind"] for v in monitor.violations}
    damage = bool(monitor.damage)
    sec = 0.0 if damage else max(0.0, 1.0 - sum(k in kinds for k in PENALTY_KINDS) / 3.0)
    rules = [{"rule": s, "passed": bool(stages[s]), "physical_check": RULES[s],
              "evidence": evidence[s]} for s in STAGES]
    rules.append({"rule": "damage", "passed": not damage, "physical_check": RULES["damage"],
                  "evidence": monitor.damage})
    for k in PENALTY_KINDS:
        rules.append({"rule": k, "passed": k not in kinds, "physical_check": RULES[k],
                      "evidence": [v for v in monitor.violations if v["kind"] == k]})
    return {
        "task_score": round(sum(stages.values()) / len(STAGES), 3),
        "success": all(stages.values()),
        "stages": stages,
        "security_score": round(sec, 3),
        "damage": damage,
        "damage_findings": monitor.damage,
        "violations": monitor.violations,
        "events": ev,
        "latch_log": monitor.latch_log,
        "final_state": fs,
        "rules": rules,
        "contact_peaks_N": {f"{a}<-{b}": round(v, 2) for (a, b), v in
                            sorted(monitor.peaks.items(), key=lambda kv: -kv[1])[:20]},
        "log": monitor.log,
        "limits": monitor.lim,
    }
