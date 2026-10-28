"""What a run's submission says about its task, in one place (server, operator console, leaderboard).

A submission describes its task in a few fields (see client/submission/submission.yaml):

    "task_id": "pick-cube-v1",
    "task": "put the cube into the yellow bin",                 # also the default instruction of every trial
    "objects": "same" | "different",                           # the objects across rollouts
    "layout":  "same" | "different",                           # their positions across rollouts (details offline)
    "rollouts": 10,
    "custom_hardware": null

Every trial is scored success / fail, with fault and intervention flags. Runs registered with a separate task card
(--task-card, "task_card" in the submission) keep the stages they were scored with.
"""
from __future__ import annotations

TYPES = ("same", "different", "fixed", "various", "randomized")
PASS_FAIL = [{"stage": 0, "name": "fail", "rule": "the task was not done"},
             {"stage": 1, "name": "success", "rule": "the task was done"}]


def spec(sub: dict) -> dict:
    """task_id, title, stages [{stage, name, rule}], success_stage, objects, layout, layouts [{id, instruction}]."""
    card = sub.get("task_card")
    if isinstance(card, dict) and card.get("stages"):  # older runs: a full task card
        return {"task_id": card.get("task_id", "custom"), "title": card.get("title", card.get("task_id", "")),
                "stages": sorted(card["stages"], key=lambda s: s["stage"]), "success_stage": int(card["success_stage"]),
                "objects": card.get("objects_setup"), "layout": card.get("layout_setup"),
                "layouts": [{"id": lay.get("id"), "instruction": lay.get("instruction", "")} for lay in card.get("layouts", [])],
                "card": card}
    stages = PASS_FAIL  # every trial is scored success / fail, plus fault and intervention flags
    return {"task_id": sub.get("task_id") or "custom", "title": sub.get("task") or sub.get("task_id") or "Custom task",
            "stages": stages, "success_stage": len(stages) - 1, "objects": setup(sub.get("objects")),
            "layout": setup(sub.get("layout")), "layouts": [], "card": None}


def setup(v) -> dict | None:
    """{"type", "note"} from the submission's objects / layout field (a dict, or just a note)."""
    if not v:
        return None
    if isinstance(v, dict):
        return {"type": str(v.get("type", "")).strip(), "note": str(v.get("note", "")).strip()}
    v = str(v).strip()
    if v.lower() in TYPES:  # submission.yaml: objects: same | different, layout: same | different
        return {"type": v.lower(), "note": ""}
    return {"type": "", "note": v}


def setup_lines(sub: dict) -> list[str]:
    """What the operator reads before each rollout."""
    s = spec(sub)
    lines = [f"Task: {s['title']}"]
    for key in ("objects", "layout"):
        v = s[key]
        if v:
            lines.append(f"{key.capitalize():8s} [{v['type'] or '-'}] {v['note']}".rstrip())
    card = s["card"]
    if card:  # older runs: positions from the task card
        lines += [f"  {f.get('name')}: x {f['xy_m'][0] * 1000:.0f}, y {f['xy_m'][1] * 1000:.0f} mm"
                  for f in card.get("fixtures", []) if "xy_m" in f]
    if len(lines) == 1:
        lines.append("(no setup notes: set up the task as the team described it)")
    return lines
