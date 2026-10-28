"""Write the illustrative example entries shown on the leaderboard (leaderboard/examples/*.json).

    uv run python fanuc_er4ia_sim/challenge/leaderboard/make_examples.py

They are runs in the cell's run format with "example": true. The page tags every one of them "example" and says
they are not real-robot results. Edit EXAMPLES below and run this again; delete examples/ to remove them.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PASS_FAIL = [{"stage": 0, "name": "fail", "rule": "the task was not done"},
             {"stage": 1, "name": "success", "rule": "the task was done"}]
OUT = HERE / "examples"
RANDOM5 = [{"id": f"R{k}", "instruction": ""} for k in range(1, 6)]
TASKS = {
    "push-cube-v1": {"task_id": "push-cube-v1", "title": "push-cube-v1: push a randomly placed cube into the yellow area",
                     "objects_setup": {"type": "same", "note": "orange cube, yellow target area on the table"},
                     "layout_setup": {"type": "various", "note": "place the cube somewhere new every rollout"},
                     "stages": [{"stage": 0, "name": "none", "rule": "the cube does not end inside the yellow area"},
                                {"stage": 1, "name": "push well", "rule": "the cube ends partly inside the yellow area"},
                                {"stage": 2, "name": "push perfect", "rule": "the cube ends fully inside the yellow area"}],
                     "success_stage": 2,
                     "layouts": [{**l, "instruction": "push the cube into the yellow area"} for l in RANDOM5],
                     "trials_per_layout": 2},
}
# task -> (policy, model_type, action, stages reached per trial in order, cycle time s)
EXAMPLES = [
    ("push-cube-v1", "VA baseline", "VA", "joints", [2, 2, 2, 2, 1, 2, 2, 2, 2, 2], 18.0),
    ("push-cube-v1", "VLM + BM baseline", "VLM + BM", "tcp", [2, 1, 2, 1, 2, 2, 1, 2, 0, 2], 26.0),
]


# (task, model_type) -> {trial index: cameras}: example trials that show a real recording from rollouts/
VIDEOS = {("push-cube-v1", "VA"): {0: ["front", "side"]}}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.json"):
        old.unlink()
    t0 = time.mktime(time.strptime("2026-10-14", "%Y-%m-%d"))
    for k, (task_id, policy, model_type, action, stages, cycle) in enumerate(EXAMPLES):
        # scored success / fail like every run: the old stage lists only say which trials succeeded
        card = {**TASKS[task_id], "stages": PASS_FAIL, "success_stage": 1}
        full, lays = TASKS[task_id]["success_stage"], card["layouts"]
        stages = [int(st >= full) for st in stages]
        top = 1
        run_id = f"example-{task_id}-{model_type.lower().replace(' + ', '-').replace(' ', '-')}"
        trials = []
        for i, st in enumerate(stages):
            t_start = t0 + 3600 * k + 60 * i
            trials.append({
                "trial_id": f"{run_id}-t{i:02d}", "index": i, "layout": lays[i % len(lays)]["id"], "status": "done",
                "stats": {"t_start": t_start, "latency_s": []},
                "outcome": {"stage": st, "success_stage": top, "success": st >= top, "intervention": False,
                            "fault": None, "cycle_time_s": cycle if st >= top else None, "notes": ""}})
            if i in VIDEOS.get((task_id, model_type), {}):
                trials[-1]["video"] = VIDEOS[(task_id, model_type)][i]
        run = {"run_id": run_id, "example": True, "created": t0 + 3600 * k, "backend": "FanucBackend",
               "task_id": task_id, "trials": trials,
               "submission": {"team": "SIDC", "authors": "SIDC", "policy": policy, "model_type": model_type,
                              "action": action, "cameras": ["table", "side", "wrist"], "date": "2026-10-14",
                              "task_card": card}}
        (OUT / f"{run_id}.json").write_text(json.dumps(run, indent=1))
    print(f"{len(EXAMPLES)} example runs -> {OUT}")


if __name__ == "__main__":
    main()
