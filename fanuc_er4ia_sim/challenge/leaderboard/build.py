"""Build the real-robot leaderboard (index.html) from the scored runs in ../runs/*.json.

    uv run python fanuc_er4ia_sim/challenge/leaderboard/build.py            # real-robot runs only
    uv run python fanuc_er4ia_sim/challenge/leaderboard/build.py --include-mock

robot_server.py rebuilds it after every score and serves it at /leaderboard. One row per run, grouped by the
run's task card (teams define their own tasks). Unscored trials are left out; a run with none scored is skipped.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import task_spec  # noqa: E402
def task_of(run: dict) -> dict:
    """The run's task (task_spec.py) for grouping and scoring; the oldest runs are the screw task."""
    sub = run["submission"]
    if not (isinstance(sub.get("task_card"), dict) or sub.get("task_id") or sub.get("stages")):
        legacy = json.loads((ROOT / "task_card.json").read_text())
        if run.get("task_id") == legacy["task_id"]:  # registered before tasks were per run
            sub = {"task_card": {**legacy, "title": legacy.get("instruction", legacy["task_id"]), "success_stage": 4}}
    s = task_spec.spec(sub)
    return {k: s[k] for k in ("task_id", "title", "stages", "success_stage", "objects", "layout", "layouts")}

def progress(j: dict) -> float:
    """Furthest stage over the task card's success stage (4 for runs scored before cards were per run)."""
    return min(1.0, j["outcome"]["stage"] / max(1, j["outcome"].get("success_stage", 4)))

def video_cams(trial: dict) -> list[str]:
    """The cameras recorded for a trial ([] if none; True from early recordings means the cell's three)."""
    v = trial.get("video")
    return list(v) if isinstance(v, list) else (["table", "side", "wrist"] if v else [])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarize(run: dict) -> dict | None:
    scored = [j for j in run["trials"] if j.get("outcome")]
    if not scored:
        return None
    n = len(scored)
    wins = [j for j in scored if j["outcome"]["success"]]
    lo, hi = wilson(len(wins), n)
    cycles = [j["outcome"]["cycle_time_s"] for j in wins if j["outcome"].get("cycle_time_s")]
    lat = [x for j in scored for x in j.get("stats", {}).get("latency_s", [])]
    per_layout = {}
    for lay_id in dict.fromkeys(j["layout"] for j in run["trials"]):  # the run's own layouts, in order
        js = [j for j in scored if j["layout"] == lay_id]
        if js:
            per_layout[lay_id] = {"n": len(js), "success": sum(j["outcome"]["success"] for j in js),
                                     "progress": 100 * statistics.mean(progress(j) for j in js)}
    sub = run["submission"]
    return {
        "run_id": run["run_id"],
        "task_id": task_of(run)["task_id"],
        "example": bool(run.get("example")),
        "complete": all(j.get("outcome") for j in run["trials"]),
        "action": sub.get("action", "joints"),
        "mock": run.get("backend") != "FanucBackend",
        "team": sub.get("team", "?"),
        "policy": sub.get("policy", "?"),
        "authors": sub.get("authors", sub.get("team", "")),
        "model_type": sub.get("model_type", "other"),
        "parameters_m": sub.get("parameters_m"),
        "training_data": sub.get("training_data", "sim"),
        "sim_episodes": sub.get("sim_episodes"),
        "real_episodes": sub.get("real_episodes"),
        "cameras": sub.get("cameras", []),
        "action_rate_hz": sub.get("action_rate_hz"),
        "sim_success_rate": sub.get("sim_success_rate"),
        "weights_url": sub.get("weights_url") or "",
        "data_url": sub.get("data_url") or "",
        "code_url": sub.get("code_url") or "",
        "date": sub.get("date") or time.strftime("%Y-%m-%d", time.localtime(run["created"])),
        "trials": n,
        "trials_planned": len(run["trials"]),
        "successes": len(wins),
        "success_rate": 100 * len(wins) / n,
        "ci": [100 * lo, 100 * hi],
        "progress": 100 * statistics.mean(progress(j) for j in scored),
        "cycle_time_s": statistics.mean(cycles) if cycles else None,
        "faults": sum(bool(j["outcome"].get("fault")) for j in scored),
        "interventions": sum(bool(j["outcome"].get("intervention")) for j in scored),
        "latency_ms": 1000 * statistics.median(lat) if lat else None,
        "per_layout": per_layout,
        "strip": [{"trial_id": j["trial_id"], "video": video_cams(j), "episode": j.get("episode"),
                   "t": j.get("stats", {}).get("t_start"),
                   "layout": j["layout"], "stage": j["outcome"]["stage"], "success": j["outcome"]["success"],
                   "fault": bool(j["outcome"].get("fault")), "intervention": j["outcome"]["intervention"],
                   "notes": j["outcome"].get("notes", "")} for j in sorted(scored, key=lambda j: j["index"])],
    }


def data(runs: list[dict], include_mock: bool = False, examples: bool = True, include_hidden: bool = False) -> dict:
    """{"tasks", "rows", "built"} for these runs: what the page renders."""
    rows, tasks = [], {}
    ex = [json.loads(f.read_text()) for f in sorted((HERE / "examples").glob("*.json"))] if examples else []  # make_examples.py
    for run in ex + sorted(runs, key=lambda r: r["created"]):
        s = None if run.get("hidden") and not include_hidden else summarize(run)  # operator_console.py hide <run_id>
        if s and (include_mock or not s["mock"]):
            rows.append(s)
            tasks.setdefault(s["task_id"], task_of(run))
    for task_id, task in tasks.items():  # the clip on the task's card: the latest recorded success, else the latest
        recorded = [(t["success"], t["t"] or 0, r["run_id"], t["trial_id"]) for r in rows if r["task_id"] == task_id
                    for t in r["strip"] if t["video"]]
        if recorded:
            _, _, run_id, trial_id = max(recorded)
            cams = next(t["video"] for r in rows if r["run_id"] == run_id for t in r["strip"] if t["trial_id"] == trial_id)
            cam = next((c for c in ("table", "front", "side") if c in cams), cams[0])
            task["highlight"] = {"run_id": run_id, "trial_id": trial_id, "cam": cam}
    return {"tasks": tasks, "rows": rows, "built": time.strftime("%Y-%m-%d %H:%M")}


def run_data(run: dict) -> dict:
    """One run's results for its team's private view (My results), hidden or mock included."""
    return data([run], include_mock=True, examples=False, include_hidden=True)


def page(d: dict, mode: str = "full") -> str:
    """The site. mode "full": the leaderboard of d. mode "private": no leaderboard; each team loads its own runs
    (results/<key>.json) into My results with the key from its link."""
    return (HERE / "template.html").read_text().replace("/*__DATA__*/null", json.dumps({**d, "mode": mode}).replace("</", "<\\/"))


def build(runs: list[dict], include_mock: bool = False) -> str:
    """The full leaderboard page for these runs."""
    return page(data(runs, include_mock))

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=ROOT / "runs")
    p.add_argument("--out", type=Path, default=HERE / "index.html")
    p.add_argument("--include-mock", action="store_true", help="also list runs scored on the mock backend")
    args = p.parse_args()
    html = build([json.loads(f.read_text()) for f in sorted(args.runs.glob("r*.json"))], args.include_mock)
    args.out.write_text(html)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
