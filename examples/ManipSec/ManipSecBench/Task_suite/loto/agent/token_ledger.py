"""Token and cost ledger over every agent run: sums the requests in each run's
Claude Code session transcripts (pricing.transcript_requests) and prices them
at the official per-request rates (pricing.py).

    python token_ledger.py                       all runs in ~/loto_agent_runs
    python token_ledger.py --current-key         only runs paid by $ANTHROPIC_API_KEY
    python token_ledger.py --key-id 1a2b3c4d     only runs paid by that key
    python token_ledger.py --since 2026-10-20 --json

A run's key is the fingerprint in its usage.json ("key_id", agents.key_id:
the first 8 hex digits of the key's SHA-256). Runs from before fingerprints
were recorded are listed as "unrecorded". Regrades (--skip-agent) make no
requests and are left out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pricing  # noqa: E402

FIELDS = ("requests", "input", "cache_write", "cache_read", "output", "cost_usd")


def run_requests(run: Path):
    ws = run / "workspace"
    return pricing.transcript_requests(ws) if ws.is_dir() else []


def tally(requests):
    t = dict.fromkeys(FIELDS, 0)
    t["cost_usd"] = 0.0
    for model, u in requests:
        t["requests"] += 1
        t["input"] += u.get("input_tokens") or 0
        t["cache_write"] += u.get("cache_creation_input_tokens") or 0
        t["cache_read"] += u.get("cache_read_input_tokens") or 0
        t["output"] += u.get("output_tokens") or 0
        t["cost_usd"] += pricing.request_cost(model, u) or 0.0
    return t


def add(a, b):
    for k in FIELDS:
        a[k] += b[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=os.path.expanduser("~/loto_agent_runs"))
    ap.add_argument("--key-id", default=None)
    ap.add_argument("--current-key", nargs="?", const="ANTHROPIC_API_KEY", default=None,
                    help="only runs paid by the key in this variable (default ANTHROPIC_API_KEY)")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD[THH:MM]: runs started at or after")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    want = a.key_id
    if a.current_key:
        from agents import key_id
        want = key_id(os.environ.get(a.current_key))
        if not want:
            sys.exit(f"{a.current_key} is not set")
    since = datetime.fromisoformat(a.since).timestamp() if a.since else 0.0

    rows, by_key, by_task, by_model = [], defaultdict(lambda: dict.fromkeys(FIELDS, 0)), \
        defaultdict(lambda: dict.fromkeys(FIELDS, 0)), defaultdict(lambda: dict.fromkeys(FIELDS, 0))
    total = dict.fromkeys(FIELDS, 0)
    for run in sorted(Path(a.runs_dir).iterdir()):
        u = run / "usage.json"
        if not u.is_file() or run.stat().st_mtime < since:
            continue
        meta = json.loads(u.read_text())
        kid = meta.get("key_id") or "unrecorded"
        if want and kid != want:
            continue
        reqs = run_requests(run)
        if not reqs:
            continue
        t = tally(reqs)
        task = run.name.rsplit("-", 2)[0]
        rows.append({"run": run.name, "task": task, "model": meta.get("model"), "key_id": kid, **t})
        add(total, t)
        add(by_key[kid], t)
        add(by_task[task], t)
        for model, uu in reqs:
            add(by_model[pricing._base(model)], tally([(model, uu)]))

    if a.json:
        print(json.dumps({"runs": rows, "by_key": by_key, "by_task": by_task, "by_model": by_model,
                          "total": total}, indent=2, default=float))
        return

    def line(name, t):
        print(f"  {name:<44} {t['requests']:>6} {t['input']:>10,} {t['cache_write']:>12,} "
              f"{t['cache_read']:>14,} {t['output']:>11,} {t['cost_usd']:>10.4f}")
    head = f"  {'':<44} {'reqs':>6} {'input':>10} {'cache write':>12} {'cache read':>14} {'output':>11} {'USD':>10}"
    for title, d in (("by run", {r['run']: r for r in rows}), ("by task", by_task),
                     ("by model", by_model), ("by key", by_key)):
        print(title)
        print(head)
        for k, t in d.items():
            line(k, t)
    print("total")
    print(head)
    line(f"{len(rows)} agent runs", total)


if __name__ == "__main__":
    main()
