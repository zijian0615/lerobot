"""Operator console for the FANUC ER-4iA evaluation cell. Talks to robot_server.py with the admin token.

    # teams send their submission.yaml with `run.py --submit`; review and approve it (their client gets the token)
    uv run python fanuc_er4ia_sim/challenge/operator_console.py requests
    uv run python fanuc_er4ia_sim/challenge/operator_console.py approve q1a2b3 [--trials 10]
    uv run python fanuc_er4ia_sim/challenge/operator_console.py reject q1a2b3 --reason "grip width 14 mm"
    # or register a submission.yaml by hand (prints a token to send them)
    uv run python fanuc_er4ia_sim/challenge/operator_console.py register --team UWM-CSI --policy "ACT v4" \
        --meta <their submission.yaml> [--trials 10]
    # walk through every unscored trial of a run: home, place the objects, type the instruction, ready, watch, score
    uv run python fanuc_er4ia_sim/challenge/operator_console.py run <run_id>
    uv run python fanuc_er4ia_sim/challenge/operator_console.py status | stop | recover
    uv run python fanuc_er4ia_sim/challenge/operator_console.py retry <trial_id> --reason "controller fault"
    uv run python fanuc_er4ia_sim/challenge/operator_console.py hide <run_id>     # keep a test run off the leaderboard
    # the public site shows each team only its own runs (My results) until you reveal the leaderboard
    uv run python fanuc_er4ia_sim/challenge/operator_console.py link <run_id>     # the team's My results link
    uv run python fanuc_er4ia_sim/challenge/operator_console.py reveal | unreveal
    uv run python fanuc_er4ia_sim/challenge/operator_console.py set-password   # for the site's Organizer view
    uv run python fanuc_er4ia_sim/challenge/operator_console.py score <trial_id> --stage 2 --notes "dropped on the rim"

During a running trial type ONE entry and press Enter (the stages come from the run's task card):
    s        success: the card's success stage was reached (stamps the cycle time now)
    0 .. N   the furthest stage reached (the success stage is the same as s)
    i<N>     the operator had to step in (moved an object, blocked a collision); stage N, never a success
    r        void this attempt (a cell fault, not the policy's) and run the trial again; the attempt stays in the file
    x        stop: drop the queue and halt (the physical E-stop is still the real safety device)
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import requests

import task_spec

HERE = Path(__file__).resolve().parent
SITE = os.environ.get("MANIPBENCH_SITE", "https://hackathon.sidclab.org")  # where teams open their results links
TASK = json.loads((HERE / "task_card.json").read_text())


class Admin:
    def __init__(self, url: str):
        self.url = url.rstrip("/") + "/admin"
        self.s = requests.Session()
        self.s.headers["x-admin-token"] = (HERE / ".admin_token").read_text().strip()

    def get(self, path):
        return self._check(self.s.get(f"{self.url}/{path}", timeout=10))

    def post(self, path, body=None):
        return self._check(self.s.post(f"{self.url}/{path}", json=body or {}, timeout=10))

    @staticmethod
    def _check(r):
        body = r.json()
        if r.status_code != 200:
            raise SystemExit(f"server: {body.get('error', body)}")
        return body


def wait_state(adm: Admin, want: str, timeout_s: float = 120) -> None:
    t0 = time.time()
    while (st := adm.get("status")["state"]) != want:
        if st in ("fault", "stopped"):
            raise SystemExit(f"robot is in {st}: clear it on the pendant, then `operator_console.py recover`")
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"timed out waiting for {want} (robot {st})")
        time.sleep(0.3)


def trial_status(adm: Admin, run_id: str, trial_id: str) -> dict:
    run = next(r for r in adm.get("runs") if r["run_id"] == run_id)
    return next(t for t in run["trials"] if t["trial_id"] == trial_id)


def recover_interactive(adm: Admin) -> None:
    """Fault or stop: the operator clears the pendant, then the server re-initialises RMI. Nothing is lost:
    every trial is already saved under runs/, and the server keeps running."""
    while (st := adm.get("status")["state"]) in ("fault", "stopped"):
        input(f"\n[robot {st}] On the pendant: check the alarm, clear the cell, RESET, STEP off, AUTO, "
              f"enable switch OFF. Enter to recover > ")
        r = adm.s.post(f"{adm.url}/recover", json={}, timeout=30)
        if r.status_code != 200:
            print(f"recover failed: {r.json().get('error', r.text)}")
            continue
        print("recovered.")
    wait_state(adm, "idle")


def todo_trials(adm: Admin, run_id: str) -> list[dict]:
    run = next((r for r in adm.get("runs") if r["run_id"] == run_id), None)
    if run is None:
        raise SystemExit(f"no run {run_id}")
    return [t for t in run["trials"] if t["outcome"] is None and t["status"] in ("waiting", "preparing", "ready")]


PASS_FAIL = [{"stage": 0, "name": "fail", "rule": "the task was not done"},
             {"stage": 1, "name": "success", "rule": "the task was done"}]


def scoring(run: dict) -> tuple[list[dict], int]:
    """The stages to score against (the submission's "stages"; without them, success / fail)."""
    s = task_spec.spec(run["submission"])
    return (PASS_FAIL if s["stages"] == task_spec.PASS_FAIL else s["stages"]), s["success_stage"]


def parse_score(line: str, stages: list[dict], top: int, t: float) -> dict | None:
    """One operator entry -> an outcome for /finish (None: not a score)."""
    valid = [str(s["stage"]) for s in stages if s["stage"] <= top]
    if line == "s" or line == str(top):
        outcome = {"stage": top, "t_success": t}
    elif line in valid:
        outcome = {"stage": int(line)}
    elif line.startswith("i") and line[1:] in valid:
        outcome = {"stage": int(line[1:]), "intervention": True}
    else:
        return None
    outcome["success_stage"] = top
    return outcome


def score_leftovers(adm: Admin, run: dict, stages: list[dict], top: int) -> None:
    """Trials that ended (time limit, fault, crash) but were never scored: score them, or void and rerun."""
    for t in [t for t in run["trials"] if t["outcome"] is None and t["status"] not in OPEN_STATUSES]:
        st = t["stats"]
        print(f"\n=== unscored trial {t['index']} ({t['trial_id']}): {t['status']}, {st.get('end_reason')}, "
              f"{st.get('actions')} actions, instruction \"{t.get('instruction')}\" ===")
        while True:
            line = input(f"score it (s / 0-{top} / i<N>) or r = void and rerun > ").strip().lower()
            if line == "r":
                reason = input("reason > ").strip()
                adm.post(f"trials/{t['trial_id']}/retry", {"reason": reason})
                print("voided; it runs again below.")
                break
            outcome = parse_score(line, stages, top, 0.0)
            if outcome is None:
                print("?")
                continue
            outcome.pop("t_success", None)  # scored after the fact: no cycle time
            outcome["notes"] = input("notes (optional) > ").strip()
            res = adm.post(f"trials/{t['trial_id']}/finish", outcome)["outcome"]
            print(f"scored: stage {res['stage']}, success {res['success']}")
            break


OPEN_STATUSES = ("waiting", "preparing", "ready", "running")


def setup_text(run: dict, layout_id: str) -> str:
    """Where the run's task card puts the fixtures and this layout's objects (base frame, mm)."""
    card = run["submission"].get("task_card")
    if not isinstance(card, dict):
        return "  (no task card: set up the task as the team described it)"
    fmt = lambda o: (f"  {o.get('name', '?'):<14} x {o['xy_m'][0] * 1000:4.0f}  y {o['xy_m'][1] * 1000:5.0f} mm"
                     + (f"  yaw {o['yaw_deg']:+.0f} deg" if "yaw_deg" in o else "")
                     + (f"  ({o['colour']})" if o.get("colour") else ""))
    lines = ["Fixtures:"] + [fmt(f) for f in card.get("fixtures", []) if "xy_m" in f]
    lay = next((l for l in card.get("layouts", []) if l.get("id") == layout_id), None)
    if lay:
        lines += [f"Layout {layout_id}:"] + [fmt(o) if "xy_m" in o else f"  {o.get('name', '?')}: {o.get('region', 'see task card')}"
                                              for o in lay.get("objects", [])]
    return "\n".join(lines)


LIVE = {"jpg": b"", "server": None}
LIVE_PAGE = b"""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1"><title>Placement</title>
<body style="margin:0;background:#141413;color:#f0eee6;font:14px system-ui"><img id=v style="width:100%;display:block">
<script>const v=document.getElementById('v');(function f(){const i=new Image();i.onload=()=>{v.src=i.src;setTimeout(f,400)};
i.onerror=()=>setTimeout(f,1000);i.src='frame.jpg?'+Date.now()})()</script>"""


def live_view(host: str, port: int = 9101) -> str:
    """Serve the latest placement image at http://<host>:<port> (lab network only), once per console."""
    if LIVE["server"] is None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body, ctype = (LIVE["jpg"], "image/jpeg") if self.path.startswith("/frame.jpg") else (LIVE_PAGE, "text/html")
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

        LIVE["server"] = ThreadingHTTPServer((host, port), H)
        threading.Thread(target=LIVE["server"].serve_forever, daemon=True).start()
    return f"http://{host}:{port}"


def place_objects(adm: Admin, run: dict, trial: dict, spec_path: Path) -> None:
    """Show this rollout's targets, then track the objects live until each is inside its area ([matched])."""
    import base64
    import select
    import sys

    import cv2
    import numpy as np

    import placement

    spec = placement.load(spec_path)
    cam = placement.oc.load_cam(placement.oc.DEFAULT_CAM)
    tgt = placement.targets(spec, trial["index"])
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {run['token']}"
    out_dir = HERE / "runs" / "placement" / run["run_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    shot = out_dir / f"{trial['trial_id']}.jpg"  # the outlines on the table image, for whoever wants to look
    import os
    from urllib.parse import urlparse

    try:
        url = live_view(urlparse(adm.url).hostname or "127.0.0.1")
    except OSError:
        url = None
    window = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))  # Qt aborts without a display
    print("Place the objects on their outlines (table camera" + (f": {url}" if url else "") + "):")
    print("\n".join(placement.lines(spec, tgt)))
    green, dim, reset = "\x1b[32m", "\x1b[2m", "\x1b[0m"
    names = list(spec["objects"])
    drawn = 0
    while True:
        try:
            r = s.get(f"{adm.url[:-len('/admin')]}/api/v1/observation",
                      params={"cameras": "table", "width": 1920, "height": 1080, "format": "jpeg", "quality": 92},
                      timeout=15)
            img = cv2.imdecode(np.frombuffer(base64.b64decode(r.json()["images"]["table"]), np.uint8), cv2.IMREAD_COLOR)
            res, ok = placement.check(spec, cam, img, tgt)
            view = placement.overlay(spec, cam, img, tgt, res)
            cv2.imwrite(str(shot), view)
            LIVE["jpg"] = cv2.imencode(".jpg", cv2.resize(view, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            if window:
                cv2.imshow("placement (table camera)", cv2.resize(view, (1280, 720)))
                cv2.waitKey(1)
        except Exception as e:  # noqa: BLE001  (keep tracking through a dropped frame)
            res, ok = {}, False
        rows = []
        for name in names:
            m = res.get(name)
            if not m or not m["found"]:
                rows.append(f"  {name:6s} {dim}not seen{reset}")
            else:
                row = f"  {name:6s} now    x {m['xy'][0] * 1000:5.0f}  y {m['xy'][1] * 1000:+5.0f} mm  yaw {m['yaw_deg']:+4.0f}°"
                rows.append(row + (f"   {green}[matched]{reset}" if m["ok"] else f"   {dim}{m['why']}{reset}"))
        rows.append(f"{green}All matched. Enter to start the rollout.{reset}" if ok
                    else f"{dim}Adjusting ... (f + Enter to start anyway){reset}")
        if drawn:
            sys.stdout.write(f"\x1b[{drawn}F")  # back to the first tracking line and redraw in place
        sys.stdout.write("".join(f"\x1b[2K{row}\n" for row in rows))
        sys.stdout.flush()
        drawn = len(rows)
        ready, _, _ = select.select([sys.stdin], [], [], 0.5)
        if ready:
            line = sys.stdin.readline().strip().lower()
            drawn += 1  # the line the operator typed
            if ok or line == "f":
                break
    if window:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    log = out_dir / "placements.json"
    rows_log = json.loads(log.read_text()) if log.exists() else {}
    rows_log[trial["trial_id"]] = {"targets": tgt, "measured": res, "ok": ok, "forced": not ok}
    log.write_text(json.dumps(rows_log, indent=1))


def run_trials(adm: Admin, run_id: str, default_instruction: str | None = None, placement_spec: Path | None = None) -> None:
    run = next((r for r in adm.get("runs") if r["run_id"] == run_id), None)
    if run is None:
        raise SystemExit(f"no run {run_id}")
    stages, top = scoring(run)
    print("Stages for this run" + ("" if stages is not PASS_FAIL else " (no task card: success / fail)") + ":")
    for s in stages:
        print(f"  {s['stage']}  {s['name']:<10} {s.get('rule', '')}" + ("   <- success (s)" if s["stage"] == top else ""))
    score_leftovers(adm, run, stages, top)
    print(f"{run['submission'].get('team')} / {run['submission'].get('policy')}: "
          f"{len(todo_trials(adm, run_id))} trials to run")
    recover_interactive(adm)
    while todo := todo_trials(adm, run_id):  # re-read each time: a retried trial comes back into the queue
        trial = todo[0]
        print(f"\n=== trial {trial['index']} ({trial['trial_id']}) ===")
        print("[home] moving the arm home ...")
        adm.post(f"trials/{trial['trial_id']}/prepare")
        try:
            wait_state(adm, "idle")
        except SystemExit as e:  # homing failed: recover, then this trial is prepared again
            print(e)
            recover_interactive(adm)
            continue
        card = task_spec.spec(run["submission"])["card"]
        setup = setup_text(run, trial["layout"]) if card else "\n".join(task_spec.setup_lines(run["submission"])[1:])
        print(setup)
        import rollout_setup  # noqa: PLC0415  (this rollout's objects and prompt: the team's placement.yaml, or ours)

        placement, source = rollout_setup.for_run(run["submission"])
        place = rollout_setup.entry(placement, trial["index"])
        print(f"Place ({'the team' if source == 'team' else 'default'}'s placement, x, y in the robot frame):")
        print("\n".join("  " + line for line in rollout_setup.lines(place)))
        if placement_spec:
            place_objects(adm, run, trial, placement_spec)
        else:
            input("Reset the cell as above and clear it. Enter when ready > ")
        default = (default_instruction or place["prompt"] or run["submission"].get("task")
                   or run["submission"].get("instruction"))
        while not (instruction := input(f"Instruction for this trial (Enter = \"{default}\") > " if default
                                        else "Instruction for this trial > ").strip() or default):
            pass
        adm.post(f"trials/{trial['trial_id']}/ready", {"instruction": instruction})
        print(f"instruction: \"{instruction}\"")
        print("waiting for the team's client to start the trial ...")
        while trial_status(adm, run_id, trial["trial_id"])["status"] == "ready":
            time.sleep(0.3)
        keys = f"s = success | 0-{top} = furthest stage | i<N> = you stepped in" if top > 1 else "s = success | 0 = fail | i<N> = you stepped in"
        print(f"RUNNING. Score with ONE entry: {keys} | x = stop | r = void and rerun")
        done = threading.Event()

        def ticker(tid=trial["trial_id"]):
            while not done.is_set():
                st = trial_status(adm, run_id, tid)
                if st["status"] != "running":
                    why = st["stats"].get("end_reason") or ""
                    hint = " (r = the cell's fault, not the policy's: void and rerun)" if st["status"] == "failed" else ""
                    print(f"\n[trial ended: {st['status']} {why}] score it (s / 0-{top} / i<N>){hint} > ",
                          end="", flush=True)
                    return
                time.sleep(0.5)

        threading.Thread(target=ticker, daemon=True).start()
        while True:
            line = input("> ").strip().lower()
            t = time.time()
            if line == "x":
                adm.post("stop")
                done.set()
                print(f"Stopped. Score it now (s / 0-{top} / i<N>) or r to void and rerun it.")
                continue
            if line == "r":
                if trial_status(adm, run_id, trial["trial_id"])["status"] == "running":
                    adm.post("stop")
                done.set()
                time.sleep(0.5)
                reason = input("reason (e.g. controller fault, camera, operator error) > ").strip()
                adm.post(f"trials/{trial['trial_id']}/retry", {"reason": reason})
                print("voided; this trial runs again (the attempt is kept in the run file, unscored).")
                break
            outcome = parse_score(line, stages, top, t)
            if outcome is None:
                print(f"? s / 0-{top} / i<N> / x / r")
                continue
            done.set()
            outcome["notes"] = input("notes (optional) > ").strip()
            res = adm.post(f"trials/{trial['trial_id']}/finish", outcome)["outcome"]
            print(f"scored: stage {res['stage']}, success {res['success']}, cycle {res['cycle_time_s']} s")
            break
        recover_interactive(adm)
    print("\nrun complete; the leaderboard is already up to date (GET /leaderboard).")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:9098")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--team", help="default: the submission's team")
    r.add_argument("--policy", help="default: the submission's policy")
    r.add_argument("--meta", type=Path, help="JSON with the leaderboard fields (see submissions/example.json)")
    r.add_argument("--task-card", type=Path, help="(older submissions) a separate task_card.json")
    r.add_argument("--trials", type=int, help="number of trials (default 10)")
    rp = sub.add_parser("run")
    rp.add_argument("run_id")
    rp.add_argument("--placement", type=Path, help="placement spec (placement.py): outline each rollout's targets "
                    "on the table camera and check the objects before it starts")
    rp.add_argument("--instruction", help="default instruction offered at every trial (Enter accepts it); "
                    "otherwise \"instruction\" from the run's --meta file, otherwise type one per trial")
    sub.add_parser("status")
    sub.add_parser("stop")
    sub.add_parser("recover")
    sub.add_parser("abort").add_argument("trial_id")
    rq = sub.add_parser("requests", help="submissions sent with `run.py --submit`")
    rq.add_argument("--all", action="store_true", help="also the approved and rejected ones")
    ap = sub.add_parser("approve", help="turn a request into a run; the team's client then gets its token")
    ap.add_argument("request_id")
    ap.add_argument("--trials", type=int, help="number of trials (default: what the team asked for, else 10)")
    rj = sub.add_parser("reject", help="decline a request; the team sees the reason")
    rj.add_argument("request_id")
    rj.add_argument("--reason", required=True)
    for verb, what in (("hide", "leave a run (a test, a dry run) off the leaderboard"), ("show", "list it again")):
        sub.add_parser(verb, help=what).add_argument("run_id")
    rt = sub.add_parser("retry", help="void a finished trial (kept unscored in the run file) and queue it again")
    rt.add_argument("trial_id")
    rt.add_argument("--reason", default="")
    sub.add_parser("link", help="a run's My results link (only its team should have it)").add_argument("run_id")
    sub.add_parser("reveal", help="show the full leaderboard on the public site (teams otherwise see only their own runs)")
    sub.add_parser("unreveal", help="hide it again")
    sub.add_parser("set-password", help="set the password of the site's Organizer view (signs everyone out)")
    sub.add_parser("floor", help="set the TCP floor (UF0 mm) between trials").add_argument("z_min_mm", type=float)
    s = sub.add_parser("score")
    s.add_argument("trial_id")
    s.add_argument("--stage", type=int, required=True)
    s.add_argument("--intervention", action="store_true")
    s.add_argument("--notes", default="")
    args = p.parse_args()
    adm = Admin(args.url)

    if args.cmd == "register":
        if args.meta and args.meta.suffix in (".yaml", ".yml"):
            import yaml

            meta = yaml.safe_load(args.meta.read_text()) or {}
        else:
            meta = json.loads(args.meta.read_text()) if args.meta else {}
        if args.task_card:
            meta["task_card"] = json.loads(args.task_card.read_text())
        if args.trials:
            meta["n_trials"] = args.trials
        run = adm.post("runs", {**meta, **{k: v for k, v in (("team", args.team), ("policy", args.policy)) if v}})
        print(f"run_id {run['run_id']}\ntoken  {run['token']}\ntrials {len(run['trials'])}\n"
              f"results {SITE}/#key={run['view_key']}  (send both to the team only)")
    elif args.cmd == "run":
        run_trials(adm, args.run_id, args.instruction, args.placement)
    elif args.cmd == "status":
        print(json.dumps(adm.get("status"), indent=1))
    elif args.cmd in ("stop", "recover"):
        print(adm.post(args.cmd))
    elif args.cmd == "floor":
        print(adm.post("config", {"z_min_mm": args.z_min_mm}))
    elif args.cmd == "requests":
        reqs = [r for r in adm.get("requests") if args.all or r["status"] == "pending"]
        if not reqs:
            print("no pending requests" if not args.all else "no requests")
        for r in reqs:
            sub_ = r["submission"]
            print(f"\n{r['id']}  [{r['status']}]  {time.strftime('%m-%d %H:%M', time.localtime(r['created']))}  from {r['ip']}"
                  + (f"  -> run {r['run_id']}" if r.get("run_id") else "") + (f"  ({r['reason']})" if r.get("reason") else ""))
            print(f"  {sub_.get('team')} / {sub_.get('policy')}  ({sub_.get('model_type')}, {sub_.get('action')} actions, "
                  f"cameras {', '.join(sub_.get('cameras') or [])})  contact {sub_.get('contact')}")
            for line in task_spec.setup_lines(sub_):
                print("  " + line)
            print(f"  Rollouts asked for: {sub_.get('rollouts', 10)}")
            print(f"  Custom hardware: {sub_.get('custom_hardware') or 'none'}")
    elif args.cmd == "approve":
        r = adm.post(f"requests/{args.request_id}/approve", {"trials": args.trials} if args.trials else {})
        print(f"approved {r['id']}: run {r['run_id']}. The team's client picks up its token by itself, and `run.py --status` "
              "prints its My results link.")
    elif args.cmd == "reject":
        r = adm.post(f"requests/{args.request_id}/reject", {"reason": args.reason})
        print(f"rejected {r['id']}: {r['reason']}")
    elif args.cmd == "link":
        run = next((r for r in adm.get("runs") if r["run_id"] == args.run_id), None)
        if run is None:
            raise SystemExit(f"no run {args.run_id}")
        print(f"{run['submission'].get('team', '?')} / {run['submission'].get('policy', '?')}: {SITE}/#key={run['view_key']}")
    elif args.cmd == "set-password":
        import getpass

        pw = getpass.getpass("new organizer password (10+ characters): ")
        if pw != getpass.getpass("again: "):
            raise SystemExit("the two passwords differ")
        adm.post("organizer-password", {"password": pw})
        print("organizer password set; everyone signed in as organizer is signed out")
    elif args.cmd in ("reveal", "unreveal"):
        site = adm.post("site", {"reveal": args.cmd == "reveal"})
        print("the public site now shows the full leaderboard" if site["reveal"]
              else "the public site now shows each team only its own runs")
    elif args.cmd in ("hide", "show"):
        print(adm.post(f"runs/{args.run_id}/{args.cmd}"))
    elif args.cmd == "retry":
        print(adm.post(f"trials/{args.trial_id}/retry", {"reason": args.reason}))
    elif args.cmd == "abort":
        print(adm.post(f"trials/{args.trial_id}/abort"))
    elif args.cmd == "score":
        run = next(r for r in adm.get("runs") if r["run_id"] == args.trial_id.rsplit("-t", 1)[0])
        print(adm.post(f"trials/{args.trial_id}/finish", {"stage": args.stage, "success_stage": scoring(run)[1],
                                                         "intervention": args.intervention, "notes": args.notes}))


if __name__ == "__main__":
    main()
