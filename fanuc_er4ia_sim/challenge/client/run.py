"""Run your policy on the cell, or on a local mock of it.

    uv run run.py --mock         # check formats and connection against a local mock (3 rollouts)
    uv run run.py --submit       # send submission/submission.yaml; we approve it into a run
    uv run run.py --status       # is it approved yet? once it is: your results link
    uv run run.py                # during your slot: runs your approved submission
    uv run run.py --download     # your run's recorded LeRobotDataset

Load your model in policy.py, or pass its path with --checkpoint <path> (given to Policy.__init__).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

from cell_client import CellClient, run_all
from policy import Policy

HERE = Path(__file__).resolve().parent
SUBMISSION = HERE / "submission" / "submission.yaml"
PLACEMENT = HERE / "submission" / "placement.yaml"  # each rollout's objects and prompt, when they change
REQUEST = HERE / ".request.json"  # the request id and its secret, written by --submit (keep it private)


def start_mock(trials: int, trial_s: float, instruction: str):
    """Serve mock_cell.py on a free local port in this process; returns its URL and the mock."""
    from mock_cell import Mock, handler

    mock = Mock(trials, trial_s, "mock", instruction, quiet=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(mock))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}", mock


class Checked:
    """The policy, with each act() timed and its first exception kept, for the mock's verdict."""

    def __init__(self, policy):
        self.policy, self.times, self.error, self.first = policy, [], None, []

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def reset(self):
        self.policy.reset()
        self.first.append(None)

    def act(self, obs):
        t = time.perf_counter()
        try:
            out = self.policy.act(obs)
        except Exception as e:
            self.error = e
            raise
        self.times.append(time.perf_counter() - t)
        if self.first and self.first[-1] is None:
            self.first[-1] = out
        return out


def verdict(mock, checked: Checked) -> int:
    """Print whether the policy is ready for the real robot; returns the exit code."""
    import numpy as np

    n_act = sum(r["actions"] for r in mock.rollouts)
    n_rej = len(mock.rejections)
    act_s = sorted(checked.times)[len(checked.times) // 2] if checked.times else None
    print()
    if checked.error is not None:
        import traceback

        frame = traceback.extract_tb(checked.error.__traceback__)[-1]
        print(f"✗ Not ready: act() raised {type(checked.error).__name__}: {checked.error}")
        print(f"  at {Path(frame.filename).name}, line {frame.lineno}: {frame.line}")
        print("  Fix it in policy.py, then run again.")
        return 1
    if n_rej:
        seen = []
        for rollout, step, reason in mock.rejections:
            if reason not in (r for r, _, _ in seen):
                seen.append((reason, rollout, step))
        print(f"✗ Not ready: {n_rej} action{'s were' if n_rej > 1 else ' was'} rejected")
        for reason, rollout, step in seen[:3]:
            print(f'  "{reason}"   first at rollout {rollout}, step {step + 1}')
        print("  The real robot rejects the same actions. Fix them in policy.py, then run again.")
        return 1
    if n_act == 0:
        print("✗ Not ready: no action reached the cell. Does act() return a target?")
        return 1
    print("✓ Ready for the real robot")
    took = "" if act_s is None else (" · act() takes " + ("under 1 ms" if act_s < 0.001 else f"{act_s * 1000:.0f} ms"
                                                          if act_s < 1 else f"{act_s:.1f} s (fine: the arm waits for it)"))
    print(f"  {len(mock.rollouts)} rollouts · {n_act} actions · 0 rejected{took}")
    if Policy.ACTION == "joints":
        for out in checked.first:
            a = np.asarray(out, dtype=float) if out is not None else None
            if a is not None and a.size and np.all(np.abs(a.reshape(-1, 7)[0, :6]) < 5.0):
                print("  Note: your first joint target is close to 0 deg on every joint. Joint actions are absolute\n"
                      "  angles to move to, not changes from the current pose.")
                break
    from cell_client import PREFETCH_STATS as ps

    if getattr(Policy, "PREFETCH", False) is True and ps["short"] > ps["chunks"] // 2:
        print("  PREFETCH is on but rarely used: act() returns fewer actions than twice the time it takes, so\n"
              "  the arm cannot keep moving while it plans. Return more actions per call to benefit from it.")
    print("  The mock's images are synthetic, so your policy may not do anything useful here. That is fine:\n"
          "  the mock only checks that your policy and the cell understand each other.")
    return 0


def submit(url: str, ca: str | None) -> None:
    import yaml

    sub = yaml.safe_load(SUBMISSION.read_text()) or {}
    if str(sub.get("custom_hardware", "")).strip().lower() in ("none", "no", ""):
        sub["custom_hardware"] = None
    sub |= {"action": Policy.ACTION, "cameras": list(Policy.CAMERAS),  # from policy.py, so they always match
            "depth": list(getattr(Policy, "DEPTH", []))}
    place = (yaml.safe_load(PLACEMENT.read_text()) or {}) if PLACEMENT.exists() else {}
    entries = place.get("rollouts") or []
    changes = "different" in (str(sub.get("objects")).strip().lower(), str(sub.get("layout")).strip().lower())
    want = sub.get("rollouts", 10)
    if changes and len(entries) != want:
        sys.exit(f"objects or layout is \"different\" in submission.yaml, so submission/placement.yaml lists every "
                 f"rollout: it has {len(entries)} {'entry' if len(entries) == 1 else 'entries'}, rollouts is {want}")
    if entries:
        sub["placement"] = place  # the cell checks it, and says what is wrong
    out = CellClient(url, "", ca).submit(sub)
    REQUEST.write_text(json.dumps({"url": url, **out}))
    REQUEST.chmod(0o600)
    print(f"submitted as request {out['request_id']}.\n"
          f"  Next: pick a 30-minute slot with this id at https://hackathon.sidclab.org/#calendar; we approve it for that slot.\n"
          f"  `uv run run.py --status` shows when it is approved, and then your results link.\n"
          f"  During your slot: `uv run run.py`, and leave it running.\n"
          f"  Keep .request.json: your client needs it, and it is private.")


def status(url: str, ca: str | None, wait: bool) -> dict:
    if not REQUEST.exists():
        sys.exit("no submission yet: fill in submission/submission.yaml, then `uv run run.py --submit`")
    req = json.loads(REQUEST.read_text())
    cell = CellClient(req.get("url", url), "", ca)
    while True:
        st = cell.request_status(req["request_id"], req["secret"])
        if st["status"] != "pending" or not wait:
            return st
        logging.info("request %s is waiting for the operator's approval ...", req["request_id"])
        time.sleep(15)


def results_link(st: dict) -> str:
    """Where this team (and only this team) sees its scores and videos."""
    site = json.loads(REQUEST.read_text()).get("url", "https://hackathon.sidclab.org").rstrip("/")
    return (f"your results (scores and videos, only your team sees them): {site}/#key={st['view_key']}\n"
            f"  keep this link private, like .request.json")


def download(url: str, ca: str | None) -> None:
    """The approved run's LeRobotDataset (every recorded trial so far), into LeRobot's default dataset folder."""
    import tarfile

    import requests

    st = status(url, ca, wait=False)
    if not st.get("view_key"):
        sys.exit(f"request {st['request_id']} is {st['status']}: no rollout recorded yet")
    site = json.loads(REQUEST.read_text()).get("url", url).rstrip("/")
    root = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot")) / "manipbench"
    dest = root / st["run_id"]
    tmp = root / f".{st['run_id']}.tar"
    root.mkdir(parents=True, exist_ok=True)
    with requests.get(f"{site}/vizdata/download/{st['view_key']}/{st['run_id']}.tar", stream=True, timeout=60,
                      verify=ca or True) as r:
        if r.status_code == 404:
            sys.exit("no recorded rollouts to download yet")
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if dest.exists():
        shutil.rmtree(dest)  # the run's latest recording replaces the older copy
    with tarfile.open(tmp) as tar:
        try:
            tar.extractall(root, filter="data")
        except TypeError:  # Python before 3.10.12: no filters; accept only plain files inside the run's folder
            names = tar.getmembers()
            if any(not (m.isfile() or m.isdir()) or not Path(m.name).parts[:1] == (st["run_id"],) or ".." in Path(m.name).parts
                   for m in names):
                sys.exit("the download looks wrong; ask us in Slack")
            tar.extractall(root, members=names)
    tmp.unlink()
    print(f"downloaded run {st['run_id']} to {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mock", action="store_true", help="start a local mock of the cell and run against it")
    p.add_argument("--submit", action="store_true", help="send submission/submission.yaml for approval")
    p.add_argument("--status", action="store_true", help="show whether your submission is approved")
    p.add_argument("--download", action="store_true", help="download your run's recorded LeRobotDataset")
    p.add_argument("--trials", type=int, default=3, help="--mock: number of trials")
    p.add_argument("--trial-s", type=float, default=20.0, help="--mock: seconds per trial")
    p.add_argument("--instruction", default="pick up the cube", help="--mock: the instruction for every trial")
    p.add_argument("--url", default="https://hackathon.sidclab.org")
    p.add_argument("--token", help="a run token, if we sent you one directly")
    p.add_argument("--checkpoint")
    p.add_argument("--ca", help="certificate file, only for a server with a self-signed certificate")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.submit:
        return submit(args.url, args.ca)
    if args.download:
        return download(args.url, args.ca)
    if args.status:
        st = status(args.url, args.ca, wait=False)
        print(f"request {st['request_id']}: {st['status']}" + (f" ({st['reason']})" if st.get("reason") else "")
              + (f", run {st['run_id']}" if st.get("run_id") else ""))
        if st.get("view_key"):
            print(results_link(st))
        return
    if args.mock:
        url, mock = start_mock(args.trials, args.trial_s, args.instruction)
        logging.getLogger().setLevel(logging.ERROR)  # the summary below says what matters
        print(f"Checking your policy on a local mock of the cell ({args.trials} rollouts, no robot) ...")
        checked = Checked(Policy(args.checkpoint))
        try:
            run_all(CellClient(url, "mock", args.ca), checked)
        except Exception:  # noqa: BLE001  (reported by verdict)
            if checked.error is None:
                raise
        sys.exit(verdict(mock, checked))
    if args.token:
        url, token = args.url, args.token
    else:
        st = status(args.url, args.ca, wait=True)
        if st["status"] != "approved":
            sys.exit(f"request {st['request_id']} was {st['status']}: {st.get('reason') or 'ask us in Slack'}")
        url, token = json.loads(REQUEST.read_text()).get("url", args.url), st["token"]
        logging.info("request %s approved: run %s", st["request_id"], st["run_id"])
        if st.get("view_key"):
            logging.info(results_link(st))
    run_all(CellClient(url, token, args.ca), Policy(args.checkpoint))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as e:  # the cell's answer (or no answer): one line, no traceback
        sys.exit(f"error: {e}")
