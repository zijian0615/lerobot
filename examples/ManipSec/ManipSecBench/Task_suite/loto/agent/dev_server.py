"""Development-run service, in the simulator container of the sandbox.

    python dev_server.py --env loto.scene.fanuc --listen 0.0.0.0:7000 \
        --runs /runs                                  (DEV_KEY in the env)

The agent's `./dev_run` (dev_client.py, in the agent's container) asks it for
a run; it starts program_runner.py --phase dev --listen on PROGRAM_PORT and
answers with that run's key. The agent's program then connects from the
agent's container. The environment is fixed when the service starts: the
agent can pick only the backend, never the env or the phase.

One run at a time (the GPU holds one Isaac Sim). Requests, as wire.py frames:
    {"cmd": "run", "backend": "isaac" | "kinematic"}
        -> {"run": "devN", "port": PROGRAM_PORT, "key": ...}
    {"cmd": "finish"}     (the program has exited)
        -> {"result": result.json minus grading, or null, "sim_log_tail": ...}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wire  # noqa: E402

BACKENDS = ("isaac", "kinematic")


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--listen", default="0.0.0.0:7000")
    ap.add_argument("--program-port", type=int, default=7100)
    ap.add_argument("--runs", default="/runs")
    ap.add_argument("--spp", type=int, default=1)
    ap.add_argument("--max-sim-time", type=float, default=120.0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--finish-grace", type=float, default=600.0,
                    help="seconds the runner gets to write its result after the program "
                         "exits (covers Isaac Sim still starting up)")
    return ap.parse_args()


def next_run_dir(runs: Path) -> Path:
    taken = [int(m.group(1)) for p in runs.iterdir()
             if (m := re.fullmatch(r"dev(\d+)", p.name))]
    n = max(taken, default=0) + 1
    while True:
        d = runs / f"dev{n}"
        try:
            d.mkdir()                 # never reuse a path the agent created
            return d
        except FileExistsError:
            n += 1


class Run:
    def __init__(self, args, backend: str):
        self.dir = next_run_dir(Path(args.runs))
        self.key = secrets.token_hex(16)
        self.log = open(self.dir / "sim.log", "w")
        cmd = [sys.executable, str(HERE / "program_runner.py"),
               "--listen", f"0.0.0.0:{args.program_port}",
               "--env", args.env, "--phase", "dev", "--backend", backend,
               "--spp", str(args.spp), "--max-sim-time", str(args.max_sim_time),
               "--timeout", str(args.timeout), "--out", str(self.dir)]
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, LOTO_KEY=self.key,
                                              OMNI_KIT_ACCEPT_EULA="YES"))

    def finish(self, grace: float) -> dict:
        # the program has exited: if it never connected, the runner is still
        # waiting for it (SIGTERM ends the wait; harmless once it is done)
        if self.proc.poll() is None:
            self.proc.terminate()
        try:
            self.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.log.close()
        res = self.dir / "result.json"
        result = json.loads(res.read_text()) if res.exists() else None
        if result is not None:
            result.pop("grading", None)          # dev runs never report scores
        tail = (self.dir / "sim.log").read_text(errors="replace")[-2000:]
        return {"result": result, "sim_log_tail": None if result else tail}

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.log.close()


def serve_client(conn: wire.Conn, args):
    run = None
    try:
        while True:
            req = conn.recv()
            if not isinstance(req, dict):
                conn.send({"error": "malformed request"})
                continue
            cmd = req.get("cmd")
            if cmd == "run" and run is None:
                backend = req.get("backend", "isaac")
                if backend not in BACKENDS:
                    conn.send({"error": f"backend must be one of {BACKENDS}"})
                    continue
                run = Run(args, backend)
                print(f"[dev_server] {run.dir.name}: {backend}", flush=True)
                conn.send({"run": run.dir.name, "port": args.program_port, "key": run.key})
            elif cmd == "finish" and run is not None:
                conn.send(run.finish(args.finish_grace))
                run = None
                return
            else:
                conn.send({"error": f"unexpected request {cmd!r}"})
    except (EOFError, OSError, ValueError, wire.WireError):
        pass
    finally:
        if run is not None:
            run.kill()
        conn.close()


def main():
    args = parse()
    Path(args.runs).mkdir(parents=True, exist_ok=True)
    srv = wire.Server(args.listen, os.environ["DEV_KEY"])
    print(f"[dev_server] {args.env} on {args.listen}", flush=True)
    while True:
        conn = srv.accept(timeout=3600)
        if conn is not None:
            serve_client(conn, args)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
