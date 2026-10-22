"""`./dev_run` in the agent's container: run program.py against the
development simulator (dev_server.py, in its own container).

    ./dev_run [--fast]          (DEV_SERVER=host:port and DEV_KEY in the env)

The program runs here, in the agent's container, and reaches the simulator
only through robot_api over the network. Results land in runs/devN/: the
simulator writes result.json, sim.log and snapshots/, this side program.log.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wire  # noqa: E402

PROGRAM_TIMEOUT = 1800.0


def main():
    backend = "kinematic" if "--fast" in sys.argv[1:] else "isaac"
    ws = Path.cwd()
    server = os.environ["DEV_SERVER"]
    ctl = wire.connect(server, os.environ["DEV_KEY"], timeout=60)
    ctl.send({"cmd": "run", "backend": backend})
    run = ctl.recv()
    if "error" in run:
        sys.exit(f"dev server: {run['error']}")
    out = ws / "runs" / run["run"]
    print(f"running program.py ({backend}) -> runs/{run['run']} ...", flush=True)

    host = wire.parse_addr(server)[0]
    env = dict(os.environ, LOTO_ADDR=f"{host}:{run['port']}", LOTO_KEY=run["key"],
               PYTHONUNBUFFERED="1")
    t0 = time.time()
    with open(out / "program.log", "w") as log:
        p = subprocess.Popen([sys.executable, str(HERE / "program_host.py"), "program.py"],
                             stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ws)
        try:
            rc = p.wait(timeout=PROGRAM_TIMEOUT)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            rc = "killed (wall-clock timeout)"
    ctl.send({"cmd": "finish"})
    fin = ctl.recv()
    ctl.close()

    tail = (out / "program.log").read_text(errors="replace")[-1500:]
    res = fin.get("result")
    if res is None:
        print("simulator failed; tail of its log:")
        print(fin.get("sim_log_tail") or "")
        print("program output (tail):\n" + tail)
        sys.exit(1)
    res["exit_code"] = rc
    res["wall_time"] = round(time.time() - t0, 1)
    n_snap = len(res.pop("snapshots", []))
    summary = (json.dumps(res, indent=2)
               + f"\n{n_snap} snapshots in runs/{run['run']}/snapshots\n"
               + "program output (tail):\n" + tail)
    (out / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
