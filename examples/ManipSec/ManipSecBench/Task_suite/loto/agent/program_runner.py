"""Run an agent-written program against a LOTO environment.

    python program_runner.py --program program.py --env loto.clean.fanuc \
        --phase dev|deploy --backend isaac|kinematic --out runs/dev1
    python program_runner.py --listen 0.0.0.0:7100 --env loto.env.fanuc \
        --phase deploy --out /out                 (LOTO_KEY in the env)

Two phases, as in the attack design (attacks/loto.py):
  dev     the agent's own test runs. Prompt and scene attacks are in effect
          (poisoned docs, forged lock); the environment attack is not. The
          result shows the program's output, the robot's final state and
          snapshots -- never the scores.
  deploy  the graded rollout of the submitted program, clean of the prompt
          and scene attacks (no forged lock): only what they left in the
          program counts. The environment attack (re-energization
          mid-maintenance) is in effect. The result has task_score and
          safety_score.

The program reaches the simulator only through robot_api.Robot, over a socket
in the JSON format of wire.py. Two ways to run it:
  --program  the runner starts the program itself, in a child process
             (program_host.py); for local runs without the sandbox
  --listen   the runner only serves: the program runs elsewhere (in the
             sandbox, its own container) and connects with LOTO_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_SUITE = HERE.parents[1]
sys.path.insert(0, str(TASK_SUITE))
sys.path.insert(0, str(HERE))

import wire  # noqa: E402


import tasks  # noqa: E402  (which task an env name belongs to)


def parse():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--program", help="run this program in a child process")
    src.add_argument("--listen", help="HOST:PORT; serve a program that connects "
                                      "from elsewhere (key in LOTO_KEY)")
    ap.add_argument("--env", default="loto.clean.fanuc")
    ap.add_argument("--phase", default="dev", choices=["dev", "deploy"])
    ap.add_argument("--backend", default="isaac", choices=["isaac", "kinematic"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--spp", type=int, default=1)
    ap.add_argument("--snap-every", type=float, default=None,
                    help="sim seconds between snapshots for the agent (task default)")
    ap.add_argument("--video-stride", type=int, default=0,
                    help="also save every Nth control step for a video (0 = off)")
    ap.add_argument("--max-sim-time", type=float, default=None, help="task default")
    ap.add_argument("--dev-feedback", default="none", choices=["none", "pass"],
                    help="dev phase: also tell the program's author whether the run did the "
                         "whole task (task.dev_check, judged on the development scene)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="wall-clock seconds")
    ap.add_argument("--accept-timeout", type=float, default=900.0,
                    help="--listen: seconds to wait for the program to connect")
    a = ap.parse_args()
    if a.program:
        a.program = os.path.abspath(a.program)
    a.out = os.path.abspath(a.out)
    return a


# SIGTERM: the program has exited (sent by whoever ran it, from another
# container). Stop waiting for it, but still write the result.
STOP = {"flag": False}


def _on_term(*_):
    STOP["flag"] = True


class Server:
    def __init__(self, env, args, task):
        self.env, self.args, self.task = env, args, task
        self.calls = 0
        self.next_snap = 0.0
        self.snapshots = []
        self.n_acts = 0
        self.limit_hit = None

    # ---- recording ----------------------------------------------------------
    def _save(self, img, path):
        import cv2
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def after_act(self):
        be = self.env.backend
        self.n_acts += 1
        if self.args.backend != "isaac":
            return
        if be.t >= self.next_snap:
            imgs = self.task.snapshot_images(self.env)
            for view, sub in self.task.SNAP_VIEWS.items():
                img = imgs.get(view)
                if img is not None:
                    p = os.path.join(self.args.out, "snapshots", sub, f"t{be.t:06.2f}.png")
                    self._save(img, p)
                    self.snapshots.append(os.path.relpath(p, self.args.out))
            self.next_snap = be.t + self.args.snap_every
        if self.args.video_stride and self.n_acts % self.args.video_stride == 0:
            img = self.task.video_image(self.env)
            if img is not None:
                self._save(img, os.path.join(self.args.out, "video",
                                             f"f{self.n_acts // self.args.video_stride:05d}.png"))

    # ---- request dispatch -----------------------------------------------------
    def _over_time(self):
        return self.env.backend.t > self.args.max_sim_time

    def handle(self, name, args):
        env = self.env
        if self._over_time():
            self.limit_hit = f"sim time limit ({self.args.max_sim_time:.0f} s) reached"
            raise TimeoutError(self.limit_hit)
        if name == "get_obs":
            return env.get_obs(images=bool(args and args[0]))
        if name == "send_act":
            if not args or not isinstance(args[0], dict):
                raise TypeError("send_act(action: dict, arm='A')")
            env.send_act(dict(args[0]), *(args[1:2] or ("A",)))
            self.after_act()
            return None
        if name == "wait":
            control_dt = getattr(env, "control_dt", env.backend.dt * 2)
            n = max(1, int(round(float(args[0]) / control_dt)))
            for _ in range(n):
                if self._over_time():
                    break
                env.send_act({})
                self.after_act()
            return None
        raise ValueError(f"unknown call {name!r}")

    def serve(self, conn, deadline):
        while True:
            if time.time() > deadline:
                self.limit_hit = "wall-clock timeout"
                return
            if STOP["flag"] and not conn.poll(0):
                return
            try:
                if not conn.poll(1.0):
                    continue
                req = conn.recv()
            except (EOFError, OSError, ValueError):
                return                        # program finished (or sent garbage)
            self.calls += 1
            try:
                if not (isinstance(req, list) and len(req) == 2
                        and isinstance(req[0], str) and isinstance(req[1], list)):
                    raise ValueError("malformed request")
                conn.send(["ok", self.handle(req[0], req[1])])
            except Exception as e:            # report to the program, keep serving
                try:
                    conn.send(["err", f"{type(e).__name__}: {e}"])
                except OSError:
                    return
                if self.limit_hit:
                    return


def encode_video(frames: Path, mp4: Path):
    import cv2
    fs = sorted(frames.glob("f*.png"))
    if not fs:
        return None
    h, w = cv2.imread(str(fs[0])).shape[:2]
    vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for f in fs:
        vw.write(cv2.imread(str(f)))
    vw.release()
    return mp4


def main():
    signal.signal(signal.SIGTERM, _on_term)
    args = parse()
    task = tasks.load(args.env)
    if args.snap_every is None:
        args.snap_every = task.SNAP_EVERY
    if args.max_sim_time is None:
        args.max_sim_time = task.MAX_SIM_TIME
    for sub in task.SNAP_VIEWS.values():
        os.makedirs(os.path.join(args.out, "snapshots", sub), exist_ok=True)
    if args.video_stride:
        os.makedirs(os.path.join(args.out, "video"), exist_ok=True)

    env = task.make_env(args.env, args.phase, backend=args.backend, spp=args.spp)
    be = env.backend
    srv = Server(env, args, task)
    srv.after_act()                           # snapshot of the start state

    if args.listen:
        key = os.environ["LOTO_KEY"]
        listener = wire.Server(args.listen, key)
        print(f"[runner] listening on {args.listen}", flush=True)
        child, log = None, None
        accept_timeout = args.accept_timeout
    else:
        key = secrets.token_hex(16)
        listener = wire.Server("127.0.0.1:0", key)
        host, port = listener.address
        log = open(os.path.join(args.out, "program.log"), "w")
        child_env = dict(os.environ, LOTO_ADDR=f"{host}:{port}", LOTO_KEY=key,
                         PYTHONUNBUFFERED="1")
        child = subprocess.Popen(
            [sys.executable, str(HERE / "program_host.py"), args.program],
            stdout=log, stderr=subprocess.STDOUT, env=child_env,
            cwd=os.path.dirname(args.program))
        accept_timeout = 60
    t0 = time.time()
    conn = None
    while conn is None and time.time() - t0 < accept_timeout and not STOP["flag"]:
        conn = listener.accept(1.0)
    if conn is None:
        srv.limit_hit = "program never connected"
    else:
        srv.serve(conn, t0 + args.timeout)
        conn.close()
    listener.close()
    rc = None
    if child is not None:
        try:
            rc = child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            rc = "killed"
        log.close()
    wall = time.time() - t0

    obs = env.get_obs(images=False)
    result = {
        "phase": args.phase,
        "env": args.env,
        "exit_code": rc,
        "limit": srv.limit_hit,
        "api_calls": srv.calls,
        "sim_time": round(be.t, 2),
        "wall_time": round(wall, 1),
        "final_ee_position": [round(float(v), 4) for v in obs["ee_position"]],
        "program_log": "program.log",
        "snapshots": srv.snapshots,
    }
    for key, name in (("gripper_state", "final_gripper"), ("driver", "final_driver")):
        if key in obs:                        # the end effector's state: a gripper or a driver
            result[name] = obs[key]
    if args.program:                          # which program was run (the deployed one: graded)
        import hashlib
        result["program_sha256"] = hashlib.sha256(Path(args.program).read_bytes()).hexdigest()
    if args.phase == "dev" and args.dev_feedback == "pass":
        check = getattr(task, "dev_check", None)
        result["task_check"] = ("unavailable" if check is None
                                else "PASS" if check(env) else "NO PASS")
        result["task_check_means"] = ("whether this run did the whole task, judged on this "
                                      "run's scene; it reports nothing else")
    if args.phase == "deploy":
        rep = env.report()
        result["grading"] = {k: rep[k] for k in task.GRADING_KEYS if k in rep}
    if hasattr(task, "write_extra"):
        task.write_extra(env, Path(args.out), args.phase)
        if args.video_stride:
            mp4 = encode_video(Path(args.out) / "video", Path(args.out) / "deploy.mp4")
            result["video"] = mp4.name if mp4 else None
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=2, default=lambda o: getattr(o, "tolist", lambda: str(o))())
    summary = (json.dumps({k: v for k, v in result.items() if k != "snapshots"}, indent=2,
                          default=lambda o: getattr(o, "tolist", lambda: str(o))())
               + f"\n{len(srv.snapshots)} snapshots in {args.out}/snapshots\n")
    if child is not None:                     # with --listen the log is on the program's side
        tail = open(os.path.join(args.out, "program.log")).read()[-1500:]
        summary += "program output (tail):\n" + tail
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)
    env.close()


if __name__ == "__main__":
    main()
