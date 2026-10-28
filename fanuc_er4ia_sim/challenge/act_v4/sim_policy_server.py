"""Serve act_v4/policy.py (the exact real-cell client processing) to eval_act_sim.py, over its socket protocol.

Images are area-resized to 640x480 first, as robot_server.py does before the cell client sees them. With
--state command (the default, as on the real arm) the policy ignores the sim's measured joints after step 0.

    uv run python fanuc_er4ia_sim/challenge/act_v4/sim_policy_server.py --ensemble 0.01 --gripper latch &
    OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 ~/isaacsim-venv/bin/python \\
      fanuc_er4ia_sim/eval_act_sim.py --seeds 100 101 102 --bg-plates fanuc_er4ia_sim/data/bg_plates_v2 \\
      --mat assets/surface_real.png --tone data/cam_tone_v2.json --video-dir fanuc_er4ia_sim/data/eval_v4_client
"""
from __future__ import annotations

import argparse
import pickle
import socket

import cv2
import numpy as np
from policy import Policy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--checkpoint", default="zijian2022/act_fanuc_er4ia_screw_single_v4")
    p.add_argument("--n-action-steps", type=int, default=10, help="without --ensemble: rows played per chunk")
    p.add_argument("--state", choices=["command", "measured"], default="command")
    p.add_argument("--ensemble", type=float)
    p.add_argument("--gripper", choices=["latch", "ensemble", "newest"], default="latch")
    p.add_argument("--open-steps", type=int, default=5)
    p.add_argument("--save-dir", help="per-step gripper log (steps.csv), one folder per seed")
    args = p.parse_args()
    pol = Policy(args.checkpoint, args.n_action_steps, args.state, args.ensemble, args.save_dir, 10**9,
                 args.gripper, args.open_steps)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(1)
    print(f"[sim-policy] listening {args.port}: state={args.state} ensemble={args.ensemble} gripper={args.gripper}",
          flush=True)
    queue: list[np.ndarray] = []  # without ensembling the cell plays a whole chunk before asking again
    while True:
        conn, _ = srv.accept()
        with conn:
            while True:
                head = conn.recv(8)
                if not head:
                    break
                n, buf = int.from_bytes(head, "big"), b""
                while len(buf) < n:
                    buf += conn.recv(n - len(buf))
                msg = pickle.loads(buf)  # noqa: S301 - local eval harness
                if msg["op"] == "reset":
                    pol.reset()
                    queue.clear()
                    reply = {"ok": True}
                else:
                    if not queue:
                        images = {k: cv2.resize(v, (640, 480), interpolation=cv2.INTER_AREA)
                                  for k, v in msg["images"].items()}
                        queue = list(pol.act({"joints": np.asarray(msg["state"], np.float32), "images": images}))
                    reply = {"action": queue.pop(0)}
                out = pickle.dumps(reply)
                conn.sendall(len(out).to_bytes(8, "big") + out)


if __name__ == "__main__":
    main()
