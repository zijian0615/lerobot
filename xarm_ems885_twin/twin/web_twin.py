"""Browser view of the EMS885 xArm twin (three.js), no Isaac Sim needed. READ-ONLY: no motion command is ever sent.

    uv run python xarm_ems885_twin/twin/web_twin.py                      # read the arms directly (on the Jetson)
    uv run python xarm_ems885_twin/twin/web_twin.py --source demo        # no robot
    uv run python xarm_ems885_twin/twin/web_twin.py --source udp         # fed by publish_joints.py
then open http://<this machine>:8770/ (e.g. http://10.20.88.64:8770/ on the Jetson's Wi-Fi, or forward the port over SSH).

Same model, same FK loop (omni_twin.run) and same scan alignment as the Isaac Sim twin; only the renderer differs. The page
loads the STL meshes, the GLB scan and the chain once, then polls /state (joint values incl. the gripper linkage, per-arm
status, flange position and TCP trail) about 30 times a second.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omni_twin  # noqa: E402
from sources import add_source_args, make_source  # noqa: E402
from xarm_model import Chain, build_spec  # noqa: E402

HERE = Path(__file__).resolve().parent
TWIN = HERE.parent
WEB = HERE / "web"
ASSETS = TWIN / "assets"
CONFIG = TWIN / "twin_config.json"


class WebWriter:
    """omni_twin writer that keeps the latest state for the HTTP handler."""

    def __init__(self, chain):
        self.chain = chain
        self.lock = threading.Lock()
        self.joint_values, self.status, self.trails = {}, {}, {}

    def set_pose(self, joint_values):
        with self.lock:
            self.joint_values = dict(joint_values)

    def set_status(self, arm, status):
        with self.lock:
            self.status[arm] = status

    def set_trail(self, arm, points):
        with self.lock:
            self.trails[arm] = [[round(float(c), 4) for c in p] for p in points[-150:]]

    def snapshot(self, src):
        with self.lock:
            vals, status, trails = dict(self.joint_values), dict(self.status), dict(self.trails)
        samples = src.latest()
        now = time.monotonic()
        arms = {}
        for arm in self.chain.arms:
            s = samples.get(arm)
            info = {"status": status.get(arm, "none"), "trail": trails.get(arm, [])}
            if s is not None and vals:
                info.update({
                    "age_ms": round(1000 * (now - s.t_recv)),
                    "q_deg": [round(float(np.degrees(v)), 2) for v in s.q],
                    "flange_mm": [round(float(v), 1) for v in self.chain.flange_in_base_mm(arm, vals)],
                    "gripper_pos": s.gripper_pos,
                    "limits": self.chain.limit_violations(arm, s.q),
                })
            arms[arm] = info
        return {"joints": {k: round(float(v), 5) for k, v in vals.items()}, "arms": arms, "source": src.status}


def make_handler(state_fn, files):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the terminal for the twin's status lines
            pass

        def _send(self, body, ctype, cache=True):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path == "/state":
                    return self._send(json.dumps(state_fn()).encode(), "application/json", cache=False)
                if path in files:
                    body, ctype = files[path]
                    return self._send(body() if callable(body) else body, ctype, cache=path != "/")
                if path.startswith("/assets/"):
                    target = (ASSETS / path[len("/assets/"):]).resolve()
                    if ASSETS.resolve() in target.parents and target.suffix == ".stl" and target.is_file():
                        return self._send(target.read_bytes(), "model/stl")
                self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.set_defaults(source="xarm")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--trail", type=int, default=300)
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = until Ctrl+C)")
    a = ap.parse_args(argv)

    cfg = json.loads(a.config.read_text())
    base = a.config.resolve().parent
    spec = build_spec(cfg)
    chain = Chain(spec)
    align = json.loads((base / cfg["scan"]["alignment"]).read_text())
    glb = (base / cfg["scan"]["glb"]).read_bytes()
    scene = {"chain": spec, "T_world_scan": align["T_world_scan"], "table_height_m": cfg.get("table_height_m", 0.75)}
    files = {
        "/": (lambda: (WEB / "index.html").read_bytes(), "text/html; charset=utf-8"),
        "/scene.json": (json.dumps(scene).encode(), "application/json"),
        "/scan.glb": (glb, mimetypes.types_map.get(".glb", "model/gltf-binary")),
    }

    src = make_source(a, chain)
    writer = WebWriter(chain)
    stop = threading.Event()
    t_end = time.monotonic() + a.duration if a.duration else None
    alive = lambda: not stop.is_set() and (t_end is None or time.monotonic() < t_end)  # noqa: E731
    loop = threading.Thread(target=omni_twin.run, args=(a, chain, src, writer, (lambda: None), alive, 1 / 30), daemon=True)
    loop.start()
    server = ThreadingHTTPServer((a.bind, a.port), make_handler(lambda: writer.snapshot(src), files))
    server.timeout = 0.5
    print(f"web twin on http://{a.bind}:{a.port}/  (source: {a.source}, read-only). Ctrl+C to stop.", flush=True)
    try:
        while alive():
            server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        src.close()
    return writer


if __name__ == "__main__":
    main()
