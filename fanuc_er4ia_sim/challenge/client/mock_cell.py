"""A local stand-in for the cell, to test your client before your slot. Nothing moves.

    uv run mock_cell.py [--trials 3] [--trial-s 20]

Same API, limits and timing as the real cell. Trials become ready on their own and end after --trial-s.
Joints follow your commands; each camera returns a synthetic test card at its native size. It checks the
plumbing (images, chunk shape, limits, timing, the trial flow), not whether a grasp would work.
"""
from __future__ import annotations

import argparse
import base64
import collections
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

NATIVE = {"table": (1920, 1080), "side": (1920, 1080), "wrist": (1280, 720)}
HOME = [-0.0065, 3.2909, -11.8967, -0.0007, -74.8218, -0.0286, 0.0]
LIMITS = [(-170, 170), (-110, 120), (-69, 205), (-190, 190), (-120, 120), (-360, 360)]
DT_RANGE, MAX_CHUNK, MAX_QUEUE = (0.1, 1.0), 100, 100



def mock_depth(camera: str = "table", q=None) -> np.ndarray:
    """The clear table as the real camera's depth stream sees it (uint16 mm, 0 = none), from calibration/; the wrist
    camera from the joints q."""
    from calibration.geometry import flange_pose, load
    cam = load().get(f"{camera}_depth")
    if cam is None:
        return np.full((480, 848) if camera == "wrist" else (400, 640), 350 if camera == "wrist" else 1000, np.uint16)
    w, h = cam["size"]
    T = flange_pose(q) @ cam["T"] if cam["frame"] == "flange" else cam["T"]
    v, u = np.mgrid[0:h, 0:w]
    ray = np.stack([(u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"], np.ones((h, w))], -1) @ T[:3, :3].T
    s = (-0.0051 - T[2, 3]) / ray[..., 2]
    return np.where((s > 0) & (s < 3), s * 1000, 0).astype(np.uint16)

class NotRunning(RuntimeError):
    """A chunk that arrives between trials: expected at the end of every trial, not counted as rejected."""


class Mock:
    def __init__(self, n_trials: int, trial_s: float, token: str, instruction: str, quiet: bool = False):
        self.token, self.trial_s = token, trial_s
        self.instruction = instruction
        self.trials = [{"trial_id": f"mock-t{i:02d}", "index": i, "layout": f"L{i % 5 + 1}", "status": "waiting",
                        "instruction": None} for i in range(n_trials)]
        self.lock = threading.Lock()
        self.queue: collections.deque = collections.deque()
        self.q, self.step, self.active, self.state = list(HOME), 0, None, "idle"
        self.tcp = None  # last TCP command; the mock has no kinematics, so tcp_pose is only known after one
        self.stats = collections.Counter()
        self.quiet = quiet  # run.py --mock prints its own summary
        self.rollouts: list[dict] = []  # per finished trial: actions, rejected, latency
        self.rejections: list[tuple[int, int, str]] = []  # (rollout, step, reason)
        self.current = 0
        self.last_obs: float | None = None
        self.latency: list[float] = []  # observation -> next chunk: the time your act() plus the network takes
        threading.Thread(target=self._operator, daemon=True).start()
        threading.Thread(target=self._play, daemon=True).start()

    def _operator(self) -> None:
        for t in self.trials:
            with self.lock:
                self.q, self.step, self.active, self.state, self.tcp = list(HOME), 0, t, "homing", None
            time.sleep(1.0)
            with self.lock:
                self.state, t["status"], t["instruction"] = "idle", "ready", self.instruction
            self.current = t["index"] + 1
            if not self.quiet:
                print(f"[mock] {t['trial_id']} ready", flush=True)
            while t["status"] == "ready":
                time.sleep(0.05)
            t0 = time.time()
            while time.time() - t0 < self.trial_s:
                time.sleep(0.05)
            with self.lock:
                self.queue.clear()
                t["status"], self.state = "done", "homing"
            lat = (f", observation -> chunk {1000 * sum(self.latency) / len(self.latency):.0f} ms mean / "
                   f"{1000 * max(self.latency):.0f} ms max") if self.latency else ""
            self.rollouts.append({"actions": self.stats["actions"], "rejected": self.stats["rejected"],
                                  "latency": list(self.latency)})
            if self.quiet:
                print(f"  rollout {t['index'] + 1}/{len(self.trials)}: {self.stats['actions']} actions, "
                      f"{self.stats['rejected']} rejected", flush=True)
            else:
                print(f"[mock] {t['trial_id']} done: {self.stats['chunks']} chunks, {self.stats['actions']} actions played, "
                      f"{self.stats['rejected']} rejected{lat}", flush=True)
            self.stats.clear()
            self.latency.clear()
            time.sleep(1.0)
        with self.lock:
            self.active, self.state = None, "idle"
        if not self.quiet:
            print("[mock] all trials done", flush=True)

    def _play(self) -> None:
        t_next = time.perf_counter()
        while True:
            with self.lock:
                item = self.queue.popleft() if self.queue and self.state == "running" else None
                if item:
                    if item[2] == "tcp":
                        self.tcp, self.q = item[0][:6], self.q[:6] + [item[0][6]]
                    else:
                        self.q, self.tcp = item[0], None
                    self.step += 1
                    self.stats["actions"] += 1
            if item is None:
                time.sleep(0.005)
                t_next = time.perf_counter()
                continue
            t_next += item[1]
            time.sleep(max(0.0, t_next - time.perf_counter()))

    def card(self, cam: str) -> np.ndarray:
        w, h = NATIVE[cam]
        base = {"table": (40, 90, 160), "side": (60, 140, 80), "wrist": (170, 90, 50)}[cam]
        img = np.empty((h, w, 3), np.uint8)
        img[:] = base
        x = int((self.step * 12) % w)
        img[:, max(0, x - 6): x + 6] = 255
        s = w / 640
        for k, (text, size) in enumerate([(f"MOCK {cam}", 1.6), (f"step {self.step}", 1.0),
                                          (" ".join(f"{v:.1f}" for v in self.q), 0.6)]):
            cv2.putText(img, text, (int(20 * s), int((60 + 45 * k) * s)), cv2.FONT_HERSHEY_SIMPLEX, size * s,
                        (255, 255, 255), max(1, int(2 * s)))
        return img


TCP_BOX = [(150.0, 650.0), (-450.0, 450.0), (-334.0, 300.0)]  # x, y, z mm in UF0


def validate_tcp(rows, dt) -> tuple[list, float]:
    a = np.asarray(rows, dtype=float)
    if a.ndim != 2 or a.shape[1] != 7 or not 1 <= len(a) <= MAX_CHUNK:
        raise ValueError(f"tcp must have shape (N, 7) with 1 <= N <= {MAX_CHUNK}, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("tcp contains NaN or inf")
    for k, (lo, hi) in enumerate(TCP_BOX):
        if a[:, k].min() < lo or a[:, k].max() > hi:
            raise ValueError(f"tcp {'xyz'[k]} outside [{lo}, {hi}] mm (UF0)")
    if np.abs(a[:, 3:6]).max() > 360:
        raise ValueError("tcp w, p, r must be within +-360 deg")
    a[:, 6] = np.clip(a[:, 6], 0.0, 1.0)
    if not DT_RANGE[0] <= float(dt) <= DT_RANGE[1]:
        raise ValueError(f"dt must be in [{DT_RANGE[0]}, {DT_RANGE[1]}] s")
    return a.tolist(), float(dt)


def validate(joints, dt) -> tuple[list, float]:
    a = np.asarray(joints, dtype=float)
    if a.ndim != 2 or a.shape[1] != 7 or not 1 <= len(a) <= MAX_CHUNK:
        raise ValueError(f"joints must have shape (N, 7) with 1 <= N <= {MAX_CHUNK}, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("joints contain NaN or inf")
    for k, (lo, hi) in enumerate(LIMITS):
        if a[:, k].min() < lo or a[:, k].max() > hi:
            raise ValueError(f"J{k + 1} outside [{lo}, {hi}] deg")
    a[:, 6] = np.clip(a[:, 6], 0.0, 1.0)
    if not DT_RANGE[0] <= float(dt) <= DT_RANGE[1]:
        raise ValueError(f"dt must be in [{DT_RANGE[0]}, {DT_RANGE[1]}] s")
    return a.tolist(), float(dt)


def handler(m: Mock):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def _route(self, method):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            parts = [p for p in url.path.split("/") if p][2:] if url.path.startswith("/api/v1/") else None
            if parts is None:
                return self._send(404, {"error": "not found"})
            if self.headers.get("Authorization") != f"Bearer {m.token}":
                return self._send(401, {"error": "unknown token"})
            try:
                with m.lock:
                    return self._api(method, parts, q)
            except NotRunning as e:
                return self._send(409, {"error": str(e)})
            except (ValueError, RuntimeError) as e:
                m.stats["rejected"] += 1
                m.rejections.append((m.current, m.step, str(e)))
                if not m.quiet:
                    print(f"[mock] rejected: {e}", flush=True)
                return self._send(400, {"error": str(e)})

        def _api(self, method, parts, q):
            t = m.active
            if method == "GET" and parts == ["run"]:
                return self._send(200, {"run_id": "mock", "trials": m.trials})
            if method == "GET" and parts == ["time"]:
                return self._send(200, {"t": time.time()})
            if parts[:1] == ["trials"] and len(parts) >= 2:
                trial = next((x for x in m.trials if x["trial_id"] == parts[1]), None)
                if trial is None:
                    return self._send(404, {"error": "not found"})
                if method == "GET" and len(parts) == 2:
                    return self._send(200, trial)
                if method == "POST" and parts[2:] == ["start"]:
                    if trial["status"] != "ready":
                        raise RuntimeError(f"trial is {trial['status']}, not ready")
                    trial["status"], m.state = "running", "running"
                    return self._send(200, {"ok": True})
            if method == "GET" and parts == ["observation"]:
                cams = [c for c in ",".join(q.get("cameras", ["table,side,wrist"])).split(",") if c]
                if any(c not in NATIVE for c in cams):
                    raise ValueError(f"unknown cameras {cams}; available {list(NATIVE)}")
                w, h = int(q.get("width", ["640"])[0]), int(q.get("height", ["360"])[0])
                if not (16 <= w <= 1920 and 16 <= h <= 1080):
                    raise ValueError("width/height out of range")
                fmt, quality = q.get("format", ["png"])[0], int(q.get("quality", ["90"])[0])
                if fmt not in ("png", "jpeg") or not 50 <= quality <= 100:
                    raise ValueError("format must be png or jpeg, quality 50-100")
                depth = [c for c in ",".join(q.get("depth", [""])).split(",") if c]
                if any(c not in ("table", "wrist") for c in depth):
                    raise ValueError(f"no depth for {[c for c in depth if c not in ('table', 'wrist')]}; "
                                     "depth is available for ['table', 'wrist']")
                imgs = {}
                for c in cams:
                    small = cv2.resize(m.card(c), (w, h), interpolation=cv2.INTER_AREA)
                    bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
                    buf = (cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]) if fmt == "jpeg"
                           else cv2.imencode(".png", bgr))[1]
                    imgs[c] = base64.b64encode(buf).decode()
                m.last_obs = time.time()
                out = {"t": m.last_obs, "status": m.state, "trial": dict(t) if t else None,
                       "queued": len(m.queue), "joints": m.q, "tcp_pose": m.tcp, "format": fmt, "images": imgs}
                if depth:
                    out["depth"] = {c: base64.b64encode(cv2.imencode(".png", mock_depth(c, m.q))[1]).decode() for c in depth}
                return self._send(200, out)
            if method == "POST" and parts == ["actions"]:
                if m.state != "running":
                    raise NotRunning(f"no trial is running (robot {m.state})")
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if ("joints" in body) == ("tcp" in body):
                    raise ValueError('send exactly one of "joints" or "tcp"')
                kind = "tcp" if "tcp" in body else "joints"
                chunk, dt = validate_tcp(body["tcp"], body.get("dt", 0.1)) if kind == "tcp" else validate(
                    body["joints"], body.get("dt", 0.1))
                skip, obs_t = 0, body.get("obs_t")
                if body.get("replace"):  # the same alignment as the cell (robot_server.Cell.post)
                    m.queue.clear()
                    if isinstance(obs_t, (int, float)) and 0 <= time.time() - obs_t < 30:
                        skip = min(int((time.time() - obs_t) / dt), len(chunk) - 1)
                    chunk = chunk[skip:]
                if len(m.queue) + len(chunk) > MAX_QUEUE:
                    raise RuntimeError("action queue is full")
                m.queue.extend((a, dt, kind) for a in chunk)
                m.stats["skipped"] += skip
                m.stats["chunks"] += 1
                if m.last_obs:
                    m.latency.append(time.time() - m.last_obs)
                    m.last_obs = None
                return self._send(200, {"ok": True, "queued": len(m.queue), "skipped": skip})
            return self._send(404, {"error": "not found"})

    return H


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=9098)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--trial-s", type=float, default=20.0)
    p.add_argument("--token", default="mock")
    p.add_argument("--instruction", default="pick one countersunk screw and drop it in the yellow bin",
                   help="what the operator would type for every trial (obs['trial']['instruction'])")
    args = p.parse_args()
    m = Mock(args.trials, args.trial_s, args.token, args.instruction)
    print(f"[mock] cell on http://127.0.0.1:{args.port}, token '{args.token}', {args.trials} trials of {args.trial_s:.0f} s",
          flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler(m)).serve_forever()


if __name__ == "__main__":
    main()
