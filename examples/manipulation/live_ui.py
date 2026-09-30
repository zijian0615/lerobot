# Copyright 2026 The Hugging Face Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Live task dashboard: top + wrist cameras, timer, phase log."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PANEL_W = 640
PANEL_H = 480

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>live</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; height: 100%;
    background: #111; color: #e8e8e8;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }
  .wrap { padding: 16px 18px 20px; max-width: 1980px; margin: 0 auto; }
  .row { display: flex; gap: 12px; align-items: stretch; }
  .panel {
    width: 640px; height: 480px; flex: 0 0 640px;
    background: #000; border: 1px solid #333; position: relative; overflow: hidden;
  }
  .panel img { width: 640px; height: 480px; object-fit: cover; display: block; }
  .tag {
    position: absolute; left: 10px; top: 10px;
    font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase;
    background: rgba(0,0,0,0.65); padding: 4px 8px;
  }
  .timer {
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    background: #161616;
  }
  .timer .clock {
    font-variant-numeric: tabular-nums;
    font-size: 92px; line-height: 1; font-weight: 600; letter-spacing: 0.04em;
    color: #f2f2f2;
  }
  .timer .clock .ms { color: #7a7a7a; font-weight: 500; }
  .timer .sub { margin-top: 28px; font-size: 42px; color: #d8d8d8; letter-spacing: 0.08em; }
  .timer button {
    margin-top: 28px;
    font-size: 18px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 12px 22px;
    background: #2c2c2c;
    color: #f0f0f0;
    border: 1px solid #5a5a5a;
    cursor: pointer;
  }
  .timer button:disabled { opacity: 0.35; cursor: not-allowed; }
  .timer button:not(:disabled):hover { background: #3a3a3a; }
  .below { margin-top: 16px; }
  .now {
    font-size: 28px; font-weight: 560; letter-spacing: 0.02em;
    padding: 14px 4px 10px; border-bottom: 1px solid #2a2a2a;
  }
  .now .phase { color: #fff; }
  .now .meta { color: #9a9a9a; font-weight: 400; font-size: 18px; }
  .log {
    margin-top: 10px; max-height: 220px; overflow: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 14px; line-height: 1.55; color: #bdbdbd;
  }
  .log .t { color: #7d7d7d; margin-right: 12px; }
  .instr { margin-bottom: 10px; line-height: 1.4; }
  .instr .label { color: #fff; font-size: 22px; font-weight: 560; margin-right: 12px; }
  .instr .body { color: #9a9a9a; font-size: 18px; font-weight: 400; }
</style>
</head>
<body>
<div class="wrap">
  <div class="row">
    <div class="panel">
      <div class="tag">top</div>
      <img id="top" src="/stream/top" alt="top"/>
    </div>
    <div class="panel" id="wristPanel">
      <div class="tag" id="wristTag">wrist</div>
      <img id="wrist" src="/stream/wrist" alt="wrist"/>
    </div>
    <div class="panel timer">
      <div class="clock" id="clock">00:00<span class="ms">.0</span></div>
      <div class="sub" id="clockState">planning....</div>
      <button type="button" id="goHome" disabled>GO HOME</button>
      <button type="button" id="rerun" disabled>RERUN GO</button>
    </div>
  </div>
  <div class="below">
    <div class="instr" id="instr"></div>
    <div class="now" id="now">—</div>
    <div class="log" id="log"></div>
  </div>
</div>
<script>
const clock = document.getElementById("clock");
const clockState = document.getElementById("clockState");
const nowEl = document.getElementById("now");
const logEl = document.getElementById("log");
const instrEl = document.getElementById("instr");
const goHomeBtn = document.getElementById("goHome");
const rerunBtn = document.getElementById("rerun");

function fmtClock(ms) {
  const t = Math.max(0, Math.floor(ms));
  const m = Math.floor(t / 60000);
  const s = Math.floor((t % 60000) / 1000);
  const milli = Math.floor((t % 1000) / 100);
  const main = String(m).padStart(2,"0") + ":" + String(s).padStart(2,"0");
  const frac = "." + String(milli);
  return main + '<span class="ms">' + frac + "</span>";
}

let shown = "planning";
function stageName(phase) {
  const p = (phase || "").toLowerCase();
  if (p === "execution") {
    shown = "execution";
  } else if (p === "done" || p === "failed") {
    shown = "done";
  } else if (p === "go_home") {
    if (shown === "execution") shown = "done";
  } else if (p === "rerun" || p === "waiting") {
    shown = "planning";
  } else if (p === "perceiving" || p === "planning" || p === "solving" || p === "task") {
    if (shown !== "done") shown = "planning";
  }
  if (shown === "execution") return "executing...";
  if (shown === "done") return "done";
  return "planning....";
}

async function tick() {
  const r = await fetch("/status");
  const s = await r.json();
  clock.innerHTML = fmtClock(s.elapsed_ms);
  clockState.textContent = stageName((s.current || {}).phase);
  if (s.instruction) {
    const text = String(s.instruction).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
    instrEl.innerHTML = '<span class="label">instruction</span><span class="body">' + text + "</span>";
  } else {
    instrEl.textContent = "";
  }
  const cur = s.current || {};
  const ts = cur.ts || "";
  const phase = (cur.phase || "idle").toUpperCase();
  const detail = cur.detail ? ("  ·  " + cur.detail) : "";
  nowEl.innerHTML = '<span class="meta">' + ts + '</span>  <span class="phase">' + phase + "</span>" +
    '<span class="meta">' + detail + "</span>";
  const events = (s.events || []).slice().reverse();
  logEl.innerHTML = events.map(e =>
    '<div><span class="t">' + e.ts + "</span>" + e.phase.toUpperCase() +
    (e.detail ? ("  ·  " + e.detail) : "") + "</div>"
  ).join("");
  goHomeBtn.disabled = !s.go_home_ready || !!s.go_home_busy;
  rerunBtn.disabled = !s.rerun_ready || !!s.go_home_busy;
}
setInterval(() => tick().catch(() => {}), 200);
tick().catch(() => {});

goHomeBtn.onclick = async () => {
  if (goHomeBtn.disabled) return;
  goHomeBtn.disabled = true;
  try {
    const r = await fetch("/go_home", { method: "POST" });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      console.warn("go home", body.detail || r.status);
    }
  } catch (err) {
    console.warn("go home", err);
  }
};

rerunBtn.onclick = async () => {
  if (rerunBtn.disabled) return;
  rerunBtn.disabled = true;
  try {
    const r = await fetch("/rerun", { method: "POST" });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      console.warn("rerun", body.detail || r.status);
    }
  } catch (err) {
    console.warn("rerun", err);
  }
};
</script>
</body>
</html>
"""


def _placeholder(label: str) -> np.ndarray:
    img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
    img[:] = (18, 18, 18)
    cv2.putText(
        img,
        label,
        (24, PANEL_H // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (90, 90, 90),
        2,
        cv2.LINE_AA,
    )
    return img


class _CamThread:
    def __init__(self, name: str, path: str, width: int, height: int, fps: int) -> None:
        self.name = name
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self._lock = threading.Lock()
        self._frame = _placeholder(f"{name}: opening {path}")
        self._stop = threading.Event()
        self._ok = False
        self._thread = threading.Thread(target=self._run, name=f"cam-{name}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def snapshot_rgb(self) -> np.ndarray:
        with self._lock:
            return np.asarray(self._frame).copy()

    def jpeg(self, quality: int = 70) -> bytes:
        bgr = cv2.cvtColor(self.snapshot_rgb(), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return b""
        return bytes(buf)

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            logger.warning("live UI camera %s failed to open %s", self.name, self.path)
            with self._lock:
                self._frame = _placeholder(f"{self.name}: no signal")
            return
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        cap.set(cv2.CAP_PROP_FPS, float(self.fps))
        self._ok = True
        logger.info("live UI camera %s opened %s", self.name, self.path)
        while not self._stop.is_set():
            ok, bgr = cap.read()
            if not ok or bgr is None:
                time.sleep(0.05)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
                rgb = cv2.resize(rgb, (self.width, self.height))
            with self._lock:
                self._frame = rgb
        cap.release()


class LiveUI:
    def __init__(
        self,
        *,
        instruction: str,
        top_path: str,
        wrist_path: str | None,
        width: int = PANEL_W,
        height: int = PANEL_H,
        fps: int = 30,
        host: str = "0.0.0.0",
        port: int = 8765,
    ) -> None:
        self.instruction = instruction
        self.host = host
        self.port = int(port)
        self._lock = threading.Lock()
        self._t0: float | None = None
        self._t1: float | None = None
        self._events: list[dict[str, str]] = []
        self._current = {"ts": "", "phase": "idle", "detail": ""}
        self.cams: dict[str, _CamThread] = {
            "top": _CamThread("top", top_path, width, height, fps),
        }
        if wrist_path:
            self.cams["wrist"] = _CamThread("wrist", wrist_path, width, height, fps)
        self._httpd: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._go_home: Callable[[], None] | None = None
        self._home_busy = False
        self._home_lock = threading.Lock()
        self._rerun = threading.Event()
        self._rerun_ready = False
        self._task_busy = False

    def start(self) -> str:
        for cam in self.cams.values():
            cam.start()
        handler = _make_handler(self)
        try:
            self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        except OSError as exc:
            raise RuntimeError(
                f"Live UI could not bind {self.host}:{self.port} ({exc})"
            ) from exc
        self._http_thread = threading.Thread(target=self._httpd.serve_forever, name="live-ui", daemon=True)
        self._http_thread.start()
        url = f"http://127.0.0.1:{self.port}/"
        cams = "+".join(self.cams) + "+timer"
        logger.info("Live UI → %s  (%s)", url, cams)
        print(f"\nLive UI → {url}  ({cams})\n")
        return url

    def close(self) -> None:
        # Let the browser poll the final done/failed status before the port dies.
        time.sleep(1.5)
        if self._httpd is not None:
            self._httpd.shutdown()
        for cam in self.cams.values():
            cam.close()

    def snapshot(self, name: str = "top") -> np.ndarray | None:
        cam = self.cams.get(name)
        if cam is None:
            return None
        return cam.snapshot_rgb()

    def start_task(self, instruction: str | None = None) -> None:
        with self._lock:
            if instruction:
                self.instruction = instruction
            self._t0 = time.perf_counter()
            self._t1 = None
        self.phase("task", "dispatched")

    def stop_task(self, detail: str = "") -> None:
        with self._lock:
            if self._t0 is not None and self._t1 is None:
                self._t1 = time.perf_counter()
        self.phase("done", detail)

    def phase(self, phase: str, detail: str = "") -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        event = {"ts": ts, "phase": str(phase), "detail": str(detail or "")}
        logger.info("ui phase=%s detail=%s", phase, detail)
        with self._lock:
            self._current = event
            self._events.append(event)
            if len(self._events) > 80:
                self._events = self._events[-80:]
            if phase in {"done", "failed"} and self._t0 is not None and self._t1 is None:
                self._t1 = time.perf_counter()

    def status(self) -> dict[str, Any]:
        with self._lock:
            t0, t1 = self._t0, self._t1
            events = list(self._events)
            current = dict(self._current)
            instruction = self.instruction
        now = time.perf_counter()
        if t0 is None:
            elapsed = 0.0
            running = False
        elif t1 is None:
            elapsed = now - t0
            running = True
        else:
            elapsed = t1 - t0
            running = False
        return {
            "instruction": instruction,
            "elapsed_ms": int(elapsed * 1000),
            "running": running,
            "started_at": t0 is not None,
            "current": current,
            "events": events,
            "go_home_ready": self._go_home is not None,
            "go_home_busy": self._home_busy,
            "rerun_ready": (
                self._rerun_ready and not self._home_busy and not self._task_busy
            ),
        }

    def set_go_home(self, fn: Callable[[], None] | None) -> None:
        self._go_home = fn

    def run_go_home(self) -> tuple[bool, str]:
        fn = self._go_home
        if fn is None:
            return False, "robot not connected"
        with self._lock:
            phase = str(self._current.get("phase") or "")
        if phase == "execution":
            return False, "busy executing"
        with self._home_lock:
            if self._home_busy:
                return False, "go home already running"
            self._home_busy = True
        try:
            self.phase("go_home", "start")
            fn()
            self.phase("go_home", "done")
            return True, "ok"
        except Exception as exc:  # noqa: BLE001
            logger.warning("go home failed: %s", exc)
            self.phase("go_home", f"error: {exc}")
            return False, str(exc)
        finally:
            self._home_busy = False

    def set_task_busy(self, busy: bool) -> None:
        self._task_busy = bool(busy)

    def set_rerun_ready(self, ready: bool) -> None:
        self._rerun_ready = bool(ready)
        if ready:
            self._rerun.clear()

    def wait_rerun(self, timeout: float | None = None) -> bool:
        return self._rerun.wait(timeout)

    def request_rerun(self) -> tuple[bool, str]:
        if self._home_busy or self._task_busy:
            return False, "busy"
        with self._lock:
            phase = str(self._current.get("phase") or "")
        if phase in {"execution", "perceiving", "planning", "solving", "task"}:
            return False, "busy"
        if not self._rerun_ready:
            return False, "not waiting"
        self.phase("rerun", self.instruction)
        self._rerun.set()
        return True, "ok"


def _make_handler(ui: LiveUI) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in {"/", "/index.html"}:
                html = _HTML
                if "wrist" not in ui.cams:
                    html = html.replace(
                        'id="wristPanel"', 'id="wristPanel" style="display:none"'
                    ).replace('src="/stream/wrist"', "")
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/status":
                body = json.dumps(ui.status()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path.startswith("/stream/"):
                name = path.rsplit("/", 1)[-1]
                self._mjpeg(name)
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(length)
            if path == "/go_home":
                ok, msg = ui.run_go_home()
                body = json.dumps({"ok": ok, "detail": msg}).encode("utf-8")
                self.send_response(200 if ok else 409)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/rerun":
                ok, msg = ui.request_rerun()
                body = json.dumps({"ok": ok, "detail": msg}).encode("utf-8")
                self.send_response(200 if ok else 409)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def _mjpeg(self, name: str) -> None:
            cam = ui.cams.get(name)
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    if cam is None:
                        bgr = cv2.cvtColor(_placeholder(f"{name}: unavailable"), cv2.COLOR_RGB2BGR)
                        jpeg = cv2.imencode(".jpg", bgr)[1].tobytes()
                    else:
                        jpeg = cam.jpeg()
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode()
                        + b"\r\n\r\n"
                        + jpeg
                        + b"\r\n"
                    )
                    time.sleep(1.0 / 15.0)
            except BrokenPipeError:
                return
            except ConnectionResetError:
                return

    return Handler
