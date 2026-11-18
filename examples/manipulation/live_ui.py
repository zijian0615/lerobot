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
import os
import queue
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import cv2
import numpy as np

from manipulation.voice_ui import (  # noqa: E402
    VOICE_CSS,
    VOICE_HTML,
    VOICE_JS,
    VOICE_PAGE,
    VoiceBridge,
    VoiceError,
    mint_client_secret,
)

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
  .wrap { padding: 16px 18px 20px; margin: 0 auto; }
  /* Camera on the left half (full 16:9 frame, not cropped); controls on the right half. */
  .layout { display: flex; gap: 16px; align-items: flex-start; }
  .left { flex: 0 0 50%; display: flex; flex-direction: column; gap: 12px; min-width: 0; }
  .right { flex: 1 1 0; display: flex; flex-direction: column; gap: 12px; min-width: 0; }
  .panel { background: #000; border: 1px solid #333; position: relative; overflow: hidden; }
  .panel.cam { width: 100%; aspect-ratio: 16 / 9; }
  .panel.cam img { width: 100%; height: 100%; object-fit: contain; display: block; }
  @media (max-width: 900px) {
    .layout { flex-direction: column; }
    .left { flex-basis: auto; width: 100%; }
  }
  .tag {
    position: absolute; left: 10px; top: 10px;
    font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase;
    background: rgba(0,0,0,0.65); padding: 4px 8px;
  }
  .timer {
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    background: #161616; padding: 18px 12px 20px;
  }
  .timer .clock {
    font-variant-numeric: tabular-nums;
    font-size: clamp(48px, 5vw, 92px); line-height: 1; font-weight: 600; letter-spacing: 0.04em;
    color: #f2f2f2;
  }
  .timer .clock .ms { color: #7a7a7a; font-weight: 500; }
  .timer .sub { margin-top: 16px; font-size: clamp(22px, 2.4vw, 42px); color: #d8d8d8; letter-spacing: 0.08em; }
  .timer button {
    margin-top: 16px;
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
  .below { margin-top: 0; }
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
  .submit { display: none; gap: 10px; margin-bottom: 14px; }
  .submit.on { display: flex; }
  .submit input {
    flex: 1; font-size: 20px; padding: 12px 14px;
    background: #1b1b1b; color: #f0f0f0; border: 1px solid #5a5a5a;
  }
  .submit input:disabled { opacity: 0.5; }
  .submit button {
    font-size: 18px; letter-spacing: 0.14em; text-transform: uppercase;
    padding: 12px 26px; background: #2c5d2c; color: #f0f0f0;
    border: 1px solid #4b8f4b; cursor: pointer;
  }
  .submit button:disabled { opacity: 0.35; cursor: not-allowed; }
  .submit .err { color: #e06c6c; align-self: center; font-size: 15px; }
  /*VOICE_CSS*/
  .instr { margin-bottom: 10px; line-height: 1.4; }
  .instr .label { color: #fff; font-size: 22px; font-weight: 560; margin-right: 12px; }
  .instr .body { color: #9a9a9a; font-size: 18px; font-weight: 400; }
</style>
</head>
<body>
<div class="wrap">
 <div class="layout">
  <div class="left">
    <div class="panel cam">
      <div class="tag" id="topTag">top</div>
      <img id="top" src="/stream/top" alt="top"/>
    </div>
    <div class="panel cam" id="wristPanel">
      <div class="tag" id="wristTag">wrist</div>
      <img id="wrist" src="/stream/wrist" alt="wrist"/>
    </div>
  </div>
  <div class="right">
    <div class="panel timer">
      <div class="clock" id="clock">00:00<span class="ms">.0</span></div>
      <div class="sub" id="clockState">planning....</div>
      <button type="button" id="goHome" disabled>GO HOME</button>
      <button type="button" id="rerun" disabled>RERUN GO</button>
    </div>
  <div class="below">
    <form class="submit" id="submitForm" autocomplete="off">
      <input id="newInstr" list="instrHistory" maxlength="500"
             placeholder="Next instruction, e.g. put all screws in the container"/>
      <datalist id="instrHistory"></datalist>
      <button type="submit" id="runBtn" disabled>RUN</button>
      <span class="err" id="submitErr"></span>
    </form>
    <!--VOICE_HTML-->
    <div class="instr" id="instr"></div>
    <div class="now" id="now">—</div>
    <div class="log" id="log"></div>
  </div>
  </div>
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
const topTag = document.getElementById("topTag");
const wristTag = document.getElementById("wristTag");
const submitForm = document.getElementById("submitForm");
const newInstr = document.getElementById("newInstr");
const runBtn = document.getElementById("runBtn");
const submitErr = document.getElementById("submitErr");
const historyEl = document.getElementById("instrHistory");
let historyKey = "";

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
  } else if (p === "rerun" || p === "waiting" || p === "new_task") {
    shown = "planning";
  } else if (p === "idle") {
    shown = "idle";
  } else if (p === "verifying") {
    shown = "verifying";
  } else if (p === "result") {
    shown = "done";
  } else if (p === "perceiving" || p === "planning" || p === "solving" || p === "task") {
    if (shown !== "done") shown = "planning";
  }
  if (shown === "execution") return "executing...";
  if (shown === "done") return "done";
  if (shown === "idle") return "ready";
  if (shown === "verifying") return "checking...";
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
  for (const [name, el] of [["top", topTag], ["wrist", wristTag]]) {
    const c = (s.cams || {})[name];
    if (!c) continue;
    el.textContent = name + (c.live ? "" : " · NO SIGNAL");
    el.style.color = c.live ? "" : "#ff8080";
  }
  submitForm.classList.toggle("on", !!s.serve);
  const canSubmit = !!s.can_submit;
  newInstr.disabled = !canSubmit;
  runBtn.disabled = !canSubmit || !newInstr.value.trim();
  const hist = (s.history || []);
  const key = hist.join("||");
  if (key !== historyKey) {
    historyKey = key;
    historyEl.innerHTML = "";
    hist.forEach(h => {
      const o = document.createElement("option");
      o.value = h;
      historyEl.appendChild(o);
    });
  }
}
newInstr.addEventListener("input", () => {
  runBtn.disabled = newInstr.disabled || !newInstr.value.trim();
});
submitForm.onsubmit = async (ev) => {
  ev.preventDefault();
  const text = newInstr.value.trim();
  if (!text || runBtn.disabled) return;
  runBtn.disabled = true;
  submitErr.textContent = "";
  try {
    const r = await fetch("/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ instruction: text }),
    });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) {
      submitErr.textContent = body.detail || ("error " + r.status);
    } else {
      newInstr.value = "";
    }
  } catch (err) {
    submitErr.textContent = String(err);
  }
};
/*VOICE_JS*/
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
        self._t_frame: float | None = None
        self._thread = threading.Thread(target=self._run, name=f"cam-{name}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def snapshot_rgb(self) -> np.ndarray:
        with self._lock:
            return np.asarray(self._frame).copy()

    def age_s(self) -> float | None:
        """Seconds since the last real camera frame (None if none yet)."""
        with self._lock:
            t = self._t_frame
        return None if t is None else time.monotonic() - t

    def live(self, max_age_s: float = 3.0) -> bool:
        age = self.age_s()
        return age is not None and age <= max_age_s

    def jpeg(self, quality: int = 70) -> bytes:
        bgr = cv2.cvtColor(self.snapshot_rgb(), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return b""
        return bytes(buf)

    def _open(self) -> cv2.VideoCapture | None:
        if str(self.path).startswith("/") and not os.path.exists(str(self.path)):
            return None  # device node absent (unplugged): don't spam OpenCV warnings
        cap = cv2.VideoCapture(self.path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        cap.set(cv2.CAP_PROP_FPS, float(self.fps))
        return cap

    def _show_placeholder(self, label: str) -> None:
        with self._lock:
            self._frame = _placeholder(label)
            self._t_frame = None

    def _run(self) -> None:
        """Camera loop that lives as long as the process: it never depends on a task and
        reopens the device whenever it cannot be opened or stops delivering frames."""
        backoff = 1.0
        warned = False
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                if not warned:
                    logger.warning("live UI camera %s cannot open %s; retrying", self.name, self.path)
                    warned = True
                self._show_placeholder(f"{self.name}: no signal")
                self._stop.wait(backoff)
                backoff = min(backoff * 1.5, 5.0)
                continue
            backoff = 1.0
            warned = False
            self._ok = True
            logger.info("live UI camera %s opened %s", self.name, self.path)
            last_ok = time.monotonic()
            while not self._stop.is_set():
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    if time.monotonic() - last_ok > 3.0:
                        logger.warning("live UI camera %s: no frames for 3 s; reopening", self.name)
                        break
                    time.sleep(0.05)
                    continue
                last_ok = time.monotonic()
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
                    rgb = cv2.resize(rgb, (self.width, self.height))
                with self._lock:
                    self._frame = rgb
                    self._t_frame = last_ok
            cap.release()
            self._ok = False
            if not self._stop.is_set():
                self._show_placeholder(f"{self.name}: no signal, reconnecting")


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
        wrist_width: int | None = None,
        wrist_height: int | None = None,
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
            self.cams["wrist"] = _CamThread(
                "wrist",
                wrist_path,
                int(wrist_width or width),
                int(wrist_height or height),
                fps,
            )
        self._httpd: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._go_home: Callable[[], None] | None = None
        self._home_busy = False
        self._home_lock = threading.Lock()
        self._rerun = threading.Event()
        self._rerun_ready = False
        self._task_busy = False
        # Serve mode: the process stays up and takes new instructions from the page.
        self._serve = False
        self._task_q: queue.Queue[str] = queue.Queue(maxsize=1)
        self._history: list[str] = []
        self._voice: VoiceBridge | None = None
        self.last_task_arms: list[str] | None = None
        self.last_task_rules: list[dict[str, Any]] = []  # [{"object": "bear", "arms": ["xarm2"]}]
        # role → arms, e.g. {"gripper": ["xarm"], "hand": ["xarm2"]} (set by run_xarm_live)
        self.arm_roles: dict[str, list[str]] = {}
        self._scene_check: Callable[[str | None], dict[str, Any]] | None = None
        self._check_lock = threading.Lock()
        self.post_check = False  # True when every task ends with a camera "result" event
        self.arms: list[str] = []
        self._preflight: Callable[[str], dict[str, Any]] | None = None

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

    def snapshot(self, name: str = "top", newer_than: float | None = None) -> np.ndarray | None:
        """Latest frame. ``newer_than`` (time.monotonic()) waits for a frame captured after
        that moment, e.g. after the arm stopped, so no frame from mid-motion is used."""
        cam = self.cams.get(name)
        if cam is None:
            return None
        # Wait briefly for a real frame (just after start-up / a reconnect), never hand a
        # "no signal" placeholder to the VLM.
        deadline = time.monotonic() + 5.0
        while not cam.live() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not cam.live():
            raise RuntimeError(f"{name} camera has no signal ({cam.path})")
        if newer_than is not None:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                age = cam.age_s()
                if age is not None and time.monotonic() - age > newer_than:
                    break
                time.sleep(0.03)
            else:
                raise RuntimeError(f"{name} camera gave no new frame after the arm stopped")
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
            already = str(self._current.get("phase") or "") in {"done", "failed", "result"}
        if not already:  # the orchestrator usually reported "done" already: don't say it twice
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
            if phase in {"done", "failed", "result"} and self._t0 is not None and self._t1 is None:
                self._t1 = time.perf_counter()

    def status(self) -> dict[str, Any]:
        with self._lock:
            t0, t1 = self._t0, self._t1
            events = list(self._events)
            current = dict(self._current)
            instruction = self.instruction
            history = list(self._history)
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
            "cams": {
                n: {"live": c.live(), "age_s": c.age_s()} for n, c in self.cams.items()
            },
            "serve": self._serve,
            "arms": list(self.arms),
            "post_check": self.post_check,
            "voice": self._voice is not None,
            "voice_info": self._voice.info() if self._voice is not None else None,
            "can_submit": self._serve and not self._home_busy and not self._task_busy,
            "history": history,
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

    def set_scene_check(self, fn: Callable[[str | None], dict[str, Any]] | None) -> None:
        self._scene_check = fn

    def run_scene_check(self, question: str | None) -> tuple[bool, Any]:
        fn = self._scene_check
        if fn is None:
            return False, "camera check not available"
        if self._task_busy:
            # Mid-task the arm is over the table; the automatic check runs once it has stopped.
            return False, "the arm is still working on a task; it is checked with the camera automatically when it finishes"
        if not self._check_lock.acquire(blocking=False):
            return False, "a camera check is already running"
        try:
            return True, fn(question)
        except Exception as exc:  # noqa: BLE001
            logger.warning("camera check failed: %s", exc)
            return False, f"{type(exc).__name__}: {exc}"
        finally:
            self._check_lock.release()

    def set_preflight(self, fn: Callable[[str], dict[str, Any]] | None) -> None:
        self._preflight = fn

    def run_preflight(self, instruction: str) -> tuple[bool, Any]:
        fn = self._preflight
        if fn is None:
            return False, "object pre-check not available"
        if self._task_busy or self._home_busy:
            return False, "the robot is busy"
        if not self._check_lock.acquire(blocking=False):
            return False, "a camera check is already running"
        try:
            return True, fn(instruction)
        except Exception as exc:  # noqa: BLE001
            logger.warning("object pre-check failed: %s", exc)
            return False, f"{type(exc).__name__}: {exc}"
        finally:
            self._check_lock.release()

    def enable_serve(self) -> None:
        self._serve = True

    def enable_voice(self, session: dict[str, Any]) -> None:
        """Show the voice panel; ``session`` is sent by the page in its first session.update."""
        self._voice = VoiceBridge(
            session,
            submit_task=self.submit_task,
            go_home=self.run_go_home,
            is_busy=lambda: self._home_busy or self._task_busy,
            arm_roles=lambda: self.arm_roles,
        )

    def submit_task(
        self,
        instruction: str,
        arms: list[str] | None = None,
        rules: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, str]:
        """Queue a new instruction for the serve loop (called from the HTTP thread).
        ``arms`` restricts which arms the planner may use for this task (None = all)."""
        if not self._serve:
            return False, "serve mode is off (start with --serve)"
        text = " ".join(str(instruction or "").split())[:500]
        if not text:
            return False, "empty instruction"
        with self._lock:
            if self._home_busy or self._task_busy:
                return False, "busy"
            self._task_busy = True
            try:
                self._task_q.put_nowait((text, list(arms) if arms else None, list(rules or [])))
            except queue.Full:
                self._task_busy = False
                return False, "an instruction is already queued"
            self._history = [h for h in self._history if h != text][-9:] + [text]
        tag = f"  [arms: {', '.join(arms)}]" if arms else ""
        if rules:
            tag += "  [" + "; ".join(f"{r['object']} → {'/'.join(r['arms'])}" for r in rules) + "]"
        self.phase("new_task", text + tag)
        return True, "queued"

    def wait_task(self, timeout: float | None = None) -> str | None:
        """Next instruction; its arm restriction (or None) is left in ``last_task_arms``."""
        try:
            text, arms, rules = self._task_q.get(timeout=timeout)
        except queue.Empty:
            return None
        self.last_task_arms = arms
        self.last_task_rules = rules
        return text

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
            path, _, query = self.path.partition("?")
            if path == "/voice" and ui._voice is not None:
                # Audio-only page for the device with the mic/speakers: no video is streamed.
                body = VOICE_PAGE.replace("__VOICE_JS__", VOICE_JS.replace("__MIC__", "true")).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path in {"/", "/index.html"}:
                voice_on = ui._voice is not None
                # The dashboard mirrors voice (log, state, confirm); ?mic=1 makes it own the mic too.
                mic = "true" if "mic=1" in query else "false"
                html = _HTML.replace("/*VOICE_CSS*/", VOICE_CSS if voice_on else "")
                html = html.replace("<!--VOICE_HTML-->", VOICE_HTML if voice_on else "")
                html = html.replace("/*VOICE_JS*/", VOICE_JS.replace("__MIC__", mic) if voice_on else "")
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
            if length > 16384:
                self.send_error(413)
                return
            raw = self.rfile.read(length) if length else b""
            if path.startswith("/voice/"):
                ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if ui._voice is None:
                    self.send_error(404)
                    return
                if ctype != "application/json":
                    self.send_error(415)
                    return
            if path in {"/voice/event", "/voice/resolve"}:
                try:
                    data = json.loads(raw or b"{}")
                    if not isinstance(data, dict):
                        raise ValueError
                except ValueError:
                    data = {}
                if path == "/voice/event":
                    ok, msg = ui._voice.event(data)
                else:
                    ok, msg = ui._voice.resolve(data.get("id"), bool(data.get("yes")), str(data.get("why") or ""))
                body = json.dumps({"ok": ok, "detail": msg}).encode("utf-8")
                self.send_response(200 if ok else 409)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/voice/preflight":
                try:
                    instr = str(json.loads(raw or b"{}").get("instruction") or "")
                except (ValueError, AttributeError):
                    instr = ""
                instr = " ".join(instr.split())[:300]
                ok, res = ui.run_preflight(instr) if instr else (False, "empty instruction")
                if ui._voice is not None:
                    if ok:
                        seen = ", ".join(res.get("objects") or []) or "nothing recognised"
                        ui._voice.event({"type": "log", "who": "robot",
                                         "text": f"object pre-check ({'ok' if res['ok'] else 'NO MATCH'}): "
                                                 f"{res.get('note') or ''} | seen: {seen}"})
                    else:
                        ui._voice.event({"type": "log", "who": "error", "text": f"object pre-check failed: {res}", "cls": "err"})
                body = json.dumps({"ok": ok, "result": res} if ok else {"ok": False, "detail": res}).encode("utf-8")
                self.send_response(200 if ok else 409)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/voice/check":
                try:
                    q = json.loads(raw or b"{}").get("question")
                except (ValueError, AttributeError):
                    q = None
                q = " ".join(str(q).split())[:300] if q else None
                ok, res = ui.run_scene_check(q)
                if ok and ui._voice is not None:
                    from manipulation.scene_check import verdict_text

                    ui._voice.event({"type": "log", "who": "robot", "text": verdict_text(res)})
                body = json.dumps({"ok": ok, "result": res} if ok else {"ok": False, "detail": res}).encode("utf-8")
                self.send_response(200 if ok else 409)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/voice/token":
                try:
                    secret = mint_client_secret()
                    payload = {**secret, "session": ui._voice.session}
                    code = 200
                except VoiceError as exc:
                    payload, code = {"ok": False, "detail": exc.detail}, exc.status
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/run":
                # Moves the arm: only accept a JSON body, which a cross-site form cannot
                # send without a CORS preflight that this server never answers.
                ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if ctype != "application/json":
                    self.send_error(415)
                    return
                try:
                    text = str(json.loads(raw or b"{}").get("instruction") or "")
                except (ValueError, AttributeError):
                    text = ""
                ok, msg = ui.submit_task(text)
                body = json.dumps({"ok": ok, "detail": msg}).encode("utf-8")
                self.send_response(200 if ok else 409)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
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
