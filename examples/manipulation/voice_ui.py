# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""Voice front end for the live UI (Boson Higgs Realtime over WebSocket).

The browser talks to ``wss://api.boson.ai/v1/realtime`` directly with a short-lived key that
this server mints (``POST /v1/realtime/client_secrets``) from ``BOSON_API_KEY``; the real key
never reaches the page. The model is only a talker: it can *ask* for a task (``run_task``),
read the robot status, or ask for go-home, and every motion still needs a click on CONFIRM.
Task progress is pushed to the model as ``[robot] ...`` text items so it can speak while the
arm works (the arm runs in the serve loop, independent of the audio session).
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

BOSON_CLIENT_SECRET_URL = "https://api.boson.ai/v1/realtime/client_secrets"
DEFAULT_MODEL = "higgs-realtime"
DEFAULT_VOICE = "default"

INSTRUCTIONS = """\
__BODY__

Style: spoken, short, plain. Reply in the language the operator uses (Chinese or English). \
No lists, no markdown, no reading out code or JSON.

Acting: when the operator asks for a task, call run_task in that same reply. Never say "I'll \
start", "I'll put them in" or similar without calling run_task: words alone do nothing. If you \
first used check_scene and work remains, call run_task right after it.

Tools:
- run_task(instruction): ask to start a manipulation task. Pass a short English imperative, \
e.g. "put all screws in the container". It does NOT start anything: the operator has to \
confirm on screen (Enter or CONFIRM; Esc cancels). Afterwards say the command back briefly and \
ask them to confirm. Never say you started until a [robot] report says so.
- get_status(): what you are doing right now (phase, last result). It does not look at the table.
- check_scene(question): take a fresh photo and let a vision model answer, e.g. "Is the bear \
in the container?". Use it when the operator asks whether something worked, where something \
is, or to double check. Report its answer including any uncertainty.
- go_home(): ask to return to your home pose (also needs confirmation).

[robot] messages are automatic reports from your own controller, not from the operator. \
Turn each into ONE short first-person sentence (about ten words) with only the new fact, e.g. \
"[robot] now: Grasp plush_bear (step 1/2)" -> "Picking up the bear now." Do not add promises \
or filler such as "I'll check the table once it's done", "I'm watching", "let me know"; the \
result check happens automatically and is reported to you. Never repeat a sentence you already \
said. Never invent positions, counts or results that no report gave you.

Which arm: when the operator says which hand/arm to use or not to use ("don't use the hand", \
"only the gripper", "use the hand"), you MUST pass run_task.arm ("gripper" or "hand"). Only that \
parameter is enforced; words in the instruction are not. When different objects go to \
different arms ("move the bear with the hand, put the screws in with the gripper"), pass \
run_task.arm_rules instead, one entry per object word, and keep arm "any".

Truthfulness: "moves finished" or status "success" only means my motion plan ran without an \
error; a grasp can miss silently. Only say the task succeeded when a camera check result says \
done. If it says NOT done or unsure, say that plainly. Never answer "is it done?" from memory: \
call check_scene.

Object names: speech recognition often mishears object names ("bear" heard as "beer", "box" as \
"fox"). run_task first photographs the table and matches every object you name to what is \
visible. If it returns state "not_started" with what it sees, do NOT retry with the same words: \
say what is on the table and ask which object they mean ("I see a plush bear, not a beer. Do \
you mean the bear?"). When it matches, it tells you the command it will run using the visible \
names; say that command back briefly.

Safety: I cannot stop myself by voice. If the operator says stop, danger or emergency, tell \
them to hit the physical emergency stop now. If a request is ambiguous (which object, where), \
ask one short question before calling run_task. Do not call run_task while a task is running.
"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "run_task",
        "description": (
            "Ask to start a robot manipulation task. The operator must confirm on screen "
            "before anything moves."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "Short English imperative, e.g. 'put all screws in the container'.",
                },
                "arm_rules": {
                    "type": "array",
                    "description": (
                        "Per-object arm assignment, enforced on every step that handles that "
                        "object, e.g. [{'object': 'bear', 'arm': 'hand'}, {'object': 'screw', "
                        "'arm': 'gripper'}] for 'move the bear with the hand, screws with the "
                        "gripper'. object = a word from the object name."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "object": {"type": "string"},
                            "arm": {"type": "string", "enum": ["gripper", "hand"]},
                        },
                        "required": ["object", "arm"],
                    },
                },
                "arm": {
                    "type": "string",
                    "enum": ["any", "gripper", "hand"],
                    "description": (
                        "Which end effector may be used. 'gripper' = only the parallel-gripper "
                        "arm, 'hand' = only the LEAP-hand arm, 'any' = planner decides. Set it "
                        "whenever the operator says which to use or NOT to use (e.g. 'don't use "
                        "the hand' -> 'gripper'); words in the instruction alone are not enforced."
                    ),
                },
            },
            "required": ["instruction"],
        },
    },
    {
        "type": "function",
        "name": "get_status",
        "description": "Read the current robot/task status.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "check_scene",
        "description": (
            "Photograph the table now and let a vision model answer a question about it, "
            "e.g. whether the task really succeeded. Takes a few seconds."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "English question, e.g. 'Is the beer inside the container?'",
                }
            },
        },
    },
    {
        "type": "function",
        "name": "go_home",
        "description": "Ask to move the arm to its home pose. The operator must confirm on screen.",
        "parameters": {"type": "object", "properties": {}},
    },
]


_EE_WORDS = {
    "parallel_gripper": "a parallel gripper",
    "gripper": "a parallel gripper",
    "leaphand": "a five-finger LEAP hand",
    "leap_hand": "a five-finger LEAP hand",
    "leap": "a five-finger LEAP hand",
}


def persona(arms: list[tuple[str, str]] | None = None) -> str:
    """First paragraph of the instructions, from the active arms ``[(name, end_effector)]``."""
    arms = arms or [("xarm", "parallel_gripper")]
    voice = (
        'Speak as yourself in the first person ("I\'m picking up the bear", "I put it in the '
        'container"), never "the arm" or "the robot". The operator stands next to you.'
    )
    if len(arms) == 1:
        ee = _EE_WORDS.get(arms[0][1].lower(), arms[0][1])
        return (
            f"You ARE the robot: an xArm with {ee} on a tabletop work cell, with an overhead "
            f"camera as your eyes. {voice}"
        )
    parts = "; ".join(
        f'arm {i + 1} ("{n}") has {_EE_WORDS.get(e.lower(), e)}' for i, (n, e) in enumerate(arms)
    )
    return (
        f"You ARE the robot: a {len(arms)}-arm tabletop work cell with an overhead camera as "
        f"your eyes. {parts}. A planner decides which arm does each step (both may work, taking "
        f"turns in the shared middle zone); reports name the arm, e.g. \"xarm2: Grasp bear\". "
        f'Speak of your arms in the first person ("I\'m picking up the bear with my hand"). '
        f"{voice} If the operator wants a specific arm, put it in the run_task instruction "
        f'(e.g. "use the hand to put the bear in the container").'
    )


def build_session(
    *,
    model: str = DEFAULT_MODEL,
    voice: str = DEFAULT_VOICE,
    arms: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """The ``session`` object the page sends in its first ``session.update``."""
    return {
        "type": "realtime",
        "model": model,
        "instructions": INSTRUCTIONS.replace("__BODY__", persona(arms)),
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "transcription": {"model": "higgs-stt-3.1"},
                "turn_detection": {"type": "server_vad"},
            },
            "output": {"voice": voice},
        },
        "tools": TOOLS,
        "tool_choice": "auto",
    }


class VoiceError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def mint_client_secret(expires_s: int = 900) -> dict[str, Any]:
    """Mint a short-lived browser key. Raises :class:`VoiceError` with a safe message."""
    key = os.environ.get("BOSON_API_KEY", "").strip()
    if not key:
        raise VoiceError(503, "BOSON_API_KEY is not set in the environment of run_xarm_live")
    req = urllib.request.Request(
        BOSON_CLIENT_SECRET_URL,
        data=json.dumps({"expires_after": {"seconds": int(expires_s)}}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "robot-voice-ui/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 (fixed https URL)
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        # Boson's own error text says *why* (bad key vs no credit vs rate limit); it never
        # contains our key, so it is safe to show.
        why = ""
        try:
            err = json.loads(exc.read(2000) or b"{}")
            err = err.get("error", err) if isinstance(err, dict) else {}
            why = " ".join(str(err.get(k)) for k in ("code", "type", "message") if err.get(k))
        except Exception:  # noqa: BLE001
            pass
        hints = {
            401: "invalid BOSON_API_KEY",
            402: "no credit: top up your Boson account",
            403: "key not allowed to use the realtime API",
            429: "rate limit or no credit left (insufficient_quota): check your Boson balance/limits",
        }
        hint = hints.get(exc.code, "")
        detail = f"Boson returned HTTP {exc.code}"
        if why:
            detail += f": {why[:200]}"
        if hint:
            detail += f" ({hint})"
        raise VoiceError(502, detail) from exc
    except Exception as exc:  # noqa: BLE001
        raise VoiceError(502, f"cannot reach Boson: {type(exc).__name__}") from exc
    value = data.get("value") if isinstance(data, dict) else None
    if not isinstance(value, str) or not value:
        raise VoiceError(502, "Boson response had no client secret")
    return {"value": value, "expires_at": data.get("expires_at")}


class VoiceBridge:
    """Server-side truth for the voice feature, shared by every open page.

    The mic page (``/voice``, e.g. on a laptop) posts transcripts, its state and *requests*
    (``pending``); the dashboard on the robot machine mirrors them. A request only ever turns
    into robot motion through :meth:`resolve`, i.e. one operator keypress/click on any page, and
    the action runs here, once, no matter which page confirmed.
    """

    CONFIRM_TTL_S = 60.0
    _WHO = {"you", "ai", "robot", "tool", "voice", "error", "confirm"}

    def __init__(
        self,
        session: dict[str, Any],
        *,
        submit_task: Callable[[str], tuple[bool, str]],
        go_home: Callable[[], tuple[bool, str]],
        is_busy: Callable[[], bool],
        arm_roles: Callable[[], dict[str, list[str]]] | None = None,
    ) -> None:
        self.session = session
        self._arm_roles = arm_roles or (lambda: {})
        self._submit = submit_task
        self._go_home = go_home
        self._is_busy = is_busy
        self._lock = threading.RLock()
        self._seq = 0
        self._log: list[dict[str, Any]] = []
        self._state = "off"
        self._state_t: float | None = None
        self._pending: dict[str, Any] | None = None
        self._pid = 0
        self._res: list[dict[str, Any]] = []

    # -- helpers (take the lock themselves) ------------------------------------------------
    def _add_log(self, who: str, text: str, cls: str = "") -> None:
        with self._lock:
            self._seq += 1
            self._log.append({"n": self._seq, "who": who, "text": text[:400], "cls": cls})
            del self._log[:-100]

    def _add_res(self, p: dict[str, Any], stage: str, detail: str = "") -> None:
        with self._lock:
            self._seq += 1
            self._res.append(
                {
                    "id": p["id"],
                    "kind": p["kind"],
                    "text": p["text"],
                    "instruction": p.get("instruction", ""),
                    "stage": stage,
                    "detail": detail[:200],
                    "n": self._seq,
                }
            )
            del self._res[:-12]

    def _expire(self) -> None:
        with self._lock:
            p = self._pending
            if p is not None and time.monotonic() > p["expires"]:
                self._pending = None
                self._add_res(p, "declined", "timed out")
                self._add_log("confirm", f"timed out, nothing started: {p['text']}")

    # -- from the mic page ----------------------------------------------------------------
    def event(self, ev: dict[str, Any]) -> tuple[bool, str]:
        kind = str(ev.get("type") or "")
        if kind == "log":
            who = str(ev.get("who") or "")
            if who not in self._WHO:
                return False, "bad who"
            self._add_log(who, str(ev.get("text") or ""), "err" if ev.get("cls") == "err" else "")
            return True, "ok"
        if kind == "state":
            with self._lock:
                self._state = str(ev.get("text") or "")[:60]
                self._state_t = time.monotonic()
            return True, "ok"
        if kind == "pending":
            what = str(ev.get("kind") or "")
            instr = " ".join(str(ev.get("instruction") or "").split())[:300]
            arms: list[str] | None = None
            role = str(ev.get("arm") or "any").lower()
            if what == "run" and role not in ("", "any", "both"):
                roles = self._arm_roles()
                if role not in roles or not roles[role]:
                    return False, f"no arm with role {role!r} (have: {sorted(roles) or 'one arm only'})"
                arms = list(roles[role])
            rules: list[dict[str, Any]] = []
            for r in list(ev.get("arm_rules") or [])[:8] if what == "run" else []:
                obj = " ".join(str((r or {}).get("object") or "").split())[:40].lower()
                rrole = str((r or {}).get("arm") or "").lower()
                if not obj or rrole in ("", "any", "both"):
                    continue
                roles = self._arm_roles()
                if rrole not in roles or not roles[rrole]:
                    return False, f"no arm with role {rrole!r} for {obj!r} (have: {sorted(roles) or 'one arm only'})"
                rules.append({"object": obj, "role": rrole, "arms": list(roles[rrole])})
            if what == "run":
                if not instr:
                    return False, "empty instruction"
                text = f"Start task: {instr}"
                if arms:
                    text = f"Start task ({role} only: {', '.join(arms)}): {instr}"
                if rules:
                    text += "   {" + "; ".join(
                        f"{r['object']} → {r['role']} ({', '.join(r['arms'])})" for r in rules
                    ) + "}"
                note = " ".join(str(ev.get("note") or "").split())[:160]
                if note:
                    text += f"   [{note}]"
            elif what == "home":
                instr, text = "", "Send the arm to its home pose"
            else:
                return False, "unknown request"
            if self._is_busy():
                return False, "the robot is busy"
            with self._lock:
                self._expire()
                if self._pending is not None:
                    self._add_res(self._pending, "replaced", "")
                self._pid += 1
                self._pending = {
                    "id": self._pid,
                    "kind": what,
                    "text": text,
                    "instruction": instr,
                    "arms": arms,
                    "rules": rules,
                    "expires": time.monotonic() + self.CONFIRM_TTL_S,
                }
            self._add_log("confirm", f"waiting for Enter/CONFIRM: {text}")
            return True, "pending"
        return False, "unknown event"

    # -- from any page ----------------------------------------------------------------------
    def resolve(self, pid: Any, yes: bool, why: str = "") -> tuple[bool, str]:
        with self._lock:
            self._expire()
            p = self._pending
            if p is None or p["id"] != pid:
                return False, "no such pending request (expired or already handled)"
            self._pending = None
        if not yes:
            self._add_res(p, "declined", why or "declined")
            self._add_log("confirm", f"cancelled ({why or 'declined'}): {p['text']}")
            return True, "declined"
        self._add_log("confirm", f"confirmed: {p['text']}")
        if self._is_busy():
            self._add_res(p, "start_failed", "the robot is busy")
            return False, "the robot is busy"
        if p["kind"] == "run":
            ok, msg = self._submit(p["instruction"], p.get("arms"), p.get("rules"))
            self._add_res(p, "started" if ok else "start_failed", "" if ok else msg)
            return ok, msg
        self._add_res(p, "home_running", "")

        def _home() -> None:
            try:
                ok, msg = self._go_home()
            except Exception as exc:  # noqa: BLE001
                ok, msg = False, f"{type(exc).__name__}: {exc}"
            self._add_res(p, "home_done" if ok else "home_failed", "" if ok else msg)

        threading.Thread(target=_home, name="voice-go-home", daemon=True).start()
        return True, "going home"

    def info(self) -> dict[str, Any]:
        with self._lock:
            self._expire()
            p = self._pending
            age = None if self._state_t is None else time.monotonic() - self._state_t
            return {
                "state": self._state,
                "client_age_s": age,
                "log": list(self._log[-40:]),
                "pending": None
                if p is None
                else {
                    "id": p["id"],
                    "kind": p["kind"],
                    "text": p["text"],
                    "remaining_s": max(0.0, p["expires"] - time.monotonic()),
                },
                "resolutions": list(self._res[-12:]),
            }


VOICE_CSS = """
  .voice { display: none; margin-bottom: 14px; padding: 12px 14px; background: #161616;
           border: 1px solid #2f2f2f; }
  .voice.on { display: block; }
  .voice .vrow { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .voice.mirror .mic { display: none; }
  .voice button { font-size: 16px; letter-spacing: 0.12em; text-transform: uppercase;
                  padding: 10px 18px; background: #2c2c2c; color: #f0f0f0;
                  border: 1px solid #5a5a5a; cursor: pointer; }
  .voice button:disabled { opacity: 0.35; cursor: not-allowed; }
  .voice button.live { background: #6b2b2b; border-color: #a04a4a; }
  .voice .vstate { font-size: 18px; color: #d8d8d8; min-width: 160px; }
  .voice .vhint { font-size: 13px; color: #7d7d7d; }
  .voice .confirm { margin-top: 10px; padding: 10px 12px; background: #2a2410;
                    border: 1px solid #8a7a2a; display: flex; gap: 12px; align-items: center; }
  .voice .confirm.idle { background: #1b1b1b; border-color: #333; }
  .voice .confirm.idle .txt { color: #7d7d7d; }
  .voice .confirm .txt { flex: 1; font-size: 18px; color: #ffe9a0; }
  .voice .confirm button.yes { background: #2c5d2c; border-color: #4b8f4b; }
  .voice .vlog { margin-top: 10px; max-height: 190px; overflow: auto; font-size: 15px;
                 line-height: 1.5; color: #bdbdbd; }
  .voice .vlog .who { color: #7d7d7d; margin-right: 8px; text-transform: uppercase;
                      font-size: 12px; letter-spacing: 0.08em; }
  .voice .vlog .err { color: #e06c6c; }
  .voice .vlog .w-you { color: #9fd0ff; }
  .voice .vlog .w-tool { color: #c9a8ff; }
  .voice .vlog .w-confirm { color: #ffe9a0; }
"""

VOICE_HTML = """
    <div class="voice" id="voicePanel">
      <div class="vrow">
        <span class="mic">
          <button type="button" id="voiceBtn">START VOICE</button>
          <button type="button" id="muteBtn" disabled>MUTE</button>
        </span>
        <span class="vstate" id="voiceState">off</span>
        <span class="vhint" id="voiceHint"></span>
      </div>
      <div class="confirm idle" id="confirmCard">
        <span class="txt" id="confirmText"></span>
        <button type="button" class="yes" id="confirmYes">CONFIRM (Enter)</button>
        <button type="button" id="confirmNo">CANCEL (Esc)</button>
      </div>
      <div class="vlog" id="voiceLog"></div>
    </div>
"""

# Audio-only page for the device that has the microphone/speakers (no video is streamed).
VOICE_PAGE = (
    """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>voice</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body { margin: 0; background: #111; color: #e8e8e8;
               font-family: ui-sans-serif, system-ui, sans-serif; }
  .wrap { padding: 16px 18px 20px; max-width: 900px; margin: 0 auto; }
  #mini { font-size: 18px; color: #9a9a9a; margin-bottom: 12px; line-height: 1.4; }
"""
    + VOICE_CSS
    + """
</style></head><body><div class="wrap">
  <div id="mini">connecting…</div>
"""
    + VOICE_HTML.replace('class="voice"', 'class="voice on"')
    + """
</div>
<script>
__VOICE_JS__
</script></body></html>
"""
)

# Raw string: the JS below contains backslash escapes that Python must not touch.
VOICE_JS = r"""
(() => {
const MIC = __MIC__;   // true: this page owns the mic/speakers (the /voice page); false: mirror only

const WORKLETS = `
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 24000;
    this.phase = 0; this.acc = 0; this.n = 0;
    this.buf = new Int16Array(2400);   // 100 ms at 24 kHz
    this.k = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.acc += ch[i]; this.n++; this.phase += 1;
      if (this.phase >= this.ratio) {            // box-filter decimation to 24 kHz
        let v = this.acc / this.n;
        v = v > 1 ? 1 : (v < -1 ? -1 : v);
        this.buf[this.k++] = v < 0 ? v * 32768 : v * 32767;
        this.acc = 0; this.n = 0; this.phase -= this.ratio;
        if (this.k === this.buf.length) {
          this.port.postMessage(this.buf.slice(0).buffer);
          this.k = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("capture-processor", CaptureProcessor);

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.q = [];            // {id, ci, data: Float32Array}
    this.cur = null; this.pos = 0;
    this.ratio = 24000 / sampleRate;
    this.frac = 0; this.prev = 0; this.next = 0;
    this.played = new Map(); // item id -> samples played (24 kHz domain)
    this.curId = null; this.curCi = 0;
    this.playing = false;
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === "push") {
        this.q.push({ id: m.id, ci: m.ci, data: m.data });
      } else if (m.type === "flush") {
        let remaining = this.cur ? this.cur.data.length - this.pos : 0;
        for (const c of this.q) remaining += c.data.length;
        const id = this.curId;
        const played = id !== null ? (this.played.get(id) || 0) : 0;
        this.port.postMessage({ type: "flushed", id, ci: this.curCi,
                                playedMs: played / 24.0, remaining });
        this.q = []; this.cur = null; this.pos = 0; this.played.clear();
        this.curId = null; this.prev = 0; this.next = 0;
        this.setPlaying(false);
      }
    };
  }
  setPlaying(p) {
    if (p !== this.playing) { this.playing = p; this.port.postMessage({ type: "state", playing: p }); }
  }
  pull() {
    while (!this.cur || this.pos >= this.cur.data.length) {
      if (!this.q.length) { this.cur = null; return null; }
      this.cur = this.q.shift(); this.pos = 0;
      if (this.curId !== this.cur.id) {
        if (this.curId !== null) this.played.delete(this.curId);
        this.curId = this.cur.id; this.curCi = this.cur.ci;
      }
    }
    const v = this.cur.data[this.pos++];
    this.played.set(this.curId, (this.played.get(this.curId) || 0) + 1);
    return v;
  }
  process(inputs, outputs) {
    const out = outputs[0][0];
    let any = false;
    for (let i = 0; i < out.length; i++) {
      this.frac += this.ratio;
      while (this.frac >= 1) {
        const v = this.pull();
        this.prev = this.next;
        if (v === null) { this.next = 0; } else { this.next = v; any = true; }
        this.frac -= 1;
      }
      out[i] = this.prev + (this.next - this.prev) * this.frac;
    }
    if (any) this.setPlaying(true);
    else if (!this.cur && !this.q.length) this.setPlaying(false);
    return true;
  }
}
registerProcessor("playback-processor", PlaybackProcessor);
`;

const $ = (id) => document.getElementById(id);
const panel = $("voicePanel"), btn = $("voiceBtn"), muteBtn = $("muteBtn");
const stateEl = $("voiceState"), logEl = $("voiceLog"), hintEl = $("voiceHint"), miniEl = $("mini");
const card = $("confirmCard"), cardText = $("confirmText");
const yesBtn = $("confirmYes"), noBtn = $("confirmNo");

async function post(path, body) {
  try {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                  body: JSON.stringify(body || {}) });
    const j = await r.json().catch(() => ({}));
    return { ok: r.ok, detail: j.detail || (r.ok ? "ok" : "error " + r.status) };
  } catch (e) { return { ok: false, detail: String(e) }; }
}

// ---- shared: render the server's voice state (log, state, pending confirmation) ----------
let lastStatus = null, curPending = null, shownAt = 0, renderedLog = -1;

function renderLog(info) {
  const log = info.log || [];
  const last = log.length ? log[log.length - 1].n : 0;
  if (last === renderedLog) return;
  renderedLog = last;
  while (logEl.firstChild) logEl.removeChild(logEl.firstChild);
  for (const e of log) {
    const d = document.createElement("div");
    const w = document.createElement("span"); w.className = "who"; w.textContent = e.who;
    const b = document.createElement("span"); b.textContent = e.text;
    b.className = e.cls === "err" ? "err" : "w-" + e.who;
    d.appendChild(w); d.appendChild(b); logEl.appendChild(d);
  }
  logEl.scrollTop = logEl.scrollHeight;
}
function renderPending(p) {
  // The bar is always visible; its buttons only work while a request is waiting.
  if (!p) {
    curPending = null;
    card.classList.add("idle"); yesBtn.disabled = true; noBtn.disabled = true;
    cardText.textContent = "No request waiting for confirmation";
    return;
  }
  if (!curPending || curPending.id !== p.id) shownAt = performance.now();
  curPending = p;
  card.classList.remove("idle"); yesBtn.disabled = false; noBtn.disabled = false;
  cardText.textContent = p.text + "   (auto-cancel in " + Math.ceil(p.remaining_s) + " s)";
}
renderPending(null);
async function resolve(yes, via) {
  if (!curPending) return;
  const id = curPending.id;
  renderPending(null);                                   // one press = one decision
  const r = await post("/voice/resolve", { id, yes, why: yes ? "" : "declined via " + via });
  if (!r.ok) stateEl.textContent = "confirm failed: " + r.detail;
}
yesBtn.onclick = () => resolve(true, "button");
noBtn.onclick = () => resolve(false, "button");
document.addEventListener("keydown", (e) => {
  if (!curPending || e.repeat) return;
  const t = ((e.target && e.target.tagName) || "").toUpperCase();
  if (t === "INPUT" || t === "TEXTAREA" || t === "SELECT") return;
  if (e.key === "Enter") {
    e.preventDefault();                                   // also stops Enter "clicking" a focused button
    if (performance.now() - shownAt < 600) return;        // never confirm something not yet seen
    resolve(true, "Enter");
  } else if (e.key === "Escape") {
    e.preventDefault();
    resolve(false, "Esc");
  }
});

function onStatus(s) {
  lastStatus = s;
  const info = s.voice_info || {};
  panel.classList.toggle("on", !!s.voice || MIC);
  renderLog(info);
  renderPending(info.pending || null);
  if (!MIC) {
    const age = info.client_age_s;
    stateEl.textContent = "voice: " + (age === null || age === undefined || age > 6
      ? "no voice page connected" : (info.state || "?"));
  }
  if (miniEl) {
    const c = s.current || {};
    miniEl.textContent = (s.running ? "RUNNING " : "") + (c.phase || "idle").toUpperCase() +
      (c.detail ? " · " + c.detail : "") + (s.instruction ? "   |   " + s.instruction : "");
  }
  if (MIC) micStatus(s, info);
}
async function poll() {
  try { onStatus(await (await fetch("/status")).json()); } catch (e) { /* retry next tick */ }
}
setInterval(poll, 250); poll();
hintEl.textContent = MIC
  ? "headphones recommended · the robot moves only after Enter / CONFIRM (Esc cancels)"
  : "Enter confirms, Esc cancels here too · to talk, open the voice page on the device with the mic: ";
if (!MIC) {
  panel.classList.add("mirror");
  const a = document.createElement("a");
  a.href = "/voice"; a.textContent = "/voice (audio only)"; a.style.color = "#9fd0ff";
  hintEl.appendChild(a);
}

// ---- mic mode only ----------------------------------------------------------------------
let ws = null, ctx = null, stream = null, micSrc = null, capNode = null, playNode = null;
let sessionCfg = null;
let ready = false, muted = false, activeResp = false, userSpeaking = false, playing = false;
let notes = [], lastNarr = 0, noteTimer = null, holdUntil = 0;
let seenKey = null, flushWaiter = null, localState = "off";
let lastAiText = "", nudgedThisTurn = false;
const PROMISE_RE = /\b(i'?ll|i will|i'?m going to|let me|starting|i'?m starting)\b[^.?!]*\b(put|move|pick|place|start|grab|take|do|begin|clear)\b/i;
const respCalls = new Map(), handledCalls = new Set();
const seenRes = new Set();
const NARRATE_GAP_MS = 8000;

let chain = Promise.resolve();
function ev(o) { chain = chain.then(() => post("/voice/event", o)); return chain; }
function addLog(who, text, cls) { if (MIC) ev({ type: "log", who, text: String(text).slice(0, 400), cls: cls || "" }); }
function setState(t) { localState = t; stateEl.textContent = t; if (MIC) ev({ type: "state", text: t }); }
if (MIC) setInterval(() => ev({ type: "state", text: localState }), 2000);

function b64(buf) {
  const bytes = new Uint8Array(buf);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  }
  return btoa(s);
}
function pcm16ToFloat(b64s) {
  const bin = atob(b64s);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const i16 = new Int16Array(bytes.buffer, 0, bytes.length >> 1);
  const f = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f[i] = i16[i] / 32768;
  return f;
}
function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

// ---- notes → the model (spoken progress) ---------------------------------------------
function note(text, urgent) {
  if (notes.length && notes[notes.length - 1].text === text) return;
  notes.push({ text, urgent: !!urgent });
  addLog("robot", text.replace(/^\[robot\]\s*/, ""));
  flushNotes();
}
function flushNotes() {
  if (!ready || !notes.length) return;
  const now = Date.now();
  if (activeResp || playing || userSpeaking) return;       // re-run on response.done / idle
  if (now < holdUntil) { clearTimeout(noteTimer); noteTimer = setTimeout(flushNotes, holdUntil - now + 20); return; }
  const urgent = notes.some((n) => n.urgent);
  const wait = NARRATE_GAP_MS - (now - lastNarr);
  if (!urgent && wait > 0) { clearTimeout(noteTimer); noteTimer = setTimeout(flushNotes, wait + 50); return; }
  const text = notes.slice(-8).map((n) => n.text).join("\n");
  notes = [];
  lastNarr = now;
  send({ type: "conversation.item.create",
         item: { type: "message", role: "user", content: [{ type: "input_text", text }] } });
  send({ type: "response.create" });
  activeResp = true;
}

// ---- tools -----------------------------------------------------------------------------
function statusSummary(s) {
  if (!s) return { ok: false, reason: "no status yet" };
  const c = s.current || {};
  const res = (s.events || []).filter((e) => e.phase === "result" || e.phase === "failed").pop();
  return { arms: s.arms || [], running: !!s.running, phase: c.phase || "idle", detail: c.detail || "",
           instruction: s.instruction || "", elapsed_s: Math.round((s.elapsed_ms || 0) / 1000),
           can_start_new_task: !!s.can_submit,
           last_result: res ? res.phase + ": " + res.detail : "none",
           note: "phase/result come from the controller; to know what is really on the table call check_scene" };
}
async function handleCall(call) {
  let args = {};
  try { args = JSON.parse(call.arguments || "{}"); } catch (e) { args = {}; }
  addLog("tool", call.name + " " + (call.arguments || ""));
  if (call.name === "get_status") {
    try { lastStatus = await (await fetch("/status")).json(); } catch (e) { /* keep last */ }
    return JSON.stringify(statusSummary(lastStatus));
  }
  if (call.name === "check_scene") {
    try {
      const r = await fetch("/voice/check", { method: "POST", headers: { "Content-Type": "application/json" },
                                              body: JSON.stringify({ question: String(args.question || "") }) });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) return JSON.stringify({ ok: false, reason: j.detail || ("error " + r.status) });
      return JSON.stringify({ ok: true, done: j.result.done, answer: j.result.answer, evidence: j.result.evidence });
    } catch (e) { return JSON.stringify({ ok: false, reason: String(e) }); }
  }
  if (call.name === "run_task" || call.name === "go_home") {
    const busyMsg = JSON.stringify({ ok: false, reason: "the robot is busy; tell the operator to wait" });
    if (!lastStatus || !lastStatus.can_submit) return busyMsg;
    let r;
    if (call.name === "run_task") {
      const said = String(args.instruction || "").trim().slice(0, 300);
      if (!said) return JSON.stringify({ ok: false, reason: "empty instruction" });
      // Camera pre-check: every named object must match something visible (misheard names!).
      let pre = null, preErr = "";
      try {
        const pr = await fetch("/voice/preflight", { method: "POST", headers: { "Content-Type": "application/json" },
                                                    body: JSON.stringify({ instruction: said }) });
        const pj = await pr.json().catch(() => ({}));
        if (pr.ok) pre = pj.result; else preErr = pj.detail || ("error " + pr.status);
      } catch (e) { preErr = String(e); }
      if (pre && !pre.ok) {
        return JSON.stringify({ ok: false, state: "not_started", said,
          seen_on_table: pre.objects, mapping: pre.mapping, camera_note: pre.note,
          todo: "Nothing was started. Tell the operator what is on the table and ask which object they mean. Do not retry with the same words." });
      }
      const instr = pre ? pre.instruction : said;
      const note = pre ? pre.mapping.map((m) => m.said + " = " + m.visible).join("; ")
                       : "camera pre-check failed: " + preErr;
      r = await ev({ type: "pending", kind: "run", instruction: instr, note, arm: String(args.arm || "any"),
                     arm_rules: Array.isArray(args.arm_rules) ? args.arm_rules : [] });
      if (r.ok) {
        return JSON.stringify({ ok: true, state: "waiting_for_operator_confirmation", will_run: instr,
          arm: String(args.arm || "any"),
          pre_check: pre ? "objects matched" : "unavailable (" + preErr + ")",
          note: "Nothing has started. Say the command back briefly and ask the operator to confirm with Enter (Esc cancels)." });
      }
    } else {
      r = await ev({ type: "pending", kind: "home" });
    }
    if (!r.ok) return JSON.stringify({ ok: false, reason: r.detail });
    return JSON.stringify({ ok: true, state: "waiting_for_operator_confirmation",
      note: "Nothing has started. Tell the operator to confirm on screen with Enter (Esc cancels); do not say it started." });
  }
  return JSON.stringify({ ok: false, reason: "unknown tool " + call.name });
}
async function onResponseDone(resp) {
  resp = resp || {};
  if (resp.status === "failed") {
    const d = (resp.status_details || {}).error || {};
    addLog("error", "response failed: " + (d.message || d.code || ""), "err");
  }
  // Function calls are collected from response.output_item.done as well as response.done, and
  // handled even when the spoken part was interrupted (status "cancelled", e.g. by speaker echo):
  // dropping them made the model say "please confirm" with nothing to confirm.
  const byId = new Map();
  for (const i of [...(respCalls.get(resp.id) || []), ...(resp.output || [])]) {
    if (i && i.type === "function_call" && i.call_id && !handledCalls.has(i.call_id)) byId.set(i.call_id, i);
  }
  respCalls.delete(resp.id);
  const calls = [...byId.values()];
  if (resp.status === "failed") calls.length = 0;
  if (!calls.length) {
    // Promised to act but called no tool: nothing would happen. Nudge once per operator turn.
    if (!nudgedThisTurn && PROMISE_RE.test(lastAiText) && lastStatus && lastStatus.can_submit &&
        !((lastStatus.voice_info || {}).pending)) {
      nudgedThisTurn = true;
      note("[robot] You said you would do it, but you did not call run_task, so nothing will happen. " +
           "If the operator asked for a task, call run_task now.", true);
    }
    flushNotes();
    return;
  }
  calls.forEach((c) => handledCalls.add(c.call_id));
  if (resp.status !== "completed") addLog("voice", "reply was interrupted (" + resp.status + "); still handling " + calls.map((c) => c.name).join(", "));
  const outs = await Promise.all(calls.map(async (c) => ({ id: c.call_id, out: await handleCall(c) })));
  for (const o of outs) {
    send({ type: "conversation.item.create",
           item: { type: "function_call_output", call_id: o.id, output: o.out } });
  }
  send({ type: "response.create" });
  activeResp = true;
}

// ---- audio in / out -------------------------------------------------------------------
function flushPlayback() {
  return new Promise((resolve) => {
    const t = setTimeout(() => { flushWaiter = null; resolve(null); }, 400);
    flushWaiter = (m) => { clearTimeout(t); flushWaiter = null; resolve(m); };
    playNode.port.postMessage({ type: "flush" });
  });
}
async function bargeIn() {
  if (!playNode) return;
  const m = await flushPlayback();
  if (m && m.id && m.remaining > 0 && m.playedMs > 0) {
    send({ type: "conversation.item.truncate", item_id: m.id, content_index: m.ci || 0,
           audio_end_ms: Math.round(m.playedMs) });
  }
}

function onEvent(e) {
  switch (e.type) {
    case "session.created":
      ready = true; setState(muted ? "muted" : "listening"); muteBtn.disabled = false;
      if (lastStatus) {
        const evs = lastStatus.events || [];
        seenKey = evs.length ? keyOf(evs[evs.length - 1]) : null;
        for (const r of ((lastStatus.voice_info || {}).resolutions || [])) seenRes.add(r.id + ":" + r.stage);
      }
      addLog("voice", "connected");
      break;
    case "input_audio_buffer.speech_started":
      userSpeaking = true; setState("hearing you"); bargeIn(); break;
    case "input_audio_buffer.speech_stopped":
      userSpeaking = false; holdUntil = Date.now() + 1500; setState(muted ? "muted" : "listening"); break;
    case "conversation.item.input_audio_transcription.completed":
      if (e.transcript && e.transcript.trim()) { addLog("you", e.transcript.trim()); nudgedThisTurn = false; }
      break;
    case "response.created":
      activeResp = true; lastAiText = ""; break;
    case "response.output_item.done":
      if (e.item && e.item.type === "function_call") {
        const k = e.response_id || "";
        respCalls.set(k, [...(respCalls.get(k) || []), e.item]);
      }
      break;
    case "response.output_audio.delta":
      if (e.delta && playNode) {
        const f = pcm16ToFloat(e.delta);
        playNode.port.postMessage({ type: "push", id: e.item_id, ci: e.content_index || 0, data: f }, [f.buffer]);
      }
      break;
    case "response.output_audio_transcript.done":
      if (e.transcript) { addLog("ai", e.transcript); lastAiText = e.transcript; }
      break;
    case "response.done":
      activeResp = false; onResponseDone(e.response); break;
    case "error": {
      const er = e.error || {};
      addLog("error", (er.type || "error") + (er.code ? "/" + er.code : "") + ": " + (er.message || ""), "err");
      if (/active_response|voice_output_task_ongoing/.test((er.code || "") + (er.type || ""))) activeResp = true;
      break;
    }
    case "session.idle_timeout":
      addLog("voice", "closed: idle for 5 minutes"); break;
    case "session.max_duration_reached":
      addLog("voice", "closed: 60 minute session limit"); break;
    default: break;
  }
}

function cleanup(code, reason) {
  ready = false; activeResp = false; userSpeaking = false; playing = false;
  clearTimeout(noteTimer); notes = [];
  try { if (stream) stream.getTracks().forEach((t) => t.stop()); } catch (e) { /* ignore */ }
  try { if (ctx) ctx.close(); } catch (e) { /* ignore */ }
  ws = null; ctx = null; stream = null; micSrc = null; capNode = null; playNode = null;
  btn.textContent = "START VOICE"; btn.classList.remove("live"); btn.disabled = false;
  muteBtn.disabled = true;
  let t = "off";
  if (code === 3000) t = "key rejected, press START";
  else if (code === 4429) t = "billing refused";
  else if (code === 1013) t = "too many sessions";
  else if (code && code !== 1000 && code !== 1005) t = "disconnected (" + code + ")";
  setState(t);
  if (reason) addLog("voice", "closed: " + reason);
}

async function start() {
  if (ws) return;
  btn.disabled = true; setState("connecting…");
  try {
    ctx = new AudioContext({ latencyHint: "interactive" });
    await ctx.resume();                      // inside the click: autoplay policy
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: false, autoGainControl: true, channelCount: 1 } });
    const url = URL.createObjectURL(new Blob([WORKLETS], { type: "application/javascript" }));
    await ctx.audioWorklet.addModule(url);
    micSrc = ctx.createMediaStreamSource(stream);
    capNode = new AudioWorkletNode(ctx, "capture-processor",
      { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
    micSrc.connect(capNode); capNode.connect(ctx.destination);   // process() only runs when connected
    capNode.port.onmessage = (e) => {
      if (ready && !muted) send({ type: "input_audio_buffer.append", audio: b64(e.data) });
    };
    playNode = new AudioWorkletNode(ctx, "playback-processor",
      { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1] });
    playNode.connect(ctx.destination);
    playNode.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === "flushed" && flushWaiter) flushWaiter(m);
      else if (m.type === "state") { playing = m.playing; if (!playing) flushNotes(); }
    };
    const r = await fetch("/voice/token", { method: "POST",
      headers: { "Content-Type": "application/json" }, body: "{}" });
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || ("token error " + r.status));
    sessionCfg = body.session;
    ws = new WebSocket("wss://api.boson.ai/v1/realtime", ["realtime", "bai-client-secret." + body.value]);
    ws.onopen = () => { send({ type: "session.update", session: sessionCfg }); setState("starting…"); };
    ws.onmessage = (m) => { try { onEvent(JSON.parse(m.data)); } catch (e) { console.warn("voice event", e); } };
    ws.onerror = () => setState("connection error");
    ws.onclose = (e) => cleanup(e.code, e.reason);
    btn.textContent = "STOP VOICE"; btn.classList.add("live"); btn.disabled = false;
  } catch (err) {
    addLog("error", String((err && err.message) || err), "err");
    cleanup(0, "");
    setState("failed");
  }
}
function stop() { if (ws) ws.close(1000); else cleanup(1000, ""); }
if (MIC) {
  btn.onclick = () => { if (ws) stop(); else start(); };
  muteBtn.onclick = () => {
    muted = !muted; muteBtn.textContent = muted ? "UNMUTE" : "MUTE";
    if (ready) setState(muted ? "muted" : "listening");
  };
}

// ---- robot status → narration (mic mode) ------------------------------------------------
const keyOf = (e) => e.ts + "|" + e.phase + "|" + e.detail;
// Only what is worth saying: each move, and the final (camera-checked) result. Looking/planning
// and "moves finished" stay silent when a camera result follows anyway.
function lineFor(e, s) {
  const d = (e.detail || "").replace(/\s+/g, " ").slice(0, 120);
  switch ((e.phase || "").toLowerCase()) {
    case "execution": return ["[robot] now: " + d, false];
    case "done": return s && s.post_check ? null
      : ["[robot] moves finished without error (not checked with the camera)", false];
    case "result": return ["[robot] final result: " + d, true];
    case "failed": return ["[robot] task failed: " + d + "; I was not homed automatically", true];
    default: return null;
  }
}
function noteForResolution(r) {
  switch (r.stage) {
    case "declined": return "[robot] operator did not confirm (" + (r.detail || "declined") + "): " + r.text + "; nothing started";
    case "started": return "[robot] operator confirmed; starting: " + r.instruction;
    case "start_failed": return "[robot] could not start (" + (r.detail || "error") + ")";
    case "home_running": return "[robot] operator confirmed; going home";
    case "home_done": return "[robot] at home";
    case "home_failed": return "[robot] go-home failed (" + (r.detail || "error") + ")";
    default: return null;
  }
}
function micStatus(s, info) {
  for (const r of (info.resolutions || [])) {
    const k = r.id + ":" + r.stage;
    if (seenRes.has(k)) continue;
    seenRes.add(k);
    const t = ready ? noteForResolution(r) : null;
    if (t) note(t, true);
  }
  if (!ready) return;
  const evs = s.events || [];
  let fresh = evs;
  if (seenKey !== null) {
    const i = evs.map(keyOf).lastIndexOf(seenKey);
    fresh = i >= 0 ? evs.slice(i + 1) : evs;
  }
  if (evs.length) seenKey = keyOf(evs[evs.length - 1]);
  for (const e of fresh) { const l = lineFor(e, s); if (l) note(l[0], l[1]); }
}
})();
"""
