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

"""Camera check: did the task actually happen?

The orchestrator's "success" only means every motion step ran without an error; nothing
checks that the object really ended up in the container (``--skip-place-verify``, and the
xArm gripper "width" is the commanded state, not a grasp sensor). This asks a VLM to judge a
fresh top-camera photo against the instruction.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import cv2
import numpy as np
from tabletop_perception.vlm import GEMINI_ROBOTICS_ER_MODEL, strip_markdown_fences

PROMPT = """\
You are checking a robot's work from a top-down camera photo of its work table.
The robot was given this task: "{instruction}"
Question: "{question}"

Judge only from what is visible in the photo. Name where the relevant objects are. If an \
object is hidden, out of view, or you cannot tell, say so instead of guessing.

Reply with JSON only:
{{"done": true | false | null, "answer": "<one short sentence for the operator>", \
"evidence": "<where the relevant objects are in the photo>"}}
"done" is true only if the task's goal is clearly satisfied in this photo, false if it \
clearly is not, null if you cannot tell.
"""


def _json_obj(text: str) -> dict[str, Any]:
    cleaned = strip_markdown_fences(text or "")
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    data = json.loads(m.group(0) if m else cleaned)
    if not isinstance(data, dict):
        raise ValueError("VLM answer is not a JSON object")
    return data


def _parse(text: str) -> dict[str, Any]:
    data = _json_obj(text)
    done = data.get("done")
    if done not in (True, False, None):
        done = None
    return {
        "done": done,
        "answer": str(data.get("answer") or "").strip()[:300],
        "evidence": str(data.get("evidence") or "").strip()[:500],
    }


def _ask(image_rgb: np.ndarray, prompt: str, parse: Any, model: str) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No Gemini API key (GOOGLE_API_KEY) for the camera check")
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("could not encode the camera image")
    client = genai.Client(api_key=key)
    last: Exception | None = None
    for _ in range(2):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=bytes(buf), mime_type="image/jpeg"), prompt],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            return parse(getattr(resp, "text", "") or "")
        except (ValueError, json.JSONDecodeError) as exc:  # malformed JSON → one retry
            last = exc
    raise RuntimeError(f"camera check: unreadable VLM answer ({last})")


def check_scene(
    image_rgb: np.ndarray,
    *,
    instruction: str,
    question: str | None = None,
    model: str = GEMINI_ROBOTICS_ER_MODEL,
) -> dict[str, Any]:
    """Return ``{"done": bool|None, "answer": str, "evidence": str}``. Raises on API errors."""
    prompt = PROMPT.format(
        instruction=instruction or "(none)",
        question=question or "Has this task been completed?",
    )
    return _ask(image_rgb, prompt, _parse, model)


PREFLIGHT_PROMPT = """You see a top-down photo of a robot work table. An operator gave this SPOKEN command; speech \
recognition may have misheard object names (e.g. "beer" for "bear", "fox" for "box"):
"{instruction}"

1. List the distinct objects on the work mat, each with a short visual name \
(e.g. "black plush bear in a yellow shirt", "black bolt", "clear plastic container").
2. For every object the command refers to, give the visible object it most plausibly means \
(allow for misheard names), or null if nothing visible fits. Never map a name onto an object \
just because it is the only one left: a bolt is not a "toy", a plush animal is not a "screw".
3. Rewrite the command as a short English imperative using those visible names, or null if \
any referenced object has no plausible visible match. Only replace the object names: keep \
everything else exactly (counts, "all", order, destinations, and constraints such as which \
hand or arm to use or not to use).

Reply with JSON only:
{{"objects": ["..."], "mapping": [{{"said": "...", "visible": "..." or null, "sure": true or false}}], \
"ok": true or false, "instruction": "..." or null, "note": "<one short sentence for the operator>"}}
"ok" is true only if every referenced object maps to exactly one visible object and you are sure.
"""


def _parse_preflight(text: str) -> dict[str, Any]:
    data = _json_obj(text)
    mapping = []
    for m in data.get("mapping") or []:
        if isinstance(m, dict):
            vis = m.get("visible")
            mapping.append(
                {
                    "said": str(m.get("said") or "")[:80],
                    "visible": str(vis)[:120] if vis else None,
                    "sure": m.get("sure") is True,
                }
            )
    instr = data.get("instruction")
    ok = (
        data.get("ok") is True
        and bool(mapping)
        and all(m["visible"] and m["sure"] for m in mapping)
        and isinstance(instr, str)
        and bool(instr.strip())
    )
    return {
        "ok": ok,
        "objects": [str(o)[:120] for o in (data.get("objects") or []) if o][:20],
        "mapping": mapping,
        "instruction": " ".join(instr.split())[:300] if ok else None,
        "note": str(data.get("note") or "").strip()[:300],
    }


def preflight_objects(
    image_rgb: np.ndarray, *, instruction: str, model: str = GEMINI_ROBOTICS_ER_MODEL
) -> dict[str, Any]:
    """Before a spoken task runs: do the objects it names exist? Returns the command rewritten
    with visible names (``instruction``) when every object matches, else ``ok: False``."""
    return _ask(image_rgb, PREFLIGHT_PROMPT.format(instruction=instruction), _parse_preflight, model)


def verdict_text(result: dict[str, Any]) -> str:
    done = result.get("done")
    head = "done" if done is True else ("NOT done" if done is False else "unsure")
    answer = result.get("answer") or result.get("evidence") or ""
    return f"camera check: {head}" + (f" — {answer}" if answer else "")
