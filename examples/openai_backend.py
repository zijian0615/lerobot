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

"""OpenAI GPT-6 Astra Responses API helper (perception + planner)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

GPT6_ASTRA_MODEL = "gpt-6-astra"
GEMINI_ROBOTICS_ER_MODEL = "gemini-robotics-er-2-preview"

# Astra likes to ask questions; keep the stack non-interactive.
GPT6_JSON_ONLY = (
    "\n\nDo not ask questions. Infer any missing details from the scene and "
    "instruction. Return only the JSON document requested above."
)

_GPT6_ALIASES = frozenset(
    {"gpt-6", "gpt6", "gpt-6-astra", "astra", "gpt6astra"}
)
_GEMINI_ALIASES = frozenset(
    {
        "gemini",
        "er",
        "gemini-robotics-er",
        "gemini-robotics-er-2-preview",
    }
)


def resolve_base_model(name: str | None) -> tuple[str, str]:
    """Return ``(backend, api_model_id)``. backend is ``gemini`` or ``gpt-6``."""
    raw = str(name or "gemini").strip() or "gemini"
    key = raw.lower().replace("_", "-").replace(" ", "")
    if key in _GPT6_ALIASES or key.startswith("gpt-6"):
        return "gpt-6", GPT6_ASTRA_MODEL if key in _GPT6_ALIASES else raw
    if key in _GEMINI_ALIASES:
        return "gemini", GEMINI_ROBOTICS_ER_MODEL
    return "gemini", raw


def reasoning_effort_from_thinking_budget(budget: int) -> str:
    """Map Gemini thinking_budget onto Astra ``reasoning.effort``.

    Astra does not support ``none``. 0 → low, -1 → medium, large cap → high.
    """
    b = int(budget)
    if b == 0:
        return "low"
    if b >= 8192:
        return "high"
    return "medium"


def _extract_output_text(payload: Mapping[str, Any]) -> str:
    text = str(payload.get("output_text") or "")
    if text.strip():
        return text
    chunks: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"output_text", "text"}:
                chunks.append(str(part.get("text") or ""))
    return "".join(chunks)


def call_openai_responses(
    *,
    model: str = GPT6_ASTRA_MODEL,
    input_payload: str | list[Any],
    api_key: str | None = None,
    reasoning_effort: str = "low",
    timeout_s: float = 180.0,
) -> str:
    """Call ``POST /v1/responses`` and return the assistant text."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set (needed for --model gpt-6)")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        response = client.responses.create(
            model=model,
            input=input_payload,
            reasoning={"effort": reasoning_effort},
        )
        text = getattr(response, "output_text", None) or ""
        if text.strip():
            return str(text)
        return _extract_output_text(
            response.model_dump() if hasattr(response, "model_dump") else {}
        )
    except ImportError:
        pass

    body = {
        "model": model,
        "input": input_payload,
        "reasoning": {"effort": reasoning_effort},
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"OpenAI Responses HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise RuntimeError(
                "Cannot reach api.openai.com (DNS/network). "
                f"{exc}. Check the Jetson's internet/DNS, then retry --model gpt-6."
            ) from exc
    else:
        raise RuntimeError(f"OpenAI request failed: {last_exc}") from last_exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"OpenAI Responses returned non-object: {payload!r}")
    if payload.get("error"):
        raise RuntimeError(f"OpenAI Responses error: {payload['error']}")
    return _extract_output_text(payload)
