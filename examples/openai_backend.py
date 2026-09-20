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

"""Shared VLM backends: GPT-6 Astra, Gemini aliases, Cosmos 3 Nano."""

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
COSMOS3_NANO_MODEL = "nvidia/cosmos3-nano-reasoner"
NVIDIA_CHAT_BASE_URL = "https://integrate.api.nvidia.com/v1"
LOCAL_NIM_BASE_URL = "http://127.0.0.1:8000/v1"

# Astra likes to ask questions; keep the stack non-interactive.
GPT6_JSON_ONLY = (
    "\n\nDo not ask questions. Infer any missing details from the scene and "
    "instruction. Return only the JSON document requested above."
)
COSMOS_JSON_ONLY = (
    "\n\nDo not reason out loud. Do not write analysis, captions, or "
    "step-by-step thoughts. Do not use <think> or markdown fences. "
    "Reply with a single JSON object only. The first character must be `{` "
    "and the last character must be `}`."
)
COSMOS_DETECTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "grasp_point": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "long_axis": {},
                    "blocked_by": {},
                },
                "required": ["name", "box_2d", "grasp_point"],
            },
        }
    },
    "required": ["objects"],
}

# Name numbered close-up tiles. Do not ask Nano to invent boxes.
COSMOS_NAME_ENUM = [
    "skip",
    "red_pen",
    "screw",
    "vial",
    "black_frame",
    "black_block",
    "purple_screwdriver",
]
COSMOS_NAMING_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string", "enum": list(COSMOS_NAME_ENUM)},
                },
                "required": ["id", "name"],
            },
        }
    },
    "required": ["objects"],
}

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
_COSMOS_ALIASES = frozenset(
    {
        "cosmos",
        "cosmos3",
        "cosmos-3",
        "cosmos3-nano",
        "cosmos-3-nano",
        "cosmos3nano",
        "cosmos-nano",
        "nvidia-cosmos",
        "nvidia/cosmos3-nano",
        "nvidia/cosmos3-nano-reasoner",
        "cosmos3-nano-reasoner",
    }
)

VLM_MODEL_HELP = (
    "Base VLM: gemini (Google Robotics-ER), gpt-6 (OpenAI gpt-6-astra), "
    "or cosmos / cosmos3-nano (NVIDIA Cosmos 3 Nano Reasoner). "
    "Cosmos uses NVIDIA_API_KEY against integrate.api.nvidia.com, or a local "
    "NIM/vLLM at COSMOS_BASE_URL (default http://127.0.0.1:8000/v1)."
)


def resolve_base_model(name: str | None) -> tuple[str, str]:
    """Return ``(backend, api_model_id)``. backend is gemini, gpt-6, or cosmos."""
    raw = str(name or "gemini").strip() or "gemini"
    key = raw.lower().replace("_", "-").replace(" ", "")
    if key in _GPT6_ALIASES or key.startswith("gpt-6"):
        return "gpt-6", GPT6_ASTRA_MODEL if key in _GPT6_ALIASES else raw
    if key in _COSMOS_ALIASES or "cosmos3-nano" in key or key.startswith("cosmos"):
        return "cosmos", COSMOS3_NANO_MODEL if key in _COSMOS_ALIASES else raw
    if key in _GEMINI_ALIASES:
        return "gemini", GEMINI_ROBOTICS_ER_MODEL
    return "gemini", raw


def nvidia_api_key(explicit: str | None = None) -> str | None:
    return (
        explicit
        or os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("NGC_API_KEY")
        or os.environ.get("NIM_API_KEY")
    )


def cosmos_base_url(explicit: str | None = None) -> str:
    raw = (
        explicit
        or os.environ.get("COSMOS_BASE_URL")
        or os.environ.get("NVIDIA_BASE_URL")
        or (NVIDIA_CHAT_BASE_URL if nvidia_api_key() else LOCAL_NIM_BASE_URL)
    )
    url = str(raw).rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url


def _extract_chat_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], Mapping):
        return ""
    message = choices[0].get("message") or {}
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, Mapping) and part.get("type") in {"text", "output_text"}:
                chunks.append(str(part.get("text") or ""))
        return "".join(chunks)
    return str(content or "")


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


def _post_chat_completions(
    *,
    endpoint: str,
    model: str,
    messages: list[Any],
    key: str,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    extra_body: Mapping[str, Any] | None = None,
    response_format: Mapping[str, Any] | None = None,
) -> str:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }
    if extra_body:
        body.update(dict(extra_body))
    if response_format:
        body["response_format"] = dict(response_format)
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Cosmos returned non-object from {endpoint}: {payload!r}")
    if payload.get("error"):
        raise RuntimeError(f"Cosmos chat error from {endpoint}: {payload['error']}")
    return _extract_chat_text(payload)


def call_nvidia_chat_completions(
    *,
    model: str = COSMOS3_NANO_MODEL,
    messages: list[Any],
    api_key: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    timeout_s: float = 120.0,
    extra_body: Mapping[str, Any] | None = None,
    response_format: Mapping[str, Any] | None = None,
) -> str:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint (NIM or NVIDIA API)."""
    preferred = cosmos_base_url(base_url)
    key = nvidia_api_key(api_key)
    hosted_preferred = "api.nvidia.com" in preferred
    if hosted_preferred and not key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not set (needed for hosted Cosmos). "
            "Export NVIDIA_API_KEY, or point COSMOS_BASE_URL at a local NIM/vLLM."
        )
    if not key:
        key = "not-used"

    bases = [preferred]
    if hosted_preferred:
        for extra in (NVIDIA_CHAT_BASE_URL, "https://ai.api.nvidia.com/v1"):
            if extra not in bases:
                bases.append(extra)

    last_error: Exception | None = None
    for base in bases:
        endpoint = f"{base.rstrip('/')}/chat/completions"
        print(f"[cosmos] POST {endpoint} model={model}", flush=True)
        if extra_body or response_format:
            print(
                f"[cosmos] structured extra={list((extra_body or {}).keys())} "
                f"response_format={None if response_format is None else response_format.get('type')}",
                flush=True,
            )
        try:
            from openai import OpenAI

            client = OpenAI(api_key=key, base_url=base)
            create_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if extra_body:
                create_kwargs["extra_body"] = dict(extra_body)
            if response_format:
                create_kwargs["response_format"] = dict(response_format)
            response = client.chat.completions.create(**create_kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                print(
                    f"[cosmos] usage prompt={getattr(usage, 'prompt_tokens', '?')} "
                    f"completion={getattr(usage, 'completion_tokens', '?')} "
                    f"max_tokens={max_tokens}",
                    flush=True,
                )
            text = ""
            if response.choices:
                content = response.choices[0].message.content
                text = content if isinstance(content, str) else (content or "")
            if str(text).strip():
                return str(text)
            dumped = response.model_dump() if hasattr(response, "model_dump") else {}
            text = _extract_chat_text(dumped if isinstance(dumped, Mapping) else {})
            if text.strip():
                return text
        except ImportError:
            try:
                text = _post_chat_completions(
                    endpoint=endpoint,
                    model=model,
                    messages=messages,
                    key=key,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout_s=timeout_s,
                    extra_body=extra_body,
                    response_format=response_format,
                )
                if text.strip():
                    return text
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                last_error = RuntimeError(
                    f"Cosmos chat HTTP {exc.code} at {endpoint}: {detail}"
                )
                if exc.code == 404:
                    continue
                raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"Cannot reach {endpoint}: {exc}")
                if not hosted_preferred:
                    raise RuntimeError(
                        f"No Cosmos server at {endpoint} ({exc}). "
                        "COSMOS_BASE_URL points at this machine, but nothing is "
                        "listening. Start NIM/vLLM first, for example:\n"
                        "  vllm serve nvidia/Cosmos3-Nano --port 8000\n"
                        "or unset COSMOS_BASE_URL to try the NVIDIA hosted API."
                    ) from exc
                continue
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            if status == 404:
                continue
            raise RuntimeError(f"Cosmos chat failed at {endpoint}: {exc}") from exc

    if hosted_preferred:
        raise RuntimeError(
            "NVIDIA hosted Cosmos 3 Nano chat endpoint returned 404 / is not wired. "
            "A local server is required:\n"
            "  export COSMOS_BASE_URL=http://127.0.0.1:8000/v1\n"
            "  vllm serve nvidia/Cosmos3-Nano --port 8000\n"
            f"Last error: {last_error}"
        ) from last_error
    raise RuntimeError(f"Cosmos request failed: {last_error}") from last_error
