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

"""Gemini Robotics-ER VLM call + JSON parsing (swappable / mockable)."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any, TypedDict

import numpy as np
from PIL import Image

from .prompts import DEFAULT_DETECTION_PROMPT

# 1.5-preview is retired; use ER 2 (or pass model= to override).
GEMINI_ROBOTICS_ER_MODEL = "gemini-robotics-er-2-preview"

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


class RawDetection(TypedDict):
    name: str
    box_2d: list[float]  # [ymin, xmin, ymax, xmax] in 0-1000
    grasp_point: list[float]  # [y, x] in 0-1000
    blocked_by: str | None


class ParsedDetection(TypedDict):
    name: str
    box_2d_px: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax in pixels
    grasp_point_px: tuple[float, float]  # u, v in pixels
    blocked_by: str | None


VlmCaller = Callable[[np.ndarray, str, str], list[RawDetection]]


def strip_markdown_fences(text: str) -> str:
    """Remove surrounding ``` / ```json fences if present."""
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    # Fallback: extract first {...} or [...] block if fences appear mid-text.
    if "```" in stripped:
        inner = re.sub(r"^.*?```(?:json)?\s*", "", stripped, count=1, flags=re.DOTALL | re.IGNORECASE)
        inner = re.sub(r"\s*```.*$", "", inner, count=1, flags=re.DOTALL)
        return inner.strip()
    return stripped


def _strip_trailing_commas(text: str) -> str:
    """Tolerate common LLM JSON: ``,}`` / ``,]`` (also nested)."""
    prev = None
    out = text
    # Repeat until stable — handles ``],`` / ``},`` chains.
    while prev != out:
        prev = out
        out = re.sub(r",(\s*[}\]])", r"\1", out)
    return out


def _loads_json(text: str) -> Any:
    cleaned = strip_markdown_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return json.loads(_strip_trailing_commas(cleaned))


def _normalize_objects_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "objects" in payload and isinstance(payload["objects"], list):
            return payload["objects"]
        # Single-object dict.
        if "name" in payload or "label" in payload:
            return [payload]
    raise ValueError(f"Unexpected VLM JSON shape: {type(payload)!r}")


def _coerce_raw_detection(item: dict[str, Any]) -> RawDetection:
    name = str(item.get("name") or item.get("label") or "").strip()
    if not name:
        raise ValueError(f"Detection missing name/label: {item!r}")

    box = item.get("box_2d")
    if box is None and all(k in item for k in ("y", "x", "y2", "x2")):
        box = [item["y"], item["x"], item["y2"], item["x2"]]
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"Invalid box_2d for {name!r}: {box!r}")

    grasp = item.get("grasp_point")
    if grasp is None and "point" in item:
        grasp = item["point"]
    if not isinstance(grasp, (list, tuple)) or len(grasp) != 2:
        # Fall back to box centre in normalized coords.
        ymin, xmin, ymax, xmax = (float(c) for c in box)
        grasp = [(ymin + ymax) / 2.0, (xmin + xmax) / 2.0]

    blocked = item.get("blocked_by", None)
    if blocked is not None:
        blocked = str(blocked).strip()
        if blocked.lower() in {"", "null", "none", "n/a"}:
            blocked = None

    return {
        "name": name,
        "box_2d": [float(c) for c in box],
        "grasp_point": [float(c) for c in grasp],
        "blocked_by": blocked,
    }


def parse_vlm_json_text(text: str) -> list[RawDetection]:
    """Parse VLM text into raw detections. Raises ``json.JSONDecodeError`` / ``ValueError``."""
    payload = _loads_json(text)
    objects = _normalize_objects_payload(payload)
    return [_coerce_raw_detection(obj) for obj in objects]


def call_gemini_robotics_er(
    image: np.ndarray,
    instruction: str,
    prompt: str | None = None,
    *,
    model: str = GEMINI_ROBOTICS_ER_MODEL,
    api_key: str | None = None,
) -> list[RawDetection]:
    """
    Call Gemini Robotics-ER 1.5 and return parsed detections.

    On malformed JSON: retry the API once, then raise.
    """
    from google import genai
    from google.genai import types

    template = prompt if prompt else DEFAULT_DETECTION_PROMPT
    prompt_text = template.format(instruction=instruction)
    key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=key) if key else genai.Client()

    if image.dtype != np.uint8:
        raise TypeError(f"image must be uint8 RGB, got dtype={image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be HxWx3 RGB, got shape={image.shape}")

    pil = Image.fromarray(image)
    # Encode as PNG bytes for the API.
    import io

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        prompt_text,
    ]
    # Prefer low-latency config; fall back if the model rejects thinking_config.
    configs = [
        types.GenerateContentConfig(
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
        types.GenerateContentConfig(temperature=0.2),
    ]

    last_error: Exception | None = None
    text = ""
    for attempt in range(2):
        for cfg in configs:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=cfg,
                )
                text = getattr(response, "text", None) or ""
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                # Try next config shape; only raise after both fail on last attempt.
                text = ""
                continue
        if not text:
            if attempt == 0:
                continue
            raise RuntimeError(f"Gemini request failed: {last_error}") from last_error
        try:
            return parse_vlm_json_text(text)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt == 0:
                continue
            raise ValueError(
                f"Malformed VLM JSON after retry. Last response was:\n{text[:2000]}"
            ) from last_error

    raise RuntimeError("Unreachable")  # pragma: no cover


def _norm_to_px_y(y_norm: float, height: int) -> float:
    return float(y_norm) / 1000.0 * float(height)


def _norm_to_px_x(x_norm: float, width: int) -> float:
    return float(x_norm) / 1000.0 * float(width)


def _point_in_box(u: float, v: float, xmin: float, ymin: float, xmax: float, ymax: float) -> bool:
    return xmin <= u <= xmax and ymin <= v <= ymax


def parse_vlm_detections(
    raw: list[RawDetection],
    image_hw: tuple[int, int],
) -> list[ParsedDetection]:
    """
    Convert normalized 0-1000 detections to pixel coordinates and apply
    post-VLM consistency fixes:
      - blocked_by not in detection list -> null
      - grasp_point outside its box_2d -> box centre
    """
    height, width = image_hw
    names = {det["name"] for det in raw}
    parsed: list[ParsedDetection] = []

    for det in raw:
        ymin_n, xmin_n, ymax_n, xmax_n = det["box_2d"]
        xmin = _norm_to_px_x(xmin_n, width)
        xmax = _norm_to_px_x(xmax_n, width)
        ymin = _norm_to_px_y(ymin_n, height)
        ymax = _norm_to_px_y(ymax_n, height)
        # Ensure ordering after conversion.
        xmin, xmax = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
        ymin, ymax = (ymin, ymax) if ymin <= ymax else (ymax, ymin)

        gy_n, gx_n = det["grasp_point"]
        u = _norm_to_px_x(gx_n, width)
        v = _norm_to_px_y(gy_n, height)
        if not _point_in_box(u, v, xmin, ymin, xmax, ymax):
            u = 0.5 * (xmin + xmax)
            v = 0.5 * (ymin + ymax)

        blocked = det["blocked_by"]
        if blocked is not None and blocked not in names:
            blocked = None
        # An object cannot block itself.
        if blocked == det["name"]:
            blocked = None

        parsed.append(
            {
                "name": det["name"],
                "box_2d_px": (xmin, ymin, xmax, ymax),
                "grasp_point_px": (u, v),
                "blocked_by": blocked,
            }
        )
    return parsed
