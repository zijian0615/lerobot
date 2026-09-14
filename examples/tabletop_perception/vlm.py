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
import logging
import os
import re
from collections.abc import Callable
from typing import Any, NotRequired, TypedDict

import numpy as np
from PIL import Image

from openai_backend import (
    GPT6_ASTRA_MODEL,
    GPT6_JSON_ONLY,
    call_openai_responses,
    reasoning_effort_from_thinking_budget,
    resolve_base_model,
)
from .prompts import DEFAULT_DETECTION_PROMPT

# 1.5-preview is retired; use ER 2 (or pass model= to override).
GEMINI_ROBOTICS_ER_MODEL = "gemini-robotics-er-2-preview"

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)
logger = logging.getLogger(__name__)


class RawDetection(TypedDict):
    name: str
    box_2d: list[float]  # [ymin, xmin, ymax, xmax] in 0-1000
    grasp_point: list[float]  # [y, x] in 0-1000
    blocked_by: str | None
    long_axis: NotRequired[list[list[float]] | None]  # [[y1,x1],[y2,x2]] in 0-1000
    polygon: NotRequired[list[list[float]] | None]  # [[y,x], ...] outline in 0-1000


class ParsedDetection(TypedDict):
    name: str
    box_2d_px: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax in pixels
    grasp_point_px: tuple[float, float]  # u, v in pixels
    blocked_by: str | None
    long_axis_px: NotRequired[tuple[tuple[float, float], tuple[float, float]] | None]
    polygon_px: NotRequired[list[tuple[float, float]] | None]


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


def _aabb_from_polygon(polygon: Any) -> list[float] | None:
    """``[ymin, xmin, ymax, xmax]`` from ``[[y,x], ...]`` if the outline is usable."""
    pts = _coerce_polygon(polygon)
    if not pts:
        return None
    ys = [p[0] for p in pts]
    xs = [p[1] for p in pts]
    return [min(ys), min(xs), max(ys), max(xs)]


def _coerce_box_2d(name: str, item: dict[str, Any]) -> list[float]:
    box = item.get("box_2d")
    if box is None and all(k in item for k in ("y", "x", "y2", "x2")):
        box = [item["y"], item["x"], item["y2"], item["x2"]]
    if isinstance(box, (list, tuple)) and len(box) == 4:
        return [float(c) for c in box]
    inferred = _aabb_from_polygon(item.get("polygon"))
    if inferred is not None:
        logger.warning(
            "VLM box_2d for %r was %r; using polygon AABB %s",
            name,
            box,
            [round(v, 1) for v in inferred],
        )
        return inferred
    raise ValueError(f"Invalid box_2d for {name!r}: {box!r}")


def _coerce_raw_detection(item: dict[str, Any]) -> RawDetection:
    name = str(item.get("name") or item.get("label") or "").strip()
    if not name:
        raise ValueError(f"Detection missing name/label: {item!r}")

    box = _coerce_box_2d(name, item)

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

    long_axis = _coerce_long_axis(item.get("long_axis"))
    polygon = _coerce_polygon(item.get("polygon"))

    return {
        "name": name,
        "box_2d": box,
        "grasp_point": [float(c) for c in grasp],
        "blocked_by": blocked,
        "long_axis": long_axis,
        "polygon": polygon,
    }


def _coerce_long_axis(raw: Any) -> list[list[float]] | None:
    """Accept [[y,x],[y,x]], flat [y1,x1,y2,x2], or null-ish → None."""
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "null", "none", "n/a"}:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 4 and all(
        isinstance(v, (int, float)) for v in raw
    ):
        return [[float(raw[0]), float(raw[1])], [float(raw[2]), float(raw[3])]]
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    pts: list[list[float]] = []
    for pt in raw:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            return None
        pts.append([float(pt[0]), float(pt[1])])
    return pts


def _coerce_polygon(raw: Any) -> list[list[float]] | None:
    """Accept [[y,x], ...] with ≥3 vertices, or null-ish → None."""
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "null", "none", "n/a"}:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    pts: list[list[float]] = []
    for pt in raw:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            return None
        pts.append([float(pt[0]), float(pt[1])])
    return pts


def parse_vlm_json_text(text: str) -> list[RawDetection]:
    """Parse VLM text into raw detections. Raises ``json.JSONDecodeError`` / ``ValueError``."""
    payload = _loads_json(text)
    objects = _normalize_objects_payload(payload)
    out: list[RawDetection] = []
    errors: list[str] = []
    for obj in objects:
        if not isinstance(obj, dict):
            errors.append(f"non-object item: {type(obj)!r}")
            continue
        try:
            out.append(_coerce_raw_detection(obj))
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
            logger.warning("Skipping malformed VLM object: %s", exc)
    if not out:
        detail = "; ".join(errors) if errors else "empty objects list"
        raise ValueError(f"No usable VLM detections ({detail})")
    return out


def _png_bytes(image: np.ndarray) -> bytes:
    if image.dtype != np.uint8:
        raise TypeError(f"image must be uint8 RGB, got dtype={image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be HxWx3 RGB, got shape={image.shape}")
    import io

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return buf.getvalue()


def call_gpt6_detection(
    image: np.ndarray,
    instruction: str,
    prompt: str | None = None,
    *,
    model: str = GPT6_ASTRA_MODEL,
    api_key: str | None = None,
    thinking_budget: int = -1,
) -> list[RawDetection]:
    """Call GPT-6 Astra (vision) and return parsed detections."""
    import base64

    template = prompt if prompt else DEFAULT_DETECTION_PROMPT
    prompt_text = template.format(instruction=instruction) + GPT6_JSON_ONLY
    image_b64 = base64.b64encode(_png_bytes(image)).decode("ascii")
    effort = reasoning_effort_from_thinking_budget(thinking_budget)
    input_payload = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_b64}",
                    "detail": "high",
                },
            ],
        }
    ]
    last_error: Exception | None = None
    text = ""
    for attempt in range(2):
        try:
            text = call_openai_responses(
                model=model,
                input_payload=input_payload,
                api_key=api_key,
                reasoning_effort=effort,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 0:
                continue
            raise RuntimeError(f"GPT-6 request failed: {last_error}") from last_error
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


def call_detection_vlm(
    image: np.ndarray,
    instruction: str,
    prompt: str | None = None,
    *,
    model: str = GEMINI_ROBOTICS_ER_MODEL,
    api_key: str | None = None,
    thinking_budget: int = -1,
) -> list[RawDetection]:
    """Dispatch perception VLM: Gemini Robotics-ER (default) or GPT-6 Astra."""
    backend, api_model = resolve_base_model(model)
    if backend == "gpt-6":
        return call_gpt6_detection(
            image,
            instruction,
            prompt=prompt,
            model=api_model,
            api_key=api_key,
            thinking_budget=thinking_budget,
        )
    return call_gemini_robotics_er(
        image,
        instruction,
        prompt=prompt,
        model=api_model,
        api_key=api_key,
        thinking_budget=thinking_budget,
    )


def call_gemini_robotics_er(
    image: np.ndarray,
    instruction: str,
    prompt: str | None = None,
    *,
    model: str = GEMINI_ROBOTICS_ER_MODEL,
    api_key: str | None = None,
    thinking_budget: int = -1,
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
    image_bytes = _png_bytes(image)

    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        prompt_text,
    ]
    # 0 = off, -1 = dynamic, >0 = token cap. Fall back if thinking_config is rejected.
    configs = [
        types.GenerateContentConfig(
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=int(thinking_budget)),
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

        long_axis_px = None
        raw_axis = det.get("long_axis")
        if raw_axis is not None and len(raw_axis) == 2:
            (y0, x0), (y1, x1) = raw_axis
            u0 = _norm_to_px_x(x0, width)
            v0 = _norm_to_px_y(y0, height)
            u1 = _norm_to_px_x(x1, width)
            v1 = _norm_to_px_y(y1, height)
            long_axis_px = ((u0, v0), (u1, v1))

        polygon_px = None
        raw_poly = det.get("polygon")
        if raw_poly is not None and len(raw_poly) >= 3:
            polygon_px = [
                (_norm_to_px_x(x_n, width), _norm_to_px_y(y_n, height))
                for y_n, x_n in raw_poly
            ]

        parsed.append(
            {
                "name": det["name"],
                "box_2d_px": (xmin, ymin, xmax, ymax),
                "grasp_point_px": (u, v),
                "blocked_by": blocked,
                "long_axis_px": long_axis_px,
                "polygon_px": polygon_px,
            }
        )
    return parsed
