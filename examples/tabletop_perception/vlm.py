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
import math
import os
import re
import time
from collections.abc import Callable
from typing import Any, NotRequired, TypedDict

import numpy as np
from PIL import Image

from openai_backend import (
    COSMOS3_NANO_MODEL,
    COSMOS_NAMING_JSON_SCHEMA,
    GPT6_ASTRA_MODEL,
    GPT6_JSON_ONLY,
    call_nvidia_chat_completions,
    call_openai_responses,
    reasoning_effort_from_thinking_budget,
    resolve_base_model,
)
from .prompts import COSMOS_NAMING_PROMPT, DEFAULT_DETECTION_PROMPT

# 1.5-preview is retired; use ER 2 (or pass model= to override).
GEMINI_ROBOTICS_ER_MODEL = "gemini-robotics-er-2-preview"

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
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


def _strip_think_blocks(text: str) -> str:
    """Drop Cosmos/Qwen ``<think>…</think>``. Unclosed think ⇒ empty (no JSON yet)."""
    stripped = _THINK_RE.sub("", text)
    if "</think>" in stripped.lower():
        parts = re.split(r"</think>", stripped, maxsplit=1, flags=re.IGNORECASE)
        stripped = parts[-1]
    elif "<think>" in stripped.lower():
        return ""
    return stripped.strip()


def _json_candidate(text: str) -> str:
    cleaned = strip_markdown_fences(_strip_think_blocks(text))
    brace = cleaned.find("{")
    bracket = cleaned.find("[")
    starts = [i for i in (brace, bracket) if i >= 0]
    if not starts:
        return cleaned
    return cleaned[min(starts) :].strip()


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
    cleaned = _json_candidate(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return json.loads(_strip_trailing_commas(cleaned))


def _salvage_object_dicts(text: str) -> list[dict[str, Any]]:
    """Pull complete object dicts from truncated ``{"objects":[...]}`` JSON."""
    cleaned = _json_candidate(text)
    start = cleaned.find("[")
    if start < 0:
        return []
    decoder = json.JSONDecoder()
    objs: list[dict[str, Any]] = []
    i = start + 1
    n = len(cleaned)
    while i < n:
        while i < n and cleaned[i] in " \t\n\r,":
            i += 1
        if i >= n or cleaned[i] in "]":
            break
        if cleaned[i] != "{":
            break
        try:
            obj, end = decoder.raw_decode(cleaned, i)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            objs.append(obj)
        i = end
    return objs


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
    bbox = item.get("bbox_2d")
    if box is None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        # Official Cosmos xyxy → internal [ymin, xmin, ymax, xmax].
        x1, y1, x2, y2 = (float(c) for c in bbox)
        box = [y1, x1, y2, x2]
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
    if grasp is None and isinstance(item.get("point_2d"), (list, tuple)):
        # Official Cosmos [x, y] → internal [y, x].
        px, py = item["point_2d"][:2]
        grasp = [py, px]
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
    try:
        payload = _loads_json(text)
        objects = _normalize_objects_payload(payload)
    except (json.JSONDecodeError, ValueError):
        salvaged = _salvage_object_dicts(text)
        if not salvaged:
            raise
        print(f"[vlm] truncated JSON, salvaged {len(salvaged)} complete objects", flush=True)
        objects = salvaged
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


def _jpeg_bytes(image: np.ndarray, *, quality: int = 85) -> bytes:
    if image.dtype != np.uint8:
        raise TypeError(f"image must be uint8 RGB, got dtype={image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be HxWx3 RGB, got shape={image.shape}")
    import io

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=int(quality), optimize=True)
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


_COSMOS_SKIP_NAMES = frozenset(
    {
        "skip",
        "none",
        "null",
        "n_a",
        "na",
        "print",
        "printed",
        "drawing",
        "grid",
        "table",
        "robot",
        "arm",
        "cable",
        "wire",
        "empty",
        "background",
        "shadow",
        "text",
        "yellow_circle",
        "yellow_diamond",
    }
)
_COSMOS_PLACEHOLDER_NAMES = frozenset(
    {
        "object",
        "objects",
        "item",
        "thing",
        "target",
        "detection",
        "label",
        "name",
        "object_name",
        "specific_name",
        "real_3d_object",
        "3d_object",
        "graspable",
        "graspable_object",
    }
)
_COSMOS_PLACEHOLDER_TOKENS = frozenset(
    {
        "real",
        "3d",
        "object",
        "objects",
        "graspable",
        "item",
        "thing",
        "target",
        "all",
        "instance",
    }
)


# Nano's native vision cap is 720p 16:9 (1280x720). Capture stays 1080p.
_COSMOS_VIEW_W = 1280
_COSMOS_VIEW_H = 720


def _letterbox_cosmos_view(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Fit ``image`` into a 1280x720 canvas. Return (view, affine meta)."""
    orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
    tw, th = _COSMOS_VIEW_W, _COSMOS_VIEW_H
    scale = min(tw / max(orig_w, 1), th / max(orig_h, 1))
    nw = max(1, int(round(orig_w * scale)))
    nh = max(1, int(round(orig_h * scale)))
    resized = Image.fromarray(image).resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tw, th), (0, 0, 0))
    ox = (tw - nw) // 2
    oy = (th - nh) // 2
    canvas.paste(resized, (ox, oy))
    return np.asarray(canvas), {
        "scale": float(scale),
        "ox": float(ox),
        "oy": float(oy),
        "tw": float(tw),
        "th": float(th),
        "orig_w": float(orig_w),
        "orig_h": float(orig_h),
    }


def _canvas_xy_to_orig_norm(
    x_n: float, y_n: float, meta: dict[str, float]
) -> tuple[float, float]:
    """Map a 0-1000 (x, y) on the 720p canvas to 0-1000 on the original frame."""
    x_c = float(x_n) / 1000.0 * meta["tw"]
    y_c = float(y_n) / 1000.0 * meta["th"]
    scale = max(float(meta["scale"]), 1e-6)
    x_o = (x_c - meta["ox"]) / scale
    y_o = (y_c - meta["oy"]) / scale
    x_o = min(max(x_o, 0.0), meta["orig_w"] - 1e-3)
    y_o = min(max(y_o, 0.0), meta["orig_h"] - 1e-3)
    return x_o / meta["orig_w"] * 1000.0, y_o / meta["orig_h"] * 1000.0


def _remap_cosmos_item(item: dict[str, Any], meta: dict[str, float]) -> dict[str, Any]:
    """Convert official ``bbox_2d``/``point_2d`` (xy on 720p) to internal yx 0-1000."""
    out = dict(item)
    if isinstance(item.get("bbox_2d"), (list, tuple)) and len(item["bbox_2d"]) == 4:
        x1, y1, x2, y2 = (float(v) for v in item["bbox_2d"])
        xa, ya = _canvas_xy_to_orig_norm(x1, y1, meta)
        xb, yb = _canvas_xy_to_orig_norm(x2, y2, meta)
        out["box_2d"] = [min(ya, yb), min(xa, xb), max(ya, yb), max(xa, xb)]
    elif isinstance(item.get("box_2d"), (list, tuple)) and len(item["box_2d"]) == 4:
        ymin, xmin, ymax, xmax = (float(v) for v in item["box_2d"])
        xa, ya = _canvas_xy_to_orig_norm(xmin, ymin, meta)
        xb, yb = _canvas_xy_to_orig_norm(xmax, ymax, meta)
        out["box_2d"] = [min(ya, yb), min(xa, xb), max(ya, yb), max(xa, xb)]

    if isinstance(item.get("point_2d"), (list, tuple)) and len(item["point_2d"]) == 2:
        x, y = (float(v) for v in item["point_2d"])
        xn, yn = _canvas_xy_to_orig_norm(x, y, meta)
        out["grasp_point"] = [yn, xn]
    elif isinstance(item.get("grasp_point"), (list, tuple)) and len(item["grasp_point"]) == 2:
        gy, gx = (float(v) for v in item["grasp_point"])
        xn, yn = _canvas_xy_to_orig_norm(gx, gy, meta)
        out["grasp_point"] = [yn, xn]

    axis = item.get("long_axis")
    if isinstance(axis, (list, tuple)) and len(axis) == 2:
        remapped: list[list[float]] = []
        for pt in axis:
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                remapped = []
                break
            yn, xn = float(pt[0]), float(pt[1])
            xo, yo = _canvas_xy_to_orig_norm(xn, yn, meta)
            remapped.append([yo, xo])
        if remapped:
            out["long_axis"] = remapped
    out.pop("bbox_2d", None)
    out.pop("point_2d", None)
    return out


def _cosmos_item_name(item: dict[str, Any]) -> str:
    """Nano often returns bbox_2d with no label. Keep the box; name it ``part``."""
    for key in ("name", "label", "category", "class_name", "class", "type", "caption"):
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"null", "none", "n/a", ""}:
            return text
    return "part"


_KEEP_HINT_TOKENS = frozenset(
    {
        "container",
        "cup",
        "mug",
        "bottle",
        "can",
        "pen",
        "frame",
        "block",
        "box",
        "bowl",
        "bin",
        "tray",
        "spray",
    }
)


_GENERIC_SURFACE_NAMES = frozenset(
    {
        "wooden_block",
        "block",
        "cube",
        "box",
        "wooden_box",
        "square_wooden_box",
        "wood_block",
        "wood_box",
    }
)


def _prefer_detector_container_name(cosmos_name: str, keep: str) -> bool:
    """DINO said container; Cosmos renamed it cube/block. Keep the hint."""
    keep_toks = set(keep.split("_"))
    if not keep_toks & {"container", "bin", "bowl", "tray"}:
        return False
    slug = slug_detection_name(cosmos_name)
    return slug in _GENERIC_SURFACE_NAMES or slug.split("_")[-1] in {
        "block",
        "cube",
        "box",
    }


def _hint_keep_name(hint: str) -> str | None:
    """Keep a detector box when Cosmos says skip, if the hint is a real part."""
    slug = slug_detection_name(hint)
    if slug == "screw" or slug.startswith("screw_"):
        return "screw"
    tokens = [t for t in slug.split("_") if t and t not in {"part", "object", "objects"}]
    if not tokens or not any(t in _KEEP_HINT_TOKENS for t in tokens):
        return None
    cleaned: list[str] = []
    for tok in tokens:
        if cleaned and tok == cleaned[-1]:
            continue
        cleaned.append(tok)
    return "_".join(cleaned) or None


def _cosmos_skip_name(name: str) -> bool:
    slug = slug_detection_name(name)
    if slug in _COSMOS_SKIP_NAMES or slug in _COSMOS_PLACEHOLDER_NAMES:
        return True
    tokens = [t for t in slug.split("_") if t]
    if tokens and set(tokens) <= _COSMOS_PLACEHOLDER_TOKENS:
        return True
    return bool(set(tokens) & {"skip", "print", "printed", "drawing", "grid", "robot", "cable"})


def _drop_cosmos_top_strip(dets: list[RawDetection]) -> list[RawDetection]:
    """Drop a padded row of boxes glued to the top of the frame (quota fill)."""
    top: list[RawDetection] = []
    rest: list[RawDetection] = []
    for det in dets:
        ymax = float(det["box_2d"][2])
        (rest if ymax >= 200.0 else top).append(det)
    if len(top) >= 4 and len(top) >= len(rest):
        print(f"[cosmos] drop {len(top)} top-strip boxes (quota fill)", flush=True)
        return rest
    return dets


def _cosmos_box_is_blank(
    image: np.ndarray, box_yx: list[float], *, mean_min: float = 225.0, std_max: float = 10.0
) -> bool:
    """True when the box sits on uniform bright table, not on a real part."""
    if image.ndim != 3 or len(box_yx) != 4:
        return False
    height, width = int(image.shape[0]), int(image.shape[1])
    ymin, xmin, ymax, xmax = (float(v) for v in box_yx)
    y0 = int(min(ymin, ymax) / 1000.0 * height)
    y1 = int(max(ymin, ymax) / 1000.0 * height)
    x0 = int(min(xmin, xmax) / 1000.0 * width)
    x1 = int(max(xmin, xmax) / 1000.0 * width)
    y0, y1 = max(0, y0), min(height, max(y1, y0 + 1))
    x0, x1 = max(0, x0), min(width, max(x1, x0 + 1))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return True
    gray = crop.mean(axis=2)
    return float(gray.mean()) >= mean_min and float(gray.std()) <= std_max


_COSMOS_MAX_BOXES = 16
_COSMOS_TILE_PX = 256


def _cosmos_object_sheet(image: np.ndarray, blobs: list[dict]) -> np.ndarray:
    """One close-up tile per detector box so Nano names the object, not a scene."""
    import cv2

    n = max(len(blobs), 1)
    cols = min(4, n)
    rows = int(math.ceil(n / cols))
    tile = _COSMOS_TILE_PX
    canvas = np.full((rows * tile, cols * tile, 3), 245, dtype=np.uint8)
    height, width = image.shape[:2]
    for i, blob in enumerate(blobs):
        r, c = divmod(i, cols)
        x0, y0, x1, y1 = (float(v) for v in blob["box_2d_px"])
        bw, bh = max(x1 - x0, 8.0), max(y1 - y0, 8.0)
        margin = int(max(8.0, 0.18 * max(bw, bh)))
        xa = max(0, int(math.floor(x0)) - margin)
        ya = max(0, int(math.floor(y0)) - margin)
        xb = min(width, int(math.ceil(x1)) + margin)
        yb = min(height, int(math.ceil(y1)) + margin)
        crop = image[ya:yb, xa:xb]
        if crop.size == 0:
            crop = np.full((tile - 16, tile - 16, 3), 245, dtype=np.uint8)
        inner = tile - 28
        ch, cw = crop.shape[:2]
        scale = min(inner / max(ch, 1), inner / max(cw, 1))
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        cell = canvas[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile]
        ox = (tile - nw) // 2
        oy = 22 + (inner - nh) // 2
        cell[oy : oy + nh, ox : ox + nw] = resized
        cv2.rectangle(cell, (2, 2), (tile - 3, tile - 3), (40, 40, 40), 2)
        cv2.putText(
            cell,
            str(i + 1),
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _blob_to_raw_detection(blob: dict, name: str, image_hw: tuple[int, int]) -> RawDetection:
    height, width = image_hw

    def yn(y: float) -> float:
        return float(y) / float(height) * 1000.0

    def xn(x: float) -> float:
        return float(x) / float(width) * 1000.0

    x0, y0, x1, y1 = blob["box_2d_px"]
    gu, gv = blob["grasp_point_px"]
    raw: RawDetection = {
        "name": name,
        "box_2d": [yn(y0), xn(x0), yn(y1), xn(x1)],
        "grasp_point": [yn(gv), xn(gu)],
        "blocked_by": None,
        "long_axis": None,
        "polygon": None,
    }
    axis = blob.get("long_axis_px")
    if axis is not None and len(axis) == 2:
        (u0, v0), (u1, v1) = axis
        raw["long_axis"] = [[yn(v0), xn(u0)], [yn(v1), xn(u1)]]
    poly = blob.get("polygon_px")
    if poly:
        raw["polygon"] = [[yn(v), xn(u)] for u, v in poly]
    return raw


def _parse_cosmos_names(text: str) -> dict[int, str]:
    payload = _loads_json(text)
    items = _normalize_objects_payload(payload)
    names: dict[int, str] = {}
    for i, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        raw_id = item.get("id", item.get("index"))
        try:
            idx = i if raw_id is None else int(raw_id)
        except (TypeError, ValueError):
            continue
        name = str(item.get("name") or item.get("label") or "").strip()
        if idx >= 1 and name:
            names[idx] = name
    return names


def call_cosmos_detection(
    image: np.ndarray,
    instruction: str,
    prompt: str | None = None,
    *,
    model: str = COSMOS3_NANO_MODEL,
    api_key: str | None = None,
    thinking_budget: int = -1,
) -> list[RawDetection]:
    """DINO/YOLO large boxes + dark-CC screws; Cosmos names crops. No invented boxes."""
    import base64

    del thinking_budget  # Cosmos chat has no Gemini-style thinking_budget.
    from .detector import detect_table_objects

    blobs = detect_table_objects(image, instruction, max_dets=_COSMOS_MAX_BOXES)
    if not blobs:
        return []

    template = prompt if prompt else COSMOS_NAMING_PROMPT
    prompt_text = (
        template.format(instruction=instruction)
        + f"\nThere are {len(blobs)} tiles, numbered 1 to {len(blobs)}."
        + "\nAfter </think>, return one JSON object "
        + '{"objects":[{"id":1,"name":"..."}]} with no markdown fences.'
    )
    view = _cosmos_object_sheet(image, blobs)
    print(
        f"[cosmos] name sheet {view.shape[1]}x{view.shape[0]} "
        f"from {len(blobs)} detector boxes on {image.shape[1]}x{image.shape[0]}",
        flush=True,
    )
    schema = dict(COSMOS_NAMING_JSON_SCHEMA)
    schema["properties"] = dict(schema["properties"])
    objects_schema = dict(schema["properties"]["objects"])
    objects_schema["minItems"] = len(blobs)
    objects_schema["maxItems"] = len(blobs)
    schema["properties"]["objects"] = objects_schema
    image_b64 = base64.b64encode(_jpeg_bytes(view, quality=90)).decode("ascii")
    messages = [
        {
            "role": "system",
            "content": "You are a perception JSON emitter. Output only one JSON object.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    last_error: Exception | None = None
    text = ""
    names: dict[int, str] = {}
    for attempt in range(2):
        t0 = time.monotonic()
        try:
            text = call_nvidia_chat_completions(
                model=model,
                messages=messages,
                api_key=api_key,
                max_tokens=512,
                extra_body={"guided_json": schema},
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "tabletop_box_names",
                        "schema": schema,
                    },
                },
            )
            print(
                f"[cosmos] name attempt={attempt + 1} {time.monotonic() - t0:.1f}s "
                f"chars={len(text)}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[cosmos] request failed: {exc}", flush=True)
            if attempt == 0:
                print("[cosmos] retrying once", flush=True)
                continue
            print(f"[cosmos] naming failed, using detector hints: {last_error}", flush=True)
            names = {}
            break
        try:
            names = _parse_cosmos_names(text)
            break
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            head = text[:160].replace("\n", " ")
            print(f"[cosmos] parse failed: {exc}; head={head!r}", flush=True)
            if attempt == 0:
                print("[cosmos] retrying once", flush=True)
                continue
            print("[cosmos] naming parse failed, using detector hints", flush=True)
            names = {}

    out: list[RawDetection] = []
    for i, blob in enumerate(blobs, start=1):
        hint = str(blob.get("hint") or "part")
        cosmos_name = names.get(i)
        keep = _hint_keep_name(hint)
        if cosmos_name and _cosmos_skip_name(cosmos_name):
            if keep:
                print(
                    f"[cosmos] keep tile {i} as {keep} (cosmos skipped {cosmos_name!r} hint={hint!r})",
                    flush=True,
                )
                name = keep
            else:
                print(
                    f"[cosmos] skip tile {i} cosmos={cosmos_name!r} hint={hint}",
                    flush=True,
                )
                continue
        elif cosmos_name:
            name = slug_detection_name(cosmos_name)
            if name in _COSMOS_PLACEHOLDER_NAMES:
                name = keep or slug_detection_name(hint)
            elif keep and _prefer_detector_container_name(name, keep):
                print(
                    f"[cosmos] keep tile {i} as {keep} (cosmos={name!r} hint={hint!r})",
                    flush=True,
                )
                name = keep
        else:
            name = keep or slug_detection_name(hint) or "part"
            print(f"[cosmos] tile {i} unnamed → {name}", flush=True)
        if _cosmos_skip_name(name) and not keep:
            print(f"[cosmos] skip tile {i} name={name!r}", flush=True)
            continue
        if keep and (_cosmos_skip_name(name) or not name):
            name = keep
        print(
            f"[cosmos] tile {i} {name} hint={hint} "
            f"gp={tuple(round(c) for c in blob['grasp_point_px'])}",
            flush=True,
        )
        out.append(_blob_to_raw_detection(blob, name, image.shape[:2]))
    return out


def call_detection_vlm(
    image: np.ndarray,
    instruction: str,
    prompt: str | None = None,
    *,
    model: str = GEMINI_ROBOTICS_ER_MODEL,
    api_key: str | None = None,
    thinking_budget: int = -1,
) -> list[RawDetection]:
    """Dispatch perception VLM: Gemini Robotics-ER, GPT-6 Astra, or Cosmos 3 Nano."""
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
    if backend == "cosmos":
        return call_cosmos_detection(
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
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "Missing Gemini SDK (`google.genai`). Install with: uv pip install google-genai"
        ) from exc

    template = prompt if prompt else DEFAULT_DETECTION_PROMPT
    prompt_text = template.format(instruction=instruction)
    key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "No Gemini API key. Export GOOGLE_API_KEY or GEMINI_API_KEY, then rerun."
        )
    client = genai.Client(api_key=key)
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
    return uniquify_detection_names(parsed)


def slug_detection_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower()).strip("_")
    return slug or "object"


def uniquify_detection_names(detections: list[ParsedDetection]) -> list[ParsedDetection]:
    """Make names unique snake_case so pick lists are usable."""
    used: set[str] = set()
    out: list[ParsedDetection] = []
    for det in detections:
        base = slug_detection_name(str(det.get("name") or "object"))
        name = base
        n = 2
        while name in used:
            name = f"{base}_{n}"
            n += 1
        used.add(name)
        if name != det.get("name"):
            print(f"[name] {det.get('name')!r} → {name}", flush=True)
        item = dict(det)
        item["name"] = name
        out.append(item)
    return out
