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

"""Open-vocabulary tabletop detector. Boxes only; Cosmos names them.

DINO/YOLO query: instruction nouns plus a generic ``object`` catch-all.
Tiny screws stay on dark connected components — DINO cannot resolve them.
"""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
from PIL import Image

# Size routing only. These words go to dark-CC; DINO hallucinates them.
_SMALL_PART_WORDS = frozenset(
    {"screw", "screws", "bolt", "bolts", "nut", "nuts", "washer", "nail"}
)
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "all",
        "and",
        "or",
        "on",
        "in",
        "of",
        "to",
        "for",
        "with",
        "from",
        "detect",
        "list",
        "pick",
        "grasp",
        "graspable",
        "objects",
        "object",
        "table",
        "tabletop",
        "put",
        "into",
        "move",
        "place",
        "then",
        "please",
    }
)
_MAX_DETS = 16
_MIN_AREA_FRAC = 1500 / (1920 * 1080)
_MIN_SIDE_PX = 36.0
_THIN_MIN_AREA = 400.0
_THIN_MIN_SIDE_PX = 6.0
_THIN_ASPECT = 3.0
_MAX_AREA_FRAC = 0.08
_DINO_ID = "IDEA-Research/grounding-dino-tiny"
_DINO_THRESHOLD = 0.12

_DINO = None
_DINO_PROC = None
_YOLO = None
_YOLO_CLASSES: tuple[str, ...] | None = None


def phrases_from_instruction(instruction: str) -> list[str]:
    """Open-vocab queries: instruction nouns/bigrams, then generic ``object``."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", str(instruction or "").lower())
    seen: list[str] = []
    keys: set[str] = set()

    def add(phrase: str) -> None:
        slug = re.sub(r"[\s_-]+", " ", phrase.strip().lower()).strip()
        key = slug.replace(" ", "_")
        if len(key) < 3 or key in keys:
            return
        words = slug.split()
        if any(w in _STOP or w in _SMALL_PART_WORDS for w in words):
            return
        keys.add(key)
        seen.append(slug)

    for i in range(len(tokens) - 1):
        add(f"{tokens[i]} {tokens[i + 1]}")
    for tok in tokens:
        add(tok)
    seen.append("object")
    return seen


def _xyxy_ok(x0: float, y0: float, x1: float, y1: float, width: int, height: int) -> bool:
    bw = max(x1 - x0, 1.0)
    bh = max(y1 - y0, 1.0)
    area = bw * bh
    aspect = max(bw, bh) / min(bw, bh)
    img_area = float(max(width, 1) * max(height, 1))
    min_area = _THIN_MIN_AREA if aspect >= _THIN_ASPECT else _MIN_AREA_FRAC * img_area
    min_side = _THIN_MIN_SIDE_PX if aspect >= _THIN_ASPECT else _MIN_SIDE_PX
    if area < min_area:
        return False
    if min(bw, bh) < min_side:
        return False
    if area > _MAX_AREA_FRAC * img_area:
        return False
    if bw > 0.55 * width and bh < 0.12 * height:
        return False
    if y0 < 0.28 * height:
        return False
    # Laptop / off-table clutter hugs the frame. Tiny edge screws stay.
    on_edge = x0 <= 0.02 * width or x1 >= 0.98 * width or y1 >= 0.97 * height
    if on_edge and area > 0.015 * img_area:
        return False
    return True


def _crop_blank(image: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> bool:
    crop = image[int(y0) : int(y1), int(x0) : int(x1)]
    if crop.size == 0:
        return True
    gray = crop.mean(axis=2)
    if float((gray < 140.0).mean()) < 0.08:
        return True
    return float(gray.mean()) >= 225.0 and float(gray.std()) <= 10.0


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(ax1 - ax0, 1.0) * max(ay1 - ay0, 1.0)
    ub = max(bx1 - bx0, 1.0) * max(by1 - by0, 1.0)
    return inter / max(ua + ub - inter, 1e-6)


def _nms(items: list[dict[str, Any]], iou_thr: float = 0.5) -> list[dict[str, Any]]:
    items = sorted(items, key=lambda b: -float(b["score"]))
    keep: list[dict[str, Any]] = []
    for item in items:
        box = item["box_2d_px"]
        if any(_iou(box, k["box_2d_px"]) >= iou_thr for k in keep):
            continue
        keep.append(item)
    return keep


def _finalize(
    image: np.ndarray,
    raw: list[tuple[str, float, tuple[float, float, float, float]]],
    max_dets: int,
    *,
    source: str = "dino",
) -> list[dict[str, Any]]:
    height, width = int(image.shape[0]), int(image.shape[1])
    scored: list[dict[str, Any]] = []
    for hint, score, (x0, y0, x1, y1) in raw:
        slug = (hint or "part").strip().lower().replace(" ", "_")
        if slug in _SMALL_PART_WORDS:
            continue
        x0 = min(max(x0, 0.0), width - 1.0)
        y0 = min(max(y0, 0.0), height - 1.0)
        x1 = min(max(x1, x0 + 1.0), float(width))
        y1 = min(max(y1, y0 + 1.0), float(height))
        if not _xyxy_ok(x0, y0, x1, y1, width, height):
            continue
        if _crop_blank(image, x0, y0, x1, y1):
            continue
        scored.append(
            {
                "box_2d_px": (x0, y0, x1, y1),
                "grasp_point_px": (0.5 * (x0 + x1), 0.5 * (y0 + y1)),
                "area": (x1 - x0) * (y1 - y0),
                "score": float(score),
                "hint": hint or "part",
                "source": source,
                "long_axis_px": None,
                "polygon_px": None,
            }
        )
    picked = _nms(scored)[: max(1, int(max_dets))]
    print(
        f"[detector] {source} keep={len(picked)}/{len(raw)} "
        + ", ".join(f"{b['hint']}@{b['score']:.2f}" for b in picked),
        flush=True,
    )
    return picked


def _load_dino():
    global _DINO, _DINO_PROC
    if _DINO is not None:
        return _DINO_PROC, _DINO
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    print(f"[detector] load Grounding DINO {_DINO_ID} (cpu)", flush=True)
    _DINO_PROC = AutoProcessor.from_pretrained(_DINO_ID)
    _DINO = AutoModelForZeroShotObjectDetection.from_pretrained(_DINO_ID)
    _DINO.eval()
    return _DINO_PROC, _DINO


def _detect_dino(image: np.ndarray, instruction: str, max_dets: int) -> list[dict[str, Any]]:
    import torch

    phrases = phrases_from_instruction(instruction)
    text = ". ".join(phrases) + "."
    print(f"[detector] Grounding DINO text={text!r}", flush=True)
    processor, model = _load_dino()
    pil = Image.fromarray(image)
    inputs = processor(images=pil, text=text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, threshold=_DINO_THRESHOLD, target_sizes=[pil.size[::-1]]
    )[0]
    labels = results.get("text_labels") or results.get("labels")
    raw: list[tuple[str, float, tuple[float, float, float, float]]] = []
    for i, box in enumerate(results["boxes"]):
        x0, y0, x1, y1 = (float(v) for v in box.tolist())
        hint = ""
        if labels is not None:
            hint = str(labels[i] or "").strip()
        raw.append((hint or "part", float(results["scores"][i]), (x0, y0, x1, y1)))
    print(f"[detector] Grounding DINO raw={len(raw)}", flush=True)
    return _finalize(image, raw, max_dets)


def _detect_yolo(image: np.ndarray, instruction: str, max_dets: int) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    global _YOLO, _YOLO_CLASSES
    phrases = phrases_from_instruction(instruction)
    if _YOLO is None:
        print("[detector] load YOLO-World yolov8m-worldv2.pt", flush=True)
        _YOLO = YOLO("yolov8m-worldv2.pt")
    key = tuple(phrases)
    if _YOLO_CLASSES != key:
        _YOLO.set_classes(list(phrases))
        _YOLO_CLASSES = key
    result = _YOLO.predict(
        Image.fromarray(image),
        conf=0.05,
        iou=0.5,
        imgsz=1920,
        verbose=False,
        device=0,
    )[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        print("[detector] YOLO-World: 0 boxes", flush=True)
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confs = boxes.conf.detach().cpu().numpy()
    clss = boxes.cls.detach().cpu().numpy()
    names = result.names or {}
    raw = []
    for i, row in enumerate(xyxy):
        hint = str(names.get(int(clss[i]), "part"))
        raw.append((hint, float(confs[i]), tuple(float(v) for v in row)))
    print(f"[detector] YOLO-World raw={len(raw)}", flush=True)
    return _finalize(image, raw, max_dets, source="yolo")


def _screws_as_dets(blobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dets: list[dict[str, Any]] = []
    for blob in blobs:
        dets.append(
            {
                "box_2d_px": tuple(float(v) for v in blob["box_2d_px"]),
                "grasp_point_px": tuple(float(v) for v in blob["grasp_point_px"]),
                "area": float(blob.get("area") or 0.0),
                "score": 1.0,
                "hint": "screw",
                "source": "dark_cc",
                "long_axis_px": blob.get("long_axis_px"),
                "polygon_px": blob.get("polygon_px"),
            }
        )
    return dets


def _point_in_box(pt: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1


def _merge_large_and_screws(
    large: list[dict[str, Any]],
    screws: list[dict[str, Any]],
    max_dets: int,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for det in large:
        n_inside = sum(
            1 for screw in screws if _point_in_box(screw["grasp_point_px"], det["box_2d_px"])
        )
        if n_inside >= 2:
            print(
                f"[detector] drop {det['hint']}@{det['score']:.2f} "
                f"covers {n_inside} screws",
                flush=True,
            )
            continue
        filtered.append(det)
    merged = list(screws)
    for det in _nms(filtered):
        if any(_iou(det["box_2d_px"], screw["box_2d_px"]) >= 0.5 for screw in merged):
            continue
        merged.append(det)
        if len(merged) >= max(1, int(max_dets)):
            break
    print(
        f"[detector] merged screws={len(screws)} large={len(merged) - len(screws)} "
        f"total={len(merged)}",
        flush=True,
    )
    return merged


def detect_table_objects(
    image: np.ndarray,
    instruction: str = "",
    *,
    max_dets: int = _MAX_DETS,
) -> list[dict[str, Any]]:
    """Large DINO/YOLO boxes + dark-CC screws. ``hint`` is not the final name."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be HxWx3 RGB, got shape={image.shape}")
    t0 = time.monotonic()
    from .segment import find_table_screws

    screws = _screws_as_dets(find_table_screws(image))
    print(
        f"[detector] dark-cc screws={len(screws)} {(time.monotonic() - t0) * 1000.0:.0f}ms",
        flush=True,
    )
    large_budget = max(1, int(max_dets) - len(screws))
    try:
        large = _detect_dino(image, instruction, large_budget)
    except ImportError as exc:
        print(f"[detector] Grounding DINO unavailable ({exc}); fallback YOLO-World", flush=True)
        large = _detect_yolo(image, instruction, large_budget)
    except Exception as exc:  # noqa: BLE001
        print(f"[detector] Grounding DINO failed ({exc}); fallback YOLO-World", flush=True)
        large = _detect_yolo(image, instruction, large_budget)
    return _merge_large_and_screws(large, screws, max_dets)
