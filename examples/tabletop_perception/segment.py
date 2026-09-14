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

"""GrabCut segmentation inside each VLM box → mask min-area rectangle."""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from .vlm import ParsedDetection

logger = logging.getLogger(__name__)


def _clip_box(
    box_2d_px: tuple[float, float, float, float],
    width: int,
    height: int,
    pad_px: float = 0.0,
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = (float(v) for v in box_2d_px)
    x0 = int(max(0, math.floor(xmin - pad_px)))
    y0 = int(max(0, math.floor(ymin - pad_px)))
    x1 = int(min(width, math.ceil(xmax + pad_px)))
    y1 = int(min(height, math.ceil(ymax + pad_px)))
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return 0, 0, width, height
    return x0, y0, x1, y1


def _grabcut_mask(
    image_rgb: np.ndarray,
    box_2d_px: tuple[float, float, float, float],
) -> np.ndarray | None:
    h, w = image_rgb.shape[:2]
    x0, y0, x1, y1 = _clip_box(box_2d_px, w, h, pad_px=2)
    bw, bh = x1 - x0, y1 - y0
    if bw < 8 or bh < 8:
        return None

    mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    mask[y0:y1, x0:x1] = cv2.GC_PR_FGD
    # Seed the core of the VLM box as definite foreground.
    mx = max(2, int(0.22 * bw))
    my = max(2, int(0.22 * bh))
    mask[y0 + my : y1 - my, x0 + mx : x1 - mx] = cv2.GC_FGD

    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    try:
        cv2.grabCut(bgr, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # Do not let GrabCut leak far outside the detection box.
    clip = np.zeros_like(fg)
    cx0, cy0, cx1, cy1 = _clip_box(box_2d_px, w, h, pad_px=max(4.0, 0.15 * max(bw, bh)))
    clip[cy0:cy1, cx0:cx1] = 255
    fg = cv2.bitwise_and(fg, clip)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    filled = np.zeros_like(fg)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _fitted_rect(mask: np.ndarray) -> tuple[
    list[tuple[float, float]],
    tuple[tuple[float, float], tuple[float, float]],
    tuple[float, float],
    float,
] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < 25.0:
        return None
    rect = cv2.minAreaRect(contour)
    pts = cv2.boxPoints(rect)
    corners = [(float(p[0]), float(p[1])) for p in pts]
    best_len = -1.0
    axis = (corners[0], corners[1])
    for i in range(4):
        p0, p1 = corners[i], corners[(i + 1) % 4]
        length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        if length > best_len:
            best_len = length
            axis = (p0, p1)
    m = cv2.moments(contour)
    if m["m00"] > 1e-6:
        center = (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"]))
    else:
        center = (float(rect[0][0]), float(rect[0][1]))
    return corners, axis, center, area


def refine_parsed_detections(
    image_rgb: np.ndarray,
    detections: list[ParsedDetection],
) -> list[ParsedDetection]:
    """
    Segment each VLM box with GrabCut and replace AABB yaw with the mask
    minimum-area rectangle. If segmentation fails, keep the VLM box.
    """
    if image_rgb.ndim != 3:
        return detections

    for det in detections:
        box = det["box_2d_px"]
        box_area = max(float(box[2] - box[0]), 1.0) * max(float(box[3] - box[1]), 1.0)
        mask = _grabcut_mask(image_rgb, box)
        if mask is None:
            logger.info("Segment skip %s (GrabCut empty)", det["name"])
            continue
        fitted = _fitted_rect(mask)
        if fitted is None:
            logger.info("Segment skip %s (no contour)", det["name"])
            continue
        corners, axis, center, area = fitted
        if area < 0.2 * box_area or area > 1.6 * box_area:
            logger.info(
                "Segment skip %s (area %.0f vs box %.0f)", det["name"], area, box_area
            )
            continue
        if not (box[0] - 6 <= center[0] <= box[2] + 6 and box[1] - 6 <= center[1] <= box[3] + 6):
            logger.info("Segment skip %s (centroid outside box)", det["name"])
            continue
        det["polygon_px"] = corners
        det["long_axis_px"] = axis
        det["grasp_point_px"] = center
        logger.info(
            "Segmented %s area=%.0f yaw_img=%.1fdeg",
            det["name"],
            area,
            math.degrees(math.atan2(axis[1][1] - axis[0][1], axis[1][0] - axis[0][0])),
        )
    return detections
