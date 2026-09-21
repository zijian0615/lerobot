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
import re

import cv2
import numpy as np

from .vlm import ParsedDetection, uniquify_detection_names

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


def _yellow_pixels(image_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    return ((h >= 15) & (h <= 42) & (s > 60) & (v > 70)).astype(np.uint8) * 255


def _arm_yellow_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Keep the Fanuc arm. Drop the printed yellow diamond on the table."""
    yellow = _yellow_pixels(image_rgb)
    closed = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    height, width = yellow.shape
    min_area = 0.02 * float(height * width)
    arm = np.zeros_like(yellow)
    for i in range(1, n_labels):
        if float(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            arm[labels == i] = 255
    return cv2.bitwise_and(arm, yellow)


def _print_yellow_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Table graphics only. Do not dilate into neighbouring screws."""
    yellow = _yellow_pixels(image_rgb)
    arm = _arm_yellow_mask(image_rgb)
    printed = cv2.bitwise_and(yellow, cv2.bitwise_not(arm))
    # k=2: 1080p used to scale to 4 and swallow screws sitting on the diamond.
    return cv2.dilate(printed, np.ones((2, 2), np.uint8))


def _blue_cable_mask(image_rgb: np.ndarray, dilate_px: int = 11) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    blue = ((h >= 90) & (h <= 135) & (s > 40) & (v > 40)).astype(np.uint8) * 255
    if dilate_px <= 1:
        return blue
    return cv2.dilate(blue, np.ones((dilate_px, dilate_px), np.uint8))


def _robot_ignore_mask(
    image_rgb: np.ndarray,
    *,
    yellow_dilate: int = 9,
    dark_dilate: int = 6,
) -> np.ndarray:
    """Yellow arm + nearby dark flange + blue cable. Not the printed diamond."""
    arm = _arm_yellow_mask(image_rgb)
    blue = _blue_cable_mask(image_rgb, dilate_px=1)
    robot = cv2.dilate(arm, np.ones((yellow_dilate, yellow_dilate), np.uint8))
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    dark = (gray < 140).astype(np.uint8) * 255
    robot = cv2.bitwise_or(
        robot, cv2.bitwise_and(dark, cv2.dilate(arm, np.ones((dark_dilate, dark_dilate), np.uint8)))
    )
    s = max(1, int(round(image_rgb.shape[0] / 480.0)))
    robot = cv2.bitwise_or(robot, cv2.dilate(blue, np.ones((11 * s, 11 * s), np.uint8)))
    robot = cv2.bitwise_or(
        robot, cv2.bitwise_and(dark, cv2.dilate(blue, np.ones((15 * s, 15 * s), np.uint8)))
    )
    return robot


def _strong_robot_mask(image_rgb: np.ndarray) -> np.ndarray:
    arm = _arm_yellow_mask(image_rgb)
    s = max(1, int(round(image_rgb.shape[0] / 480.0)))
    blue = _blue_cable_mask(image_rgb, dilate_px=max(7, 15 * s))
    return cv2.bitwise_or(cv2.dilate(arm, np.ones((22 * s, 22 * s), np.uint8)), blue)


def _blob_looks_like_cable(blob: dict) -> bool:
    """Gripper loom: saturated dark rubber, not a metal screw."""
    if float(blob.get("cable_frac") or 0.0) > 0.20:
        return True
    _h, s, v = (float(x) for x in blob["mean_hsv"])
    # Warm table light makes screws look red/saturated. A cable is longer.
    if s >= 110.0 and v <= 80.0 and _blob_long_side_px(blob) > 50.0:
        return True
    return False


def blob_is_robot_clutter(blob: dict, image_h: float | None = None) -> bool:
    if _blob_looks_like_cable(blob):
        return True
    if float(blob.get("robot_frac") or 0.0) > 0.30:
        return True
    if float(blob.get("cable_frac") or 0.0) > 0.20:
        return True
    fill = float(blob.get("fill_ratio") or 0.0)
    hh, ss, v = (float(x) for x in blob["mean_hsv"])
    long_side = _blob_long_side_px(blob)
    cable_dist = blob.get("cable_dist")
    on_yellow = 12.0 <= hh <= 40.0 and ss > 80.0
    # Screws on the yellow print sit next to the loom; gripper steel does not look yellow.
    # 20 px was too close: flange / cable nubs were exempted and became screws.
    if (
        fill >= 0.40
        and v < 110.0
        and long_side <= 80.0
        and (on_yellow or (cable_dist is not None and float(cable_dist) >= 55.0))
    ):
        return False
    if cable_dist is None:
        return False
    lim = 14.0 * (float(image_h) / 480.0 if image_h and image_h > 1.0 else 1.0)
    lim = min(lim, 24.0)
    return float(cable_dist) < lim


def _nms_blobs(blobs: list[dict], min_dist: float = 18.0) -> list[dict]:
    ordered = sorted(blobs, key=lambda b: -float(b.get("area") or 0.0))
    kept: list[dict] = []
    for blob in ordered:
        u, v = blob["grasp_point_px"]
        if any(math.hypot(u - o["grasp_point_px"][0], v - o["grasp_point_px"][1]) < min_dist for o in kept):
            continue
        kept.append(blob)
    return kept


def _collect_table_blobs(
    image_rgb: np.ndarray,
    *,
    yellow_dilate: int,
    dark_dilate: int,
) -> list[dict]:
    if image_rgb.ndim != 3:
        return []
    height, width = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    bg = cv2.GaussianBlur(gray, (61, 61), 0)
    diff = np.clip(bg.astype(np.int16) - gray.astype(np.int16), 0, 255).astype(np.uint8)
    robot = _robot_ignore_mask(
        image_rgb, yellow_dilate=yellow_dilate, dark_dilate=dark_dilate
    )
    strong_robot = _strong_robot_mask(image_rgb)
    print_yellow = _print_yellow_mask(image_rgb)
    scale_h = height / 480.0
    s = max(1, int(round(scale_h)))
    cable = _blue_cable_mask(image_rgb, dilate_px=max(7, 11 * s))
    blue_raw = _blue_cable_mask(image_rgb, dilate_px=1)
    if int(blue_raw.sum()) == 0:
        cable_dist_map = np.full((height, width), 1e6, dtype=np.float32)
    else:
        cable_dist_map = cv2.distanceTransform(cv2.bitwise_not(blue_raw), cv2.DIST_L2, 5)
    fg = (diff > 18).astype(np.uint8) * 255
    fg = cv2.bitwise_and(fg, cv2.bitwise_not(robot))
    # Printed box lines also have diff>18 but stay bright; AND with dark
    # so screws on the outline do not merge into one long contour.
    fg = cv2.bitwise_and(fg, ((gray < 155).astype(np.uint8) * 255))
    # Keep dark screws that sit on the printed yellow diamond.
    on_print = (print_yellow > 0) & (robot == 0) & (gray < 130) & (diff > 28)
    fg = cv2.bitwise_or(
        cv2.bitwise_and(fg, cv2.bitwise_not(print_yellow)),
        (on_print.astype(np.uint8) * 255),
    )
    fg[: max(1, int(0.28 * height)), :] = 0
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: list[dict] = []
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    scale_h = height / 480.0
    scale_px = (height * width) / (640.0 * 480.0)
    min_area = 36.0 * scale_h
    max_area = 2800.0 * scale_px
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        # Drop table-edge print (CAUTION tape, corner L/I marks).
        edge = max(8, int(round(0.045 * width)))
        if x <= edge or y <= edge or x + bw >= width - edge or y + bh >= height - edge:
            continue
        if y < 0.30 * height:
            continue
        if y < 0.48 * height and area < 100.0:
            continue
        rect = cv2.minAreaRect(contour)
        rw, rh = float(rect[1][0]), float(rect[1][1])
        aspect = max(rw, rh) / max(min(rw, rh), 1e-3)
        if aspect >= 4.8:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        mean_g = float(gray[mask > 0].mean())
        mean_d = float(diff[mask > 0].mean())
        if mean_d < 35.0 or mean_g > 170.0:
            continue
        fitted = _fitted_rect(mask)
        if fitted is None:
            continue
        corners, axis, center, _area = fitted
        x0, y0, x1, y1 = int(x), int(y), int(x + bw), int(y + bh)
        roi_robot = strong_robot[y0:y1, x0:x1]
        robot_frac = float(roi_robot.mean()) / 255.0 if roi_robot.size else 0.0
        roi_cable = cable[y0:y1, x0:x1]
        cable_frac = float(roi_cable.mean()) / 255.0 if roi_cable.size else 0.0
        roi_cd = cable_dist_map[y0:y1, x0:x1]
        cable_dist = float(roi_cd.min()) if roi_cd.size else 1e6
        if robot_frac > 0.40 or cable_frac > 0.20:
            continue
        if area > 600.0 * scale_px and robot_frac > 0.22:
            continue
        roi_gray = gray[y0:y1, x0:x1]
        dark = roi_gray < 160
        fill_ratio = float(dark.mean()) if dark.size else 0.0
        if int(dark.sum()) >= 8:
            mean_hsv = hsv[y0:y1, x0:x1][dark].mean(axis=0)
        else:
            mean_hsv = hsv[mask > 0].mean(axis=0)
        blob = {
            "box_2d_px": (float(x), float(y), float(x + bw), float(y + bh)),
            "grasp_point_px": center,
            "long_axis_px": axis,
            "polygon_px": corners,
            "mean_hsv": mean_hsv,
            "area": area,
            "fill_ratio": fill_ratio,
            "aspect": float(aspect),
            "robot_frac": float(robot_frac),
            "cable_frac": float(cable_frac),
            "cable_dist": cable_dist,
        }
        if _blob_looks_like_print(blob) or blob_is_robot_clutter(blob, image_h=height):
            continue
        blobs.append(blob)
    return blobs


def find_table_object_blobs(image_rgb: np.ndarray) -> list[dict]:
    """Dark compact blobs on a white table, excluding the yellow robot.

    Weak robot mask keeps pen/screw next to the gripper. Strong mask keeps
    the black frame from merging into the flange. NMS unions both.
    """
    height, width = image_rgb.shape[:2]
    s = max(1, int(round(height / 480.0)))
    near = _collect_table_blobs(image_rgb, yellow_dilate=9 * s, dark_dilate=6 * s)
    far = _collect_table_blobs(image_rgb, yellow_dilate=28 * s, dark_dilate=16 * s)
    dist = 18.0 * math.sqrt((height * width) / (640.0 * 480.0))
    return _nms_blobs(near + far, min_dist=dist)


def find_table_screws(image_rgb: np.ndarray) -> list[dict]:
    """Tiny dark screws only. DINO/YOLO keep cups, frames, and cans."""
    hw = image_rgb.shape[:2]
    screws: list[dict] = []
    for blob in find_table_object_blobs(image_rgb):
        if not _blob_looks_like_table_screw(blob, image_hw=hw):
            continue
        item = dict(blob)
        item["hint"] = "screw"
        item["score"] = 1.0
        item["source"] = "dark_cc"
        screws.append(item)
    return screws


def _blob_looks_like_print(blob: dict) -> bool:
    """Drop table graphics (yellow target, hollow printed grid)."""
    fill = float(blob.get("fill_ratio") or 0.0)
    aspect = float(blob.get("aspect") or 1.0)
    h, s, v = (float(x) for x in blob["mean_hsv"])
    # Printed yellow is bright. Screws on yellow pick up a yellow hue but stay dark.
    if s > 50 and 18 <= h <= 40 and v > 120:
        return True
    if fill < 0.35 and aspect < 1.8:
        return True
    return False


def _blob_long_side_px(blob: dict) -> float:
    x0, y0, x1, y1 = blob["box_2d_px"]
    return float(max(x1 - x0, y1 - y0))


def _blob_on_image_border(blob: dict, image_hw: tuple[int, int] | None) -> bool:
    if image_hw is None or len(image_hw) < 2:
        return False
    height, width = float(image_hw[0]), float(image_hw[1])
    edge_x = max(8.0, 0.045 * width)
    edge_y = max(8.0, 0.045 * height)
    x0, y0, x1, y1 = blob["box_2d_px"]
    return x0 <= edge_x or y0 <= edge_y or x1 >= width - edge_x or y1 >= height - edge_y


def _blob_looks_like_table_screw(
    blob: dict, image_hw: tuple[int, int] | None = None
) -> bool:
    """Small dark table screws / the large bolt. Geometry, not the VLM."""
    image_h = float(image_hw[0]) if image_hw is not None else 480.0
    s = image_h / 480.0
    if _blob_looks_like_print(blob) or blob_is_robot_clutter(blob, image_h=image_h):
        return False
    if _blob_on_image_border(blob, image_hw):
        return False
    area = float(blob.get("area") or 0.0)
    fill = float(blob.get("fill_ratio") or 0.0)
    aspect = float(blob.get("aspect") or 1.0)
    _h, _s, v = (float(x) for x in blob["mean_hsv"])
    long_side = _blob_long_side_px(blob)
    gy = float(blob["grasp_point_px"][1])
    cable_dist = float(blob.get("cable_dist") or 1e6)
    # Overhead table fills most of the frame. Top 28% is already wiped;
    # 0.50 rejected real screws sitting on the upper printed sheet.
    on_table = gy > 0.36 * image_h
    if v > 110.0 or aspect >= 4.5:
        return False
    if cable_dist < 55.0:
        return False
    if (220.0 * s * s) <= area <= (550.0 * s * s) and fill >= 0.42 and v < 90.0:
        return True
    if not on_table:
        return False
    # Print-sheet ticks are ~15–18 px. Real screws are longer and darker.
    # Thin screws leave a lot of white table in the bbox, so fill can be low.
    return (
        (36.0 * s) <= area <= (80.0 * s * s)
        and (11.0 * s) <= long_side <= (22.0 * s)
        and v < 110.0
    )


def recover_skip_name(blob: dict, image_hw: tuple[int, int] | None = None) -> str | None:
    """Name a blob Cosmos marked skip, if it is still a real table object."""
    image_h = float(image_hw[0]) if image_hw is not None else None
    if _blob_looks_like_print(blob) or blob_is_robot_clutter(blob, image_h=image_h):
        return None
    if _blob_looks_like_table_screw(blob, image_hw=image_hw):
        return "screw"
    appear = _appearance_base_name(blob)
    area = float(blob.get("area") or 0.0)
    fill = float(blob.get("fill_ratio") or 0.0)
    aspect = float(blob.get("aspect") or 1.0)
    h, s, v = (float(x) for x in blob["mean_hsv"])
    gy = float(blob["grasp_point_px"][1])
    near = True
    if image_hw is not None:
        near = gy > 0.55 * float(image_hw[0])
    if appear == "screw":
        if image_hw is not None and gy < 0.36 * float(image_hw[0]):
            return None
        return "screw"
    if appear == "red_pen":
        # Warm light makes black screws look red; a pen is much longer.
        if _blob_long_side_px(blob) <= 70.0:
            return "screw"
        return "red_pen"
    if _appearance_shape(blob) == "frame" and area >= 200.0:
        return "black_frame"
    if area >= 500.0 and fill >= 0.42:
        return "black_block"
    if 80.0 <= area <= 280.0 and v > 100.0 and aspect < 1.7 and float(blob["grasp_point_px"][0]) < 120.0:
        return "vial"
    if 85.0 <= h <= 130.0 and s > 40.0:
        return None
    # Big bolt beside the frame, or small screws in the lower half of the frame.
    if 40.0 <= area <= 550.0 and v < 120.0:
        fill_ok = fill >= 0.50 or (near and fill >= 0.38 and area <= 120.0)
        if fill_ok and near:
            return "screw"
    return None


def _name_color_hint(name: str) -> str | None:
    n = name.lower()
    if any(k in n for k in ("purple", "violet", "magenta")):
        return "purple"
    if "red" in n:
        return "red"
    if "black" in n:
        return "black"
    if "yellow" in n:
        return "yellow"
    return None


def _blob_color_hint(blob: dict) -> str:
    return _appearance_color(blob)


def _vlm_name_is_generic(name: str) -> bool:
    slug = re.sub(r"[^a-z0-9]+", "_", str(name or "").strip().lower()).strip("_")
    tokens = [t for t in slug.split("_") if t]
    if not tokens:
        return True
    specific = {
        "block",
        "plate",
        "screwdriver",
        "screw",
        "triangle",
        "cylinder",
        "frame",
        "rod",
        "bar",
        "peg",
        "cup",
        "wrench",
        "nut",
        "bolt",
    }
    if any(t in specific for t in tokens):
        return False
    generic = {"object", "piece", "item", "thing", "plastic"}
    return any(t in generic for t in tokens) or slug in {"black", "white", "grey", "gray"}


def _appearance_color(blob: dict) -> str:
    h, s, v = (float(x) for x in blob["mean_hsv"])
    if s > 45 and (h <= 12 or h >= 170):
        return "red"
    if s > 25 and 125 <= h <= 175:
        return "purple"
    if s > 40 and 85 <= h <= 130:
        return "blue"
    if s > 50 and 18 <= h <= 40 and v > 90:
        return "yellow"
    if v < 95 or (v < 145 and s < 90):
        return "black"
    if s < 35:
        return "grey"
    return "colored"


def _appearance_shape(blob: dict) -> str:
    aspect = float(blob.get("aspect") or 1.0)
    fill = float(blob.get("fill_ratio") or 1.0)
    if aspect >= 2.55:
        return "rod"
    if fill <= 0.50 and aspect < 1.8:
        return "frame"
    if aspect < 1.5 and fill >= 0.58:
        return "block"
    return "piece"


def _appearance_base_name(blob: dict) -> str:
    color = _appearance_color(blob)
    shape = _appearance_shape(blob)
    aspect = float(blob.get("aspect") or 1.0)
    if color == "purple" and aspect >= 1.55:
        return "purple_screwdriver"
    if color == "red" and aspect >= 1.8:
        # Warm-lit table screws read as red. A pen is longer than ~7 cm in-frame.
        if _blob_long_side_px(blob) <= 70.0 or float(blob.get("area") or 0.0) <= 450.0:
            return "screw"
        return "red_pen"
    if color == "black" and shape == "rod":
        x0, y0, x1, y1 = blob["box_2d_px"]
        long_side = max(x1 - x0, y1 - y0)
        if float(blob.get("area") or 0.0) <= 180.0 or long_side <= 28.0:
            return "screw"
        return "black_bar"
    if color == "black" and shape == "frame":
        return "black_frame"
    if color == "black" and shape == "block":
        return "black_block"
    return f"{color}_{shape}"


def _point_on_blob(u: float, v: float, blobs: list[dict], pad: float = 6.0) -> int | None:
    for i, blob in enumerate(blobs):
        x0, y0, x1, y1 = blob["box_2d_px"]
        if x0 - pad <= u <= x1 + pad and y0 - pad <= v <= y1 + pad:
            return i
    return None


def merge_vlm_with_table_blobs(
    image_rgb: np.ndarray,
    detections: list[ParsedDetection],
) -> list[ParsedDetection]:
    """Snap VLM boxes onto real dark objects. Unused blobs are not added."""
    blobs = find_table_object_blobs(image_rgb)
    if not blobs:
        return detections
    used: set[int] = set()
    out: list[ParsedDetection] = []
    for det in detections:
        u, v = det["grasp_point_px"]
        generic = _vlm_name_is_generic(str(det.get("name") or ""))
        hit = _point_on_blob(u, v, blobs)
        if hit is not None and hit in used:
            hit = None
        if hit is None:
            want = None if generic else _name_color_hint(str(det.get("name") or ""))
            best_i, best_d, best_key = None, 1e9, (1, 1e9)
            for i, blob in enumerate(blobs):
                if i in used:
                    continue
                cu, cv = blob["grasp_point_px"]
                dist = math.hypot(cu - u, cv - v)
                color_ok = want is None or _blob_color_hint(blob) == want
                key = (0 if color_ok else 1, dist)
                if key < best_key:
                    best_i, best_d, best_key = i, dist, key
            max_d = 220.0 if want and best_key[0] == 0 else 90.0
            if best_i is not None and best_d <= max_d:
                hit = best_i
                print(
                    f"[blob] snap {det['name']} ({u:.0f},{v:.0f}) → "
                    f"({blobs[best_i]['grasp_point_px'][0]:.0f},"
                    f"{blobs[best_i]['grasp_point_px'][1]:.0f}) d={best_d:.0f}px",
                    flush=True,
                )
            else:
                print(f"[blob] drop {det['name']} (empty table, no nearby object)", flush=True)
                continue
        used.add(hit)
        blob = blobs[hit]
        det = dict(det)
        det["box_2d_px"] = blob["box_2d_px"]
        det["grasp_point_px"] = blob["grasp_point_px"]
        det["long_axis_px"] = blob["long_axis_px"]
        det["polygon_px"] = blob["polygon_px"]
        vlm_name = str(det.get("name") or "")
        if _vlm_name_is_generic(vlm_name):
            appear = _appearance_base_name(blob)
            print(f"[name] appearance {vlm_name!r} → {appear}", flush=True)
            det["name"] = appear
        out.append(det)
    return uniquify_detection_names(out)


def refine_parsed_detections(
    image_rgb: np.ndarray,
    detections: list[ParsedDetection],
    *,
    snap_blobs: bool = False,
) -> list[ParsedDetection]:
    """
    Segment each VLM box with GrabCut and replace AABB yaw with the mask
    minimum-area rectangle. If segmentation fails, keep the VLM box.

    ``snap_blobs`` is for weak local VLMs (Cosmos). Do not enable for Gemini ER.
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
    if snap_blobs:
        return merge_vlm_with_table_blobs(image_rgb, detections)
    return detections
