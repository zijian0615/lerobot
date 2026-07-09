"""Green-cube detection on a light table using classic OpenCV.

The cube is a saturated green on a bright wooden desk, so a single HSV hue+sat+val
band segments it reliably without any learning (and naturally ignores the dark
arm and red/blue tape). We take the largest valid green blob and return its pixel
centroid.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import GreenCubeParams


@dataclass
class Detection:
    """Result of a single cube detection."""

    u: float  # pixel x of centroid
    v: float  # pixel y of centroid
    area: float  # blob area in pixels
    bbox: tuple[int, int, int, int]  # x, y, w, h
    angle: float  # orientation of the min-area rect (deg), useful for grasp yaw
    mask: np.ndarray  # binary mask (uint8, 0/255)


def _apply_roi(mask: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    if roi is None:
        return mask
    x, y, w, h = roi
    gated = np.zeros_like(mask)
    gated[y : y + h, x : x + w] = mask[y : y + h, x : x + w]
    return gated


def detect_cube(image_rgb: np.ndarray, params: GreenCubeParams) -> Detection | None:
    """Detect the largest green blob in an RGB image.

    Args:
        image_rgb: HxWx3 uint8 image in **RGB** order (as returned by the
            lerobot OpenCV camera by default).
        params: thresholds and filtering parameters.

    Returns:
        A ``Detection`` for the largest valid green blob, or ``None`` if nothing
        qualifies.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {image_rgb.shape}")

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    # Green = hue band with enough saturation and value.
    lower = np.array([params.h_min, params.s_min, params.v_min], dtype=np.uint8)
    upper = np.array([params.h_max, params.s_max, params.v_max], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    mask = _apply_roi(mask, params.roi)

    if params.morph_kernel > 0:
        k = np.ones((params.morph_kernel, params.morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: Detection | None = None
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < params.min_area_px or area > params.max_area_px:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        u = m["m10"] / m["m00"]
        v = m["m01"] / m["m00"]
        bbox = cv2.boundingRect(cnt)
        rect = cv2.minAreaRect(cnt)
        angle = float(rect[2])
        if best is None or area > best.area:
            best = Detection(u=u, v=v, area=area, bbox=bbox, angle=angle, mask=mask)

    return best


def draw_detection(image_rgb: np.ndarray, det: Detection | None) -> np.ndarray:
    """Return a copy of the image with the detection overlaid (for debugging)."""
    vis = image_rgb.copy()
    if det is None:
        cv2.putText(vis, "no cube", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        return vis
    x, y, w, h = det.bbox
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(vis, (int(det.u), int(det.v)), 5, (255, 0, 0), -1)
    cv2.putText(
        vis,
        f"({det.u:.0f},{det.v:.0f}) A={det.area:.0f}",
        (x, max(0, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )
    return vis


def _cli() -> None:
    """Quick offline test: ``python -m raprober.perception <image_path>``."""
    import argparse

    from .config import DEFAULT_CONFIG

    parser = argparse.ArgumentParser(description="Test green-cube detection on an image file.")
    parser.add_argument("image", help="path to an RGB/BGR image file")
    parser.add_argument("--out", default=None, help="path to save the annotated image")
    parser.add_argument("--h-min", type=int, default=DEFAULT_CONFIG.cube.h_min)
    parser.add_argument("--h-max", type=int, default=DEFAULT_CONFIG.cube.h_max)
    parser.add_argument("--s-min", type=int, default=DEFAULT_CONFIG.cube.s_min)
    parser.add_argument("--v-min", type=int, default=DEFAULT_CONFIG.cube.v_min)
    args = parser.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    params = DEFAULT_CONFIG.cube
    params.h_min, params.h_max, params.s_min, params.v_min = args.h_min, args.h_max, args.s_min, args.v_min
    det = detect_cube(rgb, params)
    if det is None:
        print("No cube detected.")
    else:
        print(f"Cube at pixel (u={det.u:.1f}, v={det.v:.1f}), area={det.area:.0f}, angle={det.angle:.1f}")

    out = args.out or (args.image.rsplit(".", 1)[0] + "_det.png")
    vis_bgr = cv2.cvtColor(draw_detection(rgb, det), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out, vis_bgr)
    print(f"Annotated image saved to {out}")


if __name__ == "__main__":
    _cli()
