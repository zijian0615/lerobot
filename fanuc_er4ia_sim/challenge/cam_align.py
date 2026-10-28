"""Is a fixed camera still where it was calibrated? Match its live frame to the reference image of its calibration.

refs/<camera>.png is the frame the camera's calibration (client/calibration/cameras.json, the twin's camera) was
solved on: side.png is fanuc_er4ia_sim/data/snap_20261009_011718/side.png. ORB features of both, matched and fitted
with a similarity (shift, scale, roll); the result says how far the reference has moved in the live image, in native
1920 x 1080 pixels, and which way to turn the camera to bring it back (as fanuc_er4ia_sim/align_side.py does).
"""
from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
W, H = 960, 540  # matching resolution
NATIVE = (1920, 1080)
TOL = {"px": 15.0, "scale": 0.01, "rot": 0.3}  # aligned within these
_cache: dict[str, tuple] = {}
_lock = threading.Lock()


def ref_path(camera: str) -> Path:
    return HERE / "refs" / f"{camera}.png"


def _gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def _ref(camera: str):
    with _lock:
        if camera not in _cache:
            p = ref_path(camera)
            if not p.is_file():
                raise ValueError(f"no reference image for the {camera} camera")
            orb = cv2.ORB_create(4000)
            k, d = orb.detectAndCompute(_gray(cv2.imread(str(p))), None)
            _cache[camera] = (orb, k, d)
        return _cache[camera]


def measure(camera: str, rgb: np.ndarray) -> dict:
    """The live frame (RGB) against the reference: dx, dy (px), scale, rot (deg), inliers, hints, aligned."""
    orb, k1, d1 = _ref(camera)
    k2, d2 = orb.detectAndCompute(_gray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), None)
    if d2 is None:
        return {"ok": False, "reason": "no features in the live image (dark or blocked?)"}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m for m, n in (x for x in pairs if len(x) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return {"ok": False, "reason": f"only {len(good)} matches: the view is far off or blocked"}
    A, inl = cv2.estimateAffinePartial2D(np.float32([k1[m.queryIdx].pt for m in good]),
                                         np.float32([k2[m.trainIdx].pt for m in good]), ransacReprojThreshold=2.0)
    n = 0 if inl is None else int(inl.sum())
    if A is None or n < 30:
        return {"ok": False, "reason": f"only {n} consistent matches: the view is far off or blocked"}
    scale = float(np.hypot(A[0, 0], A[1, 0]))
    rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    c = A @ np.array([W / 2, H / 2, 1.0])  # where the reference's centre is now
    dx, dy = (c[0] - W / 2) * NATIVE[0] / W, (c[1] - H / 2) * NATIVE[1] / H
    hints = []
    if abs(dx) > TOL["px"]:
        hints.append("turn left" if dx < 0 else "turn right")
    if abs(dy) > TOL["px"]:
        hints.append("tilt up" if dy < 0 else "tilt down")
    if abs(scale - 1) > TOL["scale"]:
        hints.append("move back" if scale > 1 else "move closer")
    if abs(rot) > TOL["rot"]:
        hints.append(f"roll {'clockwise' if rot > 0 else 'counter-clockwise'}")
    return {"ok": True, "dx": round(dx), "dy": round(dy), "scale": round(scale, 3), "rot": round(rot, 2),
            "inliers": n, "hints": hints, "aligned": not hints}
