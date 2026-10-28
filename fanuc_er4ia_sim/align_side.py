"""Live side-camera alignment against the calibrated view, read through the running cell server.

Every 2 s: grab the side frame from robot_server.py (no camera re-open), match it to the reference image the
side calibration was solved on, and print the offset. Turn the camera until dx, dy -> 0 and scale -> 1.

    uv run python fanuc_er4ia_sim/align_side.py --token <run token>
"""
from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path

import cv2
import numpy as np
import requests

HERE = Path(__file__).resolve().parent
W, H = 960, 540  # matching resolution; offsets are reported in native 1920x1080 pixels


def gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://10.20.8.103:9099")
    p.add_argument("--token", required=True)
    p.add_argument("--ref", type=Path, default=HERE / "data" / "snap_20261009_011718" / "side.png")
    p.add_argument("--period", type=float, default=2.0)
    args = p.parse_args()

    ref = gray(cv2.imread(str(args.ref)))
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(ref, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {args.token}"
    print(f"reference {args.ref}  (target: dx 0, dy 0, scale 1.000, rot 0.0; offsets in 1920x1080 px)")
    while True:
        try:
            r = s.get(f"{args.url}/api/v1/observation", params={"cameras": "side", "width": 1920, "height": 1080},
                      timeout=10).json()
            img = cv2.imdecode(np.frombuffer(base64.b64decode(r["images"]["side"]), np.uint8), cv2.IMREAD_COLOR)
            k2, d2 = orb.detectAndCompute(gray(img), None)
            good = [m for m, n in (x for x in bf.knnMatch(d1, d2, k=2) if len(x) == 2) if m.distance < 0.75 * n.distance]
            p1 = np.float32([k1[m.queryIdx].pt for m in good])
            p2 = np.float32([k2[m.trainIdx].pt for m in good])
            A, inl = cv2.estimateAffinePartial2D(p1, p2, ransacReprojThreshold=2.0)
            if A is None or inl.sum() < 30:
                print(f"{time.strftime('%H:%M:%S')}  too few matches ({0 if inl is None else int(inl.sum())}): "
                      "the view is far off or blocked")
            else:
                scale = float(np.hypot(A[0, 0], A[1, 0]))
                rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
                # where the reference image centre lands now, relative to the centre (native px)
                c = A @ np.array([W / 2, H / 2, 1.0])
                dx, dy = (c[0] - W / 2) * 1920 / W, (c[1] - H / 2) * 1080 / H
                hint = []
                if abs(dx) > 15:
                    hint.append("turn LEFT" if dx < 0 else "turn RIGHT")
                if abs(dy) > 15:
                    hint.append("tilt UP" if dy < 0 else "tilt DOWN")  # scene sits high: the camera points too low
                if abs(scale - 1) > 0.01:
                    hint.append("move BACK / zoom out" if scale > 1 else "move CLOSER / zoom in")
                if abs(rot) > 0.3:
                    hint.append(f"roll {'clockwise' if rot > 0 else 'counter-clockwise'}")  # undo the image rotation
                ok = "  ALIGNED" if not hint else ""
                print(f"{time.strftime('%H:%M:%S')}  dx {dx:+5.0f}  dy {dy:+5.0f}  scale {scale:.3f}  rot {rot:+.2f}"
                      f"  inliers {int(inl.sum()):4d}  {', '.join(hint)}{ok}", flush=True)
        except Exception as e:  # keep running while the camera is being handled
            print(f"{time.strftime('%H:%M:%S')}  error: {e}", flush=True)
        time.sleep(args.period)


if __name__ == "__main__":
    main()
