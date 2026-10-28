"""Open, read and close the wrist camera N times; report frames per cycle. Tests the Gemini 305 re-open failure.

    uv run python fanuc_er4ia_sim/wrist_reopen_test.py --path /dev/v4l/by-path/...:1.4-video-index0 --cycles 3

Warning: if a re-open delivers no frame, closing that stream has hung the xHCI controller before (reboot).
"""
from __future__ import annotations

import argparse
import time

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True)
    p.add_argument("--mode", default="848x480@15")
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--seconds", type=float, default=5.0)
    args = p.parse_args()
    wh, fps = args.mode.split("@")
    w, h = map(int, wh.split("x"))
    for k in range(1, args.cycles + 1):
        cam = OpenCVCamera(OpenCVCameraConfig(index_or_path=args.path, width=w, height=h, fps=int(fps), fourcc="MJPG"))
        t0 = time.time()
        try:
            cam.connect()
        except Exception as e:
            print(f"cycle {k}: OPEN FAILED after {time.time() - t0:.1f}s: {e}", flush=True)
            return
        n, t1 = 0, time.time()
        while time.time() - t1 < args.seconds:
            cam.async_read(timeout_ms=1000)
            n += 1
        cam.disconnect()
        print(f"cycle {k}: opened in {t1 - t0:.1f}s, {n} frames in {args.seconds:.0f}s ({n / args.seconds:.1f} fps), closed",
              flush=True)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
