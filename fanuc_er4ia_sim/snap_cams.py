"""One still from each real camera with the rollout settings (native resolution, side exposure fixed).

Read-only: cameras only, the robot is not touched.

    uv run python fanuc_er4ia_sim/snap_cams.py                      # -> fanuc_er4ia_sim/data/snap_<time>/
    uv run python fanuc_er4ia_sim/snap_cams.py --side-exposure 80 --out /tmp/snaps
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--side-exposure", type=float, default=65.0)
    parser.add_argument("--cams", nargs="+", default=["table", "side", "wrist"])
    args = parser.parse_args()
    out = args.out or HERE / "data" / f"snap_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    configs = {
        "table": OpenCVCameraConfig(
            index_or_path="/dev/v4l/by-path/platform-NVDA8000:00-usb-0:1.2.4.2:1.4-video-index0",
            width=1920, height=1080, fps=30, fourcc="MJPG"),
        "side": OpenCVCameraConfig(
            index_or_path="/dev/v4l/by-id/usb-SHENZHEN_AONI_ELECTRONIC_CO.__LTD_Full_HD_webcam_20200729001-video-index0",
            width=1920, height=1080, fps=30, fourcc="MJPG", exposure=args.side_exposure),
        "wrist": OpenCVCameraConfig(
            index_or_path="/dev/v4l/by-path/platform-NVDA8000:00-usb-0:1.2.3:1.4-video-index0",
            width=1280, height=720, fps=30, fourcc="MJPG"),
    }
    for name in args.cams:
        cam = OpenCVCamera(configs[name])
        cam.connect()
        time.sleep(1.5)  # let auto exposure / white balance settle
        frame = cam.read()
        cam.disconnect()
        path = out / f"{name}.png"
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"[snap] {name} {frame.shape[1]}x{frame.shape[0]} -> {path}")


if __name__ == "__main__":
    main()
