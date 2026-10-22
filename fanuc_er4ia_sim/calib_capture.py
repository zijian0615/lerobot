"""Capture camera calibration samples: the controller pose plus a frame from each real camera.

Read-only on the robot (FRC reads, no FRC_Initialize). Two ways to run it:

    # watch: jog with the pendant; every time the arm stops at a new pose, a sample is taken
    uv run python fanuc_er4ia_sim/calib_capture.py --watch

    # one sample at the current pose
    uv run python fanuc_er4ia_sim/calib_capture.py

Samples go to <out>/samples.json, frames to <out>/<n>_<camera>.png, with the ArUco corners each
camera saw (DICT_4X4_50, --marker-mm edge). Stop watch mode with Ctrl+C.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_j3_convention import ReadOnlyRmi, read_pair  # noqa: E402

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig  # noqa: E402

# Same devices and stream settings as the rollout scripts (data/rollout_*.sh), read through
# OpenCVCamera: opened once, a background thread keeps the latest frame. The Orbbec colour nodes are
# the USB interface 1.4 by-path links (by-id index0 is not stable across reboots).
CAMERAS = {
    "table": OpenCVCameraConfig(
        index_or_path="/dev/v4l/by-path/platform-NVDA8000:00-usb-0:1.1.1:1.4-video-index0",
        width=1920, height=1080, fps=30, fourcc="MJPG"),
    "side": OpenCVCameraConfig(
        index_or_path="/dev/v4l/by-id/usb-SHENZHEN_AONI_ELECTRONIC_CO.__LTD_Full_HD_webcam_20200729001-video-index0",
        width=1920, height=1080, fps=30, fourcc="MJPG", exposure=65),
    "wrist": OpenCVCameraConfig(
        index_or_path="/dev/v4l/by-path/platform-NVDA8000:00-usb-0:1.1.3:1.4-video-index0",
        width=1280, height=720, fps=30, fourcc="MJPG"),
}


def open_cameras(names: list[str]) -> dict[str, OpenCVCamera]:
    cams = {}
    try:
        for name in names:
            cam = OpenCVCamera(CAMERAS[name])
            cam.connect()
            cams[name] = cam
            print(f"[calib] {name} open: {CAMERAS[name].index_or_path}")
    except Exception:
        for cam in cams.values():  # release what did open, so no read thread outlives the process
            cam.disconnect()
        raise
    time.sleep(1.5)  # let auto exposure / white balance settle
    return cams


def grab(cam: OpenCVCamera, timeout_ms: float = 3000) -> np.ndarray | None:
    """A frame taken after this call (BGR), or None if the camera stalls."""
    try:
        cam.async_read(timeout_ms=timeout_ms)  # drop the frame that may predate the stop
        return cv2.cvtColor(cam.async_read(timeout_ms=timeout_ms), cv2.COLOR_RGB2BGR)
    except (TimeoutError, RuntimeError) as exc:
        print(f"[calib]   {exc}")
        return None


def read_pose_on(rmi: ReadOnlyRmi) -> dict:
    first = read_pair(rmi)
    time.sleep(0.3)
    second = read_pair(rmi)
    if first is None or second is None:
        raise RuntimeError("controller rejected the reads")
    drift = max(abs(a - b) for a, b in zip(first["joints_deg"], second["joints_deg"], strict=True))
    if drift > 0.05:
        raise RuntimeError(f"arm is moving ({drift:.2f} deg between reads); let it settle")
    return second


def capture(rmi: ReadOnlyRmi, cams: dict[str, OpenCVCamera], out: Path, label: str, marker_mm: float) -> dict | None:
    """Take one sample at the current pose. Returns it, or None if the arm moved meanwhile."""
    index_path = out / "samples.json"
    samples = json.loads(index_path.read_text()) if index_path.is_file() else []
    n = len(samples)
    try:
        before = read_pose_on(rmi)
    except RuntimeError as exc:
        print(f"[calib]   {exc}; not captured")
        return None
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters()
    )
    images = {}
    for name, cam in cams.items():
        frame = grab(cam)
        if frame is None:
            print(f"[calib]   {name}: no frame from {CAMERAS[name].index_or_path}")
            continue
        path = out / f"{n:02d}_{name}.png"
        cv2.imwrite(str(path), frame)
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        markers = (
            []
            if ids is None
            else [{"id": int(i), "corners": c.reshape(4, 2).tolist()} for i, c in zip(ids.ravel(), corners, strict=True)]
        )
        images[name] = {"file": path.name, "size": [frame.shape[1], frame.shape[0]], "markers": markers}
        print(f"[calib]   {name}: markers {[m['id'] for m in markers] or 'NONE'}")
    try:
        after = read_pose_on(rmi)
    except RuntimeError:
        after = None
    if after is None or max(
        abs(a - b) for a, b in zip(before["joints_deg"], after["joints_deg"], strict=True)
    ) > 0.05:
        print("[calib]   arm moved while capturing; sample dropped")
        for info in images.values():
            (out / info["file"]).unlink(missing_ok=True)
        return None
    sample = {
        "label": label,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "marker_dict": "DICT_4X4_50",
        "marker_mm": marker_mm,
        **after,
        "images": images,
    }
    samples.append(sample)
    index_path.write_text(json.dumps(samples, indent=1))
    print(f"[calib] sample #{n} saved  J={np.round(after['joints_deg'], 1).tolist()}  "
          f"TCP={np.round(after['xyzwpr'][:3], 0).tolist()} mm  ({len(samples)} total)")
    return sample


def pose_change(a: dict, b: dict) -> tuple[float, float]:
    """TCP distance (mm) and largest joint change (deg) between two samples."""
    dist = float(np.linalg.norm(np.subtract(a["xyzwpr"][:3], b["xyzwpr"][:3])))
    turn = float(np.max(np.abs(np.subtract(a["joints_deg"], b["joints_deg"]))))
    return dist, turn


def watch(rmi: ReadOnlyRmi, cams: dict[str, OpenCVCamera], args) -> None:
    index_path = args.out / "samples.json"
    samples = json.loads(index_path.read_text()) if index_path.is_file() else []
    last = samples[-1] if samples else None
    print(f"[calib] watching ({len(samples)} samples so far). Jog to a pose and stop; "
          f"a new sample needs {args.min_move_mm:.0f} mm or {args.min_turn_deg:.0f} deg of change. Ctrl+C ends.")
    still_since = None
    prev = None
    while True:
        cur = read_pair(rmi)
        if cur is None:
            time.sleep(0.5)
            continue
        moving = prev is not None and max(
            abs(a - b) for a, b in zip(cur["joints_deg"], prev["joints_deg"], strict=True)
        ) > 0.02
        prev = cur
        if moving:
            still_since = None
        elif still_since is None:
            still_since = time.time()
        new_pose = last is None or max(
            pose_change(cur, last)[0] / args.min_move_mm, pose_change(cur, last)[1] / args.min_turn_deg
        ) >= 1.0
        if still_since is not None and time.time() - still_since >= args.settle_s and new_pose:
            print("[calib] arm stopped at a new pose, capturing...")
            sample = capture(rmi, cams, args.out, args.label, args.marker_mm)
            if sample is not None:
                last = sample
            still_since = None
        time.sleep(0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("data") / "cam_calib")
    parser.add_argument("--host", default="172.30.109.22")
    parser.add_argument("--port", type=int, default=16001)
    parser.add_argument("--label", default="")
    parser.add_argument("--marker-mm", type=float, default=37.0, help="ArUco black square edge")
    parser.add_argument("--watch", action="store_true", help="capture every time the arm stops at a new pose")
    parser.add_argument("--settle-s", type=float, default=1.5)
    parser.add_argument("--min-move-mm", type=float, default=30.0)
    parser.add_argument("--min-turn-deg", type=float, default=8.0)
    parser.add_argument("--cams", nargs="+", default=list(CAMERAS), choices=list(CAMERAS))
    parser.add_argument("--side-exposure", type=float, default=65.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    CAMERAS["side"].exposure = args.side_exposure
    # Controller first: if it is unreachable, no camera stream has been started yet.
    rmi = ReadOnlyRmi(args.host, args.port, 1)
    cams = {}
    try:
        cams = open_cameras(args.cams)
        if args.watch:
            watch(rmi, cams, args)
        else:
            capture(rmi, cams, args.out, args.label, args.marker_mm)
    except KeyboardInterrupt:
        print("\n[calib] stopped")
    finally:
        rmi.close()
        for cam in cams.values():
            if cam.is_connected:
                cam.disconnect()


if __name__ == "__main__":
    main()
