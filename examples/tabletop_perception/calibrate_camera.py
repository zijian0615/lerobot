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

"""
Checkerboard calibration of the overhead camera (intrinsics + table extrinsics).

Board: 10x7 squares, 25 mm  ->  9x6 inner corners (``--inner 9 6 --square-mm 25``).

Workflow (headless, no GUI needed)::

    cd examples
    # 1. Board taped flat on the table, camera untouched: grab the extrinsics frame.
    python -m tabletop_perception.calibrate_camera snap --out calib/cam_YYYYMMDD/table_pose.png
    # 2. Hold/tilt the board by hand in many poses; frames are auto-saved.
    python -m tabletop_perception.calibrate_camera capture --out-dir calib/cam_YYYYMMDD/views
    # 3. Solve; add --write to update calib/xarm_overhead.json.
    python -m tabletop_perception.calibrate_camera solve --dir calib/cam_YYYYMMDD [--write]

The table frame is the board frame: origin at the board centre, +x image-right, +y image-up,
+z up out of the table (same axes as the placeholder ``T_cam_table`` it replaces).
``T_cam_table`` is the camera pose in the table frame (``p_table = R @ p_cam + t``), as
``tabletop_perception.geometry.to_table`` expects.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DEFAULT_CALIB = Path(__file__).resolve().parent / "calib" / "xarm_overhead.json"
# Stable path of the colour node (the by-id symlink of this Gemini only exposes the IR node).
DEFAULT_CAMERA = "/dev/v4l/by-path/platform-NVDA8000:01-usb-0:1.2:1.4-video-index0"
FLAGS_LIVE = cv2.CALIB_CB_ACCURACY
FLAGS_SOLVE = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY


def object_points(inner: tuple[int, int], square_m: float) -> np.ndarray:
    """Inner corners in the board frame (row-major as OpenCV returns them), z=0."""
    cols, rows = inner
    pts = np.zeros((rows * cols, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return pts


def detect(img: np.ndarray, inner: tuple[int, int], flags: int) -> np.ndarray | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ok, corners = cv2.findChessboardCornersSB(gray, inner, flags)
    return corners.reshape(-1, 2).astype(np.float32) if ok else None


def open_camera(path: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {path}")
    return cap


def read_frame(cap: cv2.VideoCapture, size: tuple[int, int]) -> np.ndarray | None:
    ok, frame = cap.read()
    if not ok or frame is None or frame.shape[1] != size[0] or frame.shape[0] != size[1]:
        return None
    if float(frame.mean()) < 30.0:
        return None  # dark warm-up frame
    if frame.ndim == 3 and frame.shape[2] == 3 and float(frame.std(axis=2).mean()) < 1.0:
        # Grey frame: this is an IR node, not the colour stream.
        raise RuntimeError("Camera returns a grey stream (IR node?). Use the colour node, e.g. /dev/video4.")
    return frame


def cmd_snap(args: argparse.Namespace) -> int:
    size = (args.width, args.height)
    cap = open_camera(args.camera, *size, args.fps)
    try:
        frame = None
        for _ in range(30):
            f = read_frame(cap, size)
            if f is not None:
                frame = f
        if frame is None:
            raise RuntimeError("No valid frame received")
    finally:
        cap.release()
    corners = detect(frame, tuple(args.inner), FLAGS_SOLVE)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), frame)
    print(f"saved {args.out} {frame.shape[1]}x{frame.shape[0]}  board_found={corners is not None}")
    return 0 if corners is not None else 1


GRID_COLS, GRID_ROWS = 4, 3


def _cell_name(i: int, j: int) -> str:
    return f"{('top', 'middle', 'bottom')[j]}-{('far-left', 'left', 'right', 'far-right')[i]}"


def _cells_hit(c: np.ndarray, size: tuple[int, int]) -> set[tuple[int, int]]:
    """Image cells containing any detected corner of the view."""
    i = np.clip((c[:, 0] / size[0] * GRID_COLS).astype(int), 0, GRID_COLS - 1)
    j = np.clip((c[:, 1] / size[1] * GRID_ROWS).astype(int), 0, GRID_ROWS - 1)
    return set(zip(i.tolist(), j.tolist(), strict=True))


def _board_tilt_deg(c: np.ndarray, inner: tuple[int, int], sq: float, size: tuple[int, int]) -> float:
    """Board normal vs optical axis, using a rough focal guess (only used for feedback)."""
    f = 0.58 * size[0]
    k = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]], float)
    ok, rvec, _ = cv2.solvePnP(object_points(inner, sq), c, k, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return 0.0
    r, _ = cv2.Rodrigues(rvec)
    return float(np.degrees(np.arccos(np.clip(abs(r[2, 2]), 0, 1))))


class _Preview:
    """Tiny MJPEG server (127.0.0.1) so a headless capture can be watched in a browser."""

    def __init__(self, port: int) -> None:
        self._lock = threading.Lock()
        self._jpg: bytes | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a) -> None:  # noqa: ANN002
                pass

            def do_GET(self) -> None:  # noqa: N802
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            with outer._lock:
                                jpg = outer._jpg
                            if jpg:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
                            time.sleep(0.05)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                page = b"<body style='margin:0;background:#111'><img src='/stream' style='width:100%'></body>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(page)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def update(self, img: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._lock:
                self._jpg = buf.tobytes()


def _annotate(frame: np.ndarray, c: np.ndarray | None, hit: dict, inner: tuple[int, int], status: str, info: str) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    ov = out.copy()
    for (i, j), n in hit.items():
        x0, x1 = int(i * w / GRID_COLS), int((i + 1) * w / GRID_COLS)
        y0, y1 = int(j * h / GRID_ROWS), int((j + 1) * h / GRID_ROWS)
        cv2.rectangle(ov, (x0, y0), (x1, y1), (0, 200, 0) if n else (0, 0, 220), -1)
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), 1)
        cv2.putText(out, str(n), (x0 + 8, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    out = cv2.addWeighted(ov, 0.18, out, 0.82, 0)
    if c is not None:
        cv2.drawChessboardCorners(out, inner, c.reshape(-1, 1, 2), True)
    cv2.rectangle(out, (0, h - 64), (w, h), (0, 0, 0), -1)
    cv2.putText(out, status, (12, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(out, info, (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return cv2.resize(out, (960, int(960 * h / w)))


def cmd_capture(args: argparse.Namespace) -> int:
    inner = tuple(args.inner)
    size = (args.width, args.height)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cap = open_camera(args.camera, *size, args.fps)
    saved: list[np.ndarray] = []
    hit = {(i, j): 0 for i in range(GRID_COLS) for j in range(GRID_ROWS)}
    tilts: list[float] = []
    prior: list[np.ndarray] = []
    if args.seed_dir is not None:
        for pth in sorted(args.seed_dir.glob("views*/view_*.png")):
            c0 = detect(cv2.imread(str(pth)), inner, FLAGS_LIVE)
            if c0 is not None:
                prior.append(c0)
                for cell in _cells_hit(c0, size):
                    hit[cell] += 1
                tilts.append(_board_tilt_deg(c0, inner, args.square_mm / 1000.0, size))
        print(f"seeded coverage from {len(prior)} existing views")
    preview = _Preview(args.serve) if args.serve else None
    if preview:
        print(f"live preview: http://127.0.0.1:{args.serve}/", flush=True)
    status = "move the board"
    prev: np.ndarray | None = None
    t0 = time.time()
    last_msg = 0.0
    print(f"Capturing up to {args.n} views in {args.timeout:.0f}s. Move/tilt the board; hold still ~0.5 s per pose.")
    try:
        while len(saved) < args.n and time.time() - t0 < args.timeout:
            frame = read_frame(cap, size)
            if frame is None:
                continue
            c = detect(frame, inner, FLAGS_LIVE)
            now = time.time()
            if preview:
                info = f"new {len(saved)}/{args.n}  strong-tilt(>25deg) views: {sum(t > 25 for t in tilts)}"
                preview.update(_annotate(frame, c, hit, inner, status, info))
            if c is None:
                status = "board not found"
                prev = None
                if now - last_msg > 3:
                    print("  board not found", flush=True)
                    last_msg = now
                continue
            still = prev is not None and float(np.linalg.norm(c - prev, axis=1).mean()) < args.still_px
            prev = c
            if not still:
                status = "hold still..."
                continue
            novel = all(float(np.linalg.norm(c - s, axis=1).mean()) > args.min_move_px for s in saved + prior)
            if not novel:
                status = "same pose as an earlier view - move the board"
                if now - last_msg > 3:
                    print("  same pose as an earlier view - move the board", flush=True)
                    last_msg = now
                continue
            saved.append(c)
            status = f"SAVED #{len(saved)}"
            path = args.out_dir / f"view_{len(saved):02d}.png"
            cv2.imwrite(str(path), frame)
            ctr = c.mean(axis=0)
            span = c.max(axis=0) - c.min(axis=0)
            for cell in _cells_hit(c, size):
                hit[cell] += 1
            tilt = _board_tilt_deg(c, inner, args.square_mm / 1000.0, size)
            tilts.append(tilt)
            missing = [_cell_name(i, j) for j in range(GRID_ROWS) for i in range(GRID_COLS) if hit[(i, j)] == 0]
            print(
                f"  SAVED {len(saved):2d}/{args.n}  centre=({ctr[0]:.0f},{ctr[1]:.0f}) "
                f"size=({span[0]:.0f}x{span[1]:.0f}px) tilt={tilt:.0f}deg  "
                f"strong-tilt views={sum(t > 25 for t in tilts)}  not-yet-reached: {', '.join(missing) or 'none'}",
                flush=True,
            )
    finally:
        cap.release()
    if saved:
        allc = np.vstack(saved)
        lo, hi = allc.min(axis=0), allc.max(axis=0)
        print(
            f"coverage x:[{lo[0]:.0f},{hi[0]:.0f}] of {size[0]}  y:[{lo[1]:.0f},{hi[1]:.0f}] of {size[1]}  "
            f"(aim for corners near all four edges and a few strongly tilted views)"
        )
    print(f"{len(saved)} views saved in {args.out_dir}")
    return 0 if len(saved) >= 12 else 1


def _table_frame_from_board(rvec: np.ndarray, tvec: np.ndarray, inner: tuple[int, int], square_m: float) -> np.ndarray:
    """Camera pose in the table frame: x image-right, y image-up, z up; origin at board centre."""
    r_cb, _ = cv2.Rodrigues(rvec)  # board -> camera
    t_cb = tvec.reshape(3)
    cols, rows = inner
    centre_b = np.array([(cols - 1) * square_m / 2, (rows - 1) * square_m / 2, 0.0])
    origin_cam = r_cb @ centre_b + t_cb
    # Board axes in camera coordinates.
    bx, by, bn = r_cb[:, 0], r_cb[:, 1], r_cb[:, 2]
    if bn[2] > 0:  # normal must point toward the camera (z_cam < 0 direction)
        bn = -bn
    # Choose the in-plane axis closest to image-right (+x_cam) as table x.
    cands = [bx, -bx, by, -by]
    x_t = max(cands, key=lambda v: float(v[0]))
    x_t = x_t - np.dot(x_t, bn) * bn
    x_t /= np.linalg.norm(x_t)
    z_t = bn  # up, toward the camera
    y_t = np.cross(z_t, x_t)  # right-handed; ends up image-up for a downward-looking camera
    r_tc = np.stack([x_t, y_t, z_t], axis=1)  # table -> camera (columns = table axes in camera)
    # p_cam = r_tc @ p_table + origin_cam  =>  camera pose in table frame:
    t = np.eye(4)
    t[:3, :3] = r_tc.T  # columns of R are camera axes expressed in table coordinates
    t[:3, 3] = -r_tc.T @ origin_cam
    return t


def cmd_solve(args: argparse.Namespace) -> int:
    inner = tuple(args.inner)
    sq = args.square_mm / 1000.0
    objp = object_points(inner, sq)
    d = args.dir
    paths = sorted(d.glob("views*/view_*.png"))
    if len(paths) < 8:
        print(f"Need >= 8 views in {d}/views*/, found {len(paths)}", file=sys.stderr)
        return 1
    obj, img, names = [], [], []
    size = None
    for p in paths:
        im = cv2.imread(str(p))
        size = (im.shape[1], im.shape[0])
        c = detect(im, inner, FLAGS_SOLVE)
        if c is None:
            print(f"  skip {p.name}: board not found")
            continue
        obj.append(objp)
        img.append(c.reshape(-1, 1, 2))
        names.append(f"{p.parent.name}/{p.name}")
    flags = cv2.CALIB_FIX_K3
    if args.fix_tangential:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    f0 = size[0] * 0.8
    k0 = np.array([[f0, 0, size[0] / 2], [0, f0, size[1] / 2], [0, 0, 1]], float)
    for it in range(2):
        rms, k, dist, rv, tv, std_i, _, per_view = cv2.calibrateCameraExtended(
            obj, img, size, k0.copy(), None, flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS
        )
        per_view = per_view.reshape(-1)
        thr = max(2.0 * float(np.median(per_view)), args.max_view_rms)
        bad = [i for i, e in enumerate(per_view) if e > thr]
        print(f"pass {it + 1}: {len(obj)} views  RMS={rms:.3f}px  views over {thr:.2f}px: {[names[i] for i in bad]}")
        if not bad or it == 1:
            break
        obj = [o for i, o in enumerate(obj) if i not in bad]
        img = [x for i, x in enumerate(img) if i not in bad]
        names = [n for i, n in enumerate(names) if i not in bad]
    std = std_i.reshape(-1)
    print(f"image size {size[0]}x{size[1]}  views used {len(names)}")
    print(f"fx={k[0, 0]:.1f}±{std[0]:.1f} fy={k[1, 1]:.1f}±{std[1]:.1f} cx={k[0, 2]:.1f}±{std[2]:.1f} cy={k[1, 2]:.1f}±{std[3]:.1f}")
    print(f"dist (k1,k2,p1,p2,k3) = {np.round(dist.reshape(-1), 5).tolist()}")
    hfov = 2 * np.degrees(np.arctan(size[0] / 2 / k[0, 0]))
    print(f"HFOV={hfov:.1f} deg   mean per-view RMS={float(np.mean(per_view)):.3f}px")

    out: dict = {
        "image_size": list(size),
        "K": k.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),
        "rms_px": float(rms),
        "views": len(names),
    }
    tp = d / "table_pose.png"
    if tp.exists():
        im = cv2.imread(str(tp))
        c = detect(im, inner, FLAGS_SOLVE)
        if c is None:
            print("table_pose.png: board not found", file=sys.stderr)
            return 1
        ok, rvec, tvec = cv2.solvePnP(objp, c, k, dist, flags=cv2.SOLVEPNP_IPPE)
        proj, _ = cv2.projectPoints(objp, rvec, tvec, k, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - c, axis=1)
        t_ct = _table_frame_from_board(rvec, tvec, inner, sq)
        cam_pos = t_ct[:3, 3]
        tilt = np.degrees(np.arccos(np.clip(abs(t_ct[2, 2]), 0, 1)))
        print(f"table pose: reproj RMS={float(np.sqrt((err**2).mean())):.3f}px  camera at {np.round(cam_pos, 3).tolist()} m, optical-axis tilt from vertical={tilt:.1f} deg")
        print(f"T_cam_table x-axis (cam right) in table = {np.round(t_ct[:3, 0], 3).tolist()}  y(cam down) = {np.round(t_ct[:3, 1], 3).tolist()}")
        out["T_cam_table"] = t_ct.tolist()
        out["table_pose_rms_px"] = float(np.sqrt((err**2).mean()))
    else:
        print("no table_pose.png: extrinsics skipped")
    (d / "result.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {d / 'result.json'}")

    if args.write:
        if "T_cam_table" not in out:
            print("--write needs table_pose.png", file=sys.stderr)
            return 1
        calib = json.loads(args.calib.read_text())
        cam = dict(calib.get("camera", {}))
        cam.update(
            {
                "_comment": "Orbbec Gemini colour node (the by-id symlink is the IR node: do not use). "
                f"Checkerboard-calibrated {datetime.now():%Y-%m-%d}.",
                "index_or_path": args.camera,
                "width": size[0],
                "height": size[1],
                "fps": args.fps,
                "fourcc": "MJPG",
            }
        )
        prev = {
            "K": calib.get("K"),
            "T_cam_table": calib.get("T_cam_table"),
            "table_xy_affine": calib.pop("table_xy_affine", None),
        }
        calib["camera"] = cam
        calib["K"] = out["K"]
        calib["dist_coeffs"] = out["dist_coeffs"]
        calib["T_cam_table"] = out["T_cam_table"]
        calib["_stale_before_camera_swap"] = prev
        calib["_calibration"] = {
            "date": f"{datetime.now():%Y-%m-%d}",
            "board": f"checkerboard {inner[0] + 1}x{inner[1] + 1} squares, {args.square_mm} mm",
            "rms_px": out["rms_px"],
            "views": out["views"],
            "table_pose_rms_px": out["table_pose_rms_px"],
            "note": "global table_xy_affine removed; per-arm residuals in execution_by_arm are stale until re-fit with calibrate_table_xy.py",
        }
        args.calib.write_text(json.dumps(calib, indent=2) + "\n")
        print(f"updated {args.calib}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--inner", type=int, nargs=2, default=[9, 6], metavar=("COLS", "ROWS"), help="inner corners")
        sp.add_argument("--square-mm", type=float, default=25.0)
        sp.add_argument("--camera", default=DEFAULT_CAMERA)
        sp.add_argument("--width", type=int, default=1280)
        sp.add_argument("--height", type=int, default=720)
        sp.add_argument("--fps", type=int, default=30)

    sp = sub.add_parser("snap", help="save one frame (board flat on the table) for extrinsics")
    common(sp)
    sp.add_argument("--out", type=Path, required=True)
    sp.set_defaults(fn=cmd_snap)

    sp = sub.add_parser("capture", help="auto-save varied board poses for intrinsics")
    common(sp)
    sp.add_argument("--out-dir", type=Path, required=True)
    sp.add_argument("--n", type=int, default=25)
    sp.add_argument("--timeout", type=float, default=300.0)
    sp.add_argument("--still-px", type=float, default=0.8, help="max mean corner motion between detections")
    sp.add_argument("--min-move-px", type=float, default=45.0, help="min mean corner distance to earlier views")
    sp.add_argument("--serve", type=int, default=0, help="serve a live annotated preview on 127.0.0.1:PORT")
    sp.add_argument("--seed-dir", type=Path, default=None, help="calib dir whose views*/ seed coverage and novelty")
    sp.set_defaults(fn=cmd_capture)

    sp = sub.add_parser("solve", help="solve K, distortion and T_cam_table")
    common(sp)
    sp.add_argument("--dir", type=Path, required=True, help="folder with views/ and table_pose.png")
    sp.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    sp.add_argument("--max-view-rms", type=float, default=0.8)
    sp.add_argument("--fix-tangential", action="store_true")
    sp.add_argument("--write", action="store_true", help="update the calib JSON")
    sp.set_defaults(fn=cmd_solve)

    args = p.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
