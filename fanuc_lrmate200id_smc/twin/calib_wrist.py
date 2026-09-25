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

"""Wrist camera (Innomaker, /dev/video2) calibration against the aligned table scan, from static robot poses.

    # 1. for each of 6-8 static poses (robot stopped >= 3 s; vary TCP height 5-30 cm, J6 angle and tilt; the printed
    #    table in view): read-only RMI joints/cartesian + one wrist and one overhead frame. Nothing is sent to move.
    uv run python twin/calib_wrist.py capture --out ../examples/cosmos_edge_fanuc/scanner/wrist_calib/pose_01
    # 2. solve, starting from the current wrist_camera.json (old one kept as wrist_camera.old.json)
    uv run python twin/calib_wrist.py solve --poses ../examples/cosmos_edge_fanuc/scanner/wrist_calib \\
        --scan ../examples/cosmos_edge_fanuc/scanner/scan_204107

Model: the camera is rigid to tool0 (it rotates with J6; the bracket holds it ~15 cm off the tool axis). Each pose gives
tool0 from the controller's UF0/UT1 pose (UTOOL 1 = 223 mm along tool Z). For a camera guess the table scan
(align_scan.py, table frame) is ray-cast into the wrist view and compared with the photo on blurred edge images
(normalized correlation). A grid over the distance along the tool axis x focal length comes first (the two trade off
at high poses; low poses separate them), then Powell refines rotation, position and focal length.
Line-art feature matching (SIFT, patches) between photo and scan was tried and is not reliable here.
Accuracy is about 1-2 cm on the table, limited by the scan-to-robot alignment (~8 mm): expect ~20 px at the wrist.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import shutil
import socket
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
WRIST_JSON = os.path.join(HERE, "..", "wrist_camera.json")
HOST, PORT = "172.30.109.22", 16001
TCP_M = 0.223           # controller UTOOL 1 = (0, 0, 223) mm
BASE_HEIGHT_M = 0.330   # UF0 origin (J1/J2 axes) above the base plate / table top
TOP_PX_PER_M = 1600.0
EVAL_WH = (320, 180)    # correlation is computed on downscaled frames
IMAGE_WH = (1280, 720)


# ---------------------------------------------------------------- capture (read-only)

def _rmi(commands, host=HOST, port=PORT):
    """Send read-only FRC_Read* commands on a fresh RMI session; never FRC_Initialize or motion."""
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(b'{"Communication": "FRC_Connect"}\r\n')
        reply = json.loads(sock.recv(4096).decode())
    if reply.get("ErrorID", -1) != 0:
        raise RuntimeError(f"FRC_Connect failed: {reply}")
    sock = socket.create_connection((host, int(reply["PortNumber"])), timeout=5)
    buf, out = b"", {}
    try:
        for name in commands:
            sock.sendall((json.dumps({"Command": name, "Group": 1}) + "\r\n").encode())
            while True:
                while b"\n" not in buf:
                    buf += sock.recv(65536)
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    msg = json.loads(line)
                    if msg.get("Command") == name:
                        out[name] = msg
                        break
        sock.sendall(b'{"Communication": "FRC_Disconnect"}\r\n')
        time.sleep(0.2)
    finally:
        sock.close()
    return out


def _grab(device, width, height, warmup=30):
    import cv2

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {device}")
    ok, frame = False, None
    for _ in range(warmup):
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"no frame from {device}")
    return frame


def capture(out_dir, host=HOST, port=PORT, wrist_dev="/dev/video2", overhead_dev="/dev/video0") -> dict:
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    before = _rmi(["FRC_ReadJointAngles", "FRC_ReadCartesianPosition"], host, port)
    wrist = _grab(wrist_dev, *IMAGE_WH)
    overhead = _grab(overhead_dev, 1920, 1080)
    after = _rmi(["FRC_ReadJointAngles"], host, port)
    j0 = before["FRC_ReadJointAngles"]["JointAngle"]
    j1 = after["FRC_ReadJointAngles"]["JointAngle"]
    drift = max(abs(j0[f"J{i}"] - j1[f"J{i}"]) for i in range(1, 7))
    cart = before["FRC_ReadCartesianPosition"]
    if cart["Configuration"]["UToolNumber"] != 1 or cart["Configuration"]["UFrameNumber"] != 0:
        raise RuntimeError(f"expected UT1/UF0, got {cart['Configuration']}")
    cv2.imwrite(os.path.join(out_dir, "wrist.png"), wrist)
    cv2.imwrite(os.path.join(out_dir, "overhead.jpg"), overhead)
    doc = {"joints_deg": [j0[f"J{i}"] for i in range(1, 7)], "cartesian": cart["Position"],
           "configuration": cart["Configuration"], "joints_after_deg": [j1[f"J{i}"] for i in range(1, 7)],
           "max_drift_deg": drift, "wrist_wh": [wrist.shape[1], wrist.shape[0]], "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(doc, open(os.path.join(out_dir, "pose.json"), "w"), indent=1)
    position = cart["Position"]
    print(f"[capture] joints {np.round(doc['joints_deg'], 2).tolist()} xyz {[round(position[k], 1) for k in 'XYZ']} "
          f"wpr {[round(position[k], 1) for k in 'WPR']} drift {drift:.3f} deg -> {out_dir}", flush=True)
    if drift > 0.05:
        print("[capture] WARNING: the robot moved while capturing; redo this pose", flush=True)
    return doc


# ---------------------------------------------------------------- model

def wpr_matrix(w, p, r):
    """FANUC W/P/R (deg) -> rotation, R = Rz(r) Ry(p) Rx(w)."""
    w, p, r = np.radians([w, p, r])
    rx = np.array([[1, 0, 0], [0, math.cos(w), -math.sin(w)], [0, math.sin(w), math.cos(w)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def tool0_pose(cartesian):
    """tool0 rotation and origin in the table frame from a UF0/UT1 controller pose (mm, deg)."""
    rot = wpr_matrix(cartesian["W"], cartesian["P"], cartesian["R"])
    tcp = np.array([cartesian["X"], cartesian["Y"], cartesian["Z"]]) / 1000.0 + [0.0, 0.0, BASE_HEIGHT_M]
    return rot, tcp - rot[:, 2] * TCP_M


class WristModel:
    """Parameter vector: rotvec (3) and position (3) of the OpenCV camera in tool0, focal length / 1000."""

    def __init__(self, k, dist):
        self.k = np.asarray(k, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64)

    @staticmethod
    def pack(rot, t, focal):
        import cv2

        return np.r_[cv2.Rodrigues(np.asarray(rot, dtype=np.float64))[0].ravel(), t, focal / 1000.0]

    @staticmethod
    def unpack(x):
        import cv2

        return cv2.Rodrigues(np.asarray(x[0:3], dtype=np.float64))[0], np.asarray(x[3:6]), float(x[6]) * 1000.0


def load_top(scan_dir, cache=True):
    """Table-frame top-down of the aligned scan (TOP_PX_PER_M, row = x, column = y) and the (x, y) of pixel (0, 0)."""
    import cv2
    from PIL import Image

    import align_scan

    path = os.path.join(scan_dir, "top_table.png")
    meta = os.path.join(scan_dir, "top_table.json")
    if cache and os.path.isfile(path) and os.path.isfile(meta):
        return cv2.imread(path), np.array(json.load(open(meta))["origin"])
    Image.MAX_IMAGE_PIXELS = None
    report = json.load(open(os.path.join(scan_dir, "alignment.json"), encoding="utf-8"))
    pos, uv, faces, jpg = align_scan.load_glb(report["glb"])
    m = np.array(report["glb_to_table"])
    points = pos @ m[:3, :3].T + m[:3, 3]
    texture = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB"))
    top, origin = align_scan.top_down(points, uv, faces, texture, px_per_m=TOP_PX_PER_M, per_px=6)
    top = cv2.cvtColor(top, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, top)
    json.dump({"origin": origin.tolist(), "px_per_m": TOP_PX_PER_M}, open(meta, "w"))
    return top, origin


def synth(x, pose, model, top, origin, wh=IMAGE_WH):
    """The table scan seen by the wrist camera (BGR, wh), with the lens distortion of `model`."""
    import cv2

    from lens import undistort_radial

    rot_tc, t_tc, focal = WristModel.unpack(x)
    rot_t, p_t = pose
    rot, centre = rot_t @ rot_tc, p_t + rot_t @ t_tc
    scale = wh[0] / IMAGE_WH[0]
    w, h = wh
    cx, cy = (model.k[0, 2] + 0.5) * scale - 0.5, (model.k[1, 2] + 0.5) * scale - 0.5
    f = focal * scale
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    norm = undistort_radial(np.stack([(uu - cx) / f, (vv - cy) / f], -1), model.dist)
    ray = np.concatenate([norm, np.ones((h, w, 1))], -1) @ rot.T
    lam = -centre[2] / ray[..., 2]
    xs, ys = centre[0] + lam * ray[..., 0], centre[1] + lam * ray[..., 1]
    map_x = ((ys - origin[1]) * TOP_PX_PER_M).astype(np.float32)
    map_y = ((xs - origin[0]) * TOP_PX_PER_M).astype(np.float32)
    map_x[lam <= 0] = -1
    map_y[lam <= 0] = -1
    return cv2.remap(top, map_x, map_y, cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def _edges(image, blur=1.5):
    import cv2

    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if image.shape[1::-1] != EVAL_WH:
        g = cv2.resize(g, EVAL_WH, interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (0, 0), blur)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def correlation(x, poses, photos, model, top, origin):
    """Per-pose normalized correlation of edge images, synthetic vs photo (-1 where the view misses the table)."""
    import cv2

    out = []
    for name, pose in poses.items():
        syn = synth(x, pose, model, top, origin)  # full resolution, then downscaled like the photo (thin lines)
        mask = cv2.resize((syn.max(-1) > 0).astype(np.uint8), EVAL_WH, interpolation=cv2.INTER_NEAREST) > 0
        if mask.mean() < 0.5:
            out.append(-1.0)
            continue
        a, b = _edges(syn)[mask], photos[name][mask]
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        out.append(float((a * b).mean()))
    return np.array(out)


def load_poses(poses_dir):
    import cv2

    poses, photos, raw = {}, {}, {}
    for folder in sorted(glob.glob(os.path.join(poses_dir, "pose_*"))):
        doc = json.load(open(os.path.join(folder, "pose.json")))
        if doc.get("max_drift_deg", 0) > 0.05:
            print(f"[solve] skip {folder}: moved during capture", flush=True)
            continue
        name = os.path.basename(folder)
        poses[name] = tool0_pose(doc["cartesian"])
        raw[name] = cv2.imread(os.path.join(folder, "wrist.png"))
        photos[name] = _edges(raw[name])
    if len(poses) < 4:
        raise RuntimeError(f"need >= 4 static poses, found {len(poses)} in {poses_dir}")
    return poses, photos, raw


def solve(poses_dir, scan_dir, init_json=WRIST_JSON, out_json=WRIST_JSON, grid=True, maxiter=3000) -> dict:
    import cv2
    from scipy.optimize import minimize

    init = json.load(open(init_json, encoding="utf-8"))
    if "T_tool_cam" not in init or "K" not in init:
        raise RuntimeError(f"{init_json} has no T_tool_cam/K to start from; set a rough guess first")
    model = WristModel(init["K"], init.get("dist", [0, 0, 0, 0, 0]))
    pose_tc = np.array(init["T_tool_cam"])
    x = WristModel.pack(pose_tc[:3, :3], pose_tc[:3, 3], model.k[0, 0])
    poses, photos, raw = load_poses(poses_dir)
    top, origin = load_top(scan_dir)
    start = correlation(x, poses, photos, model, top, origin)
    print(f"[solve] {len(poses)} poses; start correlation {start.mean():.3f} {np.round(start, 3).tolist()}", flush=True)
    if grid:  # distance along the tool axis x focal length
        best = (start.mean(), x)
        for dz in np.arange(-0.06, 0.061, 0.02):
            for focal in np.arange(0.8, 1.21, 0.1) * model.k[0, 0]:
                trial = x.copy()
                trial[5] = x[5] + dz
                trial[6] = focal / 1000.0
                score = correlation(trial, poses, photos, model, top, origin).mean()
                if score > best[0]:
                    best = (score, trial)
        x = best[1]
        print(f"[solve] grid best {best[0]:.3f}: along-axis {x[5]:.3f} m, focal {x[6] * 1000:.0f} px", flush=True)
    result = minimize(lambda v: -correlation(v, poses, photos, model, top, origin).mean(), x, method="Powell",
                      options={"maxiter": maxiter, "xtol": 1e-4, "ftol": 1e-5})
    x = result.x
    final = correlation(x, poses, photos, model, top, origin)
    rot, t, focal = WristModel.unpack(x)
    print(f"[solve] end correlation {final.mean():.3f} {np.round(final, 3).tolist()}", flush=True)

    backup = os.path.splitext(out_json)[0] + ".old.json"
    if os.path.isfile(out_json) and not os.path.isfile(backup):
        shutil.copyfile(out_json, backup)
    width, height = IMAGE_WH
    k = model.k.copy()
    k[0, 0] = k[1, 1] = focal
    pose_tc = np.eye(4)
    pose_tc[:3, :3], pose_tc[:3, 3] = rot, t
    aperture = 20.955
    doc = dict(init)
    doc.update({
        "_comment": "Innomaker wrist camera on a long bracket beside the gripper, rigid to tool0 (rotates with J6). "
                    "Calibrated by twin/calib_wrist.py against the aligned table scan from static poses; accuracy "
                    "about 1-2 cm on the table (scan-to-robot alignment). T_tool_cam: OpenCV camera in tool0.",
        "image_hw": [height, width], "K": k.tolist(), "dist": model.dist.tolist(), "T_tool_cam": pose_tc.tolist(),
        "hfov_deg": math.degrees(2 * math.atan(width / 2 / focal)),
        "eye_tool": t.tolist(), "target_tool": (t + rot[:, 2] * 0.3).tolist(), "up_tool": (-rot[:, 1]).tolist(),
        "focal_mm": focal * aperture / width, "horizontal_aperture": aperture, "vertical_aperture": aperture * height / width,
        "calibration": {"poses": sorted(poses), "correlation": np.round(final, 4).tolist(), "poses_dir": os.path.abspath(poses_dir),
                        "scan": os.path.abspath(scan_dir)},
    })
    json.dump(doc, open(out_json, "w", encoding="utf-8"), indent=1)
    rows = []
    for name, pose in poses.items():
        real = cv2.resize(raw[name], (640, 360), interpolation=cv2.INTER_AREA)
        syn = synth(x, pose, model, top, origin, (640, 360))
        rows.append(np.hstack([real, syn, cv2.addWeighted(real, 0.5, syn, 0.5, 0)]))
    cv2.imwrite(os.path.join(poses_dir, "calib_compare.jpg"), np.vstack(rows))
    print(f"[solve] wrote {out_json} (old copy: {backup}); camera in tool0 {np.round(t, 4).tolist()} m, focal {focal:.0f} px "
          f"(hfov {doc['hfov_deg']:.1f} deg); compare: {os.path.join(poses_dir, 'calib_compare.jpg')}", flush=True)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="read-only: joints + one wrist and one overhead frame of a static pose")
    c.add_argument("--out", required=True)
    c.add_argument("--host", default=HOST)
    c.add_argument("--port", type=int, default=PORT)
    s = sub.add_parser("solve", help="fit wrist_camera.json to the captured poses")
    s.add_argument("--poses", required=True, help="folder with pose_*/ (pose.json, wrist.png)")
    s.add_argument("--scan", required=True, help="align_scan.py output folder (alignment.json)")
    s.add_argument("--init", default=WRIST_JSON)
    s.add_argument("--out", default=WRIST_JSON)
    s.add_argument("--no-grid", action="store_true")
    s.add_argument("--maxiter", type=int, default=3000)
    a = ap.parse_args(argv)
    if a.cmd == "capture":
        capture(a.out, a.host, a.port)
    else:
        solve(a.poses, a.scan, a.init, a.out, grid=not a.no_grid, maxiter=a.maxiter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
