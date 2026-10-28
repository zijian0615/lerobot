"""Calibrate a depth stream into the robot frames; writes "<camera>_depth" in client/calibration/cameras.json.

    uv run python fanuc_er4ia_sim/challenge/depth_calibrate.py --camera table   # with or without the cell server
    uv run python fanuc_er4ia_sim/challenge/depth_calibrate.py --camera wrist   # the cell server must run (the joints)

1. The depth stream's factory intrinsics and its extrinsics to the colour camera (depth_reader.py --params).
2. Its pose = the colour camera's calibrated pose (cameras.json "table": in the base frame; "wrist": in the flange frame,
   placed in the base frame by the FK of the joints the frames came with) x those factory extrinsics.
3. Our colour model is not the factory's, so that pose is off by a few degrees: tilt it and shift it in z, about the point
   where its optical axis meets the table, until the clear table lies on the work surface (z = -5.1 mm). Its x, y and
   yaw stay those of the colour calibration. The wrist's correction is carried into the flange frame, so it holds at any
   pose; run it at a pose where the wrist camera sees mostly clear table (the home pose does).

Frames: from the running cell server (GET /admin/depth, with the joints), else from depth_reader.py directly (table
only). Depth is averaged over --frames. Clear the table first; objects can stay (the fit keeps points within 6 mm).
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import time
from pathlib import Path

import base64

import cv2
import numpy as np
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fk  # noqa: E402  (the ER-4iA's FK, the twin's joint convention)

CAMERAS = HERE / "client" / "calibration" / "cameras.json"
SURFACE_Z = -0.0051  # the work surface in the base frame (m)
READER = ["uv", "run", "-q", "--no-project", "--python", "3.12", "--with", "pyorbbecsdk2",
          "python", str(HERE / "depth_reader.py")]
HEADER = struct.Struct("<4sIdHH")


def server_frames(url: str, camera: str, n: int) -> tuple[list[np.ndarray], list[float]] | None:
    """n depth frames and the joints (mean) from the running cell server, or None if it is not running."""
    try:
        token = (HERE / ".admin_token").read_text().strip()
        out, joints = [], []
        while len(out) < n:
            r = requests.get(f"{url}/admin/depth", params={"camera": camera}, headers={"x-admin-token": token}, timeout=5)
            r.raise_for_status()
            d = r.json()
            out.append(cv2.imdecode(np.frombuffer(base64.b64decode(d["depth"]), np.uint8), cv2.IMREAD_UNCHANGED))
            joints.append(d["joints"])
            time.sleep(0.07)
    except (OSError, requests.RequestException) as e:
        if isinstance(e, requests.HTTPError):
            raise SystemExit(f"cell server: {e.response.text}") from None
        return None
    j = np.array(joints)
    if np.ptp(j[:, :6], axis=0).max() > 0.05:
        raise SystemExit("the arm moved while the frames were taken; keep it still")
    return out, j.mean(0).tolist()


def frames(n: int, camera: str, fps: int = 15) -> list[np.ndarray]:
    proc = subprocess.Popen([*READER, "--camera", camera, "--fps", str(fps)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out = []
    try:
        while len(out) < n + 5:  # the first frames, while the exposure settles, are dropped below
            head = proc.stdout.read(HEADER.size)
            if len(head) < HEADER.size:
                raise SystemExit("depth_reader.py stopped (is the cell server running? it holds the camera)")
            tag, _seq, _t, w, h = HEADER.unpack(head)
            assert tag == b"DPTH"
            out.append(np.frombuffer(proc.stdout.read(w * h * 2), np.uint16).reshape(h, w))
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    return out[5:]


def points(depth_mm: np.ndarray, k: dict, T: np.ndarray) -> np.ndarray:
    v, u = np.mgrid[0:depth_mm.shape[0], 0:depth_mm.shape[1]]
    m = depth_mm > 0
    z = depth_mm[m] / 1000.0
    pc = np.c_[(u[m] - k["cx"]) / k["fx"] * z, (v[m] - k["cy"]) / k["fy"] * z, z, np.ones(m.sum())]
    return (pc @ T.T)[:, :3]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", choices=["table", "wrist"], default="table")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--url", default="http://127.0.0.1:9099", help="the cell server (operator API on this PC)")
    p.add_argument("--dry-run", action="store_true", help="report, but do not write cameras.json")
    args = p.parse_args()
    k = json.loads(subprocess.run([*READER, "--camera", args.camera, "--params"], capture_output=True, text=True,
                                  check=True).stdout.splitlines()[-1])
    print(f"factory: {k['size'][0]}x{k['size'][1]} fx {k['fx']:.2f} cx {k['cx']:.1f} cy {k['cy']:.1f}, "
          f"depth->colour t {np.round(np.array(k['depth_to_colour']['t_m']) * 1000, 1)} mm")
    got = server_frames(args.url, args.camera, args.frames)
    if got is None and args.camera == "wrist":
        raise SystemExit("the wrist needs the running cell server (its frames come with the joints)")
    stack_list, joints = got if got else (frames(args.frames, args.camera), None)
    print(f"{len(stack_list)} frames from {'the cell server' if got else 'depth_reader.py'}"
          + (f", joints {np.round(joints[:6], 2).tolist()}" if joints else ""))
    stack = np.stack(stack_list).astype(np.float64)
    valid = (stack > 0).mean(0) > 0.8
    depth = np.where(valid, np.median(np.where(stack > 0, stack, np.nan), axis=0), 0)
    depth = np.nan_to_num(depth)
    cams = json.loads(CAMERAS.read_text())
    T_cd = np.eye(4)
    T_cd[:3, :3], T_cd[:3, 3] = np.array(k["depth_to_colour"]["R"]), np.array(k["depth_to_colour"]["t_m"])
    T_bf = np.eye(4)  # the flange in the base frame (the wrist camera is on it); identity for the table camera
    if args.camera == "wrist":
        R, pf = fk.fk(np.radians(np.asarray(joints[:6], float)))
        T_bf[:3, :3], T_bf[:3, 3] = R, pf
        T0 = T_bf @ np.array(cams["wrist"]["T_flange_cam"], float) @ T_cd
    else:
        T0 = np.array(cams["table"]["T_base_cam"], float) @ T_cd
    P = points(depth, k, T0)
    inside = (P[:, 0] > -0.30) & (P[:, 0] < 0.72) & (abs(P[:, 1]) < 0.74)  # the table, not the frame and the walls
    sel = inside & (abs(P[:, 2] - np.median(P[inside, 2])) < 0.06)
    for _ in range(6):  # plane fit on the points near it
        A = np.c_[P[sel, 0], P[sel, 1], np.ones(sel.sum())]
        c, *_ = np.linalg.lstsq(A, P[sel, 2], rcond=None)
        sel = inside & (abs(P[:, 2] - np.c_[P[:, 0], P[:, 1], np.ones(len(P))] @ c) < 0.006)
    n = np.array([-c[0], -c[1], 1.0])
    n /= np.linalg.norm(n)
    o, a = T0[:3, 3], T0[:3, 2]
    pivot = o + (c[2] + c[0] * o[0] + c[1] * o[1] - o[2]) / (a[2] - c[0] * a[0] - c[1] * a[1]) * a
    axis = np.cross(n, [0.0, 0.0, 1.0])
    ang = np.arcsin(np.linalg.norm(axis))
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    Rc = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
    C = np.eye(4)
    C[:3, :3], C[:3, 3] = Rc, pivot - Rc @ pivot
    C[2, 3] += SURFACE_Z - pivot[2]
    T = C @ T0
    r = points(depth, k, T)[sel][:, 2] - SURFACE_Z
    print(f"table plane: {sel.sum()} points; corrected {np.degrees(ang):.2f} deg and {(SURFACE_Z - pivot[2]) * 1000:+.1f} mm; "
          f"now median {np.median(r) * 1000:+.2f} mm, std {r.std() * 1000:.2f} mm, within 5 mm {100 * (abs(r) < 0.005).mean():.0f}%")
    if np.degrees(ang) > 8 or (abs(r) < 0.005).mean() < 0.9:
        sys.exit("the fit looks wrong (clear the table, check the colour calibration); cameras.json not written")
    entry = {"model": "pinhole, no distortion (rectified by the camera): u = fx X/Z + cx, v = fy Y/Z + cy",
             "size": k["size"], "fx": k["fx"], "fy": k["fy"], "cx": k["cx"], "cy": k["cy"], "units": "mm",
             "solved": time.strftime("%Y-%m-%d"), "rms_plane_mm": round(float(r.std() * 1000), 2)}
    if args.camera == "wrist":
        entry |= {"T_flange_cam": (np.linalg.inv(T_bf) @ T).round(6).tolist(),
                  "transform": "T_flange_cam: depth camera -> flange (the twin's J6_link frame); compose with the FK of the "
                               "joints the depth came with (camera: +z forward, +x right, +y down)",
                  "solved_at_joints": np.round(joints[:6], 3).tolist(),
                  "notes": "Factory intrinsics and depth-to-colour extrinsics on the wrist colour camera's flange pose, then "
                           "tilted and shifted onto the work surface at solved_at_joints. Depth: mm along the optical axis; 0 = none."}
    else:
        entry |= {"T_base_cam": T.round(6).tolist(),
                  "transform": "the depth camera's pose in the base frame (camera: +z forward, +x right, +y down)",
                  "notes": "Factory intrinsics and depth-to-colour extrinsics, on the table colour camera's pose, then tilted and "
                           "shifted in z onto the work surface. Depth is the distance along the optical axis (Z), in mm; 0 = none."}
    if args.dry_run:
        print(json.dumps(entry, indent=1))
        return
    cams[f"{args.camera}_depth"] = entry
    CAMERAS.write_text(json.dumps(cams, indent=1) + "\n")
    print(f"wrote {args.camera}_depth to {CAMERAS}")


if __name__ == "__main__":
    main()
