"""Camera geometry for the cell (numpy only): base-frame points <-> pixels with the real cameras' calibration.

    from calibration.geometry import load, project, pixel_to_base
    cams = load()
    uv = project(cams["table"], [[0.42, -0.10, -0.005]])          # base metres -> pixels
    xyz = pixel_to_base(cams["table"], (960, 540), z=-0.0051)      # pixel -> point on the plane z (base metres)
    pts = depth_to_base(cams["table_depth"], obs["depth"]["table"])  # depth image (mm) -> (N, 3) base metres
    pts = depth_to_base(cams["wrist_depth"], obs["depth"]["wrist"], joints=obs["joints"])  # the wrist moves: needs the joints

UF0 (the frame of obs["tcp_pose"] and "tcp" actions) is the base frame shifted down 330 mm: x, y equal, z_uf0 = z_base - 330 mm.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SURFACE_Z = -0.0051  # the work surface in the robot base frame (m)
UF0_Z_MM = -330.0


# The ER-4iA's joint chain (the twin's convention, the one obs["joints"] uses): translate in the parent frame, then
# rotate about the axis. The last frame is the flange (J6_link: wrist centre, tool along +x).
CHAIN = [((0.0, 0.0, 0.33), "z", +1), ((0.0, 0.0, 0.00), "y", +1), ((0.0, 0.0, 0.26), "y", -1),
         ((0.0, 0.0, 0.02), "x", -1), ((0.29, 0.0, 0.0), "y", -1), ((0.0, 0.0, 0.00), "x", -1)]


def _rot(axis: str, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def flange_pose(joints) -> np.ndarray:
    """J1..J6 in degrees (obs["joints"]; a 7th value, the gripper, is ignored) -> the flange's 4x4 pose in the base frame."""
    R, p = np.eye(3), np.zeros(3)
    for (off, axis, sign), q in zip(CHAIN, np.radians(np.asarray(joints, float)[:6])):
        p = p + R @ np.array(off)
        R = R @ _rot(axis, sign * q)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def load(path: Path = HERE / "cameras.json") -> dict:
    """The cameras: "table", "side", "wrist" (colour) and "table_depth" (the table camera's depth stream)."""
    raw = json.loads(Path(path).read_text())
    cams = {}
    for k, v in raw.items():
        if not isinstance(v, dict):
            continue
        cam = {"T": np.array(v.get("T_base_cam", v.get("T_flange_cam"))), "fx": v["fx"], "size": tuple(v["size"]),
               "frame": "base" if "T_base_cam" in v else "flange"}
        if "cx" in v:  # a depth stream: pinhole with its own principal point, no distortion
            cam |= {"fy": v["fy"], "cx": v["cx"], "cy": v["cy"], "depth": True}
        else:
            cam |= {"k1": v["k1"]}
        cams[k] = cam
    return cams


def project(cam: dict, pts) -> np.ndarray:
    """(N, 3) points in the camera's reference frame (base, or flange for the wrist) -> (N, 2) pixels."""
    pts = np.asarray(pts, float).reshape(-1, 3)
    T = np.linalg.inv(cam["T"])
    pc = pts @ T[:3, :3].T + T[:3, 3]
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    d = 1 + cam["k1"] * (x * x + y * y)
    w, h = cam["size"]
    return np.stack([cam["fx"] * x * d + w / 2, cam["fx"] * y * d + h / 2], -1)


def depth_to_base(cam: dict, depth_mm, stride: int = 1, with_pixels: bool = False, joints=None):
    """A depth image (H, W uint16 mm, 0 = no depth) -> (N, 3) base-frame points (m), one per valid pixel.
    joints: for the wrist camera, obs["joints"] from the same observation (the camera moves with the arm).
    stride > 1 takes every stride-th row and column. with_pixels: also return the (N, 2) pixels (u, v) they came from."""
    if not cam.get("depth"):
        raise ValueError("depth_to_base needs a depth camera, e.g. cams['table_depth']")
    T = cam["T"]
    if cam["frame"] == "flange":
        if joints is None:
            raise ValueError("the wrist camera moves with the arm: pass joints=obs['joints']")
        T = flange_pose(joints) @ T
    d = np.asarray(depth_mm)[::stride, ::stride].astype(np.float64)
    v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]] * stride
    m = d > 0
    z = d[m] / 1000.0
    pc = np.stack([(u[m] - cam["cx"]) / cam["fx"] * z, (v[m] - cam["cy"]) / cam["fy"] * z, z], -1)
    pts = pc @ T[:3, :3].T + T[:3, 3]
    return (pts, np.stack([u[m], v[m]], -1)) if with_pixels else pts


def pixel_to_base(cam: dict, uv, z: float = SURFACE_Z) -> np.ndarray:
    """A pixel of a fixed camera (table, side) -> the base-frame point where its ray meets the plane z (m)."""
    if cam.get("depth"):
        raise ValueError("for the depth stream use depth_to_base")
    if cam["frame"] != "base":
        raise ValueError("pixel_to_base needs a fixed camera; for the wrist compose T_flange_cam with the joints' FK")
    w, h = cam["size"]
    xd, yd = (uv[0] - w / 2) / cam["fx"], (uv[1] - h / 2) / cam["fx"]
    x, y = xd, yd
    for _ in range(10):
        d = 1 + cam["k1"] * (x * x + y * y)
        x, y = xd / d, yd / d
    ray = cam["T"][:3, :3] @ np.array([x, y, 1.0])
    o = cam["T"][:3, 3]
    return o + ray * (z - o[2]) / ray[2]
