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

"""Register a Scaniverse GLB table scan to the robot/table frame and preview it through the real overhead camera.

    # 1. align (lerobot env: numpy, Pillow, OpenCV). Needs a current 1920x1080 video0 frame of the table.
    uv run python twin/align_scan.py --glb '../examples/cosmos_edge_fanuc/scanner/Scaniverse 2026-09-22 204107.glb' \\
        --photo video0.jpg --out ../examples/cosmos_edge_fanuc/scanner/scan_204107
    # 2. make it the twin's overhead camera: overhead_camera.json (old one kept as overhead_camera.old.json) and
    #    table_texture.png, then rebuild the stage with twin/build_usd.py
    uv run python twin/align_scan.py --export-overhead --out ../examples/cosmos_edge_fanuc/scanner/scan_204107
    # 3. render the aligned scan in Isaac Sim from the calibrated video0 camera and compare it with the photo
    OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \\
      ~/isaacsim-venv/bin/python twin/align_scan.py --render --out ../examples/cosmos_edge_fanuc/scanner/scan_204107

The printed table in the scan is the calibration target; the robot frame comes only from robot-measured points:
1. level: glTF Y-up -> Z-up, plane fit of the table top -> z = 0. Rendered top-down at 800 px/m ("scan frame").
2. camera: video0 intrinsics (f, k1, k2; principal point fixed at the centre) and pose in the scan frame from
   photo <-> scan correspondences. Seeded by SIFT on a rough rectification, then dense patch matching on gradient images
   between the scan top-down and the photo re-projected with the current model, a few rounds.
3. robot: the 16 TCP-touched samples of `table_xy_calib_samples_fanuc.json` (pixel + robot xy on z = z_plane) are cast
   through that camera onto the scan's plane; a 2D similarity (Umeyama) takes scan xy to robot xy. Its scale checks the
   phone scan's metric scale, its residuals are the alignment error (and would expose a camera moved since sampling).
Outputs in --out: table.npz (points in the table frame, faces, face-varying st, texture path: the preview_scan.py format),
alignment.json (4x4 glb -> table, camera in the table frame, residuals), overlay_checker.png, video0_undistorted.png.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(HERE, "..", "fanuc_lrmate200id_smc.usda")
SAMPLES = os.path.join(HERE, "..", "..", "examples", "tabletop_perception", "calib", "table_xy_calib_samples_fanuc.json")
PX_PER_M = 800.0
SEED_BOUNDS = (-0.25, 0.75, -0.65, 0.65)  # (x_min, x_max, y_min, y_max) of the rough robot-frame rectification
TABLE_BAND_M = (0.0, 0.09)  # glTF height band that holds the table top of a Scaniverse export
IMAGE_WH = (1920, 1080)
RENDER_WH = (960, 540)
OVERHEAD_JSON = os.path.join(HERE, "..", "overhead_camera.json")
TABLE_TEXTURE = os.path.join(HERE, "..", "table_texture.png")
TEXTURE_BOUNDS = (-0.45, 0.45, -0.35, 0.35)  # the scene's table polygon (x_min, x_max, y_min, y_max)
PATCH, SEARCH, STEP = 48, 24, 32  # dense matching: half patch, search radius, grid step [top-down px]


def load_glb(path):
    """Positions, UVs, triangle indices and the embedded texture (JPEG bytes) of a single-mesh GLB."""
    blob = open(path, "rb").read()
    json_len = struct.unpack("<I", blob[12:16])[0]
    gltf = json.loads(blob[20 : 20 + json_len])
    start = 20 + json_len
    binary = blob[start + 8 : start + 8 + struct.unpack("<I", blob[start : start + 4])[0]]

    def accessor(index):
        acc = gltf["accessors"][index]
        view = gltf["bufferViews"][acc["bufferView"]]
        offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        dtype = {5126: np.float32, 5125: np.uint32, 5123: np.uint16}[acc["componentType"]]
        width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[acc["type"]]
        return np.frombuffer(binary, dtype, acc["count"] * width, offset).reshape(acc["count"], width)

    prim = gltf["meshes"][0]["primitives"][0]
    view = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
    offset = view.get("byteOffset", 0)
    return (
        accessor(prim["attributes"]["POSITION"]).astype(np.float64),
        accessor(prim["attributes"]["TEXCOORD_0"]).astype(np.float64),
        accessor(prim["indices"]).reshape(-1, 3).astype(np.int64),
        binary[offset : offset + view["byteLength"]],
    )


def level(points):
    """4x4 that takes glTF (Y-up) points to a Z-up frame with the table top on z = 0, plus the plane residual [m]."""
    y_up_to_z_up = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])
    q = points @ y_up_to_z_up.T
    band = q[(q[:, 2] > TABLE_BAND_M[0]) & (q[:, 2] < TABLE_BAND_M[1])]
    centre = band.mean(0)
    for _ in range(3):  # trim the objects standing on the table
        normal = np.linalg.svd(band - centre)[2][2]
        dist = (band - centre) @ normal
        band = band[np.abs(dist) < 3 * dist.std()]
        centre = band.mean(0)
    normal = np.linalg.svd(band - centre)[2][2]
    normal = normal if normal[2] > 0 else -normal
    axis = np.cross(normal, [0.0, 0.0, 1.0])
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + skew + skew @ skew / (1.0 + normal[2])
    out = np.eye(4)
    out[:3, :3] = rot @ y_up_to_z_up
    out[:3, 3] = -rot @ centre
    return out, float(((band - centre) @ normal).std())


def top_down(points, uv, faces, texture, px_per_m=PX_PER_M, per_px=10, seed=0):
    """Orthographic top view (highest surface wins), row = x, column = y. Returns the image and the (x, y) of pixel (0, 0)."""
    tri = points[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    per_face = np.maximum(np.ceil(area * px_per_m**2 * per_px).astype(int), 1)
    face = np.repeat(np.arange(len(faces)), per_face)
    r = np.random.default_rng(seed).random((len(face), 2))
    r[r.sum(1) > 1] = 1 - r[r.sum(1) > 1]
    w = np.column_stack([1 - r.sum(1), r])
    p = np.einsum("nc,ncd->nd", w, points[faces[face]])
    t = np.einsum("nc,ncd->nd", w, uv[faces[face]])
    h, wdt = texture.shape[:2]
    colour = texture[np.clip((t[:, 1] * h).astype(int), 0, h - 1), np.clip((t[:, 0] * wdt).astype(int), 0, wdt - 1)]
    origin = p[:, :2].min(0)
    row, col = ((p[:, 0] - origin[0]) * px_per_m).astype(int), ((p[:, 1] - origin[1]) * px_per_m).astype(int)
    image = np.zeros((row.max() + 1, col.max() + 1, 3), np.uint8)
    order = np.argsort(p[:, 2])
    image[row[order], col[order]] = colour[order]
    return image, origin


def load_samples(path=SAMPLES):
    samples = json.load(open(path, encoding="utf-8"))["samples"]
    uv = np.array([s["uv"] for s in samples], np.float64)
    xyz = np.array([[*s["true_table_xy"], s["z_plane"]] for s in samples], np.float64)
    return uv, xyz


def _gray(image):
    import cv2

    return cv2.createCLAHE(3.0, (8, 8)).apply(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))


def _grad(image):
    import cv2

    g = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.5)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


class Camera:
    """OpenCV camera (K, k1, k2) with pose X_cam = R X + t in some frame."""

    def __init__(self, k, dist, rot, t):
        self.k, self.dist, self.rot, self.t = (np.asarray(v, np.float64) for v in (k, dist, rot, t))

    def project(self, points):
        """Pixels of 3D points (N x 3); NaN where the lens model is not valid (behind, or folded past the image corners)."""
        import cv2

        cam = points @ self.rot.T + self.t
        norm = cam[:, :2] / cam[:, 2:]
        corners = np.array([[0, 0], [IMAGE_WH[0], 0], [0, IMAGE_WH[1]], [IMAGE_WH[0], IMAGE_WH[1]]], np.float64)
        r_max = np.linalg.norm(cv2.undistortPoints(corners.reshape(-1, 1, 2), self.k, self.dist).reshape(-1, 2), axis=1).max()
        pix, _ = cv2.projectPoints(points.reshape(-1, 1, 3), cv2.Rodrigues(self.rot)[0], self.t, self.k, self.dist)
        pix = pix.reshape(-1, 2)
        pix[(cam[:, 2] <= 0) | (np.linalg.norm(norm, axis=1) > 1.05 * r_max)] = np.nan
        return pix

    def rays(self, pixels):
        """Camera centre and unit-less ray directions (frame coordinates) through pixels."""
        import cv2

        und = cv2.undistortPoints(np.asarray(pixels, np.float64).reshape(-1, 1, 2), self.k, self.dist).reshape(-1, 2)
        return -self.rot.T @ self.t, np.c_[und, np.ones(len(und))] @ self.rot

    def remap_plane(self, photo, origin, shape, px_per_m=PX_PER_M):
        """The photo resampled onto z = 0 of this frame: row = x from origin[0], column = y from origin[1]."""
        import cv2

        rows, cols = shape
        rr, cc = np.mgrid[0:rows, 0:cols].astype(np.float64)
        world = np.stack([origin[0] + rr / px_per_m, origin[1] + cc / px_per_m, np.zeros_like(rr)], -1).reshape(-1, 3)
        pix = np.nan_to_num(self.project(world), nan=-1.0).reshape(rows, cols, 2).astype(np.float32)
        image = cv2.remap(photo, pix[..., 0], pix[..., 1], cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        return image, pix

    def to_json(self):
        return {"K": self.k.tolist(), "dist": self.dist.ravel().tolist(), "R_cam_table": self.rot.tolist(),
                "t_cam_table": self.t.tolist(), "centre_table": (-self.rot.T @ self.t).tolist()}


def _calibrate(obj, img, k0):
    import cv2

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    )
    rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(
        [obj.astype(np.float32)], [img.astype(np.float32)], IMAGE_WH, k0, np.zeros(5), flags=flags
    )
    return float(rms), Camera(k, dist.ravel(), cv2.Rodrigues(rvecs[0])[0], tvecs[0].ravel())


def seed_correspondences(top, origin, photo, sample_uv, sample_xyz):
    """Rough photo <-> scan pairs: rectify with the samples' homography, SIFT against the scan top-down, RANSAC."""
    import cv2

    h_xy_to_uv, _ = cv2.findHomography(sample_xyz[:, :2], sample_uv)
    x0, x1, y0, y1 = SEED_BOUNDS
    px_to_xy = np.array([[0, 1 / PX_PER_M, x0], [1 / PX_PER_M, 0, y0], [0, 0, 1]])  # (col, row, 1) -> (x, y, 1)
    plate_to_photo = h_xy_to_uv @ px_to_xy
    size = (int((y1 - y0) * PX_PER_M), int((x1 - x0) * PX_PER_M))
    plate = cv2.warpPerspective(photo, plate_to_photo, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    sift = cv2.SIFT_create(nfeatures=40000, contrastThreshold=0.01)
    kp_t, des_t = sift.detectAndCompute(_gray(top), (top.max(-1) > 0).astype(np.uint8) * 255)
    kp_p, des_p = sift.detectAndCompute(_gray(plate), (plate.max(-1) > 0).astype(np.uint8) * 255)
    pairs = [a for a, b in cv2.BFMatcher().knnMatch(des_t, des_p, k=2) if a.distance < 0.85 * b.distance]
    src = np.float64([kp_t[m.queryIdx].pt for m in pairs])
    dst = np.float64([kp_p[m.trainIdx].pt for m in pairs])
    _, inl = cv2.findHomography(src, dst, cv2.RANSAC, 25.0)
    inl = inl.ravel().astype(bool)
    photo_px = np.c_[dst[inl], np.ones(inl.sum())] @ plate_to_photo.T
    return _top_to_plane(src[inl], origin), photo_px[:, :2] / photo_px[:, 2:]


def _top_to_plane(col_row, origin):
    return np.c_[origin[0] + col_row[:, 1] / PX_PER_M, origin[1] + col_row[:, 0] / PX_PER_M, np.zeros(len(col_row))]


def dense_correspondences(top, origin, photo, camera):
    """Patch matches between the scan top-down and the photo re-projected with `camera` (both in top-down pixels)."""
    import cv2

    warped, pix = camera.remap_plane(photo, origin, top.shape[:2])
    valid = (pix[..., 0] >= 0) & (top.max(-1) > 0)
    g_top, g_warp = _grad(top), _grad(warped)
    reach = PATCH + SEARCH
    src, dst = [], []
    for r in range(reach, top.shape[0] - reach, STEP):
        for c in range(reach, top.shape[1] - reach, STEP):
            if valid[r - reach : r + reach, c - reach : c + reach].mean() < 0.97:
                continue
            tpl = g_top[r - PATCH : r + PATCH, c - PATCH : c + PATCH]
            if tpl.std() < 8:
                continue
            score = cv2.matchTemplate(g_warp[r - reach : r + reach, c - reach : c + reach], tpl, cv2.TM_CCOEFF_NORMED)
            _, best, _, (x, y) = cv2.minMaxLoc(score)
            if best < 0.55 or not (0 < x < 2 * SEARCH and 0 < y < 2 * SEARCH):
                continue
            left, mid, right = score[y, x - 1], score[y, x], score[y, x + 1]
            up, down = score[y - 1, x], score[y + 1, x]
            dx = 0.5 * (left - right) / (left - 2 * mid + right - 1e-9)
            dy = 0.5 * (up - down) / (up - 2 * mid + down - 1e-9)
            src.append((c, r))
            dst.append((c + x - SEARCH + dx, r + y - SEARCH + dy))
    src, dst = np.float64(src), np.float64(dst)
    keep = np.linalg.norm(src - dst, axis=1) < 12
    photo_px = np.stack(
        [cv2.remap(pix[..., i], dst[keep, :1].astype(np.float32), dst[keep, 1:].astype(np.float32), cv2.INTER_LINEAR).ravel()
         for i in (0, 1)], -1
    )
    return _top_to_plane(src[keep], origin), photo_px, warped


def umeyama_2d(src, dst):
    """Scale, rotation, translation with dst ~ s R src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    cov = (dst - mu_d).T @ (src - mu_s) / len(src)
    u, sig, vt = np.linalg.svd(cov)
    fix = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    rot = u @ fix @ vt
    scale = float(np.trace(np.diag(sig) @ fix) / ((src - mu_s) ** 2).sum(1).mean())
    return scale, rot, mu_d - scale * rot @ mu_s


def checker(a, b, tile=80):
    out = a.copy()
    yy, xx = np.mgrid[: a.shape[0], : a.shape[1]]
    sel = (((yy // tile) + (xx // tile)) % 2 == 0) & (b.max(-1) > 0)
    out[sel] = b[sel]
    return out


def usd_camera(camera, width=IMAGE_WH[0], height=IMAGE_WH[1], aperture_mm=20.955):
    """USD row-major camera-to-world matrix and pinhole lens (USD looks down -Z with +Y up; no distortion)."""
    rows = np.eye(4)
    rows[:3, :3] = (camera.rot.T @ np.diag([1.0, -1.0, -1.0])).T
    rows[3, :3] = -camera.rot.T @ camera.t
    return {"matrix": rows.tolist(), "focal_mm": float(camera.k[0, 0] * aperture_mm / width), "horizontal_aperture": aperture_mm,
            "vertical_aperture": aperture_mm * height / width}


def align(glb, photo_path, out_dir, rounds=4) -> None:
    import cv2
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    os.makedirs(out_dir, exist_ok=True)
    pos, uv, faces, jpg = load_glb(glb)
    tex_path = os.path.join(out_dir, "scan_texture.jpg")
    open(tex_path, "wb").write(jpg)
    to_level, plane_std = level(pos)
    levelled = pos @ to_level[:3, :3].T + to_level[:3, 3]
    top, origin = top_down(levelled, uv, faces, np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB")))
    photo = cv2.cvtColor(cv2.imread(photo_path), cv2.COLOR_BGR2RGB)
    if photo.shape[1::-1] != IMAGE_WH:
        raise ValueError(f"photo is {photo.shape[1::-1]}, the calibration samples are for {IMAGE_WH}")
    sample_uv, sample_xyz = load_samples()

    # camera in the scan frame
    obj, img = seed_correspondences(top, origin, photo, sample_uv, sample_xyz)
    k = np.array([[1000.0, 0, IMAGE_WH[0] / 2], [0, 1000.0, IMAGE_WH[1] / 2], [0, 0, 1]])
    rms, cam = _calibrate(obj, img, k)
    print(f"[align] seed: {len(obj)} pairs, rms {rms:.1f} px", flush=True)
    for i in range(rounds):
        obj, img, warped = dense_correspondences(top, origin, photo, cam)
        rms, cam = _calibrate(obj, img, cam.k)
        print(f"[align] round {i + 1}: {len(obj)} patches, rms {rms:.1f} px, f {cam.k[0, 0]:.0f}, "
              f"k1 {cam.dist[0]:.3f}, k2 {cam.dist[1]:.3f}", flush=True)

    # robot frame from the TCP-touched samples, cast onto the scan's table plane
    centre, rays = cam.rays(sample_uv)
    lam = (sample_xyz[:, 2] - centre[2]) / rays[:, 2]
    scan_xy = centre[:2] + lam[:, None] * rays[:, :2]
    scale, rot2, t2 = umeyama_2d(scan_xy, sample_xyz[:, :2])
    resid = np.linalg.norm(scale * scan_xy @ rot2.T + t2 - sample_xyz[:, :2], axis=1)
    sim = np.eye(4)
    sim[:2, :2], sim[:2, 3], sim[2, 2] = scale * rot2, t2, scale
    glb_to_table = sim @ to_level

    # camera in the table frame: X_cam = R_s X_scan + t_s, X_table = s R3 X_scan + t3 (metric camera: multiply by s)
    rot3 = np.eye(3)
    rot3[:2, :2] = rot2
    t3 = np.r_[t2, 0.0]
    cam_table = Camera(cam.k, cam.dist, cam.rot @ rot3.T, scale * cam.t - cam.rot @ rot3.T @ t3)

    points = (pos @ glb_to_table[:3, :3].T + glb_to_table[:3, 3]).astype(np.float32)
    st = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]])[faces].astype(np.float32)  # glTF v runs down, USD st up
    np.savez_compressed(os.path.join(out_dir, "table.npz"), points=points, faces=faces.astype(np.int32), st=st,
                        jpg=np.array(os.path.abspath(tex_path)))
    Image.fromarray(checker(warped, top)).save(os.path.join(out_dir, "overlay_checker.png"))
    Image.fromarray(cv2.undistort(photo, cam.k, cam.dist)).save(os.path.join(out_dir, "video0_undistorted.png"))
    report = {
        "glb": os.path.abspath(glb), "photo": os.path.abspath(photo_path), "glb_to_table": glb_to_table.tolist(),
        "scale": scale, "yaw_deg": float(np.degrees(np.arctan2(rot2[1, 0], rot2[0, 0]))),
        "table_plane_std_mm": 1000 * plane_std,
        "samples": {"n": len(resid), "rms_mm": float(1000 * np.sqrt((resid**2).mean())), "max_mm": float(1000 * resid.max()),
                    "residual_mm": (1000 * resid).round(1).tolist()},
        "camera": cam_table.to_json() | {"rms_px": rms, "patches": len(obj), "image_wh": list(IMAGE_WH),
                                         "usd": usd_camera(cam_table)},
    }
    json.dump(report, open(os.path.join(out_dir, "alignment.json"), "w"), indent=1)
    print(f"[align] plane std {1000 * plane_std:.1f} mm; robot samples rms {report['samples']['rms_mm']:.1f} mm, "
          f"max {report['samples']['max_mm']:.1f} mm; scale {scale:.4f}, yaw {report['yaw_deg']:.2f} deg; camera centre "
          f"{np.round(cam_table.to_json()['centre_table'], 3).tolist()}", flush=True)
    print(f"[align] wrote {out_dir}", flush=True)


def export_overhead(out_dir, photo_path=None, camera_json=OVERHEAD_JSON, texture_png=TABLE_TEXTURE) -> None:
    """Write the calibrated video0 camera as the twin's overhead camera and re-project the table texture with it."""
    import shutil

    import cv2

    report = json.load(open(os.path.join(out_dir, "alignment.json"), encoding="utf-8"))
    cam_json = report["camera"]
    camera = Camera(cam_json["K"], cam_json["dist"], cam_json["R_cam_table"], cam_json["t_cam_table"])
    backup = os.path.splitext(camera_json)[0] + ".old.json"
    if os.path.isfile(camera_json) and not os.path.isfile(backup):
        shutil.copyfile(camera_json, backup)
    pose = np.eye(4)
    pose[:3, :3] = camera.rot.T  # columns: camera x (right), y (down), z (look) in the table frame
    pose[:3, 3] = cam_json["centre_table"]
    photo_path = photo_path or report["photo"]
    photo = cv2.cvtColor(cv2.imread(photo_path), cv2.COLOR_BGR2RGB)
    x0, x1, y0, y1 = TEXTURE_BOUNDS
    shape = (int(round((x1 - x0) * PX_PER_M)), int(round((y1 - y0) * PX_PER_M)))
    texture, pix = camera.remap_plane(photo, (x0, y0), shape)
    inside = (pix[..., 0] >= 0) & (pix[..., 0] <= IMAGE_WH[0] - 1) & (pix[..., 1] >= 0) & (pix[..., 1] <= IMAGE_WH[1] - 1)
    covered = float(inside.mean())
    cv2.imwrite(texture_png, cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
    doc = {
        "_comment": "video0 calibrated against the table scan and the TCP-touched samples by twin/align_scan.py. "
                    "T_cam_table = camera pose in the table frame (OpenCV axes). 'usd' is the undistorted pinhole.",
        "image_hw": [IMAGE_WH[1], IMAGE_WH[0]],
        "K": cam_json["K"],
        "dist": cam_json["dist"],
        "T_cam_table": pose.tolist(),
        "tilt_deg": float(np.degrees(np.arccos(-pose[2, 2]))),
        "rms_px": cam_json["rms_px"],
        "samples_rms_mm": report["samples"]["rms_mm"],
        "usd": cam_json["usd"],
        "texture_bounds": list(TEXTURE_BOUNDS),
        "texture_px_per_m": PX_PER_M,
        "texture_covered": covered,
        "source_image": os.path.abspath(photo_path),
        "alignment": os.path.abspath(os.path.join(out_dir, "alignment.json")),
    }
    json.dump(doc, open(camera_json, "w", encoding="utf-8"), indent=1)
    json.dump({"bounds": list(TEXTURE_BOUNDS), "px_per_m": PX_PER_M, "frames": 1, "covered": covered,
               "camera": os.path.abspath(camera_json)},
              open(os.path.splitext(texture_png)[0] + ".json", "w", encoding="utf-8"), indent=1)
    print(f"[export] {camera_json} (old copy: {backup}); tilt {doc['tilt_deg']:.1f} deg; "
          f"{texture_png} covered {100 * covered:.0f}%", flush=True)


def _add_scan(stage, npz_path, lit):
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    data = np.load(npz_path, allow_pickle=True)
    points, faces, st = data["points"], data["faces"], data["st"]
    mesh = UsdGeom.Mesh.Define(stage, "/World/scan_table")
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([int(i) for i in faces.reshape(-1)])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying).Set(
        [Gf.Vec2f(float(u), float(v)) for u, v in st.reshape(-1, 2)]
    )
    root = "/World/Looks/scan_table"
    mat = UsdShade.Material.Define(stage, root)
    surf = UsdShade.Shader.Define(stage, root + "/Shader")
    surf.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, root + "/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex = UsdShade.Shader.Define(stage, root + "/tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(data["jpg"])))
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    # The scan texture already holds the room's lighting: by default show it as captured (emissive), --lit shades it again.
    surf.CreateInput("diffuseColor" if lit else "emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.ConnectableAPI(), "rgb"
    )
    if not lit:
        surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


def render(out_dir, lit=False, show_robot=False, joints_deg=None, photo_path=None) -> None:
    """`joints_deg`: pendant J1..J6 to pose the twin robot (implies show_robot). `photo_path`: a video0 frame to
    compare with instead of the alignment photo; it is undistorted with the calibrated lens."""
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    report = json.load(open(os.path.join(out_dir, "alignment.json"), encoding="utf-8"))
    app = SimulationApp({"headless": True, "width": RENDER_WH[0], "height": RENDER_WH[1]})
    try:
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image
        from pxr import Gf, UsdGeom

        ctx = omni.usd.get_context()
        ctx.open_stage(os.path.abspath(STAGE))
        for _ in range(15):
            app.update()
        stage = ctx.get_stage()
        hidden = ["/World/ground", "/World/status_light", "/World/trail"] + ([] if show_robot else ["/World/fanuc"])
        for path in hidden:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
        _add_scan(stage, os.path.join(out_dir, "table.npz"), lit)
        if joints_deg is not None:
            from joint_map import FINGER_OPEN_M, fanuc_to_model
            from omni_twin import PxrWriter
            from usd_chain import Chain

            chain = Chain.load(os.path.join(HERE, "..", "fanuc_lrmate200id_smc.twin.json"))
            PxrWriter(stage, chain).set_pose(chain.joint_values(fanuc_to_model(joints_deg, "coupled"), FINGER_OPEN_M))
        lens = report["camera"]["usd"]
        cam = UsdGeom.Camera.Define(stage, "/World/Video0Camera")
        xf = UsdGeom.Xformable(cam.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(Gf.Matrix4d(lens["matrix"]))
        cam.GetFocalLengthAttr().Set(float(lens["focal_mm"]))
        cam.GetHorizontalApertureAttr().Set(float(lens["horizontal_aperture"]))
        cam.GetVerticalApertureAttr().Set(float(lens["vertical_aperture"]))
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 10.0))
        product = rep.create.render_product("/World/Video0Camera", RENDER_WH)
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([product])
        for _ in range(20):
            app.update()
            rep.orchestrator.step()
        image = np.ascontiguousarray(np.asarray(rgb.get_data())[:, :, :3], dtype=np.uint8)
        suffix = "_robot" if show_robot else ""
        Image.fromarray(image).save(os.path.join(out_dir, f"render_video0{suffix}.png"))
        # The USD camera is a pinhole: compare against the undistorted photo.
        if photo_path:
            import cv2

            raw = cv2.cvtColor(cv2.imread(photo_path), cv2.COLOR_BGR2RGB)
            undistorted = cv2.undistort(raw, np.array(report["camera"]["K"]), np.array(report["camera"]["dist"]))
            photo = cv2.resize(undistorted, RENDER_WH, interpolation=cv2.INTER_AREA)
        else:
            photo = np.asarray(
                Image.open(os.path.join(out_dir, "video0_undistorted.png")).convert("RGB").resize(RENDER_WH, Image.BILINEAR)
            )
        blend = (0.5 * photo + 0.5 * image).astype(np.uint8)
        Image.fromarray(np.vstack([np.hstack([photo, image]), np.hstack([blend, checker(photo, image, 60)])])).save(
            os.path.join(out_dir, f"compare_video0{suffix}.png")
        )
        print(f"[render] {os.path.join(out_dir, f'compare_video0{suffix}.png')}", flush=True)
    finally:
        app.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Align a Scaniverse table scan to the table frame and preview it")
    parser.add_argument("--glb", help="Scaniverse GLB export")
    parser.add_argument("--photo", help="current 1920x1080 video0 frame of the table")
    parser.add_argument("--out", required=True)
    parser.add_argument("--render", action="store_true", help="Isaac Sim: render the aligned scan from the video0 camera")
    parser.add_argument("--lit", action="store_true", help="shade the scan texture instead of showing it as captured")
    parser.add_argument("--show-robot", action="store_true", help="keep the twin robot (at its default pose) visible")
    parser.add_argument("--joints", type=float, nargs=6, metavar="DEG", help="--render: pendant J1..J6 for the twin robot")
    parser.add_argument("--export-overhead", action="store_true",
                        help="write overhead_camera.json and table_texture.png from --out/alignment.json")
    args = parser.parse_args(argv)
    if args.export_overhead:
        export_overhead(args.out, args.photo)
    elif args.render:
        render(args.out, args.lit, args.show_robot or args.joints is not None, args.joints, args.photo)
    else:
        if not (args.glb and args.photo):
            parser.error("--glb and --photo are required to align")
        align(args.glb, args.photo, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
