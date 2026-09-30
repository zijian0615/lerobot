"""Overhead camera for the current /dev/video0 view, and the table photo projected onto z = 0.

SUPERSEDED by twin/align_scan.py (--export-overhead), which writes overhead_camera.json from the table scan and the
TCP-touched samples, lens distortion included. This fit assumed the yellow borders at y = +-0.35 m; the table is about
1.1 x 1.5 m, so its camera was off by ~400 px on the samples. Running write_assets() overwrites the calibrated file.

The stored fanuc_overhead.json K and T_cam_table are a nadir placeholder. The webcam is tilted:
the yellow side borders (table y = ±0.35) converge above the frame. This solves that pinhole
(square pixels, principal point at the image centre) and lays the photo onto the whole table.

    .venv/bin/python twin/overhead_camera.py --image /tmp/video0_now.jpg
"""

from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAM_JSON = os.path.join(HERE, "..", "overhead_camera.json")
TEXTURE_PNG = os.path.join(HERE, "..", "table_texture.png")
TABLE = (-0.45, 0.45, -0.35, 0.35)  # calibrated table polygon, metres
BORDER_Y = 0.35
FOCAL_MM = 24.0


def _pose(theta: float, tx: float, ty: float, tz: float):
    s, c = np.sin(theta), np.cos(theta)
    right = np.array([0.0, 1.0, 0.0])          # image right = table +y
    look = np.array([-s, 0.0, -c])             # toward -x (the robot) and down
    down = np.cross(look, right)               # OpenCV +Y
    rotation = np.c_[right, down, look]        # columns are camera axes in the table frame
    return rotation, np.array([tx, ty, tz], float)


def load_camera(path: str = CAM_JSON) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def usd_matrix(rotation: np.ndarray, eye: np.ndarray) -> np.ndarray:
    """USD camera looks along -Z, +Y up. Row-vector matrix, translation in the last row."""
    rows = np.zeros((4, 4))
    rows[0, :3] = rotation[:, 0]
    rows[1, :3] = -rotation[:, 1]
    rows[2, :3] = -rotation[:, 2]
    rows[3, :3] = eye
    rows[3, 3] = 1.0
    return rows


def project_photo(image: np.ndarray, rotation: np.ndarray, eye: np.ndarray, focal_px: float, bounds, px_per_m: int):
    """Lay a 1920x1080 photo onto z = 0. Rows run along +x, columns along +y. Unseen texels stay white."""
    x0, x1, y0, y1 = bounds
    height, width = image.shape[:2]
    rows = int(round((x1 - x0) * px_per_m))
    cols = int(round((y1 - y0) * px_per_m))
    xs = x0 + (np.arange(rows) + 0.5) / px_per_m
    ys = y0 + (np.arange(cols) + 0.5) / px_per_m
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    points = np.c_[grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)]
    camera = (points - eye) @ rotation
    valid = camera[:, 2] > 0.05
    u = focal_px * camera[:, 0] / np.maximum(camera[:, 2], 1e-6) + width / 2
    v = focal_px * camera[:, 1] / np.maximum(camera[:, 2], 1e-6) + height / 2
    valid &= (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    out = np.full((rows, cols, 3), 248, np.uint8)
    if valid.any():
        import cv2

        mapped = cv2.remap(
            image,
            u.reshape(rows, cols).astype(np.float32),
            v.reshape(rows, cols).astype(np.float32),
            cv2.INTER_LINEAR,
            borderValue=(248, 248, 248),
        )
        out[valid.reshape(rows, cols)] = mapped[valid.reshape(rows, cols)]
    return out, float(valid.mean())


def fit_borders(image: np.ndarray) -> dict:
    """Solve tilt, position and focal length so the yellow borders land on y = ±0.35."""
    import cv2
    from scipy.optimize import least_squares

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, (15, 50, 100), (42, 255, 255))
    ys, xs = np.where(yellow > 0)
    left = np.c_[xs[(xs < 450) & (ys > 40) & (ys < 750)][::8], ys[(xs < 450) & (ys > 40) & (ys < 750)][::8]]
    right = np.c_[xs[(xs > 1500) & (ys < 900)][::8], ys[(xs > 1500) & (ys < 900)][::8]]
    left, right = left.astype(float), right.astype(float)

    def unproject(uv, rotation, eye, focal):
        ray = np.c_[(uv[:, 0] - width / 2) / focal, (uv[:, 1] - height / 2) / focal, np.ones(len(uv))]
        ray_table = ray @ rotation.T
        scale = -eye[2] / ray_table[:, 2]
        return eye + scale[:, None] * ray_table, scale

    def residual(params):
        theta, tx, ty, tz, focal = params
        # The optical axis falls on the open table, in front of the base. A fit that
        # sits over the robot satisfies the borders by collapsing the view.
        if not (np.radians(18) < theta < np.radians(50) and 900 < focal < 2600 and 0.55 < tz < 1.4):
            return np.full(6, 50.0)
        if not (0.35 < tx < 0.8 and abs(ty) < 0.15):
            return np.full(6, 50.0)
        rotation, eye = _pose(theta, tx, ty, tz)
        direction = rotation.T @ np.array([-1.0, 0.0, 0.0])
        if direction[2] <= 0.05:
            return np.full(6, 50.0)
        vanishing = np.array([focal * direction[0] / direction[2] + width / 2, focal * direction[1] / direction[2] + height / 2])
        xy_l, scale_l = unproject(left, rotation, eye, focal)
        xy_r, scale_r = unproject(right, rotation, eye, focal)
        if np.any(scale_l < 0.05) or np.any(scale_r < 0.05):
            return np.full(6, 50.0)
        err_l = (xy_l[:, 1] + BORDER_Y) * 1000
        err_r = (xy_r[:, 1] - BORDER_Y) * 1000
        return np.r_[
            err_l[:: max(1, len(err_l) // 40)] / 15,
            err_r[:: max(1, len(err_r) // 40)] / 15,
            (vanishing - np.array([949.0, -2280.0])) / 40,
        ]

    best = None
    for tilt in (25, 32, 37, 42):
        for tz in (0.65, 0.85, 1.05):
            for tx in (0.40, 0.55, 0.70):
                focal0 = float(np.clip(2820 * np.tan(np.radians(tilt)), 700, 2600))
                solved = least_squares(residual, [np.radians(tilt), tx, 0.0, tz, focal0], max_nfev=50)
                if best is None or solved.cost < best.cost:
                    best = solved
    theta, tx, ty, tz, focal = (float(v) for v in best.x)
    rotation, eye = _pose(theta, tx, ty, tz)
    corners = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], float)
    footprint, _ = unproject(corners, rotation, eye, focal)
    # The quad is the whole calibrated table. Image-corner rays that graze the plane
    # must not stretch it; unseen texels stay the table white.
    bounds = TABLE
    horizontal = FOCAL_MM * width / focal
    vertical = FOCAL_MM * height / focal
    matrix = usd_matrix(rotation, eye)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = eye
    return {
        "image_hw": [height, width],
        "K": [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]],
        "T_cam_table": transform.tolist(),
        "tilt_deg": float(np.degrees(theta)),
        "border_y_m": BORDER_Y,
        "fit_cost": float(best.cost),
        "texture_bounds": list(bounds),
        "image_corners_xy": np.round(footprint[:, :2], 4).tolist(),
        "usd": {
            "matrix": matrix.tolist(),
            "focal_mm": FOCAL_MM,
            "horizontal_aperture": horizontal,
            "vertical_aperture": vertical,
        },
    }


def write_assets(image_path: str, out_json: str = CAM_JSON, out_png: str = TEXTURE_PNG, px_per_m: int = 800) -> dict:
    import cv2
    from PIL import Image

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(image_path)
    camera = fit_borders(image)
    rotation = np.array(camera["T_cam_table"], float)[:3, :3]
    eye = np.array(camera["T_cam_table"], float)[:3, 3]
    focal = camera["K"][0][0]
    texture, covered = project_photo(image, rotation, eye, focal, camera["texture_bounds"], px_per_m)
    Image.fromarray(cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)).save(out_png)
    camera["source_image"] = os.path.abspath(image_path)
    camera["texture_px_per_m"] = px_per_m
    camera["texture_covered"] = covered
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(camera, handle, indent=1)
    sidecar = os.path.splitext(out_png)[0] + ".json"
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "bounds": camera["texture_bounds"],
                "px_per_m": px_per_m,
                "frames": 1,
                "covered": covered,
                "camera": os.path.abspath(out_json),
            },
            handle,
            indent=1,
        )
    print(
        f"tilt {camera['tilt_deg']:.1f} deg  eye {np.round(eye, 3).tolist()}  f {focal:.0f}px  "
        f"texture {texture.shape[1]}x{texture.shape[0]} covered {100 * covered:.0f}%"
    )
    return camera


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-json", default=CAM_JSON)
    parser.add_argument("--out-png", default=TEXTURE_PNG)
    args = parser.parse_args(argv)
    write_assets(args.image, args.out_json, args.out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
