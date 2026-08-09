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

"""Pixel → table-plane geometry helpers (no depth estimation)."""

from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient


def to_table(
    uv: tuple[float, float],
    K: np.ndarray,
    T_cam_table: np.ndarray,
) -> tuple[float, float]:
    """
    Intersect the camera ray through pixel ``(u, v)`` with the table plane z=0.

    ``T_cam_table`` is the camera pose expressed in the table frame
    (columns of R are camera axes in table coordinates; ``t`` is camera origin).
    """
    u, v = float(uv[0]), float(uv[1])
    k = np.asarray(K, dtype=float)
    t_ct = np.asarray(T_cam_table, dtype=float)
    if k.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {k.shape}")
    if t_ct.shape != (4, 4):
        raise ValueError(f"T_cam_table must be 4x4, got {t_ct.shape}")

    ray_cam = np.linalg.inv(k) @ np.array([u, v, 1.0], dtype=float)
    r = t_ct[:3, :3]
    t = t_ct[:3, 3]
    ray_tab = r @ ray_cam
    if ray_tab[2] >= 0.0:
        raise ValueError(
            f"Ray through pixel ({u:.1f}, {v:.1f}) does not point at the table "
            f"(ray_tab[2]={ray_tab[2]:.6f} >= 0)."
        )
    s = -t[2] / ray_tab[2]
    xy = t + s * ray_tab
    return float(xy[0]), float(xy[1])


def apply_table_xy_affine(
    xy: tuple[float, float],
    A: np.ndarray | None,
    b: np.ndarray | None,
) -> tuple[float, float]:
    """Apply optional 2D affine ``xy' = A @ xy + b`` in the table frame."""
    if A is None or b is None:
        return float(xy[0]), float(xy[1])
    a = np.asarray(A, dtype=float).reshape(2, 2)
    bb = np.asarray(b, dtype=float).reshape(2)
    out = a @ np.asarray(xy, dtype=float) + bb
    return float(out[0]), float(out[1])


def transform_polygon_xy(
    footprint: Polygon,
    A: np.ndarray | None,
    b: np.ndarray | None,
) -> Polygon:
    """Transform a table-frame footprint by an optional affine."""
    if A is None or b is None:
        return footprint
    coords = [
        apply_table_xy_affine((float(x), float(y)), A, b)
        for x, y in list(footprint.exterior.coords)[:-1]
    ]
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return orient(poly, sign=1.0) if not poly.is_empty else footprint


def fit_table_xy_affine(
    pred_xy: list[tuple[float, float]],
    true_xy: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Least-squares affine ``true = A @ pred + b`` (needs ≥3 non-collinear points).

    Returns ``(A, b, metrics)`` with residual RMSE in metres.
    """
    if len(pred_xy) != len(true_xy):
        raise ValueError("pred_xy and true_xy length mismatch")
    if len(pred_xy) < 3:
        raise ValueError("Need at least 3 correspondences for a full affine")

    pred = np.asarray(pred_xy, dtype=float)
    true = np.asarray(true_xy, dtype=float)
    # [x y 1] @ [[a11,a21], [a12,a22], [bx,by]] = [tx, ty]
    design = np.column_stack([pred[:, 0], pred[:, 1], np.ones(len(pred))])
    sol, _, _, _ = np.linalg.lstsq(design, true, rcond=None)
    a = np.array([[sol[0, 0], sol[1, 0]], [sol[0, 1], sol[1, 1]]], dtype=float)
    bb = np.array([sol[2, 0], sol[2, 1]], dtype=float)
    fitted = (design @ sol)
    err = fitted - true
    rmse = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
    max_err = float(np.max(np.linalg.norm(err, axis=1)))
    return a, bb, {"rmse_m": rmse, "max_err_m": max_err, "n": float(len(pred_xy))}


def bbox_corners_px(
    box_2d_px: tuple[float, float, float, float],
) -> list[tuple[float, float]]:
    """Return four bbox corners as (u, v) in pixel coordinates."""
    xmin, ymin, xmax, ymax = box_2d_px
    return [
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
    ]


def footprint_from_box(
    box_2d_px: tuple[float, float, float, float],
    K: np.ndarray,
    T_cam_table: np.ndarray,
    *,
    table_xy_affine: tuple[np.ndarray, np.ndarray] | None = None,
) -> Polygon:
    """Project the four bbox corners to the table and form a polygon."""
    a = b = None
    if table_xy_affine is not None:
        a, b = table_xy_affine
    corners_xy = [
        apply_table_xy_affine(to_table(uv, K, T_cam_table), a, b)
        for uv in bbox_corners_px(box_2d_px)
    ]
    poly = Polygon(corners_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0.0:
        # Degenerate projection: tiny square around the centre.
        cx = sum(p[0] for p in corners_xy) / 4.0
        cy = sum(p[1] for p in corners_xy) / 4.0
        eps = 1e-3
        poly = Polygon(
            [
                (cx - eps, cy - eps),
                (cx + eps, cy - eps),
                (cx + eps, cy + eps),
                (cx - eps, cy + eps),
            ]
        )
    return orient(poly, sign=1.0)


def yaw_from_footprint(footprint: Polygon) -> float:
    """
    Orientation of the minimum-area rectangle of ``footprint``,
    wrapped to ``[-pi/2, pi/2]``.
    """
    mrr = footprint.minimum_rotated_rectangle
    if mrr.is_empty:
        return 0.0
    coords = list(mrr.exterior.coords)
    if len(coords) < 2:
        return 0.0

    # Longest edge defines the principal axis.
    best_len = -1.0
    best_yaw = 0.0
    for i in range(len(coords) - 1):
        x0, y0 = coords[i]
        x1, y1 = coords[i + 1]
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length > best_len:
            best_len = length
            best_yaw = math.atan2(dy, dx)

    # Wrap to [-pi/2, pi/2].
    while best_yaw > math.pi / 2.0:
        best_yaw -= math.pi
    while best_yaw <= -math.pi / 2.0:
        best_yaw += math.pi
    return float(best_yaw)
