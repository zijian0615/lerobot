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

"""Perception module for monocular tabletop manipulation (no depth)."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .geometry import (
    apply_table_xy_affine,
    footprint_from_box,
    footprint_from_polygon_px,
    to_table,
    yaw_from_axis,
    yaw_from_footprint,
)


# Calib keys that also apply to other name substrings (one offset for both).
_GRASP_OFFSET_ALIASES = {
    "stand": ("ring",),
}


def matches_height_key(object_name: str, key: str) -> bool:
    """True if ``key`` (or an alias such as stand→ring) appears in the object name."""
    if str(key).startswith("_"):
        return False
    hay = object_name.lower().replace(" ", "_")
    needle = str(key).lower().replace(" ", "_")
    if not needle:
        return False
    if needle in hay:
        return True
    return any(alias in hay for alias in _GRASP_OFFSET_ALIASES.get(needle, ()))


def resolve_grasp_height_m(
    object_name: str,
    default_height_m: float,
    height_offsets_m: dict[str, float] | None = None,
) -> float:
    """
    ``z = default + offset`` where ``offset`` is the first matching key in
    ``height_offsets_m`` (case-insensitive substring on the object name).
    ``stand`` also matches names containing ``ring``. Negative offset lowers
    the gripper.
    """
    z = float(default_height_m)
    if not height_offsets_m:
        return z
    for key, offset in height_offsets_m.items():
        if matches_height_key(object_name, str(key)):
            return z + float(offset)
    return z


def resolve_object_top_z_m(
    object_name: str,
    heights_m: dict[str, float] | None = None,
    default_m: float = 0.0,
) -> float:
    """Visible top-face height used when unprojecting grasp pixels.

    This is the object's physical height above the table, not the gripper
    approach height. Keys match as case-insensitive substrings (same as
    :func:`resolve_grasp_height_m`). Unmatched names use ``default_m``.
    """
    if not heights_m:
        return float(default_m)
    for key, height in heights_m.items():
        if matches_height_key(object_name, str(key)):
            return float(height)
    return float(default_m)


def object_top_z_from_calib(calib: dict) -> tuple[float, dict[str, float]]:
    """Load ``(default_m, {name: height_m})`` from a calib dict."""
    default_m = float(calib.get("object_top_z_m_default", 0.0))
    raw = dict(calib.get("object_top_z_m") or {})
    heights = {
        str(k): float(v)
        for k, v in raw.items()
        if not str(k).startswith("_")
    }
    return default_m, heights


from .segment import refine_parsed_detections
from .vlm import VlmCaller, call_detection_vlm, parse_vlm_detections


class SymbolicObject(TypedDict):
    name: str
    blocked_by: str | None
    in_workspace: list[str]
    preferred_arm: str | None


class SymbolicView(TypedDict):
    objects: list[SymbolicObject]


class GeometricObject(TypedDict):
    name: str
    xy: tuple[float, float]
    yaw: float
    footprint: Polygon
    grasp_pose: tuple[float, float, float, float]  # x, y, z, yaw


class GeometricView(TypedDict):
    objects: list[GeometricObject]
    free_space: Polygon
    table_polygon: Polygon
    workspaces: dict[str, Polygon]


def resolve_preferred_arm(
    xy: tuple[float, float],
    candidates: list[str],
    arm_workspaces: dict[str, Polygon],
    table_polygon: Polygon | None = None,
    *,
    center_margin_m: float = 0.03,
) -> str | None:
    """
    Among reachable arms, pick who should preferably Grasp this object.

    1. If the object is clearly on one side of the table center (along the
       axis joining the two farthest arm centroids), prefer that side's arm.
    2. Otherwise (near center / ties): prefer the arm whose workspace
       centroid is nearest to the object.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    pt = np.asarray(xy, dtype=float)
    cents = {
        a: np.asarray(
            [arm_workspaces[a].centroid.x, arm_workspaces[a].centroid.y],
            dtype=float,
        )
        for a in candidates
        if a in arm_workspaces
    }
    if len(cents) < len(candidates):
        # Missing workspace poly — fall back to first reachable.
        return candidates[0]
    pool = list(candidates)

    if table_polygon is not None and not table_polygon.is_empty and len(pool) >= 2:
        tc = np.asarray(
            [table_polygon.centroid.x, table_polygon.centroid.y], dtype=float
        )
        best_pair: tuple[str, str] | None = None
        best_d = -1.0
        for i, a0 in enumerate(pool):
            for a1 in pool[i + 1 :]:
                d = float(np.linalg.norm(cents[a0] - cents[a1]))
                if d > best_d:
                    best_d = d
                    best_pair = (a0, a1)
        if best_pair is not None and best_d > 1e-6:
            a0, a1 = best_pair
            axis = cents[a1] - cents[a0]
            axis = axis / float(np.linalg.norm(axis))
            o_side = float(np.dot(pt - tc, axis))
            if abs(o_side) >= float(center_margin_m):
                same = [
                    a
                    for a in pool
                    if float(np.dot(cents[a] - tc, axis)) * o_side > 0
                ]
                if len(same) == 1:
                    return same[0]
                if len(same) > 1:
                    pool = same

    return min(pool, key=lambda a: float(np.sum((pt - cents[a]) ** 2)))


class Perception:
    """
    Monocular tabletop perception.

    Assumptions:
      - single known table plane (z=0 in table frame)
      - object pixels are unprojected onto ``z = object_top_z`` (class height),
        not the table; table_polygon stays at z=0
      - top-down grasps at a fixed ``grasp_height`` (gripper z, separate)
      - VLM box is only a ROI hint; yaw/footprint come from an image mask
        min-area rectangle when segmentation succeeds
    """

    def __init__(
        self,
        *,
        footprint_buffer_m: float = 0.02,
        vlm_caller: VlmCaller | None = None,
        prompt: str | None = None,
        table_xy_affine: tuple[np.ndarray, np.ndarray] | None = None,
        grasp_height_offsets_m: dict[str, float] | None = None,
        object_top_z_m: dict[str, float] | None = None,
        object_top_z_m_default: float = 0.0,
        thinking_budget: int = -1,
        model: str = "gemini",
    ) -> None:
        self.footprint_buffer_m = float(footprint_buffer_m)
        self.thinking_budget = int(thinking_budget)
        self.model = str(model)
        self.vlm_caller: VlmCaller = vlm_caller or (
            lambda image, instruction, prompt_text: call_detection_vlm(
                image,
                instruction,
                prompt=prompt_text or None,
                model=self.model,
                thinking_budget=self.thinking_budget,
            )
        )
        self.prompt = prompt
        self.table_xy_affine = table_xy_affine
        self.grasp_height_offsets_m = dict(grasp_height_offsets_m or {})
        self.object_top_z_m = dict(object_top_z_m or {})
        self.object_top_z_m_default = float(object_top_z_m_default)
        self.last_detections: list = []

    def __call__(
        self,
        image: np.ndarray,
        K: np.ndarray,
        T_cam_table: np.ndarray,
        grasp_height: float,
        instruction: str,
        arm_workspaces: dict[str, Polygon],
        table_polygon: Polygon,
    ) -> tuple[SymbolicView, GeometricView]:
        return self.run(
            image=image,
            K=K,
            T_cam_table=T_cam_table,
            grasp_height=grasp_height,
            instruction=instruction,
            arm_workspaces=arm_workspaces,
            table_polygon=table_polygon,
        )

    def run(
        self,
        image: np.ndarray,
        K: np.ndarray,
        T_cam_table: np.ndarray,
        grasp_height: float,
        instruction: str,
        arm_workspaces: dict[str, Polygon],
        table_polygon: Polygon,
    ) -> tuple[SymbolicView, GeometricView]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must be HxWx3 RGB, got shape={image.shape}")
        height, width = int(image.shape[0]), int(image.shape[1])

        # STEP 1 — VLM call (swappable).
        # Empty / None prompt lets the default VLM caller use prompts.DEFAULT_DETECTION_PROMPT.
        prompt_text = self.prompt if self.prompt is not None else ""
        raw = self.vlm_caller(image, instruction, prompt_text)
        detections = parse_vlm_detections(raw, image_hw=(height, width))
        snap_blobs = "cosmos" in str(self.model).lower()
        detections = refine_parsed_detections(image, detections, snap_blobs=snap_blobs)
        self.last_detections = detections

        a = b = None
        if self.table_xy_affine is not None:
            a, b = self.table_xy_affine

        # STEP 2 — pixel → table projection (+ optional affine correction).
        geometric_objects: list[GeometricObject] = []
        footprints: list[Polygon] = []
        for det in detections:
            z_top = resolve_object_top_z_m(
                det["name"], self.object_top_z_m, self.object_top_z_m_default
            )
            grasp_xy = apply_table_xy_affine(
                to_table(det["grasp_point_px"], K, T_cam_table, z_plane=z_top), a, b
            )
            poly_px = det.get("polygon_px")
            footprint = None
            if poly_px:
                footprint = footprint_from_polygon_px(
                    list(poly_px),
                    K,
                    T_cam_table,
                    table_xy_affine=self.table_xy_affine,
                    z_plane=z_top,
                )
            if footprint is None:
                footprint = footprint_from_box(
                    det["box_2d_px"],
                    K,
                    T_cam_table,
                    table_xy_affine=self.table_xy_affine,
                    z_plane=z_top,
                )
            yaw = yaw_from_footprint(footprint)
            axis_px = det.get("long_axis_px")
            if axis_px is not None:
                p0 = apply_table_xy_affine(
                    to_table(axis_px[0], K, T_cam_table, z_plane=z_top), a, b
                )
                p1 = apply_table_xy_affine(
                    to_table(axis_px[1], K, T_cam_table, z_plane=z_top), a, b
                )
                axis_yaw = yaw_from_axis(p0, p1)
                if axis_yaw is not None:
                    yaw = axis_yaw
            z = resolve_grasp_height_m(
                det["name"], grasp_height, self.grasp_height_offsets_m
            )
            footprints.append(footprint)
            geometric_objects.append(
                {
                    "name": det["name"],
                    "xy": grasp_xy,
                    "yaw": yaw,
                    "footprint": footprint,
                    "grasp_pose": (grasp_xy[0], grasp_xy[1], z, yaw),
                }
            )

        # STEP 3 — predicates.
        name_to_blocked = {det["name"]: det["blocked_by"] for det in detections}
        symbolic_objects: list[SymbolicObject] = []
        for geo in geometric_objects:
            name = geo["name"]
            grasp_pt = Point(geo["xy"])
            in_ws = [
                arm
                for arm, poly in arm_workspaces.items()
                if grasp_pt.within(poly)
            ]
            preferred = resolve_preferred_arm(
                geo["xy"], in_ws, arm_workspaces, table_polygon
            )
            # Put preferred arm first so planners that scan the list bias correctly.
            if preferred is not None and preferred in in_ws:
                in_ws = [preferred] + [a for a in in_ws if a != preferred]
            symbolic_objects.append(
                {
                    "name": name,
                    "blocked_by": name_to_blocked.get(name),
                    "in_workspace": in_ws,
                    "preferred_arm": preferred,
                }
            )

        free_space = _compute_free_space(
            table_polygon=table_polygon,
            footprints=footprints,
            margin_m=self.footprint_buffer_m,
        )

        # STEP 4 — dual views.
        symbolic_view: SymbolicView = {"objects": symbolic_objects}
        geometric_view: GeometricView = {
            "objects": geometric_objects,
            "free_space": free_space,
            "table_polygon": table_polygon,
            "workspaces": dict(arm_workspaces),
        }
        return symbolic_view, geometric_view


def _compute_free_space(
    table_polygon: Polygon,
    footprints: list[Polygon],
    margin_m: float,
) -> Polygon:
    if not footprints:
        return table_polygon
    buffered = [fp.buffer(margin_m) for fp in footprints]
    occupied = unary_union(buffered)
    free = table_polygon.difference(occupied)
    if free.is_empty:
        return Polygon()
    return free


def run_perception(
    image: np.ndarray,
    K: np.ndarray,
    T_cam_table: np.ndarray,
    grasp_height: float,
    instruction: str,
    arm_workspaces: dict[str, Polygon],
    table_polygon: Polygon,
    *,
    footprint_buffer_m: float = 0.02,
    vlm_caller: VlmCaller | None = None,
    prompt: str | None = None,
    table_xy_affine: tuple[np.ndarray, np.ndarray] | None = None,
    grasp_height_offsets_m: dict[str, float] | None = None,
    object_top_z_m: dict[str, float] | None = None,
    object_top_z_m_default: float = 0.0,
    thinking_budget: int = -1,
    model: str = "gemini",
) -> tuple[SymbolicView, GeometricView]:
    """Functional entry point wrapping :class:`Perception`."""
    return Perception(
        footprint_buffer_m=footprint_buffer_m,
        vlm_caller=vlm_caller,
        prompt=prompt,
        table_xy_affine=table_xy_affine,
        grasp_height_offsets_m=grasp_height_offsets_m,
        object_top_z_m=object_top_z_m,
        object_top_z_m_default=object_top_z_m_default,
        thinking_budget=thinking_budget,
        model=model,
    ).run(
        image=image,
        K=K,
        T_cam_table=T_cam_table,
        grasp_height=grasp_height,
        instruction=instruction,
        arm_workspaces=arm_workspaces,
        table_polygon=table_polygon,
    )
