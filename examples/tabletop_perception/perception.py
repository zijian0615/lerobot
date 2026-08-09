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

from .geometry import apply_table_xy_affine, footprint_from_box, to_table, yaw_from_footprint


def resolve_grasp_height_m(
    object_name: str,
    default_height_m: float,
    height_offsets_m: dict[str, float] | None = None,
) -> float:
    """
    ``z = default + offset`` where ``offset`` is the first matching key in
    ``height_offsets_m`` (case-insensitive substring on the object name).
    Negative offset lowers the gripper.
    """
    z = float(default_height_m)
    if not height_offsets_m:
        return z
    name = object_name.lower().replace(" ", "_")
    for key, offset in height_offsets_m.items():
        if str(key).startswith("_"):
            continue
        needle = str(key).lower().replace(" ", "_")
        if needle and needle in name:
            return z + float(offset)
    return z
from .vlm import VlmCaller, call_gemini_robotics_er, parse_vlm_detections


class SymbolicObject(TypedDict):
    name: str
    blocked_by: str | None
    in_workspace: list[str]


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


class Perception:
    """
    Monocular tabletop perception.

    Assumptions:
      - single known table plane (z=0 in table frame)
      - top-down grasps at a fixed ``grasp_height``
      - geometry from 2D boxes only (no SAM / depth)
    """

    def __init__(
        self,
        *,
        footprint_buffer_m: float = 0.02,
        vlm_caller: VlmCaller | None = None,
        prompt: str | None = None,
        table_xy_affine: tuple[np.ndarray, np.ndarray] | None = None,
        grasp_height_offsets_m: dict[str, float] | None = None,
    ) -> None:
        self.footprint_buffer_m = float(footprint_buffer_m)
        self.vlm_caller: VlmCaller = vlm_caller or (
            lambda image, instruction, prompt_text: call_gemini_robotics_er(
                image,
                instruction,
                prompt=prompt_text or None,
            )
        )
        self.prompt = prompt
        self.table_xy_affine = table_xy_affine
        self.grasp_height_offsets_m = dict(grasp_height_offsets_m or {})

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

        a = b = None
        if self.table_xy_affine is not None:
            a, b = self.table_xy_affine

        # STEP 2 — pixel → table projection (+ optional affine correction).
        geometric_objects: list[GeometricObject] = []
        footprints: list[Polygon] = []
        for det in detections:
            grasp_xy = apply_table_xy_affine(
                to_table(det["grasp_point_px"], K, T_cam_table), a, b
            )
            footprint = footprint_from_box(
                det["box_2d_px"],
                K,
                T_cam_table,
                table_xy_affine=self.table_xy_affine,
            )
            yaw = yaw_from_footprint(footprint)
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
            symbolic_objects.append(
                {
                    "name": name,
                    "blocked_by": name_to_blocked.get(name),
                    "in_workspace": in_ws,
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
) -> tuple[SymbolicView, GeometricView]:
    """Functional entry point wrapping :class:`Perception`."""
    return Perception(
        footprint_buffer_m=footprint_buffer_m,
        vlm_caller=vlm_caller,
        prompt=prompt,
        table_xy_affine=table_xy_affine,
        grasp_height_offsets_m=grasp_height_offsets_m,
    ).run(
        image=image,
        K=K,
        T_cam_table=T_cam_table,
        grasp_height=grasp_height,
        instruction=instruction,
        arm_workspaces=arm_workspaces,
        table_polygon=table_polygon,
    )
