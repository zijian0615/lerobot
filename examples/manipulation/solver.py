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

"""Constraint solver: bind numeric poses to a symbolic plan (shapely only)."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

try:
    from shapely import polylabel as _polylabel
except ImportError:  # pragma: no cover
    from shapely.ops import polylabel as _polylabel


class GraspUnreachable(Exception):
    def __init__(self, step: int, message: str) -> None:
        self.step = step
        super().__init__(f"[GraspUnreachable] step {step}: {message}")


class NoLegalPlacement(Exception):
    def __init__(self, step: int, message: str) -> None:
        self.step = step
        super().__init__(f"[NoLegalPlacement] step {step}: {message}")


@dataclass(frozen=True)
class SolverConfig:
    grasp_height: float
    margin: float = 0.03
    handover_radius: float = 0.08
    polylabel_tolerance: float = 0.005
    # Destination name substring → Δz (m). Positive raises Place height.
    place_height_offsets_m: dict[str, float] = field(default_factory=dict)
    # Per-arm overrides merged on top of place_height_offsets_m for that arm.
    place_height_offsets_by_arm: dict[str, dict[str, float]] = field(default_factory=dict)
    # Per-arm object-name substring → Δz (m), applied at bind time for that arm.
    # Use for EE-specific tip offsets (e.g. LeapHand bear=-0.05; parallel gripper 0).
    grasp_height_offsets_by_arm: dict[str, dict[str, float]] = field(default_factory=dict)


def _resolve_object_height_offset_m(
    object_name: str,
    base_z_m: float,
    offsets: Mapping[str, float] | None,
) -> float:
    """``z = base + first matching substring offset`` (same rule as perception)."""
    z = float(base_z_m)
    if not offsets:
        return z
    name = object_name.lower().replace(" ", "_")
    for key, offset in offsets.items():
        if str(key).startswith("_"):
            continue
        needle = str(key).lower().replace(" ", "_")
        if needle and needle in name:
            return z + float(offset)
    return z


def _arm_object_z_m(
    *,
    arm: str,
    object_name: str,
    base_z_m: float,
    config: SolverConfig,
) -> float:
    return _resolve_object_height_offset_m(
        object_name,
        base_z_m,
        config.grasp_height_offsets_by_arm.get(arm),
    )


def _match_name_offset_m(
    name: str,
    offsets: Mapping[str, float] | None,
) -> float | None:
    """First matching substring offset for ``name``, or None if no match."""
    if not offsets:
        return None
    hay = name.lower().replace(" ", "_")
    for key, offset in offsets.items():
        if str(key).startswith("_"):
            continue
        needle = str(key).lower().replace(" ", "_")
        if needle and needle in hay:
            return float(offset)
    return None


def _resolve_place_height_m(
    destination: str,
    base_z_m: float,
    offsets: dict[str, float] | None,
) -> float:
    z = float(base_z_m)
    matched = _match_name_offset_m(destination, offsets)
    if matched is not None:
        return z + matched
    return z


def _resolve_place_z_m(
    *,
    arm: str,
    obj: str,
    dest: str,
    state: Mapping[str, ObjectState],
    config: SolverConfig,
) -> float:
    """
    Place tip height.

    - Onto a named scene object with a place_height_offset match:
      ``support.grasp_z + offset`` (so stand stacks rise each layer;
      per-arm overrides in ``place_height_offsets_by_arm`` win over global).
    - Else: held-object grasp z (+ EE offsets) + optional symbolic dest offset.
    """
    base_z = float(state[obj].grasp_pose[2])
    base_z = _arm_object_z_m(arm=arm, object_name=obj, base_z_m=base_z, config=config)
    place_offs = dict(config.place_height_offsets_m)
    place_offs.update(config.place_height_offsets_by_arm.get(arm) or {})
    if dest in state:
        layer = _match_name_offset_m(dest, place_offs)
        if layer is not None:
            support_z = float(state[dest].grasp_pose[2])
            return support_z + layer
        return base_z
    return _resolve_place_height_m(dest, base_z, place_offs)


@dataclass
class ObjectState:
    name: str
    xy: tuple[float, float]
    yaw: float
    footprint: Polygon
    grasp_pose: tuple[float, float, float, float]  # x, y, z, yaw


BoundStep = dict[str, Any]
FootprintFn = Callable[[Mapping[str, Any], dict[str, ObjectState], Mapping[str, Any], SolverConfig], Polygon]


def _empty() -> Polygon:
    return Polygon()


def _as_polygon(geom) -> Polygon | MultiPolygon:
    if geom is None or geom.is_empty:
        return _empty()
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            return _empty()
        return unary_union(polys)
    return geom.buffer(0)


def _buffer(poly: Polygon, margin: float) -> Polygon:
    if poly is None or poly.is_empty:
        return _empty()
    return _as_polygon(poly.buffer(margin))


def translate_footprint(
    footprint: Polygon, old_xy: tuple[float, float], new_xy: tuple[float, float]
) -> Polygon:
    from shapely.affinity import translate

    dx = new_xy[0] - old_xy[0]
    dy = new_xy[1] - old_xy[1]
    return _as_polygon(translate(footprint, xoff=dx, yoff=dy))


def _footprint_grasp(
    step: Mapping[str, Any],
    state: dict[str, ObjectState],
    _gv: Mapping[str, Any],
    config: SolverConfig,
) -> Polygon:
    obj = str(step["args"]["object"])
    return _buffer(state[obj].footprint, config.margin)


def _footprint_place(
    step: Mapping[str, Any],
    state: dict[str, ObjectState],
    _gv: Mapping[str, Any],
    config: SolverConfig,
) -> Polygon:
    """Consumed area at the object's *current* pose (updated after binding)."""
    obj = str(step["args"]["object"])
    return _buffer(state[obj].footprint, config.margin)


def _footprint_lift(
    _step: Mapping[str, Any],
    _state: dict[str, ObjectState],
    _gv: Mapping[str, Any],
    _config: SolverConfig,
) -> Polygon:
    return _empty()


PRIMITIVE_REGISTRY: dict[str, FootprintFn] = {
    "Grasp": _footprint_grasp,
    "Place": _footprint_place,
    "LiftUp": _footprint_lift,
}


def _init_state(geometric_view: Mapping[str, Any], grasp_height: float) -> dict[str, ObjectState]:
    state: dict[str, ObjectState] = {}
    for obj in geometric_view["objects"]:
        name = str(obj["name"])
        xy = (float(obj["xy"][0]), float(obj["xy"][1]))
        yaw = float(obj["yaw"])
        fp = obj["footprint"]
        gp = obj.get("grasp_pose")
        if gp is None:
            gp = (xy[0], xy[1], float(grasp_height), yaw)
        else:
            gp = (float(gp[0]), float(gp[1]), float(gp[2]), float(gp[3]))
        state[name] = ObjectState(name=name, xy=xy, yaw=yaw, footprint=fp, grasp_pose=gp)
    return state


def _region_polygon(
    dest: str,
    geometric_view: Mapping[str, Any],
    config: SolverConfig,
    state: dict[str, ObjectState],
) -> Polygon:
    if dest == "free_space":
        return _as_polygon(geometric_view["free_space"])
    if dest == "handover":
        # Intersection of all arm workspaces, or a disk at their centroid.
        workspaces = list(geometric_view["workspaces"].values())
        if not workspaces:
            return _empty()
        inter = workspaces[0]
        for ws in workspaces[1:]:
            inter = inter.intersection(ws)
        inter = _as_polygon(inter)
        if not inter.is_empty:
            return inter
        # Fallback: disk around mean of workspace centroids.
        cx = float(np.mean([ws.centroid.x for ws in workspaces]))
        cy = float(np.mean([ws.centroid.y for ws in workspaces]))
        return Point(cx, cy).buffer(config.handover_radius)
    if dest in state:
        return _as_polygon(state[dest].footprint)
    raise KeyError(f"Unknown Place destination {dest!r}")


def _union(geoms: Sequence) -> Polygon | MultiPolygon:
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return _empty()
    return _as_polygon(unary_union(geoms))


def _obstacles(
    state: dict[str, ObjectState],
    config: SolverConfig,
    *,
    exclude: str | set[str] | None = None,
) -> Polygon | MultiPolygon:
    if exclude is None:
        skip: set[str] = set()
    elif isinstance(exclude, str):
        skip = {exclude}
    else:
        skip = set(exclude)
    fps = [
        _buffer(obj.footprint, config.margin)
        for name, obj in state.items()
        if name not in skip
    ]
    return _union(fps)


def _farthest_from_boundary(legal: Polygon | MultiPolygon, tolerance: float) -> tuple[float, float]:
    geom = _as_polygon(legal)
    if geom.is_empty:
        raise ValueError("empty legal region")
    if isinstance(geom, MultiPolygon):
        # Choose the component with the largest inscribed-circle radius.
        best_pt = None
        best_r = -1.0
        for part in geom.geoms:
            if part.is_empty or part.area <= 0:
                continue
            pt = _polylabel(part, tolerance=tolerance)
            r = float(pt.distance(part.boundary))
            if r > best_r:
                best_r = r
                best_pt = pt
        if best_pt is None:
            raise ValueError("no non-empty component")
        return float(best_pt.x), float(best_pt.y)
    pt = _polylabel(geom, tolerance=tolerance)
    return float(pt.x), float(pt.y)


def _objects_on_support(
    state: dict[str, ObjectState],
    support: Polygon,
    *,
    exclude: set[str],
) -> set[str]:
    """Names whose grasp/centroid already lies on a support footprint."""
    on: set[str] = set()
    for name, obj in state.items():
        if name in exclude:
            continue
        pt = Point(obj.xy)
        if support.contains(pt) or support.covers(pt) or support.intersects(pt.buffer(1e-6)):
            on.add(name)
    return on


def _compute_future(
    steps_after: Sequence[Mapping[str, Any]],
    state: dict[str, ObjectState],
    geometric_view: Mapping[str, Any],
    config: SolverConfig,
) -> Polygon | MultiPolygon:
    """
    Reserve space that later steps will need.

    Grasp reserves the object's footprint. Place onto a **support object**
    (named object already in state, e.g. container) does **not** reserve the
    whole footprint — multiple items may share it. Place onto other named
    regions still reserves the region. Place(..., free_space) and
    Place(..., handover) reserve nothing: both are reusable zones (overlap
    for sequential handoffs; free_space for parking).
    """
    sim = copy.deepcopy(state)
    reserved: list = []

    for step in steps_after:
        prim = step["primitive"]
        args = step.get("args") or {}

        if prim == "Grasp":
            obj = str(args["object"])
            if obj in sim:
                reserved.append(_buffer(sim[obj].footprint, config.margin))
        elif prim == "Place":
            dest = str(args["destination"])
            obj = str(args["object"])
            if dest == "free_space":
                continue
            region = _region_polygon(dest, geometric_view, config, sim)
            # Support object / handover overlap: shareable — do not carve whole region.
            if dest not in sim and dest != "handover":
                reserved.append(region)
            if obj in sim and not region.is_empty:
                new_xy = (float(region.centroid.x), float(region.centroid.y))
                old = sim[obj]
                new_fp = translate_footprint(old.footprint, old.xy, new_xy)
                z = old.grasp_pose[2]
                sim[obj] = ObjectState(
                    name=obj,
                    xy=new_xy,
                    yaw=old.yaw,
                    footprint=new_fp,
                    grasp_pose=(new_xy[0], new_xy[1], z, old.yaw),
                )
        elif prim == "LiftUp":
            pass
        else:
            raise ValueError(f"Unknown primitive in future pass: {prim!r}")

    return _union(reserved)


def _workspace_for(arm: str, geometric_view: Mapping[str, Any]) -> Polygon:
    workspaces = geometric_view["workspaces"]
    if arm not in workspaces:
        raise KeyError(f"Arm {arm!r} missing from geometric_view['workspaces']")
    return _as_polygon(workspaces[arm])


def solve(
    plan: Sequence[Mapping[str, Any]],
    geometric_view: Mapping[str, Any],
    config: SolverConfig | Mapping[str, Any],
    *,
    lookahead: bool = True,
    debug_step: int | None = None,
    debug_out: dict[str, Any] | None = None,
) -> list[BoundStep]:
    """
    Pure function: bind numeric params for every symbolic plan step.

    When ``lookahead`` is False, ``future`` is always empty (ablation).
    """
    if isinstance(config, Mapping):
        place_offs = {
            str(k): float(v)
            for k, v in dict(config.get("place_height_offsets_m") or {}).items()
            if not str(k).startswith("_")
        }
        place_by_arm_raw = dict(config.get("place_height_offsets_by_arm") or {})
        place_by_arm = {
            str(arm): {
                str(k): float(v)
                for k, v in dict(offs or {}).items()
                if not str(k).startswith("_")
            }
            for arm, offs in place_by_arm_raw.items()
            if not str(arm).startswith("_")
        }
        by_arm_raw = dict(config.get("grasp_height_offsets_by_arm") or {})
        by_arm = {
            str(arm): {
                str(k): float(v)
                for k, v in dict(offs or {}).items()
                if not str(k).startswith("_")
            }
            for arm, offs in by_arm_raw.items()
            if not str(arm).startswith("_")
        }
        config = SolverConfig(
            grasp_height=float(config["grasp_height"]),
            margin=float(config.get("margin", 0.03)),
            handover_radius=float(config.get("handover_radius", 0.08)),
            polylabel_tolerance=float(config.get("polylabel_tolerance", 0.005)),
            place_height_offsets_m=place_offs,
            place_height_offsets_by_arm=place_by_arm,
            grasp_height_offsets_by_arm=by_arm,
        )

    state = _init_state(geometric_view, config.grasp_height)
    free_space = _as_polygon(geometric_view["free_space"])
    bound: list[BoundStep] = []

    for i, step in enumerate(plan):
        sid = int(step["step"])
        arm = str(step["arm"])
        prim = str(step["primitive"])
        args = dict(step.get("args") or {})
        depends_on = [int(d) for d in (step.get("depends_on") or [])]
        ws = _workspace_for(arm, geometric_view)

        future = (
            _compute_future(plan[i + 1 :], state, geometric_view, config)
            if lookahead
            else _empty()
        )

        params: dict[str, Any] = {}

        if prim == "Grasp":
            obj = str(args["object"])
            if obj not in state:
                raise GraspUnreachable(sid, f"Unknown object {obj!r}")
            pose = state[obj].grasp_pose
            # Perception z (global offsets) + per-arm EE offsets (e.g. LeapHand bear).
            z = float(pose[2]) if pose[2] is not None else float(config.grasp_height)
            z = _arm_object_z_m(arm=arm, object_name=obj, base_z_m=z, config=config)
            pose = (pose[0], pose[1], z, pose[3])
            pt = Point(pose[0], pose[1])
            if not (ws.contains(pt) or ws.covers(pt) or ws.intersects(pt.buffer(1e-6))):
                raise GraspUnreachable(
                    sid,
                    f"Grasp point ({pose[0]:.3f}, {pose[1]:.3f}) outside workspace of {arm!r}",
                )
            params = {"pose": list(pose), "object": obj}

        elif prim == "Place":
            obj = str(args["object"])
            dest = str(args["destination"])
            if obj not in state:
                raise NoLegalPlacement(sid, f"Unknown object {obj!r}")

            if dest == "free_space":
                region = free_space
                # Relocating obj: ignore its old footprint only.
                exclude_names = {obj}
            else:
                region = _region_polygon(dest, geometric_view, config, state)
                # Placing onto a named object/region: that target is support, not obstacle.
                exclude_names = {obj, dest} if dest in state else {obj}
                # Items already parked on the same support (multi-place) must not
                # erase the whole container via margin buffers.
                if dest in state:
                    exclude_names |= _objects_on_support(
                        state, _as_polygon(region), exclude=exclude_names
                    )

            obstacles = _obstacles(state, config, exclude=exclude_names)
            legal = region.difference(future).difference(obstacles).intersection(ws)
            legal = _as_polygon(legal)

            if debug_step is not None and sid == debug_step and debug_out is not None:
                debug_out.update(
                    {
                        "step": sid,
                        "free_space": free_space,
                        "region": region,
                        "future": future,
                        "obstacles": obstacles,
                        "legal": legal,
                        "workspace": ws,
                    }
                )

            if legal.is_empty:
                raise NoLegalPlacement(
                    sid,
                    f"No legal placement for {obj!r} at destination {dest!r}",
                )

            # Handover: park as close as possible to the dual-arm overlap
            # centroid. Obstacles may carve the center out of ``legal``; use the
            # nearest legal point instead of representative_point (often an edge).
            if dest == "handover":
                from shapely.ops import nearest_points

                seed = _as_polygon(region).centroid
                if legal.contains(seed) or legal.covers(seed):
                    x, y = float(seed.x), float(seed.y)
                else:
                    pt = nearest_points(legal, seed)[0]
                    x, y = float(pt.x), float(pt.y)
            else:
                x, y = _farthest_from_boundary(legal, config.polylabel_tolerance)
            yaw = float(state[obj].yaw)
            z = _resolve_place_z_m(
                arm=arm, obj=obj, dest=dest, state=state, config=config
            )
            pose = [x, y, z, yaw]
            params = {"pose": pose, "object": obj, "destination": dest}

            if debug_step is not None and sid == debug_step and debug_out is not None:
                debug_out["chosen"] = (x, y)

            # Update simulated layout after Place.
            old = state[obj]
            new_fp = translate_footprint(old.footprint, old.xy, (x, y))
            state[obj] = ObjectState(
                name=obj,
                xy=(x, y),
                yaw=yaw,
                footprint=new_fp,
                grasp_pose=(x, y, z, yaw),
            )

        elif prim == "LiftUp":
            params = {}
        else:
            raise ValueError(f"Unknown primitive {prim!r} at step {sid}")

        # Registry available for introspection / tests (consumed area at bind time).
        _ = PRIMITIVE_REGISTRY[prim](step, state, geometric_view, config)

        bound.append(
            {
                "step": sid,
                "arm": arm,
                "primitive": prim,
                "params": params,
                "depends_on": depends_on,
            }
        )

    return bound
