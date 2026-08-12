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

"""Multi-arm overlap zone: mutex + post-task retract."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

RetractFn = Callable[[], None]


class OverlapBusy(RuntimeError):
    def __init__(self, holder: str, requester: str) -> None:
        self.holder = holder
        self.requester = requester
        super().__init__(
            f"overlap zone busy: held by {holder!r}, requested by {requester!r}"
        )


def overlap_from_workspaces(
    workspaces: Mapping[str, Any],
    *,
    arms: Sequence[str] | None = None,
    table_polygon: Any | None = None,
) -> Polygon:
    """
    overlap = ∩ workspaces[arms]  (optionally ∩ table).

    Default ``arms`` = all keys in ``workspaces``.
    """
    names = list(arms) if arms is not None else list(workspaces.keys())
    polys = []
    for name in names:
        if name not in workspaces:
            raise KeyError(f"overlap arm {name!r} missing from workspaces")
        polys.append(workspaces[name])
    if len(polys) < 2:
        return Polygon()
    ov = polys[0]
    for p in polys[1:]:
        ov = ov.intersection(p)
    if table_polygon is not None and not getattr(table_polygon, "is_empty", True):
        ov = ov.intersection(table_polygon)
    if ov is None or ov.is_empty:
        return Polygon()
    if ov.geom_type == "Polygon":
        return ov
    if ov.geom_type == "MultiPolygon":
        return max(ov.geoms, key=lambda g: g.area)
    # GeometryCollection etc.
    parts = [g for g in getattr(ov, "geoms", []) if g.geom_type == "Polygon"]
    return max(parts, key=lambda g: g.area) if parts else Polygon()


def xy_in_overlap(xy: tuple[float, float], overlap: Polygon, *, margin_m: float = 0.0) -> bool:
    if overlap is None or overlap.is_empty:
        return False
    region = overlap.buffer(margin_m) if margin_m > 0 else overlap
    pt = Point(float(xy[0]), float(xy[1]))
    return bool(region.contains(pt) or region.covers(pt) or region.intersects(pt.buffer(1e-6)))


def exclusive_retract_xy(
    workspace: Polygon,
    overlap: Polygon,
) -> tuple[float, float] | None:
    """Pick a point in ``workspace \\ overlap`` (own side, outside mutex zone)."""
    if workspace is None or workspace.is_empty:
        return None
    exclusive = workspace
    if overlap is not None and not overlap.is_empty:
        exclusive = workspace.difference(overlap)
    if exclusive.is_empty:
        # Fall back to workspace centroid if difference vanished.
        c = workspace.centroid
        return float(c.x), float(c.y)
    if exclusive.geom_type == "MultiPolygon":
        exclusive = max(exclusive.geoms, key=lambda g: g.area)
    pt = exclusive.representative_point()
    return float(pt.x), float(pt.y)


class OverlapGuard:
    """
    Policy:
      1. Poses inside overlap require the mutex (no two arms in-zone at once
         during a step).
      2. After Place (or any step that should clear the zone): retract to
         exclusive workspace, then release.
      3. After Grasp when a later Place of the same object by this arm exists:
         release the mutex **without** retracting (arm stays at lift) so
         Grasp→Place is direct even if other arms' steps are interleaved.
         Plans are assumed sequential (no true parallel overlap use).
    """

    def __init__(
        self,
        overlap: Polygon,
        *,
        retract: Mapping[str, RetractFn] | None = None,
        margin_m: float = 0.0,
        enabled: bool = True,
    ) -> None:
        self.overlap = overlap if overlap is not None else Polygon()
        self.retract = dict(retract or {})
        self.margin_m = float(margin_m)
        self.enabled = enabled
        self._holder: str | None = None

    @property
    def holder(self) -> str | None:
        return self._holder

    def update_overlap(self, overlap: Polygon) -> None:
        self.overlap = overlap if overlap is not None else Polygon()

    def in_overlap(self, xy: tuple[float, float] | None) -> bool:
        if xy is None:
            return False
        return xy_in_overlap(xy, self.overlap, margin_m=self.margin_m)

    def acquire(self, arm: str, xy: tuple[float, float] | None) -> bool:
        """
        Returns True if this step is inside the overlap (caller must
        ``release_after``).
        """
        if not self.enabled:
            return False
        if not self.in_overlap(xy):
            return False
        if self._holder is not None and self._holder != arm:
            raise OverlapBusy(self._holder, arm)
        if self._holder != arm:
            logger.info("overlap acquire arm=%s xy=(%.3f, %.3f)", arm, xy[0], xy[1])
        self._holder = arm
        return True

    def release_after(self, arm: str, *, retract: bool = True) -> None:
        """
        Finish an overlap step and always clear the mutex.

        ``retract=True``: move to exclusive workspace first.
        ``retract=False``: stay put (e.g. post-Grasp lift) then free the lock
        so another arm's sequential step can run before this arm's Place.
        """
        if self._holder != arm:
            return
        try:
            if retract:
                fn = self.retract.get(arm)
                if fn is not None:
                    logger.info("overlap retract arm=%s", arm)
                    fn()
                logger.info("overlap release arm=%s", arm)
            else:
                logger.info("overlap release arm=%s (skip retract)", arm)
        finally:
            self._holder = None
