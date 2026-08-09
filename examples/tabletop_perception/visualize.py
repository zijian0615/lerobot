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

"""2D table-plane visualisation for debugging projection / predicates."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .perception import GeometricView


def _plot_polygon(ax, polygon, *, facecolor, edgecolor, alpha, linewidth=1.5, label=None) -> None:
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

    if polygon is None or polygon.is_empty:
        return

    geoms: list = []
    if isinstance(polygon, Polygon):
        geoms = [polygon]
    elif isinstance(polygon, (MultiPolygon, GeometryCollection)):
        geoms = [g for g in polygon.geoms if isinstance(g, Polygon)]
    else:
        geoms = [polygon]

    first = True
    for geom in geoms:
        x, y = geom.exterior.xy
        ax.fill(
            x,
            y,
            facecolor=facecolor,
            edgecolor=edgecolor,
            alpha=alpha,
            linewidth=linewidth,
            label=label if first else None,
        )
        first = False
        for interior in geom.interiors:
            ix, iy = interior.xy
            ax.fill(ix, iy, facecolor="white", edgecolor=edgecolor, alpha=1.0, linewidth=0.8)


def visualize_table_plane(
    geometric_view: GeometricView,
    *,
    ax=None,
    show: bool = True,
    save_path: str | Path | None = None,
    title: str = "Table-plane perception",
):
    """
    Draw footprints, grasp points, workspace boundaries, and free space
    onto the table plane (z=0) for debugging the monocular projection.
    """
    import matplotlib.pyplot as plt

    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(8, 8))

    # Free space first (background).
    _plot_polygon(
        ax,
        geometric_view["free_space"],
        facecolor="#c8e6c9",
        edgecolor="#2e7d32",
        alpha=0.45,
        label="free_space",
    )

    # Table outline.
    _plot_polygon(
        ax,
        geometric_view["table_polygon"],
        facecolor="none",
        edgecolor="#424242",
        alpha=1.0,
        linewidth=2.0,
        label="table",
    )

    # Workspaces.
    workspace_colors = ("#90caf9", "#ce93d8", "#ffcc80", "#80cbc4")
    for i, (name, poly) in enumerate(geometric_view["workspaces"].items()):
        color = workspace_colors[i % len(workspace_colors)]
        _plot_polygon(
            ax,
            poly,
            facecolor=color,
            edgecolor=color,
            alpha=0.2,
            linewidth=1.5,
            label=f"workspace:{name}",
        )
        # Emphasise boundary.
        if not poly.is_empty:
            x, y = poly.exterior.xy
            ax.plot(x, y, color=color, linewidth=2.0, linestyle="--")

    # Multi-arm overlap (mutex zone).
    overlap = geometric_view.get("overlap") if hasattr(geometric_view, "get") else None
    if overlap is not None and not getattr(overlap, "is_empty", True):
        _plot_polygon(
            ax,
            overlap,
            facecolor="#ff8a65",
            edgecolor="#d84315",
            alpha=0.35,
            linewidth=2.0,
            label="overlap(mutex)",
        )

    # Object footprints + grasp points.
    for obj in geometric_view["objects"]:
        _plot_polygon(
            ax,
            obj["footprint"],
            facecolor="#ef9a9a",
            edgecolor="#c62828",
            alpha=0.55,
            linewidth=1.5,
            label=None,
        )
        gx, gy = obj["xy"]
        yaw = obj["yaw"]
        ax.scatter([gx], [gy], c="#b71c1c", s=40, zorder=5)
        # Small orientation tick.
        arrow_len = 0.03
        ax.arrow(
            gx,
            gy,
            arrow_len * np.cos(yaw),
            arrow_len * np.sin(yaw),
            head_width=0.01,
            head_length=0.01,
            fc="#b71c1c",
            ec="#b71c1c",
            length_includes_head=True,
            zorder=6,
        )
        ax.annotate(obj["name"], (gx, gy), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m, table frame)")
    ax.set_ylabel("y (m, table frame)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")

    if show and created_fig:
        plt.show()

    return ax
