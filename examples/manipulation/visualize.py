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

"""Debug plot for one solver step: free_space, future, legal, chosen point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


def _plot_poly(ax, geom, *, facecolor, edgecolor, alpha, label=None, linewidth=1.5) -> None:
    if geom is None or geom.is_empty:
        return
    geoms: list = []
    if isinstance(geom, Polygon):
        geoms = [geom]
    elif isinstance(geom, (MultiPolygon, GeometryCollection)):
        geoms = [g for g in geom.geoms if isinstance(g, Polygon)]
    else:
        geoms = [geom]
    first = True
    for g in geoms:
        x, y = g.exterior.xy
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


def plot_solver_step(
    debug: dict[str, Any],
    *,
    ax=None,
    show: bool = True,
    save_path: str | Path | None = None,
    title: str | None = None,
):
    """
    ``debug`` is the dict filled by ``solve(..., debug_step=..., debug_out=...)``.
    """
    import matplotlib.pyplot as plt

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(7, 7))

    _plot_poly(ax, debug.get("free_space"), facecolor="#c8e6c9", edgecolor="#2e7d32", alpha=0.35, label="free_space")
    _plot_poly(ax, debug.get("future"), facecolor="#ffcc80", edgecolor="#ef6c00", alpha=0.45, label="future")
    _plot_poly(ax, debug.get("obstacles"), facecolor="#ef9a9a", edgecolor="#c62828", alpha=0.35, label="obstacles")
    _plot_poly(ax, debug.get("workspace"), facecolor="#90caf9", edgecolor="#1565c0", alpha=0.15, label="workspace")
    _plot_poly(ax, debug.get("legal"), facecolor="#ce93d8", edgecolor="#6a1b9a", alpha=0.55, label="legal")

    chosen = debug.get("chosen")
    if chosen is not None:
        ax.scatter([chosen[0]], [chosen[1]], c="#4a148c", s=60, zorder=5, label="chosen")

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title or f"Solver debug step {debug.get('step')}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    if show and created:
        plt.show()
    return ax
