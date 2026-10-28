"""Draw the work surface for teams: client/submission/table_plan.png (axes, table, reach, the default scene).

    uv run python fanuc_er4ia_sim/challenge/placements/table_plan.py

Top view as the table camera sees it (the robot at the top): +x points down, +y to the right. Sizes from the scan
(scene_config: TABLE_X, TABLE_Y, EDGE_BAND_W, MAX_REACH); positions from placements/default.yaml.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrow, Polygon, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rollout_setup as rs  # noqa: E402

TABLE_X, TABLE_Y, BAND, REACH = (-371, 759), (-772, 792), 48, 550
INK, MUTED, GOLD, TAPE = "#141413", "#6b6a63", "#c99300", "#f2cf3a"
OUT = HERE.parent / "client" / "submission" / "table_plan.png"


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 7.6), dpi=160)
    P = lambda x, y: (y, x)  # robot (x, y) -> plot (horizontal = y, vertical = x, inverted below)

    # the table, its CAUTION tape (front and both sides) and the usable area inside it
    ax.add_patch(Rectangle(P(TABLE_X[0], TABLE_Y[0]), TABLE_Y[1] - TABLE_Y[0], TABLE_X[1] - TABLE_X[0],
                           fc="#f4f3ee", ec=INK, lw=1.2))
    for x0, y0, dx, dy in ((TABLE_X[1] - BAND, TABLE_Y[0], BAND, TABLE_Y[1] - TABLE_Y[0]),
                           (TABLE_X[0], TABLE_Y[0], TABLE_X[1] - TABLE_X[0], BAND),
                           (TABLE_X[0], TABLE_Y[1] - BAND, TABLE_X[1] - TABLE_X[0], BAND)):
        ax.add_patch(Rectangle(P(x0, y0), dy, dx, fc=TAPE, ec="none"))
    (ux0, ux1), (uy0, uy1) = rs.TABLE_MM
    ax.add_patch(Rectangle(P(ux0, uy0), uy1 - uy0, ux1 - ux0, fc="none", ec=MUTED, lw=0.8, ls=(0, (4, 3))))

    # 100 mm grid and ticks
    for v in range(-700, 801, 100):
        ax.axvline(v, color="#e3e1d8", lw=0.6, zorder=0)
    for v in range(-300, 801, 100):
        ax.axhline(v, color="#e3e1d8", lw=0.6, zorder=0)

    # reach and the base
    ax.add_patch(Circle((0, 0), REACH, fc="none", ec=GOLD, lw=1.4, ls=(0, (6, 4))))
    ax.text(REACH * math.sin(math.radians(-38)), REACH * math.cos(math.radians(-38)) + 25,
            "reach ≈ 550 mm", color=GOLD, fontsize=9, ha="center", rotation=38, weight="bold")
    ax.add_patch(Circle((0, 0), rs.BASE_R_MM, fc="#d6d3c8", ec=INK, lw=1, hatch="///"))
    ax.text(0, -rs.BASE_R_MM - 20, "robot base (origin)", ha="center", va="bottom", fontsize=9, color=INK)

    # axes from the origin
    ax.add_patch(FancyArrow(0, 0, 0, 260, width=7, head_width=30, head_length=40, color="#b5452f", length_includes_head=True))
    ax.add_patch(FancyArrow(0, 0, 260, 0, width=7, head_width=30, head_length=40, color="#3f7d58", length_includes_head=True))
    ax.text(0, 300, "+x", color="#b5452f", fontsize=13, weight="bold", ha="center", va="top")
    ax.text(290, 0, "+y", color="#3f7d58", fontsize=13, weight="bold", ha="left", va="center")

    # the default scene: 8 screws (head -> tip) and the bin
    for name, ps in rs.validate(rs.yaml.safe_load(rs.DEFAULT.read_text()))["rollouts"][0]["objects"].items():
        for x, y, yaw in ps:
            a = math.radians(yaw)
            if name == "screw":
                tx, ty = x + rs.SCREW_MM * math.cos(a), y + rs.SCREW_MM * math.sin(a)
                ax.plot([y, ty], [x, tx], color=INK, lw=2.2, solid_capstyle="round")
                ax.add_patch(Circle((y, x), 6, color=INK))
            else:
                d = rs.BIN_MM / 2
                corners = [(x + c * math.cos(a) - s * math.sin(a), y + c * math.sin(a) + s * math.cos(a))
                           for c, s in ((-d, -d), (d, -d), (d, d), (-d, d))]
                ax.add_patch(Polygon([P(cx, cy) for cx, cy in corners], fc="#ffbd00", ec=INK, lw=1))
    ax.annotate("default scene:\n8 screws + bin", xy=(130, 470), xytext=(330, 600), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    # yaw: 0 along +x, 90 along +y
    cx, cy = 520, -540
    for deg, lab in ((0, "0°"), (90, "90°"), (180, "180°"), (-90, "−90°")):
        a = math.radians(deg)
        ax.add_patch(FancyArrow(cy, cx, 70 * math.sin(a), 70 * math.cos(a), width=3, head_width=16, head_length=18,
                                color=MUTED, length_includes_head=True))
        ax.text(cy + 100 * math.sin(a), cx + 100 * math.cos(a), lab, ha="center", va="center", fontsize=8, color=MUTED)
    ax.text(cy, cx - 135, "yaw (degrees)", ha="center", fontsize=9, color=MUTED, weight="bold")

    # edges
    ax.text(0, TABLE_X[0] - 18, "back edge (enclosure)", ha="center", va="bottom", fontsize=9, color=MUTED)
    ax.text(0, TABLE_X[1] + 18, "front edge (CAUTION tape)", ha="center", va="top", fontsize=9, color=MUTED)
    ax.text(uy1 - 8, ux0 + 14, f"usable area: x {ux0:.0f} to {ux1:.0f}, y {uy0:.0f} to {uy1:.0f}",
            ha="right", va="top", fontsize=8, color=MUTED)

    ax.set_xlim(TABLE_Y[0] - 40, TABLE_Y[1] + 40)
    ax.set_ylim(TABLE_X[1] + 80, TABLE_X[0] - 70)  # +x down, as in the table camera
    ax.set_aspect("equal")
    ax.set_xticks(range(-700, 801, 100))
    ax.set_yticks(range(-300, 701, 100))
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.set_xlabel("y (mm)", color=MUTED, fontsize=9)
    ax.set_ylabel("x (mm)", color=MUTED, fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Work surface, top view as the table camera sees it\n"
                 "x, y in mm in the robot frame, as obs[\"tcp_pose\"] and placement.yaml use them",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
