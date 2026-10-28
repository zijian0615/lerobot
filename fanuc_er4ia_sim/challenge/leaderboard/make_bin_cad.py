"""Write files/manipbench-bin-50x50x55.step and .stl for the CAD Models page.

    uv run --no-project --with cadquery python fanuc_er4ia_sim/challenge/leaderboard/make_bin_cad.py

The cell's bins, as measured: 50 x 50 mm outside, 55 mm tall, a 35 x 35 mm pocket 25 mm deep (scene_config.py BIN_*).
The pocket's corner radius is not measured (6.2 mm, as in the twin). Millimetres; +Z up, the base's centre at the origin.
"""
from pathlib import Path

import cadquery as cq

OUT = Path(__file__).resolve().parent / "files"
OUTER, HEIGHT, POCKET, DEPTH, POCKET_R = 50.0, 55.0, 35.0, 25.0, 6.2

block = cq.Workplane("XY").box(OUTER, OUTER, HEIGHT, centered=(True, True, False))
pocket = (cq.Workplane("XY").workplane(offset=HEIGHT - DEPTH).rect(POCKET, POCKET).extrude(DEPTH)
          .edges("|Z").fillet(POCKET_R))
bin_ = block.cut(pocket)
bb = bin_.val().BoundingBox()
print("bbox mm: x %.2f..%.2f  y %.2f..%.2f  z %.2f..%.2f" % (bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax))
print("volume %.0f mm3, valid %s, solids %d" % (bin_.val().Volume(), bin_.val().isValid(), len(bin_.solids().vals())))
cq.exporters.export(bin_, str(OUT / "manipbench-bin-50x50x55.step"))
cq.exporters.export(bin_, str(OUT / "manipbench-bin-50x50x55.stl"), tolerance=0.01, angularTolerance=0.1)
