"""Write files/manipbench-cube-10mm and -16mm .step / .stl for the CAD Models page.

    uv run --no-project --with cadquery python fanuc_er4ia_sim/challenge/leaderboard/make_cube_cad.py

The cell's cubes: any edge size, now 10 mm and 16 mm. The 16 mm cube ("die") has 1 mm round edges (its CAD,
'GENERIC die'); the 10 mm cube's edge radius is not measured (0.6 mm, scaled). Millimetres; the bottom face's centre at
the origin, +Z up (as the starter's cube.usda).
"""
from pathlib import Path

import cadquery as cq

OUT = Path(__file__).resolve().parent / "files"
for edge, r in ((10.0, 0.6), (16.0, 1.0)):
    cube = cq.Workplane("XY").box(edge, edge, edge, centered=(True, True, False)).edges().fillet(r)
    bb, v = cube.val().BoundingBox(), cube.val().Volume()
    print(f"{edge:.0f} mm: bbox {bb.xlen:.2f} x {bb.ylen:.2f} x {bb.zlen:.2f}, volume {v:.0f} mm3, valid {cube.val().isValid()}")
    name = f"manipbench-cube-{edge:.0f}mm"
    cq.exporters.export(cube, str(OUT / f"{name}.step"))
    cq.exporters.export(cube, str(OUT / f"{name}.stl"), tolerance=0.005, angularTolerance=0.1)
