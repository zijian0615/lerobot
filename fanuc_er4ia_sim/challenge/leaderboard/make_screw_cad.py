"""Write files/ISO10642-M6x30-countersunk-screw.step and .stl for the CAD Models page.

    uv run --no-project --with cadquery python fanuc_er4ia_sim/challenge/leaderboard/make_screw_cad.py
"""
import math
from pathlib import Path

import cadquery as cq

OUT = Path(__file__).resolve().parent / "files"
# ISO 10642 M6 x 30 countersunk socket screw, as in the twin (scene_config.py SCREW_*), mm.
# Axis +Z, head face at z = 0, tip at z = -30. Threads not modelled (plain 6 mm shank).
L, HEAD_D, HEAD_H, SHANK_D = 30.0, 12.0, 3.3, 6.0
SOCKET_S, SOCKET_T = 4.0, 2.2          # hex socket across flats, depth (ISO 10642 M6)
hr, sr = HEAD_D / 2, SHANK_D / 2
land = HEAD_H - (hr - sr)              # 90 deg cone: radial drop = axial drop
prof = [(0, 0), (hr, 0), (hr, -land), (sr, -HEAD_H), (sr, -(L - 0.8)), (sr * 0.8, -L), (0, -L)]
body = cq.Workplane("XZ").polyline(prof).close().revolve(360, (0, 0, 0), (0, 1, 0))
socket = cq.Workplane("XY").polygon(6, SOCKET_S / math.cos(math.pi / 6)).extrude(-SOCKET_T)
screw = body.cut(socket)
bb = screw.val().BoundingBox()
print("bbox mm: x %.2f..%.2f  y %.2f..%.2f  z %.2f..%.2f" % (bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax))
v = screw.val().Volume()
print("volume %.0f mm3 -> steel %.1f g (twin SCREW_MASS 7.6 g)" % (v, v * 7.85e-3))
print("valid solid:", screw.val().isValid(), "solids:", len(screw.solids().vals()))
cq.exporters.export(screw, str(OUT / "ISO10642-M6x30-countersunk-screw.step"))
cq.exporters.export(screw, str(OUT / "ISO10642-M6x30-countersunk-screw.stl"), tolerance=0.01, angularTolerance=0.1)
