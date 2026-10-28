"""Write files/csi-vial.stl and csi-vial-cap.stl (millimetres) from the USD meshes beside them (inches).

    uv run --no-project --with trimesh --with scipy --with networkx python fanuc_er4ia_sim/challenge/leaderboard/convert_vial_usda.py

The .usda files are the hackathon starter's assets/usd_assets/vial.usda and vial_cap.usda: one triangle mesh each, no
transform on it. Parsed as text, since usd-core has no Linux ARM wheel.
"""
from pathlib import Path
import re

import numpy as np
import trimesh

F = Path(__file__).resolve().parent / "files"
def grab(text, decl):
    i = text.index(decl); j = text.index("[", i + len(decl)); depth = 0
    for k in range(j, len(text)):
        depth += {"[": 1, "]": -1}.get(text[k], 0)
        if depth == 0: return text[j + 1:k]
for src, out in [(F / "csi-vial.usda", F / "csi-vial.stl"), (F / "csi-vial-cap.usda", F / "csi-vial-cap.stl")]:
    t = open(src).read()
    assert t.count("def Mesh") == 1 and "metersPerUnit = 0.0254" in t
    pts = np.array([[float(v) for v in p.split(",")] for p in re.findall(r"\(([^()]*)\)", grab(t, "point3f[] points"))]) * 25.4
    counts = np.array(re.findall(r"-?\d+", grab(t, "int[] faceVertexCounts")), int)
    idx = np.array(re.findall(r"-?\d+", grab(t, "int[] faceVertexIndices")), int)
    assert (counts == 3).all() and len(idx) == 3 * len(counts)
    mesh = trimesh.Trimesh(pts, idx.reshape(-1, 3), process=True); mesh.merge_vertices(); mesh.fix_normals()
    mesh.export(out)
    print(out.name, "faces", len(mesh.faces), "watertight", mesh.is_watertight, "extents mm", np.round(mesh.extents, 2), "z %.2f..%.2f" % tuple(mesh.bounds[:, 2]), "volume mm3 %.0f" % abs(mesh.volume))
