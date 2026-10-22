"""Build onrobot_screwdriver.npz: OnRobot's published CAD of its Screwdriver,
tessellated and put in the xArm6 flange (link6) frame.

Source: https://onrobot.com/storage/screwdriver-v1.step (OnRobot A/S, free
download for integration; not redistributed here beyond this tessellation for
the simulation). Needs OpenCascade, kept out of the Isaac venv:

    python3 -m venv /tmp/cadvenv && /tmp/cadvenv/bin/pip install cadquery-ocp numpy trimesh
    curl -L -o /tmp/screwdriver.step https://onrobot.com/storage/screwdriver-v1.step
    /tmp/cadvenv/bin/python assembly/instrument/assets/build_driver.py /tmp/screwdriver.step

CAD frame (mm): the robot-side mounting face is the plane y = +8.3 with the
body toward -y; the screwdriving axis is x = 0, y = -84.3, pointing -z, its
nozzle tip at z = -159.4. Flange frame: x = -z_cad (the tool axis), y = x_cad,
z = -(y_cad - 8.3) (out of the flange). So the tip is at (0.1594, 0, 0.0926) m
(kinematics.TOOL_TIP). Solids under 6 mm (pins, screws) are dropped; the rest
are grouped by material: nozzle (below the housing), mount (the tool-side
changer plate), housing, small.
"""

import sys
from pathlib import Path

import numpy as np

R = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], float)   # CAD -> flange frame
MOUNT_Y = 0.0083


def main(step_path):
    import trimesh
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    r = STEPControl_Reader()
    r.ReadFile(str(step_path))
    r.TransferRoots()
    shape = r.OneShape()
    BRepMesh_IncrementalMesh(shape, 0.15, False, 0.3, True)
    groups = {}
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        V, F = [], []
        fx = TopExp_Explorer(ex.Current(), TopAbs_FACE)
        while fx.More():
            f = TopoDS.Face(fx.Current())
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(f, loc)
            if tri is not None:
                tr, o = loc.Transformation(), len(V)
                for i in range(1, tri.NbNodes() + 1):
                    p = tri.Node(i).Transformed(tr)
                    V.append((p.X(), p.Y(), p.Z()))
                rev = f.Orientation() == TopAbs_REVERSED
                for i in range(1, tri.NbTriangles() + 1):
                    a, b, c = tri.Triangle(i).Get()
                    F.append((o + a - 1, o + c - 1, o + b - 1) if rev else (o + a - 1, o + b - 1, o + c - 1))
            fx.Next()
        ex.Next()
        if not F:
            continue
        m = trimesh.Trimesh(np.array(V) / 1000.0, np.array(F), process=True)
        (x0, y0, z0), (x1, y1, z1) = m.bounds
        ext = max(x1 - x0, y1 - y0, z1 - z0)
        if ext < 0.006:
            continue
        g = ("nozzle" if z1 < -0.085 else "mount" if (y0 > -0.04 and ext < 0.09)
             else "small" if ext < 0.02 else "housing")
        groups.setdefault(g, []).append(m)
    out = {}
    for g, ms in groups.items():
        m = trimesh.util.concatenate(ms)
        out[f"{g}_v"] = ((m.vertices - [0, MOUNT_Y, 0]) @ R.T).astype(np.float32)
        out[f"{g}_f"] = m.faces.astype(np.int32)
        print(g, len(ms), "solids", len(m.faces), "triangles")
    np.savez_compressed(Path(__file__).with_name("onrobot_screwdriver.npz"), **out)


if __name__ == "__main__":
    main(sys.argv[1])
