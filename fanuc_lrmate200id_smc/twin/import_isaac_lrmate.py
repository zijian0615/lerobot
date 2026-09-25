"""Export the NVIDIA Isaac Sim LR Mate 200iD/4S visual meshes to STL, one file per link and colour.

Run with Isaac Sim's Python (needs pxr):

    ~/isaacsim-venv/bin/python twin/import_isaac_lrmate.py            # downloads the asset from the Isaac content server
    ~/isaacsim-venv/bin/python twin/import_isaac_lrmate.py --usd /path/to/lrmate200id4s.usd

Writes assets/lrmate200id4s/<link>_<colour>.stl (metres, in the link frame) and parts.json. The Isaac asset's link frames
(robot_base, J1_link .. J6_link) sit on the joint origins with no rotation at the zero pose, like the MuJoCo bodies
base_link, link_1 .. link_6, so the vertices are used as they are. Materials are grouped by their OmniPBR diffuse colour.
Asset: Assets/Isaac/6.1/Isaac/Robots/Fanuc/lrmate200id4s, CC BY 4.0 (NVIDIA), see assets/lrmate200id4s/README.md.
"""

import argparse
import json
import os
import struct
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "lrmate200id4s")
BASE_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/Robots/Fanuc/lrmate200id4s"
FILES = ("lrmate200id4s.usd", "configuration/lrmate200id4s_base.usd", "configuration/lrmate200id4s_physics.usd",
         "configuration/lrmate200id4s_sensor.usd", "LICENSE")
LINKS = {"robot_base": "base_link", "J1_link": "link_1", "J2_link": "link_2", "J3_link": "link_3",
         "J4_link": "link_4", "J5_link": "link_5", "J6_link": "link_6"}
COLOUR_NAMES = {(0.0, 0.0, 0.0): "black", (1.0, 1.0, 0.0): "yellow", (0.506, 0.529, 0.549): "gray",
                (0.733, 0.733, 0.733): "silver", (0.8, 0.0, 0.0): "red"}


def download(cache):
    for rel in FILES:
        path = os.path.join(cache, rel)
        if not os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(f"{BASE_URL}/{rel}", path)
    return os.path.join(cache, FILES[0])


def write_stl(path, tri):
    """Binary STL of triangles (N x 3 x 3)."""
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = normals, tri
    with open(path, "wb") as fh:
        fh.write(b"lrmate200id4s from NVIDIA Isaac Sim assets (CC BY 4.0)".ljust(80, b" "))
        fh.write(struct.pack("<I", len(tri)))
        fh.write(rec.tobytes())


def colour_of(material):
    from pxr import Usd

    if material:
        for shader in Usd.PrimRange(material.GetPrim(), Usd.TraverseInstanceProxies()):
            attr = shader.GetAttribute("inputs:diffuse_color_constant")
            if attr and attr.Get() is not None:
                return tuple(round(float(c), 3) for c in attr.Get())
    return (0.5, 0.5, 0.5)


def export(usd_path, out_dir):
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim().GetPath()
    cache = UsdGeom.XformCache()
    os.makedirs(out_dir, exist_ok=True)
    parts = []
    for usd_link, body in LINKS.items():
        link = stage.GetPrimAtPath(root.AppendChild(usd_link))
        to_link = cache.GetLocalToWorldTransform(link).GetInverse()
        groups = {}
        for prim in Usd.PrimRange(link.GetChild("visuals"), Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            if counts.min() != 3 or counts.max() != 3:
                raise NotImplementedError(f"{prim.GetPath()} has non-triangle faces")
            m = np.array(cache.GetLocalToWorldTransform(prim) * to_link)  # row-vector convention
            pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            pts = pts @ m[:3, :3] + m[3, :3]
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            subsets = UsdGeom.Subset.GetAllGeomSubsets(mesh)
            if not subsets:
                subsets_faces = [(np.arange(len(faces)), UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0])]
            else:
                subsets_faces = [(np.asarray(s.GetIndicesAttr().Get()),
                                  UsdShade.MaterialBindingAPI(s.GetPrim()).ComputeBoundMaterial()[0]) for s in subsets]
            for face_ids, material in subsets_faces:
                groups.setdefault(colour_of(material), []).append(pts[faces[face_ids]])
        for colour, chunks in sorted(groups.items()):
            name = f"{body}_{COLOUR_NAMES.get(colour, 'c%d_%d_%d' % tuple(round(c * 255) for c in colour))}"
            tri = np.concatenate(chunks)
            write_stl(os.path.join(out_dir, name + ".stl"), tri)
            parts.append({"body": body, "mesh": name, "colour": list(colour), "triangles": len(tri)})
            print(f"[import] {name}: {len(tri)} triangles", flush=True)
    with open(os.path.join(out_dir, "parts.json"), "w") as fh:
        json.dump({"source": BASE_URL, "license": "CC BY 4.0", "parts": parts}, fh, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usd", help="local lrmate200id4s.usd (default: download into --cache)")
    ap.add_argument("--cache", default=os.path.join(os.path.expanduser("~"), ".cache", "isaac_lrmate200id4s"))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    export(a.usd or download(a.cache), a.out)


if __name__ == "__main__":
    main()
