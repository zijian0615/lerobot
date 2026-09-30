"""Build the Omniverse / Isaac Sim stage of the EMS885 dual-xArm tabletop twin. numpy + OpenCV only (writes text .usda).

    uv run python xarm_ems885_twin/twin/build_usd.py

Inputs:  twin_config.json (arms, base poses, end effectors), assets/ (xArm6 + xArm Gripper STL from xarm_ros2),
         the Scaniverse GLB and its scan_alignment.json (from align_scan.py).
Writes:  xarm_ems885.usda       stage: both arms, the textured table scan, ground, lights, cameras, status lights, trails
         xarm_ems885_scan.jpg   the scan texture (referenced by the stage)
         xarm_ems885.twin.json  kinematic chain for omni_twin.py (pxr-free)

Same conventions as fanuc_lrmate200id_smc/twin/build_usd.py (verified in Isaac Sim 6.1): every articulated body is an Xform
[xformOp:translate, xformOp:orient, xformOp:orient:joint] and omni_twin.py only rewrites the ":joint" op. World = Robot1 base
frame, +Z up, metres, table top at z = 0.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glb import load_glb  # noqa: E402
from xarm_model import build_spec  # noqa: E402

HERE = Path(__file__).resolve().parent
TWIN = HERE.parent
CONFIG = TWIN / "twin_config.json"
ASSETS = TWIN / "assets"
OUT_USD = TWIN / "xarm_ems885.usda"
OUT_JSON = TWIN / "xarm_ems885.twin.json"
OUT_TEX = TWIN / "xarm_ems885_scan.jpg"

MATERIALS = {                       # name: (rgb, roughness, metallic)
    "xarm_white": ((0.86, 0.86, 0.84), 0.35, 0.0),
    "gripper_dark": ((0.12, 0.12, 0.13), 0.5, 0.2),
    "leaphand_black": ((0.08, 0.08, 0.09), 0.6, 0.0),
    "ground": ((0.30, 0.31, 0.33), 0.9, 0.0),
}
STATUS = {"live": (0.1, 0.9, 0.2), "stale": (1.0, 0.6, 0.0), "none": (0.9, 0.1, 0.1)}
TRAIL_RGB = {"xarm": (0.1, 0.6, 1.0), "xarm2": (1.0, 0.35, 0.1)}
STATUS_ABOVE_BASE_M = 0.85
CREASE_BIN = 0.5                    # face normals are bucketed per component to 0.5 (~30 deg) -> sharp CAD edges stay sharp
# Views. Operator stands behind the two bases (-x), looking across the table (+x).
PERSP_EYE, PERSP_TARGET = (-0.95, -0.43, 1.05), (0.35, -0.43, 0.05)
TOP_EYE = (0.30, -0.45, 1.70)       # straight down; image up = +x, image right = -y, like the real overhead camera
CAM_FOCAL_MM, CAM_FOVY_DEG, CAM_ASPECT = 24.0, 45.0, 16 / 9


def f(x):
    return f"{float(x):.7g}"


def vec(v):
    return "(" + ", ".join(f(x) for x in v) + ")"


class Usda:
    def __init__(self):
        self.lines, self.depth = [], 0

    def w(self, s=""):
        self.lines.append("    " * self.depth + s)

    def open(self, header, meta=()):
        if meta:
            self.w(header + " (")
            self.depth += 1
            for m in meta:
                self.w(m)
            self.depth -= 1
            self.w(")")
        else:
            self.w(header)
        self.w("{")
        self.depth += 1

    def close(self):
        self.depth -= 1
        self.w("}")

    def text(self):
        return "\n".join(self.lines) + "\n"


# ------------------------------------------------------------------------------------------------------------- meshes
def load_stl(path):
    """Binary STL -> (N, 3, 3) triangle corners (metres, as in xarm_ros2)."""
    data = Path(path).read_bytes()
    if data.startswith(b"version https://git-lfs"):
        raise SystemExit(f"{path} is a Git LFS pointer; fetch the meshes with  git lfs install && git lfs pull")
    n = int(np.frombuffer(data, np.uint32, 1, 80)[0])
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    if 84 + n * rec.itemsize != len(data):
        raise ValueError(f"{path}: not a binary STL")
    return np.frombuffer(data, rec, n, 84)["v"].astype(float)


def weld(tris):
    """Share vertices between triangles whose normals fall in the same ~30 deg bucket. -> points, faces, vertex normals."""
    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    area2 = np.linalg.norm(fn, axis=1)
    keep = area2 > 1e-14
    tris, fn, area2 = tris[keep], fn[keep], area2[keep]
    unit = fn / area2[:, None]
    corners = tris.reshape(-1, 3)
    bucket = np.repeat(np.round(unit / CREASE_BIN).astype(np.int64), 3, axis=0)
    qpos = np.round(corners / 1e-6).astype(np.int64)
    key = np.concatenate([qpos, bucket], axis=1)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    pts = corners[first]
    normals = np.zeros_like(pts)
    np.add.at(normals, inverse, np.repeat(fn, 3, axis=0))       # area-weighted
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return pts, inverse.reshape(-1, 3), normals


def write_mesh(u, name, pts, faces, normals=None, material=None, st=None, extra=()):
    meta = ['prepend apiSchemas = ["MaterialBindingAPI"]'] if material else []
    u.open(f'def Mesh "{name}"', meta)
    u.w(f"float3[] extent = [{vec(pts.min(0))}, {vec(pts.max(0))}]")
    u.w("int[] faceVertexCounts = [" + ", ".join(["3"] * len(faces)) + "]")
    u.w("int[] faceVertexIndices = [" + ", ".join(map(str, faces.ravel())) + "]")
    u.w("point3f[] points = [" + ", ".join(vec(p) for p in pts) + "]")
    if normals is not None:
        u.w("normal3f[] normals = [" + ", ".join(vec(n) for n in normals) + '] ( interpolation = "vertex" )')
    if st is not None:
        u.w("texCoord2f[] primvars:st = [" + ", ".join(vec(t) for t in st) + '] ( interpolation = "vertex" )')
    u.w('uniform token subdivisionScheme = "none"')
    u.w("uniform bool doubleSided = true")
    for line in extra:
        u.w(line)
    if material:
        u.w(f"rel material:binding = </World/Looks/{material}>")
    u.close()


def write_material(u, name, rgb, roughness=0.5, metallic=0.0, emissive=False):
    path = f"/World/Looks/{name}"
    u.open(f'def Material "{name}"')
    u.w(f"token outputs:surface.connect = <{path}/Shader.outputs:surface>")
    u.open('def Shader "Shader"')
    u.w('uniform token info:id = "UsdPreviewSurface"')
    u.w(f"color3f inputs:diffuseColor = {vec(rgb)}")
    if emissive:
        u.w(f"color3f inputs:emissiveColor = {vec(rgb)}")
    u.w(f"float inputs:roughness = {f(roughness)}")
    u.w(f"float inputs:metallic = {f(metallic)}")
    u.w("token outputs:surface")
    u.close()
    u.close()


def write_texture_material(u, name, tex_file):
    """UsdPreviewSurface fed by a UsdUVTexture reading primvars:st (same network as omni_twin's table plate)."""
    p = f"/World/Looks/{name}"
    u.open(f'def Material "{name}"')
    u.w(f"token outputs:surface.connect = <{p}/Shader.outputs:surface>")
    u.open('def Shader "Shader"')
    u.w('uniform token info:id = "UsdPreviewSurface"')
    u.w(f"color3f inputs:diffuseColor.connect = <{p}/tex.outputs:rgb>")
    u.w("float inputs:roughness = 0.9")
    u.w("token outputs:surface")
    u.close()
    u.open('def Shader "st"')
    u.w('uniform token info:id = "UsdPrimvarReader_float2"')
    u.w('string inputs:varname = "st"')
    u.w("float2 outputs:result")
    u.close()
    u.open('def Shader "tex"')
    u.w('uniform token info:id = "UsdUVTexture"')
    u.w(f"asset inputs:file = @./{tex_file}@")
    u.w(f"float2 inputs:st.connect = <{p}/st.outputs:result>")
    u.w('token inputs:sourceColorSpace = "sRGB"')
    u.w('token inputs:wrapS = "clamp"')
    u.w('token inputs:wrapT = "clamp"')
    u.w("float3 outputs:rgb")
    u.close()
    u.close()


def xform_ops(u, pos, quat, joint=False):
    u.w(f"double3 xformOp:translate = {vec(pos)}")
    u.w(f"quatf xformOp:orient = {vec(quat)}")
    order = ["xformOp:translate", "xformOp:orient"]
    if joint:
        u.w("quatf xformOp:orient:joint = (1, 0, 0, 0)")
        order.append("xformOp:orient:joint")
    u.w("uniform token[] xformOpOrder = [" + ", ".join(f'"{o}"' for o in order) + "]")


def look_at_matrix(eye, target, up=(0.0, 0.0, 1.0)):
    """USD camera transform (row vectors, translation in the last row). The camera looks down its -Z with +Y up."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    back = eye - target
    back /= np.linalg.norm(back)
    x = np.cross(up, back)
    x /= np.linalg.norm(x)
    y = np.cross(back, x)
    return [list(x) + [0], list(y) + [0], list(back) + [0], list(eye) + [1]]


def write_camera(u, name, matrix, focal=CAM_FOCAL_MM):
    aperture_v = 2 * focal * math.tan(math.radians(CAM_FOVY_DEG / 2))
    u.open(f'def Camera "{name}"')
    u.w(f"float focalLength = {f(focal)}")
    u.w(f"float horizontalAperture = {f(aperture_v * CAM_ASPECT)}")
    u.w(f"float verticalAperture = {f(aperture_v)}")
    u.w("float2 clippingRange = (0.01, 100)")
    u.w("matrix4d xformOp:transform = (" + ", ".join(vec(r) for r in matrix) + ")")
    u.w('uniform token[] xformOpOrder = ["xformOp:transform"]')
    u.close()


# -------------------------------------------------------------------------------------------------------------- build
def scan_geometry(cfg_dir, scan_cfg):
    """(points in world, faces, st, texture bytes, alignment dict) for the aligned GLB."""
    glb = (cfg_dir / scan_cfg["glb"]).resolve()
    align_path = (cfg_dir / scan_cfg["alignment"]).resolve()
    if not align_path.exists():
        raise SystemExit(f"{align_path} missing: run twin/align_scan.py first")
    align = json.loads(align_path.read_text())
    T = np.asarray(align["T_world_scan"], float)
    mesh = load_glb(glb)
    pts = mesh.positions @ T[:3, :3].T + T[:3, 3]
    st = np.c_[mesh.uv[:, 0], 1.0 - mesh.uv[:, 1]]           # glTF v runs down the image, USD st up
    return pts, mesh.faces, st, mesh.texture_bytes, align


def build(cfg_path=CONFIG, out_usd=OUT_USD, out_json=OUT_JSON, out_tex=OUT_TEX, with_scan=True):
    cfg = json.loads(Path(cfg_path).read_text())
    cfg_dir = Path(cfg_path).resolve().parent
    spec = build_spec(cfg)
    u = Usda()
    u.w("#usda 1.0")
    u.w("(")
    u.w('    defaultPrim = "World"')
    u.w("    metersPerUnit = 1")
    u.w('    upAxis = "Z"')
    u.w('    doc = "Generated by xarm_ems885_twin/twin/build_usd.py. Do not edit; rebuild."')
    u.w(")")
    u.w()
    u.open('def Xform "World"')

    scan = scan_geometry(cfg_dir, cfg["scan"]) if with_scan else None
    u.open('def Scope "Looks"')
    for name, (rgb, rough, metal) in MATERIALS.items():
        write_material(u, name, rgb, rough, metal)
    for arm in spec["arms"]:
        for k, rgb in STATUS.items():
            write_material(u, f"status_{arm}_{k}", rgb, emissive=True)
        write_material(u, f"trail_{arm}", TRAIL_RGB.get(arm, (0.2, 0.8, 0.8)), emissive=True)
    if scan is not None:
        write_texture_material(u, "scan", Path(out_tex).name)
    u.close()

    mesh_cache = {}
    children = {}
    for b in spec["bodies"]:
        children.setdefault(b["parent"], []).append(b)

    def emit(b, path):
        path = f"{path}/{b['name']}"
        b["path"] = path
        u.open(f'def Xform "{b["name"]}"')
        xform_ops(u, b["pos"], b["quat"], joint=b["joint"] is not None)
        for i, g in enumerate(b["geoms"]):
            if g["type"] == "mesh":
                if g["file"] not in mesh_cache:
                    mesh_cache[g["file"]] = weld(load_stl(ASSETS / g["file"]))
                pts, faces, normals = mesh_cache[g["file"]]
                write_mesh(u, "visual" if i == 0 else f"visual_{i}", pts, faces, normals, b["material"])
            elif g["type"] == "box":
                u.open(f'def Cube "{g["name"]}"', ['prepend apiSchemas = ["MaterialBindingAPI"]'])
                u.w("double size = 1")
                u.w(f"rel material:binding = </World/Looks/{b['material']}>")
                u.w(f"double3 xformOp:translate = {vec(g['pos'])}")
                u.w(f"float3 xformOp:scale = {vec(g['size'])}")
                u.w('uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]')
                u.close()
        for c in children.get(b["name"], []):
            emit(c, path)
        u.close()

    u.open('def Xform "robots"')
    for top in children[None]:
        emit(top, spec["root"])
    u.close()

    ground_z = -float(cfg.get("table_height_m", 0.75))   # the scan does not reach the floor
    if scan is not None:
        pts, faces, st, tex, align = scan
        write_mesh(u, "scan", pts, faces, st=st, material="scan")
        Path(out_tex).write_bytes(tex)
        spec["scan"] = {"alignment_metrics": align.get("metrics"), "ground_z": ground_z}

    u.open('def Mesh "ground"', ['prepend apiSchemas = ["MaterialBindingAPI"]'])
    u.w("int[] faceVertexCounts = [4]")
    u.w("int[] faceVertexIndices = [0, 1, 2, 3]")
    u.w(f"point3f[] points = [(-3, -3, {f(ground_z)}), (3, -3, {f(ground_z)}), (3, 3, {f(ground_z)}), (-3, 3, {f(ground_z)})]")
    u.w('normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] ( interpolation = "faceVarying" )')
    u.w('uniform token subdivisionScheme = "none"')
    u.w("rel material:binding = </World/Looks/ground>")
    u.close()

    status_prims, trail_prims = {}, {}
    for arm, cfg_arm in cfg["arms"].items():
        bx, by, bz = cfg_arm.get("base_xyz", (0, 0, 0))
        status_prims[arm] = f"/World/status_{arm}"
        u.open(f'def Sphere "status_{arm}"', ['prepend apiSchemas = ["MaterialBindingAPI"]'])
        u.w("double radius = 0.03")
        u.w(f"rel material:binding = </World/Looks/status_{arm}_none>")
        u.w(f"double3 xformOp:translate = {vec((bx - 0.12, by, bz + STATUS_ABOVE_BASE_M))}")
        u.w('uniform token[] xformOpOrder = ["xformOp:translate"]')
        u.close()
        trail_prims[arm] = f"/World/trail_{arm}"
        u.open(f'def BasisCurves "trail_{arm}"', ['prepend apiSchemas = ["MaterialBindingAPI"]'])
        u.w('uniform token type = "linear"')
        u.w("int[] curveVertexCounts = [2]")
        u.w("point3f[] points = [(0, 0, 0), (0, 0, 0)]")
        u.w('float[] widths = [0.006] ( interpolation = "constant" )')
        u.w(f"rel material:binding = </World/Looks/trail_{arm}>")
        u.close()

    u.open('def DomeLight "dome"')
    u.w("float inputs:intensity = 900")
    u.close()
    u.open('def DistantLight "sun"')
    u.w("float inputs:intensity = 2500")
    u.w("float inputs:angle = 1.5")
    u.w("float3 xformOp:rotateXYZ = (-50, 0, 35)")
    u.w('uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]')
    u.close()
    write_camera(u, "Camera", look_at_matrix(PERSP_EYE, PERSP_TARGET))
    tx, ty, tz = TOP_EYE
    write_camera(u, "TopCamera", [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [tx, ty, tz, 1]], focal=18.0)
    u.close()   # World

    spec["prims"] = {"status": status_prims, "trail": trail_prims,
                     "cameras": {"persp": "/World/Camera", "top": "/World/TopCamera"}}
    Path(out_usd).write_text(u.text())
    Path(out_json).write_text(json.dumps(spec, indent=1) + "\n")
    return spec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--out", type=Path, default=OUT_USD)
    ap.add_argument("--no-scan", action="store_true", help="robots only (no GLB / alignment needed)")
    a = ap.parse_args(argv)
    out_json = a.out.with_suffix("").with_suffix(".twin.json") if a.out != OUT_USD else OUT_JSON
    spec = build(a.config, a.out, out_json, a.out.with_name(OUT_TEX.name), with_scan=not a.no_scan)
    size = a.out.stat().st_size / 1e6
    print(f"wrote {a.out} ({size:.1f} MB), {out_json} ({len(spec['bodies'])} bodies, arms {list(spec['arms'])})")


if __name__ == "__main__":
    main()
