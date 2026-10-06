"""
Convert the scanned site assets (RealityCapture OBJ + textures) into USD.

    /isaacsim/python.sh import_assets.py --src <CSI-site-digital-model dir>

Writes assets/<name>.usd plus a copy of the diffuse texture. These are
photogrammetry scans of the real bins, so they come in Y-up and need rotating
into the Z-up stage; the conversion also re-centres each bin on its own
footprint and drops it so its base sits at z=0, which is what the scene code
expects to place.
"""
import argparse, os, shutil, sys

parser = argparse.ArgumentParser()
parser.add_argument("--src", default=os.path.expanduser(
    "~/Downloads/CSI-site-digital-model"))
parser.add_argument("--out", default=None)
args = parser.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(args.out or os.path.join(HERE, "assets"))

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf

os.makedirs(OUT, exist_ok=True)

# name -> (obj relative path, texture relative path)
SOURCES = {
    "orange_bin": ("org/Org.obj", "org/tex_u1_v1_diffuse.jpg"),
    "blue_bin":   ("blue/Bluebin.obj", "blue/tex_u1_v1_diffuse.jpg"),
}


def _compact(v, f, uv):
    """Drop unreferenced vertices and renumber the faces."""
    used = np.unique(f)
    remap = -np.ones(len(v), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return v[used], remap[f].astype(np.int32), (None if uv is None else uv[used])


def convert(name, obj_path, tex_path, open_mouth=True):
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)

    # RealityCapture exports Y-up; the stage is Z-up
    v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)
    uv_all = getattr(mesh.visual, "uv", None)
    uv_all = None if uv_all is None else np.asarray(uv_all)

    v[:, 2] -= v[:, 2].min()

    # --- 1. drop the captured tabletop skirt -------------------------------
    # Photogrammetry grabs a patch of the surface the object stood on. Measure
    # the footprint from the part that actually stands up, then cut to it.
    height = v[:, 2].max() - v[:, 2].min()
    tall = v[v[:, 2] > 0.35 * height]
    if len(tall) > 50:
        # Near-full extent, rejecting only isolated outliers. Tight percentiles
        # (2/98) trim the real corners and shrank a 68 mm bin to 57 mm; raw
        # min/max lets stray fragments stretch it to 85 x 77 mm. 0.5/99.5 keeps
        # the corners and drops the specks.
        lo = np.percentile(tall[:, :2], 0.5, axis=0) - 0.001
        hi = np.percentile(tall[:, :2], 99.5, axis=0) + 0.001
        inside = ((v[:, 0] >= lo[0]) & (v[:, 0] <= hi[0]) &
                  (v[:, 1] >= lo[1]) & (v[:, 1] <= hi[1]))
        f = f[inside[f].all(axis=1)]
        v, f, uv_all = _compact(v, f, uv_all)
        print(f"   skirt cut: {len(v)} verts / {len(f)} tris")

    # NOTE: do not try to keep only the largest connected component here. This
    # is RealityCapture output and the orange bin alone splits into 657 shells,
    # so "largest" is a small fragment -- doing that reduced a 68 mm bin to a
    # 49 x 11 x 43 mm sliver.

    # --- 2. open the mouth -------------------------------------------------
    # The scan could not see into the bin so it capped the opening with a
    # shallow dish. Remove only that cap: near the top, inside the mouth, and
    # facing upward. The normal test is what keeps this off the side walls --
    # without it the crop tore a hole straight through one of them.
    if open_mouth:
        height = v[:, 2].max() - v[:, 2].min()
        cx = 0.5 * (v[:, 0].min() + v[:, 0].max())
        cy = 0.5 * (v[:, 1].min() + v[:, 1].max())
        hx = 0.5 * (v[:, 0].max() - v[:, 0].min())
        hy = 0.5 * (v[:, 1].max() - v[:, 1].min())
        tri = v[f]
        cen = tri.mean(axis=1)
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(n, axis=1)
        nz = np.where(ln > 1e-12, n[:, 2] / np.maximum(ln, 1e-12), 0.0)
        # Conservative on purpose. The walls are ~5 mm on a ~58 mm bin, so the
        # mouth is about 0.82 of the half-extent; anything wider than ~0.65
        # starts eating wall, and the scan's walls are not perfectly vertical so
        # a loose normal test eats them too. Both thresholds were tuned down
        # after looser ones punched holes clean through the front face.
        dish = ((np.abs(cen[:, 0] - cx) < 0.65 * hx) &
                (np.abs(cen[:, 1] - cy) < 0.65 * hy) &
                (cen[:, 2] > 0.55 * height) &
                (np.abs(nz) > 0.70))
        f = f[~dish]
        v, f, uv_all = _compact(v, f, uv_all)
        print(f"   mouth opened: removed {int(dish.sum())} cap tris")

    # --- 3. trim scan noise above the rim ----------------------------------
    # Photogrammetry leaves wispy fragments sticking up past the rim. The rim
    # itself is a dense ring of faces; the wisps are not, so cut at the highest
    # level where there is still a substantial amount of geometry.
    cen_z = v[f].mean(axis=1)[:, 2]
    hist, edges = np.histogram(cen_z, bins=60)
    thresh = 0.04 * hist.max()
    top_bin = max(i for i in range(len(hist)) if hist[i] >= thresh)
    cut = edges[min(top_bin + 1, len(edges) - 1)]
    n_before = len(f)
    f = f[cen_z <= cut]
    v, f, uv_all = _compact(v, f, uv_all)
    if n_before != len(f):
        print(f"   trimmed {n_before - len(f)} tris above the rim")

    # centre on the footprint, sit the base on z = 0
    v[:, 0] -= 0.5 * (v[:, 0].min() + v[:, 0].max())
    v[:, 1] -= 0.5 * (v[:, 1].min() + v[:, 1].max())
    v[:, 2] -= v[:, 2].min()
    size = v.max(0) - v.min(0)

    stage = Usd.Stage.CreateNew(os.path.join(OUT, f"{name}.usd"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())
    m = UsdGeom.Mesh.Define(stage, f"/{name}/mesh")
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in v])
    m.CreateFaceVertexCountsAttr([3] * len(f))
    m.CreateFaceVertexIndicesAttr(f.reshape(-1).tolist())
    m.CreateSubdivisionSchemeAttr().Set("none")

    uv = uv_all
    if uv is not None and len(uv) == len(v):
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.vertex)
        pv.Set([Gf.Vec2f(float(a), float(b)) for a, b in uv])

    tex_name = f"{name}_diffuse.jpg"
    shutil.copy(tex_path, os.path.join(OUT, tex_name))

    mat = UsdShade.Material.Define(stage, f"/{name}/Looks/mat")
    sh = UsdShade.Shader.Define(stage, f"/{name}/Looks/mat/Shader")
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(f"./{tex_name}"))
    sh.CreateInput("reflection_roughness_constant",
                   Sdf.ValueTypeNames.Float).Set(0.55)
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)

    stage.GetRootLayer().Save()
    print(f"[assets] {name}: {len(v)} verts, {len(f)} tris, "
          f"size {np.round(size*1000, 1)} mm -> {name}.usd")
    return size


OPEN = os.environ.get("OPEN_MOUTH", "1") != "0"
for name, (rel_obj, rel_tex) in SOURCES.items():
    obj = os.path.join(args.src, rel_obj)
    tex = os.path.join(args.src, rel_tex)
    if not os.path.exists(obj):
        print(f"[assets] SKIP {name}: {obj} not found", file=sys.stderr)
        continue
    convert(name, obj, tex, open_mouth=OPEN)

print(f"[assets] written to {OUT}")
sim_app.close()
