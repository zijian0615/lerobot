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

"""Cut a scanned object out of its photogrammetry scan (object lying on a table) and write a physics USD asset.

    uv run python twin/crop_object.py --scan ../examples/cosmos_edge_fanuc/scanner/objects/Pen.zip \\
        --name pen --mass 0.015 --out ../examples/cosmos_edge_fanuc/scanner/objects/pen

Input: a zip (or folder) with one textured OBJ (RealityCapture / Object Capture export: metres, Y up) of the object
lying on a flat surface. Steps (numpy + Pillow; writes text .usda, no pxr needed):
1. RANSAC plane fit of the surface; heights above it.
2. faces whose three vertices are > --cut-mm above the surface, split into connected pieces; the object is the piece
   holding the most vertices > --seed-mm (thin strings, hair and scan-border clutter stay out).
3. object frame: z = surface normal (up), x = the long axis of the footprint, origin = footprint centre on the surface.
4. visual mesh = the cropped scan (open underneath: the surface hid it), textured with the scan's diffuse map;
   collision = an upright box or, for elongated objects, a capsule along x, fitted to the crop.
Writes <name>.usda (Xform with RigidBodyAPI + MassAPI, visual Mesh, invisible collider, UsdPreviewSurface material),
the texture, <name>.json (dimensions, collider, mass, source) and preview.png (top and side views).

--container (open boxes such as bins): the scan's background often fuses into floating sheets touching the object, and
the table can lose to them in a plane fit. So: keep the faces whose texture is saturated (--min-saturation), take the
largest connected piece, orient the box by its dominant face normals with the axis nearest the scan's +Y as up (Object
Capture exports are gravity aligned), put the origin at the bottom centre, and collide with a floor slab plus four walls
whose thickness and floor height come from the inward-facing wall and floor faces, so things can be dropped inside.
    uv run python twin/crop_object.py --scan ../examples/cosmos_edge_fanuc/scanner/objects/Bluebin.zip \
        --name blue_bin --mass 0.03 --container --out ../examples/cosmos_edge_fanuc/scanner/objects/blue_bin
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import zipfile

import numpy as np


def load_obj(path):
    """Positions, UVs and triangles (vertex and UV indices) of a triangulated OBJ, plus its diffuse texture path."""
    verts, uvs, fv, ft = [], [], [], []
    mtl = None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            uvs.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("f "):
            parts = [p.split("/") for p in line.split()[1:]]
            for k in range(1, len(parts) - 1):  # fan-triangulate polygons
                tri = (parts[0], parts[k], parts[k + 1])
                fv.append([int(p[0]) - 1 for p in tri])
                ft.append([int(p[1]) - 1 if len(p) > 1 and p[1] else 0 for p in tri])
        elif line.startswith("mtllib"):
            mtl = os.path.join(os.path.dirname(path), line.split(None, 1)[1].strip())
    texture = None
    if mtl and os.path.isfile(mtl):
        for line in open(mtl, encoding="utf-8", errors="replace"):
            if line.strip().startswith("map_Kd"):
                texture = os.path.join(os.path.dirname(path), line.split(None, 1)[1].strip())
    return np.array(verts), np.array(uvs), np.array(fv), np.array(ft), texture


def fit_plane(points, tol=0.0015, iters=3000, seed=0):
    """Unit normal (pointing to the side most points stand on, i.e. +Y-ish for an OBJ) and a point of the plane."""
    rng = np.random.default_rng(seed)
    best = (-1, None, None)
    for _ in range(iters):
        s = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        count = int((np.abs((points - s[0]) @ n) < tol).sum())
        if count > best[0]:
            best = (count, n, s[0])
    inl = np.abs((points - best[2]) @ best[1]) < tol
    centre = points[inl].mean(0)
    normal = np.linalg.svd(points[inl] - centre)[2][2]
    above = (points - centre) @ normal
    if np.percentile(above, 99) < -np.percentile(above, 1):  # the object sticks out on the normal's side
        normal = -normal
    return normal, centre, float(((points[inl] - centre) @ normal).std())


def components(faces, n_verts):
    """Connected-component label per face (faces sharing a vertex are connected)."""
    parent = np.arange(n_verts)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b, c in faces:
        for x, y in ((a, b), (b, c)):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
    return np.array([find(f[0]) for f in faces])


def crop(verts, faces, normal, centre, cut_m, seed_m):
    height = (verts - centre) @ normal
    keep = (height[faces] > cut_m).all(1)
    sub = faces[keep]
    label = components(sub, len(verts))
    seeds = height > seed_m
    best, best_count = None, 0
    for lab in np.unique(label):
        count = int(seeds[np.unique(sub[label == lab])].sum())
        if count > best_count:
            best, best_count = lab, count
    if best is None:
        raise RuntimeError("nothing stands above the surface; lower --seed-mm")
    return np.flatnonzero(keep)[label == best]


def object_frame(verts, normal, centre):
    """4x4 from scan coordinates to the object frame (z up = normal, x = footprint long axis, origin on the surface)."""
    rel = verts - centre
    ref = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 0, 1.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    uv = np.c_[rel @ u, rel @ v]
    mean = uv.mean(0)
    axis = np.linalg.svd(uv - mean)[2][0]
    x = axis[0] * u + axis[1] * v
    y = np.cross(normal, x)
    rot = np.stack([x, y, normal])  # rows: object axes in scan coordinates
    local = rel @ rot.T
    mid = (local[:, :2].min(0) + local[:, :2].max(0)) / 2
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = -rot @ centre - np.r_[mid, 0.0]
    return out


def face_colours(uvs, ft, texture):
    """HSV (0..1) of the texture at each face's UV centroid."""
    from PIL import Image

    tex = np.asarray(Image.open(texture).convert("HSV")).astype(np.float64) / 255.0
    h, w = tex.shape[:2]
    c = uvs[ft].mean(1)
    return tex[np.clip(((1 - c[:, 1]) * h).astype(int), 0, h - 1), np.clip((c[:, 0] * w).astype(int), 0, w - 1)]


def _dominant_direction(normals, areas, mask, cos=math.cos(math.radians(8)), samples=400, seed=0):
    idx = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    cand = normals[rng.choice(idx, min(samples, len(idx)), p=areas[idx] / areas[idx].sum(), replace=False)]
    best = max(cand, key=lambda d: areas[idx][np.abs(normals[idx] @ d) > cos].sum())
    sel = idx[np.abs(normals[idx] @ best) > cos]
    m = (normals[sel] * np.sign(normals[sel] @ best)[:, None] * areas[sel, None]).sum(0)
    return m / np.linalg.norm(m)


def container_frame(verts, faces, up_hint=(0.0, 1.0, 0.0)):
    """4x4 scan -> box frame (axes = dominant face normals, z = the one nearest up_hint, origin = bottom centre)."""
    tri = verts[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(2 * areas, 1e-15)[:, None]
    d1 = _dominant_direction(normals, areas, np.ones(len(faces), bool))
    d2 = _dominant_direction(normals, areas, np.abs(normals @ d1) < 0.2)
    d2 = d2 - d1 * (d2 @ d1)
    d2 /= np.linalg.norm(d2)
    axes = [d1, d2, np.cross(d1, d2)]
    up = int(np.argmax([abs(a @ np.asarray(up_hint)) for a in axes]))
    z = axes[up] * np.sign(axes[up] @ np.asarray(up_hint))
    x = axes[(up + 1) % 3]
    y = np.cross(z, x)
    rot = np.stack([x, y, z])
    local = tri.mean(1) @ rot.T
    lo, hi = np.percentile(local, 0.5, 0), np.percentile(local, 99.5, 0)
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = -np.r_[(lo[:2] + hi[:2]) / 2, lo[2]]
    return out


def fit_container(points, faces):
    """Floor slab + four walls from the box outline, the inward-facing wall faces and the upward-facing floor faces."""
    tri = points[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = cross / np.maximum(np.linalg.norm(cross, axis=1), 1e-15)[:, None]
    cen = tri.mean(1)
    lo, hi = np.percentile(cen, 0.5, 0), np.percentile(cen, 99.5, 0)
    half = (hi[:2] - lo[:2]) / 2
    height = float(hi[2])
    inner = []
    for k in range(2):
        # wall faces facing the centre: their normal points against their own offset along this axis
        sel = (np.abs(normals[:, k]) > 0.8) & (normals[:, k] * cen[:, k] < 0) & (np.abs(cen[:, k]) < 0.97 * half[k]) \
            & (cen[:, 2] > 0.2 * height) & (cen[:, 2] < 0.9 * height)
        inner.append(float(np.percentile(np.abs(cen[sel, k]), 50)) if sel.sum() > 20 else 0.8 * float(half[k]))
    floor_sel = (normals[:, 2] > 0.8) & (np.abs(cen[:, 0]) < inner[0]) & (np.abs(cen[:, 1]) < inner[1]) & (cen[:, 2] < 0.9 * height)
    floor = float(np.percentile(cen[floor_sel, 2], 50)) if floor_sel.sum() > 20 else 0.1 * height
    tx, ty = float(half[0] - inner[0]), float(half[1] - inner[1])
    boxes = [{"size": [2 * float(half[0]), 2 * float(half[1]), floor], "center": [0.0, 0.0, floor / 2]}]
    for sx in (-1, 1):
        boxes.append({"size": [tx, 2 * float(half[1]), height], "center": [sx * (float(half[0]) - tx / 2), 0.0, height / 2]})
    for sy in (-1, 1):
        boxes.append({"size": [2 * inner[0], ty, height], "center": [0.0, sy * (float(half[1]) - ty / 2), height / 2]})
    return {"type": "container", "outer": [2 * float(half[0]), 2 * float(half[1]), height],
            "opening": [2 * inner[0], 2 * inner[1]], "wall": [tx, ty], "floor": floor, "boxes": boxes}


def fit_collider(points):
    lo, hi = points.min(0), points.max(0)
    size = hi - lo
    height = float(hi[2])
    if size[0] > 2.5 * max(size[1], height):  # elongated: capsule along x lying on the surface
        # a round object lying down is as tall as it is thick; the width is inflated by the scan's flare at the surface
        radius = height / 2
        return {"type": "capsule", "axis": "X", "radius": radius, "height": float(max(size[0] - 2 * radius, 0.0)),
                "center": [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), radius]}
    return {"type": "box", "size": [float(size[0]), float(size[1]), height],
            "center": [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), height / 2]}


def _vec(v):
    return "(" + ", ".join(f"{float(x):.9g}" for x in v) + ")"


def write_usda(path, name, points, faces, st, texture_name, collider, mass, friction=0.8):
    lines = ["#usda 1.0", "(", f'    defaultPrim = "{name}"', "    metersPerUnit = 1", '    upAxis = "Z"',
             '    doc = "Generated by fanuc_lrmate200id_smc/twin/crop_object.py from a photogrammetry scan."', ")", "",
             f'def Xform "{name}" (', '    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]', ")", "{",
             f"    float physics:mass = {mass:.6g}"]
    lines += ['    def Scope "Looks"', "    {", '        def Material "scan"', "        {",
              f"            token outputs:surface.connect = </{name}/Looks/scan/Shader.outputs:surface>",
              '            def Shader "Shader"', "            {", '                uniform token info:id = "UsdPreviewSurface"',
              f"                color3f inputs:diffuseColor.connect = </{name}/Looks/scan/tex.outputs:rgb>",
              "                float inputs:roughness = 0.6", "                token outputs:surface", "            }",
              '            def Shader "st"', "            {", '                uniform token info:id = "UsdPrimvarReader_float2"',
              '                string inputs:varname = "st"', "                float2 outputs:result", "            }",
              '            def Shader "tex"', "            {", '                uniform token info:id = "UsdUVTexture"',
              f"                asset inputs:file = @./{texture_name}@",
              f"                float2 inputs:st.connect = </{name}/Looks/scan/st.outputs:result>",
              '                token inputs:sourceColorSpace = "sRGB"', "                float3 outputs:rgb", "            }",
              "        }", '        def Material "physics" (', '            prepend apiSchemas = ["PhysicsMaterialAPI"]', "        )",
              "        {", f"            float physics:staticFriction = {friction}", f"            float physics:dynamicFriction = {friction}",
              "            float physics:restitution = 0", "        }", "    }"]
    lines += ['    def Mesh "visual" (', '        prepend apiSchemas = ["MaterialBindingAPI"]', "    )", "    {",
              f"        float3[] extent = [{_vec(points.min(0))}, {_vec(points.max(0))}]",
              "        int[] faceVertexCounts = [" + ", ".join(["3"] * len(faces)) + "]",
              "        int[] faceVertexIndices = [" + ", ".join(str(int(i)) for i in faces.ravel()) + "]",
              "        point3f[] points = [" + ", ".join(_vec(p) for p in points) + "]",
              "        texCoord2f[] primvars:st = [" + ", ".join(_vec(t) for t in st.reshape(-1, 2)) + '] ( interpolation = "faceVarying" )',
              '        uniform token subdivisionScheme = "none"', "        uniform bool doubleSided = true",
              f"        rel material:binding = </{name}/Looks/scan>", "    }"]
    c = collider
    if c["type"] == "container":
        body = []
        for i, box in enumerate(c["boxes"]):
            body += [f'    def Cube "collision_{i}" (', '        prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]',
                     "    )", "    {", "        double size = 1", f"        float3 xformOp:scale = {_vec(box['size'])}",
                     f"        double3 xformOp:translate = {_vec(box['center'])}",
                     '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
                     '        uniform token purpose = "guide"', f"        rel material:binding:physics = </{name}/Looks/physics>", "    }"]
        lines += body + ["}", ""]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        return
    if c["type"] == "capsule":
        body = ['    def Capsule "collision" (', '        prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]', "    )",
                "    {", f'        uniform token axis = "{c["axis"]}"', f"        double radius = {c['radius']:.6g}",
                f"        double height = {c['height']:.6g}"]
    else:
        body = ['    def Cube "collision" (', '        prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]', "    )",
                "    {", "        double size = 1", f"        float3 xformOp:scale = {_vec(c['size'])}"]
    body += [f"        double3 xformOp:translate = {_vec(c['center'])}",
             '        uniform token[] xformOpOrder = ["xformOp:translate"' + (', "xformOp:scale"' if c["type"] == "box" else "") + "]",
             '        uniform token purpose = "guide"', f'        rel material:binding:physics = </{name}/Looks/physics>', "    }"]
    lines += body + ["}", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def preview(path, points, faces, st, texture, px_per_m=2500):
    from PIL import Image

    tex = np.asarray(Image.open(texture).convert("RGB"))
    h, w = tex.shape[:2]
    tri = points[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    per = np.maximum(np.ceil(area * px_per_m**2 * 6).astype(int), 1)
    fi = np.repeat(np.arange(len(faces)), per)
    r = np.random.default_rng(0).random((len(fi), 2))
    r[r.sum(1) > 1] = 1 - r[r.sum(1) > 1]
    wts = np.c_[1 - r.sum(1), r]
    p = np.einsum("nc,ncd->nd", wts, points[faces[fi]])
    t = np.einsum("nc,ncd->nd", wts, st[fi])
    col = tex[np.clip(((1 - t[:, 1]) * h).astype(int), 0, h - 1), np.clip((t[:, 0] * w).astype(int), 0, w - 1)]
    views = []
    for a, b, depth, flip in ((0, 1, 2, True), (0, 2, 1, True)):  # top (x right, y up), side (x right, z up)
        u = ((p[:, a] - p[:, a].min()) * px_per_m).astype(int)
        v = ((p[:, b] - p[:, b].min()) * px_per_m).astype(int)
        img = np.full((v.max() + 1, u.max() + 1, 3), 40, np.uint8)
        order = np.argsort(p[:, depth] if a == 0 and b == 1 else -p[:, depth])
        img[v[order], u[order]] = col[order]
        views.append(img[::-1] if flip else img)
    width = max(v.shape[1] for v in views)
    pad = [np.pad(v, ((0, 10), (0, width - v.shape[1]), (0, 0)), constant_values=40) for v in views]
    Image.fromarray(np.vstack(pad)).save(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True, help="zip or folder with the OBJ, MTL and textures")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mass", type=float, required=True, help="kg")
    ap.add_argument("--cut-mm", type=float, default=2.0, help="drop faces lower than this above the surface")
    ap.add_argument("--seed-mm", type=float, default=4.0, help="the object is the piece with most vertices above this")
    ap.add_argument("--friction", type=float, default=0.8)
    ap.add_argument("--container", action="store_true", help="open box (bin): colour crop, box axes, hollow collider")
    ap.add_argument("--min-saturation", type=float, default=0.35, help="--container: keep faces at least this saturated")
    a = ap.parse_args(argv)

    work = tempfile.mkdtemp()
    try:
        src = a.scan
        if zipfile.is_zipfile(src):
            with zipfile.ZipFile(src) as zf:
                zf.extractall(work)
            src = work
        obj = next(os.path.join(root, f) for root, _, files in os.walk(src) for f in files if f.lower().endswith(".obj"))
        verts, uvs, fv, ft, texture = load_obj(obj)
        if texture is None:
            raise RuntimeError(f"{obj}: no map_Kd texture")
        if a.container:
            hsv = face_colours(uvs, ft, texture)
            saturated = np.flatnonzero((hsv[:, 1] > a.min_saturation) & (hsv[:, 2] > 0.08))
            label = components(fv[saturated], len(verts))
            labels, counts = np.unique(label, return_counts=True)
            chosen = saturated[label == labels[counts.argmax()]]
            to_obj = container_frame(verts, fv[chosen])
            plane_std = float("nan")
        else:
            normal, centre, plane_std = fit_plane(verts)
            chosen = crop(verts, fv, normal, centre, a.cut_mm / 1000, a.seed_mm / 1000)
            to_obj = object_frame(verts[np.unique(fv[chosen])], normal, centre)
        used, remap = np.unique(fv[chosen], return_inverse=True)
        points = verts[used] @ to_obj[:3, :3].T + to_obj[:3, 3]
        faces = remap.reshape(-1, 3)
        st = uvs[ft[chosen]]
        collider = fit_container(points, faces) if a.container else fit_collider(points)
        size = points.max(0) - points.min(0)

        os.makedirs(a.out, exist_ok=True)
        tex_name = f"{a.name}_diffuse" + os.path.splitext(texture)[1]
        shutil.copyfile(texture, os.path.join(a.out, tex_name))
        write_usda(os.path.join(a.out, f"{a.name}.usda"), a.name, points, faces, st, tex_name, collider, a.mass, a.friction)
        preview(os.path.join(a.out, "preview.png"), points, faces, st, texture)
        info = {"name": a.name, "source": os.path.abspath(a.scan), "size_m": size.round(4).tolist(), "mass_kg": a.mass,
                "collider": collider, "triangles": int(len(faces)), "surface_fit_std_mm": round(1000 * plane_std, 2),
                "frame": "z up from the surface it lay on, x along the long axis, origin at the footprint centre on the surface",
                "note": "visual mesh is open underneath (the surface hid it); collide with the primitive"}
        json.dump(info, open(os.path.join(a.out, f"{a.name}.json"), "w"), indent=1)
        print(f"[crop] {a.name}: {len(faces)} triangles, size {np.round(size * 1000, 1).tolist()} mm, "
              f"collider {collider['type']}" + (f" (opening {np.round(np.array(collider['opening']) * 1000, 1).tolist()} mm, "
              f"wall {np.round(np.array(collider['wall']) * 1000, 1).tolist()} mm, floor {collider['floor'] * 1000:.1f} mm)"
              if a.container else f", surface fit {1000 * plane_std:.2f} mm") + f" -> {a.out}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
