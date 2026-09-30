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

"""Place one Scaniverse table mesh in the twin and render the overhead camera.

    uv run python twin/preview_scan.py --prepare
    OMNI_KIT_ACCEPT_EULA=YES LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 \\
      ~/isaacsim-venv/bin/python twin/preview_scan.py
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from joint_map import FINGER_OPEN_M  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(HERE, "..", "fanuc_lrmate200id_smc.usda")
CHAIN = os.path.join(HERE, "..", "fanuc_lrmate200id_smc.twin.json")
SCAN_ZIP = os.path.join(
    HERE, "..", "..", "examples", "cosmos_edge_fanuc", "scanner", "Scaniverse 2026-09-22 183745.zip"
)
WORK = os.path.join(HERE, "..", "..", "examples", "cosmos_edge_fanuc", "scanner", "scan_183745")
# Arm folded over the base, matching the live video0/video2 pair (tool +Z down).
FOLDED_Q = np.array([0.0, -0.611, -1.035, 0.0, -1.147, 0.0])
# The scan's yellow tape is one border. The calibrated workspace puts that border at y = +0.35 m.
BORDER_Y = 0.35
LIFT_M = 0.003


def _load_obj(path):
    verts, uvs, faces, face_uv = [], [], [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("vt "):
            uvs.append([float(x) for x in line.split()[1:3]])
        elif line.startswith("f "):
            fv, ft = [], []
            for part in line.split()[1:]:
                bits = part.split("/")
                fv.append(int(bits[0]) - 1)
                ft.append(int(bits[1]) - 1 if len(bits) > 1 and bits[1] else 0)
            faces.append(fv)
            face_uv.append(ft)
    return (
        np.asarray(verts, np.float64),
        np.asarray(uvs, np.float64),
        np.asarray(faces, np.int32),
        np.asarray(face_uv, np.int32),
    )


def prepare(zip_path=SCAN_ZIP, work=WORK) -> str:
    """Unpack the scan and write a table-frame mesh. Scaniverse is Y-up; the twin is Z-up."""
    from PIL import Image

    os.makedirs(work, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(work)
    folder = next(name for name in os.listdir(work) if name.startswith("Scaniverse"))
    root = os.path.join(work, folder)
    obj = next(os.path.join(root, name) for name in os.listdir(root) if name.endswith(".obj"))
    jpg = next(os.path.join(root, name) for name in os.listdir(root) if name.endswith(".jpg"))
    pos, uv, faces, face_uv = _load_obj(obj)
    tex = np.asarray(Image.open(jpg).convert("RGB").resize((2048, 1024), Image.BILINEAR))
    height, width = tex.shape[:2]
    acc = np.zeros((len(pos), 2))
    count = np.zeros(len(pos))
    for fv, ft in zip(faces, face_uv, strict=True):
        for vertex, texcoord in zip(fv, ft, strict=True):
            acc[vertex] += uv[texcoord]
            count[vertex] += 1
    good = count > 0
    acc[good] /= count[good, None]
    px = np.clip((acc[:, 0] * (width - 1)).astype(int), 0, width - 1)
    py = np.clip(((1.0 - acc[:, 1]) * (height - 1)).astype(int), 0, height - 1)
    red, green, blue = tex[py, px].astype(np.int16).T
    yellow = (red > 160) & (green > 140) & (blue < 120) & (red + green > blue * 2.2)
    if int(yellow.sum()) < 50:
        raise RuntimeError(f"yellow tape not found ({int(yellow.sum())} vertices)")
    tape_z = float(np.median(pos[yellow, 2]))
    # Scan X runs along the tape, scan Z runs across it, scan Y is up.
    sim = np.column_stack(
        [
            pos[:, 0] - pos[:, 0].mean(),
            pos[:, 2] - tape_z + BORDER_Y,
            pos[:, 1] - np.median(pos[:, 1]) + LIFT_M,
        ]
    ).astype(np.float32)
    face_st = uv[face_uv].astype(np.float32)
    out = os.path.join(work, "table.npz")
    np.savez_compressed(out, points=sim, faces=faces, st=face_st, jpg=np.array(jpg))
    print(
        f"[prepare] {len(sim)} verts, yellow {int(yellow.sum())} at scan z {tape_z:.3f}, "
        f"xy [{sim[:, 0].min():.2f},{sim[:, 0].max():.2f}] [{sim[:, 1].min():.2f},{sim[:, 1].max():.2f}]",
        flush=True,
    )
    return out


def _add_mesh(stage, npz_path):
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    data = np.load(npz_path, allow_pickle=True)
    points, faces, st = data["points"], data["faces"], data["st"]
    jpg = str(data["jpg"])
    mesh = UsdGeom.Mesh.Define(stage, "/World/scan_table")
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([int(i) for i in faces.reshape(-1)])
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * len(points))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying
    )
    primvar.Set([Gf.Vec2f(float(u), float(v)) for u, v in st.reshape(-1, 2)])
    root = "/World/Looks/scan_table"
    mat = UsdShade.Material.Define(stage, root)
    surf = UsdShade.Shader.Define(stage, root + "/Shader")
    surf.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, root + "/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    tex = UsdShade.Shader.Define(stage, root + "/tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(jpg))
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)
    return points


def render(npz_path, out_dir) -> None:
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    from record_overhead import HEIGHT, WIDTH, WRIST, _frame_cameras, _rgb

    app = SimulationApp({"headless": True, "width": WIDTH, "height": HEIGHT})
    try:
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image

        from omni_twin import PxrWriter
        from scene import Scene
        from usd_chain import Chain

        ctx = omni.usd.get_context()
        ctx.open_stage(os.path.abspath(STAGE))
        for _ in range(15):
            app.update()
        stage = ctx.get_stage()
        chain = Chain.load(CHAIN)
        writer = PxrWriter(stage, chain)
        _frame_cameras(stage)
        points = _add_mesh(stage, npz_path)
        pad = 0.02
        writer.set_scene(
            Scene(
                "scan",
                (
                    float(points[:, 0].min()) - pad,
                    float(points[:, 0].max()) + pad,
                    float(points[:, 1].min()) - pad,
                    float(points[:, 1].max()) + pad,
                ),
                (),
            )
        )
        # set_scene rebuilds /World/scene and would drop nothing at /World/scan_table, but rebuild the slab under it.
        writer.set_pose(chain.joint_values(FOLDED_Q, FINGER_OPEN_M))
        overhead_rp = rep.create.render_product("/World/TopCamera", (WIDTH, HEIGHT))
        wrist_rp = rep.create.render_product(WRIST, (1280, 720))
        overhead_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        wrist_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        overhead_rgb.attach([overhead_rp])
        wrist_rgb.attach([wrist_rp])
        for _ in range(12):
            app.update()
            rep.orchestrator.step()
        os.makedirs(out_dir, exist_ok=True)
        overhead_path = os.path.join(out_dir, "preview_overhead.png")
        wrist_path = os.path.join(out_dir, "preview_wrist.png")
        Image.fromarray(_rgb(overhead_rgb, "overhead")).save(overhead_path)
        wrist_image = np.asarray(wrist_rgb.get_data())
        if wrist_image.shape[-1] == 4:
            wrist_image = wrist_image[:, :, :3]
        Image.fromarray(np.ascontiguousarray(wrist_image[:, :, :3], dtype=np.uint8)).save(wrist_path)
        print(f"[preview] {overhead_path}", flush=True)
        print(f"[preview] {wrist_path}", flush=True)
    finally:
        app.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Preview the Scaniverse table mesh in the twin")
    parser.add_argument("--prepare", action="store_true", help="Unpack and align the mesh, then exit.")
    parser.add_argument("--zip", default=SCAN_ZIP)
    parser.add_argument("--work", default=WORK)
    parser.add_argument("--out", default=os.path.join(WORK, ".."))
    args = parser.parse_args(argv)
    npz_path = os.path.join(args.work, "table.npz")
    if args.prepare or not os.path.isfile(npz_path):
        npz_path = prepare(args.zip, args.work)
        if args.prepare:
            return 0
    render(npz_path, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
