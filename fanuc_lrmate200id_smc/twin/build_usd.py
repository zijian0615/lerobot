"""Build the Omniverse / USD model of the twin from the MuJoCo model. Needs mujoco + numpy only (writes text .usda, no pxr).

    .venv-twin/bin/python twin/build_usd.py
Writes  fanuc_lrmate200id_smc.usda  (stage: visual geoms, materials, ground, lights, camera, status light, TCP trail)
and     fanuc_lrmate200id_smc.twin.json  (kinematic chain that omni_twin.py uses, pxr-free).

Every articulated body is an Xform  [xformOp:translate, xformOp:orient, xformOp:<orient|translate>:joint]; omni_twin.py
only ever rewrites the ":joint" op. USD applies xformOpOrder outermost-first, so this is  T * R_static * R_joint,
the same composition as MuJoCo's body frame.
"""
import argparse
import json
import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_XML = os.path.join(HERE, "..", "fanuc_lrmate200id_smc.xml")
OUT_USD = os.path.join(HERE, "..", "fanuc_lrmate200id_smc.usda")
OUT_JSON = os.path.join(HERE, "..", "fanuc_lrmate200id_smc.twin.json")
ROOT = "/World/fanuc"
VISIBLE_GROUP = 2                                  # visual meshes and the gripper primitives; group 3 = collision hulls
STATUS_POS, STATUS_RADIUS = (-0.30, 0.0, 0.95), 0.03     # same status light as the MuJoCo twin
CAM_AZ_EL_DIST, CAM_LOOKAT = (140.0, -20.0, 1.9), (0.25, 0.0, 0.4)   # same view as the MuJoCo twin
CAM_FOCAL_MM, CAM_FOVY_DEG, CAM_ASPECT = 24.0, 45.0, 16 / 9          # MuJoCo's default 45 deg vertical FOV
OVERHEAD_JSON = os.path.join(HERE, "..", "overhead_camera.json")


def f(x):
    return f"{float(x):.9g}"


def vec(v):
    return "(" + ", ".join(f(x) for x in v) + ")"


class Usda:
    def __init__(self):
        self.lines, self.depth = [], 0

    def w(self, s=""):
        self.lines.append("    " * self.depth + s)

    def open(self, header, meta=()):
        """`def Xform "name"` + optional prim metadata lines, then `{`."""
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


def xform_ops(u, pos, quat, joint=None):
    """Static pose ops plus, for articulated bodies, the ':joint' op that omni_twin.py rewrites every frame."""
    u.w(f"double3 xformOp:translate = {vec(pos)}")
    u.w(f"quatf xformOp:orient = {vec(quat)}")
    order = ["xformOp:translate", "xformOp:orient"]
    if joint is not None:
        if joint["type"] == "hinge":
            u.w("quatf xformOp:orient:joint = (1, 0, 0, 0)")
            order.append("xformOp:orient:joint")
        else:
            u.w("double3 xformOp:translate:joint = (0, 0, 0)")
            order.append("xformOp:translate:joint")
    u.w("uniform token[] xformOpOrder = [" + ", ".join(f'"{o}"' for o in order) + "]")


def flat_normals(pts, faces):
    tri = pts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return np.repeat(n, 3, axis=0)


def write_geom(u, model, g, name, mat_path):
    gtype, size = model.geom_type[g], model.geom_size[g]
    bind = [f"rel material:binding = <{mat_path}>"]
    if gtype == mujoco.mjtGeom.mjGEOM_MESH:
        m = model.geom_dataid[g]
        v0, nv = model.mesh_vertadr[m], model.mesh_vertnum[m]
        f0, nf = model.mesh_faceadr[m], model.mesh_facenum[m]
        pts = model.mesh_vert[v0:v0 + nv].astype(float)
        faces = model.mesh_face[f0:f0 + nf].astype(int)
        u.open(f'def Mesh "{name}"', ['prepend apiSchemas = ["MaterialBindingAPI"]'])
        u.w(f"float3[] extent = [{vec(pts.min(0))}, {vec(pts.max(0))}]")
        u.w("int[] faceVertexCounts = [" + ", ".join(["3"] * nf) + "]")
        u.w("int[] faceVertexIndices = [" + ", ".join(map(str, faces.ravel())) + "]")
        u.w("point3f[] points = [" + ", ".join(vec(p) for p in pts) + "]")
        u.w("normal3f[] normals = [" + ", ".join(vec(n) for n in flat_normals(pts, faces)) + '] ( interpolation = "faceVarying" )')
        u.w('uniform token subdivisionScheme = "none"')
        u.w("uniform bool doubleSided = true")
    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
        u.open(f'def Cube "{name}"', ['prepend apiSchemas = ["MaterialBindingAPI"]'])
        u.w("double size = 2")                      # unit cube of edge 2, scaled to the MuJoCo half-sizes
        u.w(f"float3 xformOp:scale = {vec(size)}")
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        u.open(f'def Cylinder "{name}"', ['prepend apiSchemas = ["MaterialBindingAPI"]'])
        u.w(f"double radius = {f(size[0])}")
        u.w(f"double height = {f(2 * size[1])}")
        u.w('uniform token axis = "Z"')
    else:
        raise NotImplementedError(f"visible geom '{name}' has unsupported type {mujoco.mjtGeom(gtype).name}")
    for b in bind:
        u.w(b)
    u.w(f"double3 xformOp:translate = {vec(model.geom_pos[g])}")
    u.w(f"quatf xformOp:orient = {vec(model.geom_quat[g])}")
    order = ["xformOp:translate", "xformOp:orient"] + (["xformOp:scale"] if gtype == mujoco.mjtGeom.mjGEOM_BOX else [])
    u.w("uniform token[] xformOpOrder = [" + ", ".join(f'"{o}"' for o in order) + "]")
    u.close()


def joint_spec(model, body):
    n = model.body_jntnum[body]
    if n == 0:
        return None
    if n > 1:
        raise NotImplementedError(f"body '{model.body(body).name}' has {n} joints; only one joint per body is supported")
    j = model.body_jntadr[body]
    jtype = {int(mujoco.mjtJoint.mjJNT_HINGE): "hinge", int(mujoco.mjtJoint.mjJNT_SLIDE): "slide"}.get(int(model.jnt_type[j]))
    if jtype is None:
        raise NotImplementedError(f"joint '{model.joint(j).name}' has unsupported type")
    if np.linalg.norm(model.jnt_pos[j]) > 1e-9:
        raise NotImplementedError(f"joint '{model.joint(j).name}' is not at its body origin")
    return {"name": model.joint(j).name, "type": jtype, "axis": model.jnt_axis[j].tolist(),
            "range": model.jnt_range[j].tolist()}


def material_table(model, geoms):
    """{material key: (usd name, rgba)} for the visible geoms."""
    table = {}
    for g in geoms:
        mid = model.geom_matid[g]
        key = model.mat(mid).name if mid >= 0 else f"geom_{g}"
        table[g] = (key, model.mat_rgba[mid] if mid >= 0 else model.geom_rgba[g])
    return table


def write_material(u, name, rgba, roughness=0.5, metallic=0.0, emissive=False):
    path = f"/World/Looks/{name}"
    u.open(f'def Material "{name}"')
    u.w(f"token outputs:surface.connect = <{path}/Shader.outputs:surface>")
    u.open('def Shader "Shader"')
    u.w('uniform token info:id = "UsdPreviewSurface"')
    u.w(f"color3f inputs:diffuseColor = {vec(rgba[:3])}")
    if emissive:
        u.w(f"color3f inputs:emissiveColor = {vec(rgba[:3])}")
    u.w(f"float inputs:roughness = {f(roughness)}")
    u.w(f"float inputs:metallic = {f(metallic)}")
    u.w("token outputs:surface")
    u.close()
    u.close()


def camera_matrix(az_el_dist, lookat):
    """USD camera looks down its -Z with +Y up; row-vector matrix, translation in the last row. Same convention as mujoco.viewer."""
    az, el, dist = np.radians(az_el_dist[0]), np.radians(az_el_dist[1]), az_el_dist[2]
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    eye = np.asarray(lookat) - dist * fwd
    x = np.cross(fwd, [0, 0, 1])
    x /= np.linalg.norm(x)
    y = np.cross(-fwd, x)
    return [list(x) + [0], list(y) + [0], list(-fwd) + [0], list(eye) + [1]]


def build(model):
    bodies, u = [], Usda()
    geoms_by_body = {}
    visible = [g for g in range(model.ngeom) if model.geom_group[g] == VISIBLE_GROUP]
    for g in visible:
        geoms_by_body.setdefault(int(model.geom_bodyid[g]), []).append(g)
    mats = material_table(model, visible)

    u.w("#usda 1.0")
    u.w("(")
    u.w('    defaultPrim = "World"')
    u.w("    metersPerUnit = 1")
    u.w('    upAxis = "Z"')
    u.w('    doc = "Generated by twin/build_usd.py from fanuc_lrmate200id_smc.xml. Do not edit; rebuild."')
    u.w(")")
    u.w()
    u.open('def Xform "World"')

    u.open('def Scope "Looks"')
    for key, rgba in {v[0]: v[1] for v in mats.values()}.items():
        write_material(u, key, rgba, metallic=0.1 if "gray" in key or "smc" in key else 0.0)
    write_material(u, "ground", (0.30, 0.31, 0.33, 1), roughness=0.9)
    write_material(u, "status_live", (0.1, 0.9, 0.2, 1), emissive=True)
    write_material(u, "status_stale", (1.0, 0.6, 0.0, 1), emissive=True)
    write_material(u, "status_none", (0.9, 0.1, 0.1, 1), emissive=True)
    write_material(u, "trail", (0.1, 0.6, 1.0, 1), emissive=True)
    u.close()

    u.open('def Xform "fanuc"')

    def emit(body, parent_name, path):
        name = model.body(body).name
        path = f"{path}/{name}"
        joint = joint_spec(model, body)
        pos, quat = model.body_pos[body], model.body_quat[body]
        bodies.append({"name": name, "parent": parent_name, "path": path, "pos": pos.tolist(),
                       "quat": quat.tolist(), "joint": joint})
        u.open(f'def Xform "{name}"')
        xform_ops(u, pos, quat, joint)
        for g in geoms_by_body.get(body, []):
            write_geom(u, model, g, model.geom(g).name or f"{name}_visual", f"/World/Looks/{mats[g][0]}")
        for child in range(model.nbody):
            if model.body_parentid[child] == body and child != body:
                emit(child, name, path)
        u.close()

    for top in range(1, model.nbody):
        if model.body_parentid[top] == 0:
            emit(top, None, ROOT)
    u.close()   # fanuc

    sites = {}
    for s in ("flange", "tcp"):
        sid = model.site(s).id
        sites[s] = {"body": model.body(int(model.site_bodyid[sid])).name, "pos": model.site_pos[sid].tolist()}

    u.open('def Mesh "ground"', ['prepend apiSchemas = ["MaterialBindingAPI"]'])
    u.w("int[] faceVertexCounts = [4]")
    u.w("int[] faceVertexIndices = [0, 1, 2, 3]")
    u.w("point3f[] points = [(-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)]")
    u.w('normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] ( interpolation = "faceVarying" )')
    u.w('uniform token subdivisionScheme = "none"')
    u.w("rel material:binding = </World/Looks/ground>")
    u.w("double3 xformOp:translate = (0, 0, 0)")           # omni_twin lowers it under the table slab when a scene is shown
    u.w('uniform token[] xformOpOrder = ["xformOp:translate"]')
    u.close()

    u.open('def Sphere "status_light"', ['prepend apiSchemas = ["MaterialBindingAPI"]'])
    u.w(f"double radius = {f(STATUS_RADIUS)}")
    u.w("rel material:binding = </World/Looks/status_none>")
    u.w(f"double3 xformOp:translate = {vec(STATUS_POS)}")
    u.w('uniform token[] xformOpOrder = ["xformOp:translate"]')
    u.close()

    u.open('def BasisCurves "trail"', ['prepend apiSchemas = ["MaterialBindingAPI"]'])
    u.w('uniform token type = "linear"')
    u.w("int[] curveVertexCounts = [2]")
    u.w("point3f[] points = [(0, 0, 0), (0, 0, 0)]")
    u.w('float[] widths = [0.006] ( interpolation = "constant" )')
    u.w("rel material:binding = </World/Looks/trail>")
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

    aperture = 2 * CAM_FOCAL_MM * np.tan(np.radians(CAM_FOVY_DEG / 2)) * CAM_ASPECT
    with open(OVERHEAD_JSON, encoding="utf-8") as handle:
        overhead = json.load(handle)["usd"]
    cameras = (
        ("Camera", camera_matrix(CAM_AZ_EL_DIST, CAM_LOOKAT), CAM_FOCAL_MM, aperture, None),
        (
            "TopCamera",
            overhead["matrix"],
            overhead["focal_mm"],
            overhead["horizontal_aperture"],
            overhead["vertical_aperture"],
        ),
    )
    for name, matrix, focal, horizontal, vertical in cameras:
        u.open(f'def Camera "{name}"')
        u.w("matrix4d xformOp:transform = (" + ", ".join(vec(r) for r in matrix) + ")")
        u.w('uniform token[] xformOpOrder = ["xformOp:transform"]')
        u.w(f"float focalLength = {f(focal)}")
        u.w(f"float horizontalAperture = {f(horizontal)}")
        if vertical is not None:
            u.w(f"float verticalAperture = {f(vertical)}")
        u.w("float2 clippingRange = (0.01, 100)")
        u.close()

    u.close()   # World
    return u.text(), {"version": 1, "root": ROOT, "bodies": bodies, "sites": sites}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", default=ROBOT_XML)
    ap.add_argument("--out", default=OUT_USD)
    ap.add_argument("--chain", default=OUT_JSON)
    a = ap.parse_args(argv)
    model = mujoco.MjModel.from_xml_path(a.xml)
    usda, chain = build(model)
    with open(a.out, "w") as fh:
        fh.write(usda)
    with open(a.chain, "w") as fh:
        json.dump(chain, fh, indent=1)
    print(f"wrote {a.out} ({len(usda) / 1e6:.1f} MB) and {a.chain}")


if __name__ == "__main__":
    main()
