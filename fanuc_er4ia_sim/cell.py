"""
Scene construction for the ER-4iA cell, shared by build_scene.py and make_video.py.

Assumes SimulationApp has already been started by the caller.
"""
import os
import sys
import math
import json
import numpy as np
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Sdf, Gf
import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

import scene_config as C
from gripper_geometry import finger_profile, clip_profile

FLANGE = C.ROBOT_PRIM + "/J6_link"
WRIST_CAM_PATH = C.WRIST_CAM["parent"] + "/WristCam"
TABLE_CAM_PATH = "/World/TableCam"
SIDE_CAM_PATH = "/World/SideCam"
CLOSEUP_CAM_PATH = "/World/CloseupCam"


# ---------------------------------------------------------------- materials
# UsdPreviewSurface rather than displayColor: the renders are meant to be fed to
# zero-shot localisation models, so specular response and material separation
# between machined aluminium, anodised bodies, black oxide and moulded plastic
# actually matter.
MATS = {}

_TEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures")

# Texture sets for the surfaces where a constant colour is not convincing.
# project_uvw is essential: none of this geometry carries UVs (scaled
# UsdGeom.Cubes and hand-built meshes), so without triplanar projection the
# maps simply do not appear. Object space rather than world space so the tool
# marks travel with the part instead of swimming as the arm moves.
_TEXTURED = {
    "alu_machined": dict(albedo="alu_albedo.png", rough="alu_roughness.png",
                         normal="alu_normal.png", scale=60.0, bump=0.22),
    "alu_anodised": dict(albedo="anod_albedo.png", rough="anod_roughness.png",
                         normal="alu_normal.png", scale=40.0, bump=0.12),
    "steel_dark":   dict(albedo=None, rough="alu_roughness.png",
                         normal="alu_normal.png", scale=26.0, bump=0.30),
}

_MATERIAL_SPEC = {
    # CAD gripper (merged from the hackathon starter): blades, air lines, camera cable
    "uwm_finger": ((1.0, 1.0, 1.0), 0.88, 0.38),
    "finger_milled": ((1.0, 1.0, 1.0), 0.92, 0.44),
    "air_line_blue": ((0.025, 0.30, 0.68), 0.00, 0.30),
    "camera_cable_orange": ((0.85, 0.22, 0.025), 0.00, 0.36),
    #                        diffuse                metallic roughness
    "alu_machined":     ((0.910, 0.915, 0.930), 0.92, 0.26),
    "alu_anodised":     ((0.780, 0.790, 0.800), 0.70, 0.44),
    "steel_dark":       ((0.230, 0.240, 0.265), 0.88, 0.38),
    "black_oxide":      ((0.055, 0.055, 0.062), 0.75, 0.42),
    "black_plastic":    ((0.035, 0.035, 0.040), 0.00, 0.48),
    "plastic_red":      ((0.640, 0.055, 0.045), 0.00, 0.28),
    "plastic_yellow":   ((0.780, 0.620, 0.040), 0.00, 0.28),
    "plastic_blue":     ((0.045, 0.165, 0.620), 0.00, 0.28),
    "table_white":      ((0.880, 0.880, 0.875), 0.00, 0.55),
    "floor_grey":       ((0.230, 0.230, 0.245), 0.00, 0.80),
    "tray_dark":        ((0.060, 0.058, 0.062), 0.00, 0.62),
    "fixture_grey":     ((0.105, 0.105, 0.110), 0.00, 0.70),
    "bottle_white":     ((0.880, 0.880, 0.870), 0.00, 0.40),
    "white_plastic":    ((0.860, 0.860, 0.850), 0.00, 0.45),
    "label_white":      ((0.900, 0.900, 0.890), 0.00, 0.60),
    "black_anodised":   ((0.030, 0.030, 0.033), 0.35, 0.40),
    "base_grey":        ((0.200, 0.205, 0.210), 0.00, 0.50),
    # "BLOCK-03", 3D-printed amber-yellow PLA, as in the physical cell
    # REAL (2026-10-08): the printed bin reads amber, not lemon, from the table camera; from the side it is
    # paler still (translucent PLA against the window), which an albedo cannot do -- this splits the two.
    "bin_yellow":       ((0.850, 0.500, 0.120), 0.00, 0.70),
    "duct_tape":        ((0.380, 0.390, 0.395), 0.05, 0.78),
}


def recolor_robot(stage):
    """Repaint the asset's pure-yellow arm materials with C.ROBOT_YELLOW."""
    looks = stage.GetPrimAtPath(C.ROBOT_PRIM + "/Looks")
    if not looks.IsValid() or getattr(C, "ROBOT_YELLOW", None) is None:
        return
    for prim in Usd.PrimRange(looks):
        if not prim.IsA(UsdShade.Shader):
            continue
        inp = UsdShade.Shader(prim).GetInput("diffuse_color_constant")
        col = inp.Get() if inp else None
        if col is not None and abs(col[0] - 1) < 1e-3 and abs(col[1] - 1) < 1e-3 and col[2] < 1e-3:
            inp.Set(Gf.Vec3f(*C.ROBOT_YELLOW))


def init_materials(stage):
    """Define every material once under /World/Looks.

    OmniPBR (MDL) rather than UsdPreviewSurface -- it is what the RTX renderer
    is tuned for and gives a more predictable metallic/roughness response.

    NOTE: MDL materials take a few frames to compile after the stage is built.
    Render too early and everything comes out default white, which looks exactly
    like a broken binding. Warm up with several world.step(render=True) calls
    before capturing.
    """
    from isaacsim.core.api.materials import OmniPBR
    MATS.clear()
    UsdGeom.Scope.Define(stage, "/World/Looks")
    for name, (diff, metal, rough) in _MATERIAL_SPEC.items():
        path = f"/World/Looks/{name}"
        pbr = OmniPBR(prim_path=path, name=name, color=np.array(diff, float))
        pbr.set_metallic_constant(float(metal))
        # 0..1. (This used to pass rough * 100, which the shader clamps to 1:
        # every material rendered fully matte whatever the table below says.)
        pbr.set_reflection_roughness(float(rough))

        t = _TEXTURED.get(name)
        if t:
            # OmniPBR does not necessarily name its shader prim "<mat>/Shader";
            # take the handle the helper already holds.
            sh = pbr.shaders_list[0]

            def _set(key, tname, val):
                sh.CreateInput(key, tname).Set(val)

            if t["albedo"]:
                _set("diffuse_texture", Sdf.ValueTypeNames.Asset,
                     Sdf.AssetPath(os.path.join(_TEX, t["albedo"])))
            if t["rough"]:
                _set("reflectionroughness_texture", Sdf.ValueTypeNames.Asset,
                     Sdf.AssetPath(os.path.join(_TEX, t["rough"])))
                _set("reflection_roughness_texture_influence",
                     Sdf.ValueTypeNames.Float, 1.0)
            if t["normal"]:
                _set("normalmap_texture", Sdf.ValueTypeNames.Asset,
                     Sdf.AssetPath(os.path.join(_TEX, t["normal"])))
                _set("bump_factor", Sdf.ValueTypeNames.Float, float(t["bump"]))
            _set("project_uvw", Sdf.ValueTypeNames.Bool, True)
            _set("world_or_object", Sdf.ValueTypeNames.Bool, False)
            _set("texture_scale", Sdf.ValueTypeNames.Float2,
                 Gf.Vec2f(float(t["scale"]), float(t["scale"])))
        MATS[name] = pbr.material

    # the printed mat: the real artwork from the site scan, on explicit UVs
    tex = os.path.join(os.path.dirname(os.path.abspath(__file__)), C.MAT_TEXTURE)
    if os.path.exists(tex):
        pbr = OmniPBR(prim_path="/World/Looks/mat_print", name="mat_print",
                      color=np.array((1.0, 1.0, 1.0)))
        pbr.set_metallic_constant(0.0)
        pbr.set_reflection_roughness(0.32)        # glossy laminate
        # the helper turns on triplanar projection for every new material,
        # which would ignore the quad's UVs and tile the artwork
        pbr.set_project_uvw(False)
        pbr.shaders_list[0].CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(tex))
        MATS["mat_print"] = pbr.material
    else:
        print(f"[cell] WARNING: {tex} missing; run make_mat_texture.py", file=sys.stderr)


PHYS_MAT_PATH = "/World/Physics/ObjectMaterial"
TAPE_MAT_PATH = "/World/Physics/TapeMaterial"
SOFT_TIP_MAT_PATH = "/World/Physics/SoftTipMaterial"


def enable_ccd(world):
    """Continuous collision detection.

    The bin walls are 4 mm; without CCD a fast object can pass straight through
    one between solver steps rather than bouncing off it.
    """
    try:
        ctx = world.get_physics_context()
        ctx.enable_ccd(True)
    except Exception as e:
        print(f"[cell] could not enable CCD: {e}", file=sys.stderr)


def init_physics(stage):
    """One shared physics material for the task objects and the table.

    Without a material PhysX uses its defaults, which are slippery enough that a
    parallel-jaw grasp cannot hold anything.
    """
    UsdGeom.Scope.Define(stage, "/World/Physics")
    m = UsdPhysics.MaterialAPI.Apply(
        UsdGeom.Scope.Define(stage, PHYS_MAT_PATH).GetPrim())
    m.CreateStaticFrictionAttr().Set(0.9)
    m.CreateDynamicFrictionAttr().Set(0.8)
    m.CreateRestitutionAttr().Set(0.05)

    t = UsdPhysics.MaterialAPI.Apply(
        UsdGeom.Scope.Define(stage, TAPE_MAT_PATH).GetPrim())
    t.CreateStaticFrictionAttr().Set(C.TAPE_STATIC_FRIC)
    t.CreateDynamicFrictionAttr().Set(C.TAPE_DYNAMIC_FRIC)
    t.CreateRestitutionAttr().Set(0.0)

    # the blades' high-friction distal tips (CAD gripper)
    tip = UsdPhysics.MaterialAPI.Apply(UsdShade.Material.Define(stage, SOFT_TIP_MAT_PATH).GetPrim())
    tip.CreateStaticFrictionAttr().Set(C.SOFT_TIP_STATIC_FRIC)
    tip.CreateDynamicFrictionAttr().Set(C.SOFT_TIP_DYNAMIC_FRIC)
    tip.CreateRestitutionAttr().Set(0.0)
    from pxr import PhysxSchema
    PhysxSchema.PhysxMaterialAPI.Apply(tip.GetPrim()).CreateFrictionCombineModeAttr().Set("max")


def bind_physics_material(prim, path=PHYS_MAT_PATH):
    mat_prim = prim.GetStage().GetPrimAtPath(path)
    if not mat_prim.IsValid():
        return
    rel = UsdShade.MaterialBindingAPI.Apply(prim)
    rel.Bind(UsdShade.Material(mat_prim), bindingStrength=UsdShade.Tokens.weakerThanDescendants,
             materialPurpose="physics")


def make_dynamic(prim, mass, angular_damping=0.08, linear_damping=0.02,
                 collide_self=True):
    """Turn a prim into a dynamic rigid body so it falls and collides.

    collide_self=False when the collision geometry lives on child meshes, which
    need their own convex-hull approximation -- PhysX refuses a raw triangle
    mesh on a dynamic body and falls back with an error otherwise.
    """
    if collide_self:
        UsdPhysics.CollisionAPI.Apply(prim)
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateRigidBodyEnabledAttr().Set(True)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
    from pxr import PhysxSchema
    px = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    px.CreateAngularDampingAttr().Set(angular_damping)
    px.CreateLinearDampingAttr().Set(linear_damping)
    px.CreateSolverPositionIterationCountAttr().Set(16)
    px.CreateSolverVelocityIterationCountAttr().Set(2)
    bind_physics_material(prim)
    return prim


def bind_mat(prim, name):
    if name and name in MATS:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(MATS[name])
    return prim


def place(prim, pos, yaw_deg=0.0):
    """Set translate/orient on a prim that may already carry xform ops.

    Referenced assets (the robot) arrive with [translate, orient, scale] already
    in xformOpOrder, so AddXformOp would raise. Reuse what is there.
    """
    x = UsdGeom.Xformable(prim)
    ops = {o.GetOpName(): o for o in x.GetOrderedXformOps()}
    (ops.get("xformOp:translate") or x.AddTranslateOp()).Set(Gf.Vec3d(*pos))
    q = Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw_deg).GetQuat()
    if "xformOp:orient" in ops:
        op = ops["xformOp:orient"]
        cast = Gf.Quatf if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat else Gf.Quatd
        op.Set(cast(q))
    elif yaw_deg:
        x.AddRotateZOp().Set(yaw_deg)


def box(stage, path, size, pos, color, kinematic=True, rot=None, collision=True,
        mat=None):
    """Axis-aligned box centred at pos. size = (x, y, z) extents.

    collision=False leaves the prim out of the physics scene entirely. That
    matters for anything we animate by writing its xform: touching the transform
    of a collider forces PhysX to resync, which invalidates the tensor
    simulation view and breaks the articulation mid-run.
    """
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    x = UsdGeom.Xformable(cube.GetPrim())
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    if rot is not None:
        x.AddRotateXYZOp().Set(Gf.Vec3f(*rot))
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    bind_mat(cube.GetPrim(), mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    if not kinematic:
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(0.05)
    return cube.GetPrim()


def camera(stage, path, cfg):
    """Camera posed by eye/look_at. USD cameras look down local -Z with +Y up."""
    cam = UsdGeom.Camera.Define(stage, path)
    eye, tgt, up = (Gf.Vec3d(*cfg[k]) for k in ("eye", "look_at", "up"))
    view = Gf.Matrix4d().SetLookAt(eye, tgt, up)
    UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(view.GetInverse())
    w, h = cfg["resolution"]
    cam.CreateFocalLengthAttr(cfg["focal_length"])
    cam.CreateHorizontalApertureAttr(cfg["horiz_aperture"])
    cam.CreateVerticalApertureAttr(cfg["horiz_aperture"] * h / w)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 20.0))
    return cam.GetPrim()


def cyl(stage, path, radius, height, pos, color, collision=False, mat=None):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateAxisAttr("Z"); c.CreateRadiusAttr(radius); c.CreateHeightAttr(height)
    c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdGeom.Xformable(c.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*pos))
    bind_mat(c.GetPrim(), mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    return c.GetPrim()


def add_mat_print(stage):
    """The printed mat as a textured quad over the whole tabletop.

    assets/mat_print.png is the real artwork, rectified into the robot frame
    from the site scan by make_mat_texture.py: row 0 is the front (+X) edge,
    column 0 the -Y edge. It includes the yellow CAUTION edge bands and the
    black tape marks.
    """
    if "mat_print" not in MATS:
        return
    (x0, x1), (y0, y1) = C.TABLE_X, C.TABLE_Y
    z = C.TABLE_TOP_Z + C.SURFACE_DZ + 0.0002
    m = UsdGeom.Mesh.Define(stage, "/World/MatPrint")
    m.CreatePointsAttr([Gf.Vec3f(x0, y0, z), Gf.Vec3f(x0, y1, z),
                        Gf.Vec3f(x1, y1, z), Gf.Vec3f(x1, y0, z)])
    m.CreateFaceVertexCountsAttr([4])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    m.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    st = UsdGeom.PrimvarsAPI(m).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    # s runs along +Y, t along +X (t = 1 is the image's top row, the front edge)
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
    bind_mat(m.GetPrim(), "mat_print")


def add_props(stage):
    """The dark fixture at the front-left corner of the bench (VIDEO).

    Grey base plate, a raised tray with a recessed top, a block with two round
    holes, and a pill bottle. Static colliders: the arm never goes there, but
    the fixture is solid in the real cell.
    """
    Z = C.TABLE_TOP_Z + C.SURFACE_DZ
    fx, fy = C.FIXTURE_XY
    pw, pd, pt = C.FIXTURE_PLATE
    UsdGeom.Xform.Define(stage, "/World/Fixture")
    box(stage, "/World/Fixture/Plate", (pw, pd, pt), (fx, fy, Z + pt / 2),
        (0.1, 0.1, 0.1), mat="fixture_grey")
    Z += pt
    # raised tray on the back half of the plate, recess suggested by a rim
    tw, td, th = C.FIXTURE_TRAY
    tx, ty = fx - pw / 2 + tw / 2 + 0.012, fy + 0.0
    rim = 0.012
    box(stage, "/World/Fixture/TrayFloor", (tw, td, th * 0.45),
        (tx, ty, Z + th * 0.225), (0.06, 0.06, 0.06), mat="tray_dark")
    for tag, sz, off in (("n", (tw, rim), (0, td / 2 - rim / 2)),
                         ("s", (tw, rim), (0, -td / 2 + rim / 2)),
                         ("e", (rim, td), (tw / 2 - rim / 2, 0)),
                         ("w", (rim, td), (-tw / 2 + rim / 2, 0))):
        box(stage, f"/World/Fixture/TrayRim_{tag}", (sz[0], sz[1], th),
            (tx + off[0], ty + off[1], Z + th / 2), (0.06, 0.06, 0.06), mat="tray_dark")
    # two-hole block at the front of the plate
    bw, bd, bh = C.FIXTURE_BLOCK
    bx, by = fx + pw / 2 - bw / 2 - 0.012, fy
    box(stage, "/World/Fixture/Block", (bw, bd, bh), (bx, by, Z + bh / 2),
        (0.06, 0.06, 0.06), mat="tray_dark")
    for i, dy in enumerate((-bd / 4, bd / 4)):
        cyl(stage, f"/World/Fixture/Hole{i}", bw * 0.34, 0.001,
            (bx, by + dy, Z + bh + 0.0004), (0.0, 0.0, 0.0), mat="black_plastic")
    # pill bottle standing in the tray
    zt = Z + th * 0.45
    cyl(stage, "/World/Fixture/Bottle", C.BOTTLE_R, C.BOTTLE_H,
        (tx, ty - td / 4, zt + C.BOTTLE_H / 2), (0.88, 0.88, 0.87),
        collision=True, mat="bottle_white")
    cyl(stage, "/World/Fixture/BottleCap", C.BOTTLE_R * 0.92, 0.016,
        (tx, ty - td / 4, zt + C.BOTTLE_H + 0.008), (0.95, 0.95, 0.95),
        collision=True, mat="white_plastic")


def build(stage, with_objects=True):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    init_materials(stage)
    init_physics(stage)
    add_lighting(stage)

    box(stage, "/World/Floor", (6.0, 6.0, 0.02), (0, 0, -0.01),
        (0.23, 0.23, 0.25), mat="floor_grey")

    tw, td, tt = C.TABLE_SIZE
    top = C.TABLE_TOP_Z + C.SURFACE_DZ - C.MAT_T  # the table top, under the surface sheet
    box(stage, "/World/Table", (tw, td, tt),
        (C.TABLE_CENTER[0], C.TABLE_CENTER[1], top - tt / 2.0),
        (0.88, 0.88, 0.875), mat="table_white")
    # The printed laminate covers the whole top, and is what objects rest on.
    # (Do not parent anything under /World/Mat -- it is a scaled Cube, and
    # children inherit the scale.)
    matp = box(stage, "/World/Mat", (tw, td, C.MAT_T),
               (C.TABLE_CENTER[0], C.TABLE_CENTER[1], top + C.MAT_T / 2),
               (0.90, 0.90, 0.895), mat="table_white")
    bind_physics_material(matp)

    robot_usd = get_assets_root_path() + C.ROBOT_USD_SUBPATH
    print(f"[cell] referencing robot: {robot_usd}")
    add_reference_to_stage(usd_path=robot_usd, prim_path=C.ROBOT_PRIM)
    place(stage.GetPrimAtPath(C.ROBOT_PRIM), (0.0, 0.0, C.TABLE_TOP_Z), 0.0)
    # The asset's visuals are instanceable, and a binding on an ancestor does
    # not reach an instance's prototype in the renderer -- de-instance the base
    # visuals locally so the override takes.
    base = stage.GetPrimAtPath(C.ROBOT_PRIM + "/robot_base/visuals")
    if C.ROBOT_BASE_GREY and base.IsValid() and "base_grey" in MATS:
        base.SetInstanceable(False)
        UsdShade.MaterialBindingAPI.Apply(base).Bind(
            MATS["base_grey"], bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    recolor_robot(stage)

    if C.MAT_MARKINGS:
        add_mat_print(stage)
    if with_objects:
        if C.INCLUDE_FIXTURE:
            add_props(stage)
        add_grasp_cubes(stage)
        add_screws(stage)
        add_bin(stage)

    if stage.GetPrimAtPath(FLANGE).IsValid():
        attach_gripper(stage)
    else:
        print(f"[cell] WARNING: {FLANGE} not found; gripper not attached", file=sys.stderr)

    camera(stage, TABLE_CAM_PATH, C.TABLE_CAM)
    camera(stage, SIDE_CAM_PATH, C.SIDE_CAM)
    camera(stage, CLOSEUP_CAM_PATH, C.CLOSEUP_CAM)
    if stage.GetPrimAtPath(C.WRIST_CAM["parent"]).IsValid():
        camera(stage, WRIST_CAM_PATH, C.WRIST_CAM)


def add_lighting(stage):
    """Studio-ish three-point rig.

    A flat dome alone gives no specular response, which flattens the machined
    aluminium and the moulded plastic into the same look -- bad for the
    localisation work these renders feed.
    """
    # A textured studio environment, not a flat colour. Metal is mostly what it
    # reflects -- against a featureless dome, machined aluminium renders as dead
    # grey however the metallic and roughness constants are set.
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(1100.0)
    env = os.path.join(_TEX, "studio_env.png")
    if os.path.exists(env):
        dome.CreateTextureFileAttr().Set(Sdf.AssetPath(env))
        dome.CreateTextureFormatAttr().Set("latlong")
    else:
        dome.CreateColorAttr(Gf.Vec3f(0.88, 0.92, 1.0))

    cx, cy = C.TABLE_CENTER
    Z = C.TABLE_TOP_Z
    for name, pos, size, inten, warm, rot in (
            ("Key",  (cx + 0.30, cy - 0.50, Z + 1.25), 0.55, 260000.0,
             (1.0, 0.96, 0.90), (150.0, 0.0, 25.0)),
            ("Fill", (cx - 0.50, cy + 0.70, Z + 1.05), 1.00,  65000.0,
             (0.90, 0.94, 1.0), (165.0, 0.0, -30.0)),
            ("Rim",  (cx - 0.35, cy - 0.05, Z + 1.55), 0.45,  70000.0,
             (1.0, 1.0, 1.0), (180.0, 0.0, 0.0))):
        lt = UsdLux.RectLight.Define(stage, f"/World/{name}Light")
        lt.CreateIntensityAttr(inten)
        lt.CreateWidthAttr(size); lt.CreateHeightAttr(size)
        lt.CreateColorAttr(Gf.Vec3f(*warm))
        lt.CreateNormalizeAttr(True)
        xf = UsdGeom.Xformable(lt.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*rot))


def _rounded_square(half, r, per_corner=9):
    """Counter-clockwise outline of a square with rounded corners.

    per_corner is odd so each arc includes its 45 deg midpoint, which is what
    lets _ray_to_square land exactly on the outer square's corners.
    """
    pts = []
    for k, (cx, cy) in enumerate(((half - r, half - r), (-half + r, half - r),
                                  (-half + r, -half + r), (half - r, -half + r))):
        for j in range(per_corner):
            a = math.radians(90 * k + 90.0 * j / (per_corner - 1))
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _ray_to_square(p, half):
    """Push a point radially out onto an axis-aligned square of half-size half."""
    m = max(abs(p[0]), abs(p[1]))
    return (p[0] * half / m, p[1] * half / m)


def add_bin(stage):
    """The collection bin, "BLOCK-03": a thick-walled yellow 3D-printed cube
    with a rounded-square pocket (VIDEO). Visual mesh plus an invisible box
    collider shell.

    Built procedurally because the scanned bins in assets/ are different parts
    (BLOCK-06 orange and BLOCK-02 blue, both thin-walled) that never appear in
    the cell video.
    """
    Z = C.TABLE_TOP_Z + C.SURFACE_DZ
    bx, by = C.BIN_XY
    ow, od = C.BIN_OUTER
    h, fl = C.BIN_WALL_H, C.BIN_FLOOR_T
    inner = _rounded_square(C.BIN_POCKET / 2, C.BIN_POCKET_R)
    outer = [_ray_to_square(p, ow / 2) for p in inner]
    n = len(inner)
    pts, counts, idx = [], [], []

    def ring(poly, z):
        base = len(pts)
        pts.extend(Gf.Vec3f(float(x), float(y), float(z)) for x, y in poly)
        return base

    o_top, i_top = ring(outer, h), ring(inner, h)
    o_bot, i_bot = ring(outer, 0.0), ring(inner, fl)
    c_bot = len(pts); pts.append(Gf.Vec3f(0, 0, 0))
    c_fl = len(pts); pts.append(Gf.Vec3f(0, 0, fl))
    for i in range(n):
        j = (i + 1) % n
        counts += [4, 4, 4, 3, 3]
        idx += [o_top + i, o_top + j, i_top + j, i_top + i]      # rim
        idx += [o_bot + i, o_bot + j, o_top + j, o_top + i]      # outer wall
        idx += [i_top + i, i_top + j, i_bot + j, i_bot + i]      # pocket wall
        idx += [c_bot, o_bot + j, o_bot + i]                      # underside
        idx += [c_fl, i_bot + i, i_bot + j]                       # pocket floor
    m = UsdGeom.Mesh.Define(stage, "/World/Bin")
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateDoubleSidedAttr(True)
    m.CreateSubdivisionSchemeAttr("none")
    UsdGeom.Xformable(m.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(bx, by, Z))
    bind_mat(m.GetPrim(), "bin_yellow")
    # the printed "CSI / BLOCK-03" barcode label on the front face
    lw, lh = 0.034, 0.022
    box(stage, "/World/BinLabel", (0.0006, lw, lh),
        (bx + ow / 2 + 0.0003, by, Z + h * 0.58), (0.9, 0.9, 0.9),
        collision=False, mat="label_white")
    box(stage, "/World/BinLabelHeader", (0.0007, lw, lh * 0.28),
        (bx + ow / 2 + 0.00035, by, Z + h * 0.58 + lh * 0.36), (0.05, 0.05, 0.05),
        collision=False, mat="black_plastic")

    # Collision: a box shell with a pocket-sized cavity. A convex hull or
    # decomposition of the visual mesh seals the pocket, so objects would rest
    # on the rim instead of dropping in.
    t = C.BIN_WALL_T
    prox = UsdGeom.Xform.Define(stage, "/World/BinCollider")
    UsdGeom.Imageable(prox.GetPrim()).CreateVisibilityAttr().Set("invisible")
    box(stage, "/World/BinCollider/Floor", (ow, od, fl),
        (bx, by, Z + fl / 2), (0.5, 0.5, 0.5))
    for tag, sz, off in (("n", (ow, t), (0.0,  od / 2 - t / 2)),
                         ("s", (ow, t), (0.0, -od / 2 + t / 2)),
                         ("e", (t, od - 2 * t), ( ow / 2 - t / 2, 0.0)),
                         ("w", (t, od - 2 * t), (-ow / 2 + t / 2, 0.0))):
        p = box(stage, f"/World/BinCollider/Wall_{tag}", (sz[0], sz[1], h),
                (bx + off[0], by + off[1], Z + h / 2), (0.5, 0.5, 0.5))
        bind_physics_material(p)


def add_grasp_cubes(stage):
    """Row of moulded plastic cubes in front of the robot for the pick task.

    Dynamic rigid bodies: they rest on the table under gravity and drop when
    released.
    """
    Z = C.TABLE_TOP_Z
    e = C.CUBE_EDGE
    for i, y in enumerate(C.CUBE_ROW_Y):
        m = C.CUBE_MATS[i % len(C.CUBE_MATS)]
        # start a hair above the table so the first physics step settles them
        # onto it rather than resolving an initial interpenetration
        p = box(stage, f"/World/Cube{i}", (e, e, e),
                (C.CUBE_ROW_X, y, Z + e / 2 + 0.0005), (0.5, 0.5, 0.5),
                collision=False, mat=m)
        make_dynamic(p, C.CUBE_MASS)


def _ear_clip(poly):
    """Triangulate a simple polygon. Needed because the finger profile is
    concave (the scallop), and a naive fan triangulation would fill the pocket
    back in."""
    idx = list(range(len(poly)))
    area = sum(poly[i][0] * poly[(i + 1) % len(poly)][1]
               - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))
    if area < 0:
        idx.reverse()

    def cross(o, a, b):
        return ((a[0]-o[0]) * (b[1]-o[1])) - ((a[1]-o[1]) * (b[0]-o[0]))

    def inside(p, a, b, c):
        d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
        neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (neg and pos)

    tris, guard = [], 0
    while len(idx) > 3 and guard < 5000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k-1], idx[k], idx[(k+1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross(a, b, c) <= 0:
                continue                      # reflex vertex, not an ear
            if any(inside(poly[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue                      # another vertex inside the ear
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


def prism(stage, path, profile, thickness, color, mat=None):
    """Extrude a 2D profile (list of (u, v)) by `thickness` along local Z."""
    n = len(profile)
    h = thickness / 2.0
    pts = [Gf.Vec3f(float(u), float(v), -h) for u, v in profile] + \
          [Gf.Vec3f(float(u), float(v), +h) for u, v in profile]
    counts, idx = [], []
    for i in range(n):                                   # side walls
        j = (i + 1) % n
        counts.append(4); idx += [i, j, j + n, i + n]
    for t in _ear_clip(profile):                         # caps
        counts.append(3); idx += [t[2], t[1], t[0]]
        counts.append(3); idx += [t[0] + n, t[1] + n, t[2] + n]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    m.CreateDoubleSidedAttr(True)
    bind_mat(m.GetPrim(), mat)
    return m.GetPrim()


def revolve(stage, path, profile, color, segments=48, mat=None):
    """Revolve a 2D (radius, axial) profile about the local Z axis.

    Used for the countersunk screws, whose tapered head is not expressible with
    the USD cone/cylinder primitives.
    """
    n = len(profile)
    pts, counts, idx = [], [], []
    for k in range(segments):
        a = 2 * math.pi * k / segments
        ca, sa = math.cos(a), math.sin(a)
        for r, z in profile:
            pts.append(Gf.Vec3f(float(r * ca), float(r * sa), float(z)))
    for k in range(segments):
        k2 = (k + 1) % segments
        for i in range(n - 1):
            a0 = k * n + i
            a1 = k * n + i + 1
            b0 = k2 * n + i
            b1 = k2 * n + i + 1
            counts.append(4); idx += [a0, a1, b1, b0]
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    m.CreateDoubleSidedAttr(True)
    bind_mat(m.GetPrim(), mat)
    return m.GetPrim()


def add_screws(stage):
    """Loose M6 x 30 countersunk socket screws (ISO 10642) lying on the mat.

    90 deg conical head, as seen side-on in the cell video. Lying on its side a
    countersunk screw rests on the head rim and the shank, so it settles with
    its axis tilted a few degrees -- the physics works that out.
    """
    Z = C.TABLE_TOP_Z + C.SURFACE_DZ
    hr, sr = C.SCREW_HEAD_D / 2, C.SCREW_SHANK_D / 2
    L, hh = C.SCREW_LENGTH, C.SCREW_HEAD_H
    edge = hh - (hr - sr)             # short cylindrical land at the head face
    # profile in (radius, axial): head face -> land -> 90 deg cone -> shank -> tip
    prof = [(0.0, 0.0), (hr, 0.0), (hr, -edge), (sr, -hh),
            (sr, -(L - 0.0008)), (sr * 0.8, -L), (0.0, -L)]
    for i, (x, y, yaw) in enumerate(C.SCREWS):
        g = UsdGeom.Xform.Define(stage, f"/World/Screw{i}")
        xf = UsdGeom.Xformable(g.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, Z + hr + 0.0005))
        # lie the screw on its side: tip along +X of the local frame, then yaw
        xf.AddRotateZOp().Set(float(yaw))
        xf.AddRotateYOp().Set(-90.0)
        body = revolve(stage, f"/World/Screw{i}/Body", prof, (0.06, 0.06, 0.07),
                       mat="black_oxide")
        # Convex DECOMPOSITION, not a convex hull: the hull merges the head and
        # shank into one long taper, and parallel jaws closing on a taper wedge
        # the part out axially instead of gripping it.
        UsdPhysics.CollisionAPI.Apply(body)
        UsdPhysics.MeshCollisionAPI.Apply(body).CreateApproximationAttr().Set(
            "convexDecomposition")
        from pxr import PhysxSchema
        cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(body)
        cd.CreateHullVertexLimitAttr().Set(64)
        cd.CreateMaxConvexHullsAttr().Set(8)
        cd.CreateErrorPercentageAttr().Set(0.5)
        bind_physics_material(body)
        make_dynamic(g.GetPrim(), C.SCREW_MASS, collide_self=False)
        # 4 mm hex socket, drawn as a dark decal just proud of the head face
        hexp = UsdGeom.Mesh.Define(stage, f"/World/Screw{i}/Socket")
        r = 0.0023
        pts, counts, idx = [], [], []
        for k in range(6):
            a = 2 * math.pi * k / 6
            pts.append(Gf.Vec3f(float(r * math.cos(a)), float(r * math.sin(a)), 0.0002))
        counts.append(6); idx += list(range(6))
        hexp.CreatePointsAttr(pts)
        hexp.CreateFaceVertexCountsAttr(counts)
        hexp.CreateFaceVertexIndicesAttr(idx)
        hexp.CreateDoubleSidedAttr(True)
        hexp.GetPrim().SetInstanceable(False)
        bind_mat(hexp.GetPrim(), "black_plastic")


# ---- gripper: SMC MHZL2-16D2 CAD body and photo-reconstructed blades, merged from the hackathon
# starter (2026-10-09); lengths and jaw gap set to the real cell in scene_config.py.

METAL = (0.62, 0.63, 0.66)
ANODISED = (0.72, 0.73, 0.74)  # light anodised aluminium gripper body
DARK_METAL = (0.30, 0.31, 0.34)


def gripper_layout():
    """(carriage_x, finger_x) along TOOL_AXIS from the J6_link origin (the faceplate is FLANGE_OFFSET out)."""
    x = C.FLANGE_OFFSET + C.ADAPTER_T + C.ZCYL_LEN + C.ZFLANGE_T + C.BODY_LEN + C.PLATE_T
    return x, x + C.CARRIAGE_LEN - C.CARRIAGE_OVERLAP


def gripper_transforms(jaw):
    """Translations for the moving jaw prims at a given per-jaw opening."""
    cx, fx = gripper_layout()
    ax = np.array(C.TOOL_AXIS, float)
    sp = np.array(C.FINGER_SPREAD_AXIS, float)
    out = {}
    for sign, name in ((+1, "L"), (-1, "R")):
        out[FLANGE + f"/Gripper/Carriage{name}"] = ax * (
            cx + C.CARRIAGE_LEN / 2
        ) + sp * sign * (jaw + C.FINGER_WIDTH / 2)
        out[FLANGE + f"/Gripper/Finger{name}"] = ax * fx + sp * (sign * jaw)
    return out


def grasp_offset():
    """Flange face to the centre of the scallop pocket.

    Correct for cradling a cylinder in mid-air, WRONG for picking something off
    a surface: the pocket sits 12 mm above the fingertips, so commanding it to an
    object's centre height drives the tips through the table. Now that the
    fingers carry colliders that is a hard collision, not a cosmetic issue.
    Use tip_offset() for table-top picks.
    """
    _, fx = gripper_layout()
    return fx + C.SCALLOP_AT * C.FINGER_LENGTH


def tip_offset():
    """Flange face to the fingertips -- the reference for table-top grasps."""
    _, fx = gripper_layout()
    return fx + C.FINGER_LENGTH


JAW_PRIMS = {"L": C.ROBOT_PRIM + "/JawL", "R": C.ROBOT_PRIM + "/JawR"}
JAW_JOINTS = {"L": "jaw_l", "R": "jaw_r"}


def smc_visual(stage, path, part_name, translation, material):
    """Load a pre-tessellated supplied STEP part; physics uses explicit proxies."""
    source = os.path.join(os.path.dirname(os.path.abspath(__file__)), C.GRIPPER_CAD)
    with open(source) as f:
        part = next(p for p in json.load(f)["parts"] if p["name"] == part_name)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in part["points"]])
    mesh.CreateFaceVertexCountsAttr([3] * (len(part["indices"]) // 3))
    mesh.CreateFaceVertexIndicesAttr(part["indices"])
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(*translation))
    bind_mat(mesh.GetPrim(), material)
    return mesh.GetPrim()


def detailed_finger_visual(stage, jaw_path, sign, offset):
    """Photo-guided visual machining; collision mesh is deliberately separate."""
    source = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "assets/gripper/photo_finger_detail.json",
    )
    with open(source) as f:
        parts = json.load(f)["parts"]
    for part in parts:
        mesh = UsdGeom.Mesh.Define(stage, jaw_path + "/Detail/" + part["name"])
        mesh.CreatePointsAttr(
            [Gf.Vec3f(p[0], sign * p[1], p[2]) for p in part["points"]]
        )
        indices = np.array(part["indices"]).reshape(-1, 3)
        if sign < 0:
            indices = indices[:, ::-1]
        mesh.CreateFaceVertexCountsAttr([3] * len(indices))
        mesh.CreateFaceVertexIndicesAttr(indices.flatten().tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        uv.Set([Gf.Vec2f(*p) for p in part["uv"]])
        UsdGeom.Xformable(mesh).AddTranslateOp().Set(
            Gf.Vec3d(offset, -sign * C.FINGER_WIDTH / 2, 0)
        )
        bind_mat(
            mesh.GetPrim(),
            {
                "plain": "finger_milled",
                "marked": "uwm_finger",
                "socket": "alu_machined",
            }[part["name"]],
        )


def attach_gripper(stage, jaw=None, articulated=True):
    """EOAT modelled from the bench photos, with real physics.

    Static parts (adapter, cylinder, flange, gripper body, jaw plate, camera and
    bracket) are children of J6_link and carry CollisionAPI, so PhysX treats
    them as collision geometry belonging to that articulation link and moves
    them with it -- no transform writes, no resync.

    The two jaws are separate rigid bodies joined to J6_link by PRISMATIC joints,
    which makes them part of the articulation. That is what lets them carry
    colliders and still open and close: writing xforms on a collider forces a
    PhysX resync that invalidates the tensor view mid-run, whereas a joint target
    is exactly what the physics engine expects.

    Jaw bodies live at /World/ER4iA/Jaw{L,R}, NOT under J6_link -- nesting a
    rigid body inside another rigid body's subtree is invalid.
    articulated=False builds a static, zero-origin assembly for inspection.
    """
    if jaw is None:
        jaw = C.JAW_OPEN
    ax = np.array(C.TOOL_AXIS, float)
    sp = np.array(C.FINGER_SPREAD_AXIS, float)

    def at(d, lateral=0.0):
        return tuple(ax * d + sp * lateral)

    def disc(path, radius, length, d, color, mat="alu_machined"):
        c = UsdGeom.Cylinder.Define(stage, path)
        c.CreateAxisAttr("X")
        c.CreateRadiusAttr(radius)
        c.CreateHeightAttr(length)
        c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdGeom.Xformable(c.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(*at(d + length / 2))
        )
        bind_mat(c.GetPrim(), mat)
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        return c.GetPrim()

    x = C.FLANGE_OFFSET  # COMPAT PATCH: the stack starts at the J6 faceplate
    disc(
        FLANGE + "/Gripper/Adapter",
        C.ADAPTER_R,
        C.ADAPTER_T,
        x,
        (0.05, 0.05, 0.05),
        mat="black_anodised",
    )
    x += C.ADAPTER_T
    if C.ZCYL_RECTANGULAR:
        box(
            stage,
            FLANGE + "/Gripper/ZCylinder",
            (C.ZCYL_LEN, C.ZCYL_BODY_W, C.ZCYL_BODY_D),
            at(x + C.ZCYL_LEN / 2),
            ANODISED,
            mat="alu_machined",
        )
        box(
            stage,
            FLANGE + "/Gripper/SlideRail",
            (C.ZCYL_LEN * 0.85, 0.009, 0.003),
            (x + C.ZCYL_LEN / 2, 0, C.ZCYL_BODY_D / 2 + 0.0015),
            METAL,
            mat="alu_machined",
        )
    else:
        disc(FLANGE + "/Gripper/ZCylinder", C.ZCYL_R, C.ZCYL_LEN, x, METAL)
    disc(
        FLANGE + "/Gripper/Collar",
        C.COLLAR_R,
        C.COLLAR_T,
        x + C.COLLAR_AT,
        METAL,
        mat="alu_machined",
    )
    x += C.ZCYL_LEN
    disc(FLANGE + "/Gripper/ZFlange", C.ZFLANGE_R, C.ZFLANGE_T, x, METAL)
    x += C.ZFLANGE_T

    body = box(
        stage,
        FLANGE + "/Gripper/Body",
        (C.BODY_LEN, C.BODY_W, C.BODY_D),
        at(x + C.BODY_LEN / 2),
        ANODISED,
        mat="alu_anodised",
    )
    UsdGeom.Imageable(body).CreateVisibilityAttr().Set("invisible")
    smc_visual(stage, FLANGE + "/Gripper/SMCBody", "body", at(x), "alu_anodised")
    x += C.BODY_LEN
    if C.PLATE_T > 0:
        box(
            stage,
            FLANGE + "/Gripper/JawPlate",
            (C.PLATE_T, C.PLATE_W, C.PLATE_D),
            at(x + C.PLATE_T / 2),
            DARK_METAL,
            mat="steel_dark",
        )
    x += C.PLATE_T

    carriage_x, finger_x = gripper_layout()

    for sign, tag in ((+1, "L"), (-1, "R")):
        jaw_path = JAW_PRIMS[tag]
        g = UsdGeom.Xform.Define(stage, jaw_path)
        # place the jaw where it sits with the arm at its zero pose
        home = (
            np.array([0.29, 0.0, 0.61 + C.TABLE_TOP_Z]) if articulated else np.zeros(3)
        )
        UsdGeom.Xformable(g.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(
                *(home + np.array(at(carriage_x, sign * (jaw + C.FINGER_WIDTH / 2))))
            )
        )

        # CAD open jaw centres are +/-13.9 mm. Their inward motion is 6 mm.
        lateral = sign * (0.0139 - C.JAW_OPEN - C.FINGER_WIDTH / 2)
        carriage = box(
            stage,
            jaw_path + "/Carriage",
            (C.CARRIAGE_LEN, C.CARRIAGE_W, C.CARRIAGE_D),
            (0.006725, lateral, 0),
            DARK_METAL,
            mat="steel_dark",
        )
        UsdGeom.Imageable(carriage).CreateVisibilityAttr().Set("invisible")
        smc_visual(
            stage,
            jaw_path + "/SMCJaw",
            "jaw_l" if sign > 0 else "jaw_r",
            (0, lateral, 0),
            "steel_dark",
        )

        prof = [(u, sign * v) for u, v in finger_profile()]
        boundary = C.FINGER_LENGTH - C.SOFT_TIP_LENGTH
        sections = [
            (
                clip_profile(prof, boundary, False),
                "Finger",
                "uwm_finger",
                PHYS_MAT_PATH,
            ),
            (
                clip_profile(prof, boundary, True),
                "SoftTip",
                "uwm_finger",
                SOFT_TIP_MAT_PATH,
            ),
        ]
        for section, name, visual_material, physics_material in sections:
            fprim = prism(
                stage,
                jaw_path + "/" + name,
                section,
                C.FINGER_THICK,
                METAL,
                mat=visual_material,
            )
            UsdGeom.Xformable(fprim).AddTranslateOp().Set(
                Gf.Vec3d(finger_x - carriage_x, -sign * C.FINGER_WIDTH / 2, 0)
            )
            uv = UsdGeom.PrimvarsAPI(fprim).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            )
            uv.Set(
                [
                    Gf.Vec2f(
                        float(p[0] / C.FINGER_LENGTH),
                        float(sign * p[1] / C.FINGER_WIDTH),
                    )
                    for p in UsdGeom.Mesh(fprim).GetPointsAttr().Get()
                ]
            )
            UsdGeom.Imageable(fprim).CreateVisibilityAttr().Set("invisible")
        detailed_finger_visual(stage, jaw_path, sign, finger_x - carriage_x)
        # COLLISION: straight blades -- the grasping validated in this cell (every recorded episode). The CAD
        # / photo profile above is visual only: its hooked AA socket stalled the jaws on a lying screw. The
        # inner face sits at the real fingertip gap, FINGER_INNER_TIP outside the jaw's straight face, along
        # the whole blade (4 mm closed / 16 mm open); the distal SOFT_TIP_LENGTH carries the high friction.
        from pxr import PhysxSchema

        lateral = -sign * C.FINGER_WIDTH / 2 + sign * (C.FINGER_INNER_TIP + C.FINGER_COLLIDER_T / 2)
        for name, u0, u1, material in (
            ("Blade", 0.0, boundary, PHYS_MAT_PATH),
            ("BladeTip", boundary, C.FINGER_LENGTH, SOFT_TIP_MAT_PATH),
        ):
            collider = box(
                stage,
                jaw_path + f"/Collision/{name}",
                (u1 - u0, C.FINGER_COLLIDER_T, C.FINGER_THICK),
                (finger_x - carriage_x + (u0 + u1) / 2, lateral, 0.0),
                METAL,
            )
            UsdGeom.Imageable(collider).CreateVisibilityAttr().Set("invisible")
            contact = PhysxSchema.PhysxCollisionAPI.Apply(collider)
            contact.CreateContactOffsetAttr().Set(0.0002)
            contact.CreateRestOffsetAttr().Set(0.0)
            bind_physics_material(collider, material)
        for number, u in enumerate(C.FINGER_MOUNT_HOLES):
            for face in (-1, 1):
                cyl(
                    stage,
                    jaw_path + f"/MountScrew{number}_{'front' if face>0 else 'back'}",
                    0.0022,
                    0.0008,
                    (
                        finger_x - carriage_x + u,
                        0,
                        face * (C.FINGER_THICK / 2 - 0.0007),
                    ),
                    DARK_METAL,
                    mat="steel_dark",
                )
                # Hex drive inset within the recessed cap screw.
                h = [
                    (
                        finger_x - carriage_x + u + 0.0012 * math.cos(k * math.pi / 3),
                        0.0012 * math.sin(k * math.pi / 3),
                    )
                    for k in range(6)
                ]
                inset = prism(
                    stage,
                    jaw_path + f"/ScrewSocket{number}_{'front' if face>0 else 'back'}",
                    h,
                    0.00004,
                    (0.035, 0.035, 0.04),
                    mat="black_oxide",
                )
                UsdGeom.Xformable(inset).AddTranslateOp().Set(
                    Gf.Vec3d(0, 0, face * (C.FINGER_THICK / 2 - 0.00028))
                )
        outer_bolt = cyl(
            stage,
            jaw_path + "/OuterCrossBolt",
            0.0022,
            0.0018,
            (finger_x - carriage_x + 0.012, sign * (C.FINGER_WIDTH / 2 + 0.0009), 0),
            DARK_METAL,
            mat="black_oxide",
        )
        UsdGeom.Xformable(outer_bolt).AddRotateXOp().Set(90.0)
        if tag == "R" and C.SHOW_TOOL_DETAIL:
            carrier = prism(
                stage,
                jaw_path + "/SensorCarrier",
                [
                    (0.024, 0),
                    (0.054, 0),
                    (0.056, 0.0015),
                    (0.053, 0.0028),
                    (0.027, 0.0028),
                ],
                0.0036,
                METAL,
                mat="alu_machined",
            )
            UsdGeom.Xformable(carrier).AddTranslateOp().Set(
                Gf.Vec3d(finger_x - carriage_x, C.FINGER_WIDTH / 2, 0)
            )
            box(
                stage,
                jaw_path + "/SensorRedNose",
                (0.006, 0.0025, 0.003),
                (finger_x - carriage_x + 0.054, C.FINGER_WIDTH / 2 + 0.0008, 0),
                (0.7, 0.035, 0.025),
                collision=False,
                mat="plastic_red",
            )

        if C.FINGER_TAPE:
            # duct tape wrapped round the working end: slightly proud of the blade
            # so it is the surface that actually touches the part, and bound to a
            # high-friction material. It covers the scallop, which is what the real
            # wrap does too.
            pad, u0 = C.TAPE_PAD, C.FINGER_LENGTH - C.TAPE_LEN
            tape_prof = [
                (u0, sign * -pad),
                (C.FINGER_LENGTH, sign * -pad),
                (C.FINGER_LENGTH, sign * (C.FINGER_WIDTH + pad)),
                (u0, sign * (C.FINGER_WIDTH + pad)),
            ]
            tprim = prism(
                stage,
                jaw_path + "/Tape",
                tape_prof,
                C.FINGER_THICK + 2 * pad,
                (0.38, 0.39, 0.40),
                mat="duct_tape",
            )
            UsdGeom.Xformable(tprim).AddTranslateOp().Set(
                Gf.Vec3d(finger_x - carriage_x, -sign * C.FINGER_WIDTH / 2, 0.0)
            )
            UsdPhysics.CollisionAPI.Apply(tprim)
            UsdPhysics.MeshCollisionAPI.Apply(tprim).CreateApproximationAttr().Set(
                "convexHull"
            )
            bind_physics_material(tprim, TAPE_MAT_PATH)

        if articulated:
            rb = UsdPhysics.RigidBodyAPI.Apply(g.GetPrim())
            rb.CreateRigidBodyEnabledAttr().Set(True)
            UsdPhysics.MassAPI.Apply(g.GetPrim()).CreateMassAttr(C.JAW_MASS)

            j = UsdPhysics.PrismaticJoint.Define(
                stage, FLANGE + f"/Gripper/{JAW_JOINTS[tag]}"
            )
            j.CreateBody0Rel().SetTargets([FLANGE])
            j.CreateBody1Rel().SetTargets([jaw_path])
            j.CreateAxisAttr().Set("Y")
            j.CreateLocalPos0Attr().Set(
                Gf.Vec3f(*at(carriage_x, sign * (jaw + C.FINGER_WIDTH / 2)))
            )
            j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            travel = C.JAW_OPEN - C.JAW_CLOSED
            # q = 0 is fully open; closing moves each jaw toward the tool axis
            if sign > 0:
                j.CreateLowerLimitAttr().Set(-travel)
                j.CreateUpperLimitAttr().Set(0.0)
            else:
                j.CreateLowerLimitAttr().Set(0.0)
                j.CreateUpperLimitAttr().Set(travel)
            drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
            drive.CreateTypeAttr().Set("force")
            drive.CreateStiffnessAttr().Set(C.JAW_DRIVE_STIFF)
            drive.CreateDampingAttr().Set(C.JAW_DRIVE_DAMP)
            drive.CreateMaxForceAttr().Set(C.JAW_MAX_FORCE)
            drive.CreateTargetPositionAttr().Set(0.0)

    # Wrist vision: compact grey camera body on a bracket to one side of the
    # tool. PHOTO 2026-10-08: no ring light — earlier model was wrong.
    side = np.array(C.WRISTCAM_DIR, float)

    def at_side(d, lateral):
        return tuple(ax * d + side * lateral)

    lat = C.WRISTCAM_LATERAL
    cb = C.CAMBODY
    cam_bot = C.TOOL_LENGTH - C.CAM_ABOVE_TIP  # distance from flange to camera bottom
    cam_ctr = cam_bot - cb[2] / 2
    # Camera bracket orientation follows the fitted mount direction in Y/Z.
    bracket_roll = math.degrees(math.atan2(-side[1], side[2]))
    # Housing and sensor roll together around the tool axis. The bracket
    # follows the rotated mount offset; finger orientation stays on local Y.
    for name, depth, axial, material in (
        ("WristCamBody", cb[2], cam_ctr, "alu_anodised"),
        ("WristCamSeam", 0.0005, cam_bot - cb[2] - 0.00025, "black_plastic"),
        (
            "WristCamLid",
            C.CAM_LID_T,
            cam_bot - cb[2] - 0.0005 - C.CAM_LID_T / 2,
            "alu_machined",
        ),
    ):
        housing = prism(
            stage,
            FLANGE + "/Gripper/" + name,
            _rounded_square(cb[0] / 2, C.CAM_CORNER_R),
            depth,
            C.CAMBODY_GREY,
        )
        bind_mat(housing, material)
        xf = UsdGeom.Xformable(housing)
        xf.AddTranslateOp().Set(Gf.Vec3d(*at_side(axial, lat)))
        xf.AddRotateXOp().Set(C.WRISTCAM_ROLL_DEG)
        xf.AddRotateYOp().Set(90.0)
        UsdPhysics.CollisionAPI.Apply(housing)
        UsdPhysics.MeshCollisionAPI.Apply(housing).CreateApproximationAttr().Set(
            "convexHull"
        )
    for y in (-0.018, 0.018):
        for z in (-0.018, 0.018):
            screw = disc(
                FLANGE
                + f"/Gripper/CamScrew_{y}_{z}".replace("-", "m").replace(".", "p"),
                0.0018,
                0.001,
                cam_bot - cb[2] - 0.003,
                DARK_METAL,
                "steel_dark",
            )
            UsdGeom.Xformable(screw).GetOrderedXformOps()[0].Set(
                Gf.Vec3d(cam_bot - cb[2] - 0.0025, side[1] * lat + y, side[2] * lat + z)
            )
    # small lens cylinder on the bottom face of the camera body, pointing down
    lens = UsdGeom.Cylinder.Define(stage, FLANGE + "/Gripper/WristCamLens")
    lens.CreateAxisAttr("X")
    lens.CreateRadiusAttr(0.008)
    lens.CreateHeightAttr(0.006)
    UsdGeom.Xformable(lens.GetPrim()).AddTranslateOp().Set(
        Gf.Vec3d(*at_side(cam_bot + 0.003, lat))
    )
    bind_mat(lens.GetPrim(), "black_plastic")
    # bracket: an arm out from the rod guide, then a plate down beside the camera
    arm_x = C.FLANGE_OFFSET + C.ADAPTER_T + C.ZCYL_LEN + C.ZFLANGE_T / 2
    box(
        stage,
        FLANGE + "/Gripper/CamBracketArm",
        (C.BRACKET_T, 0.018, lat - cb[1] / 2),
        at_side(arm_x, (lat - cb[1] / 2) / 2 + C.ZFLANGE_R / 2),
        DARK_METAL,
        rot=(bracket_roll, 0.0, 0.0),
        mat="steel_dark",
    )
    drop = cam_bot - arm_x
    box(
        stage,
        FLANGE + "/Gripper/CamBracketPlate",
        (drop, 0.018, C.BRACKET_T),
        at_side(arm_x + drop / 2, lat - cb[1] / 2 - C.BRACKET_T / 2),
        DARK_METAL,
        rot=(bracket_roll, 0.0, 0.0),
        mat="steel_dark",
    )

    if C.SHOW_TOOL_DETAIL:
        for n, color in enumerate(((0.95, 0.32, 0.04), (0.08, 0.35, 0.8))):
            # Two stacked pneumatic sensor/valve units visible above the jaws.
            box(
                stage,
                FLANGE + f"/Gripper/PneumaticBlock{n}",
                (0.012, 0.022, 0.014),
                (finger_x - 0.027 + n * 0.014, -0.004, 0.022),
                ANODISED,
                mat="alu_anodised",
            )
            box(
                stage,
                FLANGE + f"/Gripper/PneumaticInsert{n}",
                (0.006, 0.019, 0.002),
                (finger_x - 0.027 + n * 0.014, -0.004, 0.030),
                color,
                collision=False,
            )
        for n, (along, color) in enumerate(
            ((0.036, (0.08, 0.35, 0.8)), (0.065, (0.95, 0.32, 0.04)))
        ):
            cyl(
                stage,
                FLANGE + f"/Gripper/AirFitting{n}",
                0.0034,
                0.010,
                (along, C.ZCYL_BODY_W / 2, 0),
                DARK_METAL,
                mat="steel_dark",
            )
            hose = UsdGeom.BasisCurves.Define(stage, FLANGE + f"/Gripper/AirTube{n}")
            hose.CreateTypeAttr("linear")
            hose.CreateWrapAttr("nonperiodic")
            hose.CreateCurveVertexCountsAttr([5])
            hose.CreatePointsAttr(
                [
                    Gf.Vec3f(along, 0.019, 0.006),
                    Gf.Vec3f(along - 0.005, 0.027, 0.015),
                    Gf.Vec3f(0.022, 0.029 + n * 0.005, 0.022),
                    Gf.Vec3f(-0.012, 0.031 + n * 0.005, 0.021),
                    Gf.Vec3f(-0.035, 0.026, 0.016),
                ]
            )
            hose.CreateWidthsAttr([0.003])
            hose.SetWidthsInterpolation("constant")
            hose.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            bind_mat(hose.GetPrim(), "air_line_blue")
        laser = disc(
            FLANGE + "/Gripper/LaserBody",
            C.LASER_RADIUS,
            C.LASER_LENGTH,
            finger_x - 0.010,
            (0.65, 0.035, 0.025),
            None,
        )
        UsdGeom.Xformable(laser).GetOrderedXformOps()[0].Set(
            Gf.Vec3d(finger_x - 0.010 + C.LASER_LENGTH / 2, -0.016, 0.011)
        )
        # Keep the existing collider; reproduce the visible slim metal carrier
        # and red nose from the JPG independently of collision geometry.
        UsdGeom.Imageable(laser).CreateVisibilityAttr().Set("invisible")
        box(
            stage,
            FLANGE + "/Gripper/CameraConnector",
            (0.012, 0.010, 0.009),
            at_side(cam_ctr - 0.005, lat + cb[0] / 2 + 0.004),
            (0.95, 0.32, 0.04),
            collision=False,
            mat="camera_cable_orange",
        )
        across = np.cross(ax, side)

        def photo_line(name, points, width, material):
            line = UsdGeom.BasisCurves.Define(stage, FLANGE + "/Gripper/" + name)
            line.CreateTypeAttr("cubic")
            line.CreateBasisAttr("bspline")
            line.CreateWrapAttr("nonperiodic")
            line.CreateCurveVertexCountsAttr([len(points)])
            line.CreatePointsAttr(
                [Gf.Vec3f(*(ax * d + side * l + across * t)) for d, l, t in points]
            )
            line.CreateWidthsAttr([width])
            line.SetWidthsInterpolation("constant")
            bind_mat(line.GetPrim(), material)

        # Broad blue loops and orange camera cable visible in both close-up JPGs.
        # These details have no colliders and do not alter the grasp envelope.
        for n in range(2):
            t = 0.012 + n * 0.010
            photo_line(
                f"PhotoBlueAirLoop{n}",
                [
                    (-0.040, 0.045, t),
                    (-0.020, 0.055, t),
                    (0.025, 0.105, t),
                    (0.080, 0.122, t),
                    (0.133, 0.105, t),
                    (0.152, 0.070, t),
                    (0.127, 0.049, t),
                    (0.080, 0.032, t),
                    (0.036, 0.023, t),
                    (0.036, 0.023, t),
                ],
                0.0055,
                "air_line_blue",
            )
        photo_line(
            "PhotoOrangeCameraCable",
            [
                (cam_ctr - 0.005, lat + 0.031, 0),
                (cam_ctr - 0.005, lat + 0.031, 0),
                (0.065, 0.095, 0.024),
                (0.015, 0.100, 0.040),
                (-0.025, 0.065, 0.035),
                (-0.040, 0.055, 0.028),
            ],
            0.004,
            "camera_cable_orange",
        )
        photo_line(
            "PhotoBlackSensorLead",
            [
                (0.095, 0.002, 0.017),
                (0.095, 0.002, 0.017),
                (0.084, -0.025, 0.025),
                (0.035, -0.043, 0.035),
                (0.010, -0.029, 0.030),
                (-0.025, -0.015, 0.020),
                (-0.030, -0.015, 0.020),
            ],
            0.003,
            "black_plastic",
        )

    # Default centimetre-scale contact offsets are too large for this small
    # actuator: the carriage proxies can repel each other before closing.
    from pxr import PhysxSchema

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.CollisionAPI) and (
            path.startswith(FLANGE + "/Gripper/")
            or any(path.startswith(root + "/") for root in JAW_PRIMS.values())
        ):
            contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            contact.CreateContactOffsetAttr().Set(0.0002)
            contact.CreateRestOffsetAttr().Set(0.0)


def jaw_targets(jaw):
    """Joint targets (metres) for a requested per-side opening."""
    d = C.JAW_OPEN - float(jaw)
    return {JAW_JOINTS["L"]: -d, JAW_JOINTS["R"]: +d}


def _oriented(size):
    """Re-order a (thick, wide, long) extent so 'long' lies along TOOL_AXIS."""
    ax = int(np.argmax(np.abs(C.TOOL_AXIS)))
    out = [size[0], size[1], size[2]]
    if ax != 2:
        out[2], out[ax] = out[ax], out[2]
    return tuple(out)
