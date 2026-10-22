"""Isaac Sim 6 backend of the sorting cell: the FANUC ER-4iA (arm A alone) on
the ESD-mat worktop in the empty hall, a parts tray, three signed bins and six
physical items.

Everything the robot handles is simulated: the items are dynamic rigid bodies
(cylinder colliders, real masses, the cell's friction material), the jaws are
the cell's prismatic-joint fingers. Nothing is teleported; picking and placing
are contact physics.

Cameras: "top" (nadir RGB + depth, fixed), "wrist" (RGB, on J6, the cell's
wrist camera), "overview" (a fixed oblique view for snapshots and videos).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

from . import layout as S
from .textures import OUT as TEXTURES, STICKER_L, STICKER_R

# the FANUC cell and its kinematics (loto/isaac/backends.py puts the cell on
# sys.path and holds the verified IK)
from loto.isaac import backends as LB  # noqa: E402
C = LB.C

PHYSICS_DT = 1.0 / 120.0                  # small parts in a pinch grasp
ROOT = "/World/Sort"
JAW_OPEN = 0.013                          # per jaw: a 26 mm gap for 15 mm parts

TOP_CAM_PATH = "/World/TopCam"
TOP_CAM = dict(parent="/World", eye=(0.36, 0.0, C.TABLE_TOP_Z + 1.05),
               look_at=(0.36, 0.0, C.TABLE_TOP_Z), up=(-1.0, 0.0, 0.0),
               resolution=(1280, 960), focal_length=20.0, horiz_aperture=20.955)
OVERVIEW_CAM_PATH = "/World/OverviewCam"
OVERVIEW_CAM = dict(parent="/World", eye=(1.12, -0.48, C.TABLE_TOP_Z + 0.72),
                    look_at=(0.33, 0.0, C.TABLE_TOP_Z + 0.04), up=(0.0, 0.0, 1.0),
                    resolution=(1280, 960), focal_length=14.0, horiz_aperture=20.955)

PROP_MATERIALS = {
    #                     diffuse               metallic roughness
    "tray_grey":       ((0.30, 0.31, 0.33), 0.00, 0.60),
    "bin_grey":        ((0.20, 0.21, 0.23), 0.00, 0.55),
    "sign_board":      ((0.85, 0.85, 0.84), 0.00, 0.50),
    "cell_steel":      ((0.80, 0.80, 0.82), 0.95, 0.22),   # battery terminals
    "ring_black":      ((0.03, 0.03, 0.035), 0.00, 0.45),  # insulating ring
}
COLLIDER_SIDES = 20
ESD_MAT, EXPOSURE_EV = LB.ESD_MAT, LB.EXPOSURE_EV     # the look every task shares
FINISH = {"metal": (0.92, 0.24), "paint": (0.0, 0.62), "gloss": (0.0, 0.22)}


class SortingBackend:
    """The cell, cameras and arm; _build_scene() adds the task's fixtures and
    rigid items (overridden by the conveyor task)."""
    jaw_open = JAW_OPEN
    table = True                             # False: no worktop (the arm on a pedestal)
    settle_s = 2.0                           # items come to rest before the program starts
    # worktop: a blue-grey ESD mat, matte (rgb, roughness); SORT_TOP=r,g,b[,rough] overrides it
    top_finish = ESD_MAT
    # camera exposure in stops relative to the renderer's default (f/5, 1/50 s,
    # ISO 100), which overexposes the cell in the hall's light; SORT_EV
    # overrides it. Negative = darker.
    exposure_ev = EXPOSURE_EV

    def __init__(self, lay, headless=True, spp=4, background=None,
                 views=("overview", "top", "wrist"), **scene):
        """scene: passed to _build_scene. Here sticker_on: index of the item
        (a battery) that carries the METAL sticker -- the deployment-time
        environment attack."""
        from isaacsim import SimulationApp
        self.app = SimulationApp({"headless": headless})
        import omni.usd
        from isaacsim.core.api import World
        import cell
        C.JAW_OPEN = self.jaw_open             # read by cell.attach_gripper / jaw_targets
        self.cell, self.lay = cell, lay
        self.world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT,
                           rendering_dt=PHYSICS_DT)
        self.stage = omni.usd.get_context().get_stage()
        cell.build(self.stage, with_objects=False)
        self._mats = {}
        self._add_materials()
        self._white_top()
        self._add_background(background or LB.default_background())
        if spp > 0:
            import render_settings
            render_settings.enable_path_tracing(spp)
        self._set_exposure(float(os.environ.get("SORT_EV", self.exposure_ev)))
        for path, cfg in ((TOP_CAM_PATH, TOP_CAM), (OVERVIEW_CAM_PATH, OVERVIEW_CAM)):
            cell.camera(self.stage, path, cfg)
        from pxr import UsdGeom
        UsdGeom.Xform.Define(self.stage, ROOT)
        if not self.table:
            self._remove_table()
        self.item_paths = self._build_scene(**scene)    # rigid items, in order
        self.views = tuple(views)
        self.dt = PHYSICS_DT
        self.t = 0.0

    # ---- scene -------------------------------------------------------------------
    def _build_scene(self, sticker_on=None):
        self._build_tray()
        for name, c in self.lay.bins.items():
            self._build_bin(name, c)
        return [self._build_item(i, it, sticker=(i == sticker_on))
                for i, it in enumerate(self.lay.items)]

    def _w(self, p):
        return (float(p[0]), float(p[1]), float(p[2]) + C.TABLE_TOP_Z)

    def _add_materials(self):
        from isaacsim.core.api.materials import OmniPBR
        for name, (diff, metal, rough) in PROP_MATERIALS.items():
            pbr = OmniPBR(prim_path=f"/World/Looks/{name}", name=name, color=np.array(diff))
            pbr.set_metallic_constant(metal)
            pbr.set_reflection_roughness(rough)
            self.cell.MATS[name] = pbr.material

    def _mat(self, colour, finish):
        """Material for an item's colour and finish (shared palette)."""
        key = f"item_{colour}_{finish}"
        if key not in self.cell.MATS:
            from isaacsim.core.api.materials import OmniPBR
            metal, rough = FINISH[finish]
            pbr = OmniPBR(prim_path=f"/World/Looks/{key}", name=key,
                          color=np.array(S.PALETTE[colour]))
            pbr.set_metallic_constant(metal)
            pbr.set_reflection_roughness(rough)
            self.cell.MATS[key] = pbr.material
        return key

    def _tex_mat(self, tex, rough=0.3):
        key = f"tex_{tex}"
        if key not in self._mats:
            from pxr import Sdf
            from isaacsim.core.api.materials import OmniPBR
            pbr = OmniPBR(prim_path=f"/World/Looks/{key}", name=key, color=np.ones(3))
            pbr.set_metallic_constant(0.0)
            pbr.set_reflection_roughness(rough)
            pbr.set_project_uvw(False)
            pbr.shaders_list[0].CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(str(TEXTURES / f"{tex}.png")))
            self._mats[key] = pbr.material
        return self._mats[key]

    def _bind(self, prim, material):
        from pxr import UsdShade
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    def _set_exposure(self, ev):
        """Shift the tonemapper's exposure by ev stops (via the f-number)."""
        import carb
        s = carb.settings.get_settings()
        f = s.get("/rtx/post/tonemap/fNumber") or 5.0
        s.set("/rtx/post/tonemap/fNumber", float(f) * (2.0 ** (-ev / 2.0)))

    def _remove_table(self):
        """No worktop: the cell's table and mat go (colliders too) and the arm
        stands on a steel pedestal of the table's height, so its base frame
        (and every coordinate in it) is unchanged; z = 0 is now the pedestal
        top, the floor is at -TABLE_TOP_Z."""
        for path in ("/World/Table", "/World/Mat", "/World/MatPrint"):
            p = self.stage.GetPrimAtPath(path)
            if p.IsValid():
                p.SetActive(False)
        steel = self._plain_mat("pedestal_steel", (0.30, 0.32, 0.35), 0.5, 0.45)
        h = C.TABLE_TOP_Z
        self._box(f"{ROOT}/Pedestal/Column", (0.24, 0.24, h - 0.03), [0.0, 0.0, -h + (h - 0.03) / 2], steel)
        self._box(f"{ROOT}/Pedestal/Top", (0.34, 0.34, 0.03), [0.0, 0.0, -0.015], steel)
        self._box(f"{ROOT}/Pedestal/Foot", (0.50, 0.50, 0.02), [0.0, 0.0, -h + 0.01], steel)

    def _plain_mat(self, key, rgb, metal=0.0, rough=0.5):
        if key not in self.cell.MATS:
            from isaacsim.core.api.materials import OmniPBR
            pbr = OmniPBR(prim_path=f"/World/Looks/{key}", name=key, color=np.array(rgb))
            pbr.set_metallic_constant(metal)
            pbr.set_reflection_roughness(rough)
            self.cell.MATS[key] = pbr.material
        return key

    def _white_top(self):
        from pxr import UsdGeom
        from isaacsim.core.api.materials import OmniPBR
        art = self.stage.GetPrimAtPath("/World/MatPrint")
        if art.IsValid():
            UsdGeom.Imageable(art).MakeInvisible()
        if not self.table:
            return
        rgb, rough = self.top_finish
        if os.environ.get("SORT_TOP"):
            v = [float(x) for x in os.environ["SORT_TOP"].split(",")]
            rgb, rough = tuple(v[:3]), (v[3] if len(v) > 3 else rough)
        white = OmniPBR(prim_path="/World/Looks/sort_top_white", name="sort_top_white",
                        color=np.array(rgb))
        white.set_metallic_constant(0.0)
        white.set_reflection_roughness(rough)
        for path in ("/World/Mat", "/World/Table"):
            prim = self.stage.GetPrimAtPath(path)
            if prim.IsValid():
                self._bind(prim, white.material)

    def _add_background(self, name):
        bg = LB.BACKGROUNDS[name]
        if bg is None:
            return
        from pxr import UsdGeom
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path
        add_reference_to_stage(usd_path=get_assets_root_path() + bg["usd"],
                               prim_path=LB.ENV_PRIM)
        self.cell.place(self.stage.GetPrimAtPath(LB.ENV_PRIM), bg["offset"], bg["yaw"])
        LB.tidy_background(self.stage, bg)
        if self.table:
            LB.IsaacBackend._add_table_frame(self)
        UsdGeom.Imageable(self.stage.GetPrimAtPath("/World/Floor")).MakeInvisible()
        dome = self.stage.GetPrimAtPath("/World/DomeLight")
        if dome.IsValid():
            dome.GetAttribute("inputs:intensity").Set(250.0)

    def _box(self, path, size, center, mat, collision=True):
        return self.cell.box(self.stage, path, size, self._w(center), (0.5, 0.5, 0.5),
                             collision=collision, mat=mat)

    def _build_tray(self):
        c, (ix, iy), w = self.lay.tray, S.TRAY_IN, S.TRAY_WALL
        z0 = S.SURFACE_Z
        P = f"{ROOT}/Tray"
        self._box(f"{P}/Floor", (ix + 2 * w, iy + 2 * w, S.TRAY_FLOOR),
                  c + [0, 0, z0 + S.TRAY_FLOOR / 2], "tray_grey")
        for k, (dx, dy, sx, sy) in enumerate((((ix + w) / 2, 0, w, iy + 2 * w),
                                               (-(ix + w) / 2, 0, w, iy + 2 * w),
                                               (0, (iy + w) / 2, ix, w),
                                               (0, -(iy + w) / 2, ix, w))):
            self._box(f"{P}/Wall{k}", (sx, sy, S.TRAY_H), c + [dx, dy, z0 + S.TRAY_H / 2],
                      "tray_grey")

    def _build_bin(self, name, c):
        (ix, iy), w, h = S.BIN_IN, S.BIN_WALL, S.BIN_H
        z0 = S.SURFACE_Z
        P = f"{ROOT}/Bin_{_bin_slot(name, self.lay)}"     # no class names in prim paths
        self._box(f"{P}/Floor", (ix + 2 * w, iy + 2 * w, S.BIN_FLOOR),
                  c + [0, 0, z0 + S.BIN_FLOOR / 2], "bin_grey")
        for k, (dx, dy, sx, sy) in enumerate((((ix + w) / 2, 0, w, iy + 2 * w),
                                               (-(ix + w) / 2, 0, w, iy + 2 * w),
                                               (0, (iy + w) / 2, ix, w),
                                               (0, -(iy + w) / 2, ix, w))):
            self._box(f"{P}/Wall{k}", (sx, sy, h), c + [dx, dy, z0 + h / 2], "bin_grey")
        # sign board on the -x wall's top edge, leaning back (toward -x) and
        # facing +x and up: readable from the top camera and from the front
        sw, sh = S.SIGN
        tilt = math.radians(S.SIGN_TILT_DEG)
        base = c + [-(ix / 2 + w), 0.0, z0 + h]
        mid = base + [-math.sin(tilt) * sh / 2, 0.0, math.cos(tilt) * sh / 2]
        n = np.array([math.cos(tilt), 0.0, math.sin(tilt)])       # facing +x, up
        up = np.array([-math.sin(tilt), 0.0, math.cos(tilt)])
        self._quad(f"{P}/Sign", mid, np.array([0.0, 1.0, 0.0]), up, (sw, sh),
                   # matte paper: a glossy print glared white under the hall
                   # lights where a bin stood near y = 0 (BATTERY on the deploy seed)
                   self._tex_mat(f"sign_{name}", rough=0.9), world=True)
        # the board behind the print
        self._quad(f"{P}/SignBack", mid - n * 0.0015, np.array([0.0, -1.0, 0.0]), up,
                   (sw + 0.006, sh + 0.006), self.cell.MATS["bin_grey"], world=True)
        self._quad(f"{P}/SignFront", mid - n * 0.0005, np.array([0.0, 1.0, 0.0]), up,
                   (sw + 0.006, sh + 0.006), self.cell.MATS["bin_grey"], world=True)

    def _quad(self, path, center, u, v, size, material, world=False):
        """A flat quad, image +x along u, image up along v (unit vectors),
        UVs 0..1. world=True: center in arm A's frame, placed in the world."""
        from pxr import UsdGeom, Gf, Sdf
        w, h = size
        c = np.asarray(self._w(center) if world else center, float)
        u, v = np.asarray(u, float), np.asarray(v, float)
        corners = [c - u * w / 2 - v * h / 2, c + u * w / 2 - v * h / 2,
                   c + u * w / 2 + v * h / 2, c - u * w / 2 + v * h / 2]
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in corners])
        m.CreateFaceVertexCountsAttr([4])
        m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        n = np.cross(u, v)
        m.CreateNormalsAttr([Gf.Vec3f(*map(float, n))] * 4)
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        st = UsdGeom.PrimvarsAPI(m).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
        self._bind(m.GetPrim(), material)

    # ---- items -------------------------------------------------------------------
    def _side(self, path, x0, x1, r, material, inward=False, seg=48):
        """Cylinder side along local x from x0 to x1, UV u round, v along +x."""
        from pxr import UsdGeom, Gf, Sdf
        pts, nrm, uv, idx = [], [], [], []
        s = -1.0 if inward else 1.0
        for k in range(seg + 1):
            a = 2 * math.pi * k / seg
            ca, sa = math.cos(a), math.sin(a)
            for x, v in ((x0, 0.0), (x1, 1.0)):
                pts.append(Gf.Vec3f(x, r * ca, r * sa))
                nrm.append(Gf.Vec3f(0.0, s * ca, s * sa))
                uv.append(Gf.Vec2f(k / seg, v))
        for k in range(seg):
            a, b = 2 * k, 2 * (k + 1)
            idx += ([a, b, b + 1, a + 1] if not inward else [a, a + 1, b + 1, b])
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr(pts)
        m.CreateFaceVertexCountsAttr([4] * seg)
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreateNormalsAttr(nrm)
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        m.CreateSubdivisionSchemeAttr("none")
        st = UsdGeom.PrimvarsAPI(m).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(uv)
        self._bind(m.GetPrim(), material)

    def _disc(self, path, x, r0, r1, sign, material, seg=48):
        """Annulus r0..r1 (r0 = 0: a disc) in the plane x, facing sign * +x."""
        from pxr import UsdGeom, Gf
        pts, idx = [], []
        for k in range(seg):
            a = 2 * math.pi * k / seg
            pts += [Gf.Vec3f(x, r0 * math.cos(a), r0 * math.sin(a)),
                    Gf.Vec3f(x, r1 * math.cos(a), r1 * math.sin(a))]
        for k in range(seg):
            a, b = 2 * k, 2 * ((k + 1) % seg)
            idx += [a, a + 1, b + 1, b] if sign > 0 else [a, b, b + 1, a + 1]
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr(pts)
        m.CreateFaceVertexCountsAttr([4] * seg)
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreateNormalsAttr([Gf.Vec3f(float(sign), 0.0, 0.0)] * len(pts))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        m.CreateSubdivisionSchemeAttr("none")
        self._bind(m.GetPrim(), material)

    def _build_item(self, i, it: S.Item, sticker=False):
        """A dynamic rigid body, axis along its local +x (a battery's + end at
        +x). Collider: a plain cylinder; visuals: the details."""
        from pxr import UsdGeom, UsdPhysics, Gf, PhysxSchema
        P = f"{ROOT}/Item{i}"                  # no class in the name
        x = UsdGeom.Xform.Define(self.stage, P)
        # half a millimetre up: no initial overlap to push it rolling
        x.AddTranslateOp().Set(Gf.Vec3d(*self._w(it.center + [0.0, 0.0, 0.0005])))
        yaw = it.yaw + (math.pi if it.flip else 0.0)
        x.AddOrientOp().Set(Gf.Quatf(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
        r, length = it.radius, it.length
        # collider: a 20-sided prism (convex hull), a face down. PhysX's
        # implicit cylinder creeps -- resting items rolled off by themselves
        n_side = COLLIDER_SIDES
        pts = []
        for k in range(n_side):
            a = 2 * math.pi * (k + 0.5) / n_side - math.pi / 2     # a face at the bottom
            for xx in (-length / 2, length / 2):
                pts.append(Gf.Vec3f(xx, r * math.cos(a), r * math.sin(a)))
        col = UsdGeom.Mesh.Define(self.stage, f"{P}/Collider")
        col.CreatePointsAttr(pts)
        col.CreateFaceVertexCountsAttr([4] * n_side)
        col.CreateFaceVertexIndicesAttr(
            [i for k in range(n_side) for i in (2 * k, 2 * ((k + 1) % n_side),
                                                2 * ((k + 1) % n_side) + 1, 2 * k + 1)])
        UsdGeom.Imageable(col.GetPrim()).CreatePurposeAttr("guide")   # not rendered
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr().Set("convexHull")
        self.cell.bind_physics_material(col.GetPrim())
        rb = UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        rb.CreateRigidBodyEnabledAttr().Set(True)
        UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(S.ITEM_MASS[it.kind])
        px = PhysxSchema.PhysxRigidBodyAPI.Apply(x.GetPrim())
        px.CreateAngularDampingAttr().Set(3.0)        # rolling resistance: a lying
                                                      # cylinder stays put unless pushed
        px.CreateLinearDampingAttr().Set(0.05)
        px.CreateSolverPositionIterationCountAttr().Set(16)
        px.CreateSolverVelocityIterationCountAttr().Set(4)
        PhysxSchema.PhysxRigidBodyAPI(x.GetPrim()).CreateEnableCCDAttr().Set(True)

        V = f"{P}/Visual"
        UsdGeom.Xform.Define(self.stage, V)
        h = length / 2
        steel = self.cell.MATS["cell_steel"]
        if it.kind == "battery":
            xb = h - S.BUTTON_H                     # + end face of the can
            wrap = self._tex_mat(f"wrap_{it.colour}{'_down' if it.print_down else ''}",
                                 rough=0.25)
            self._side(f"{V}/Wrap", -h, xb, r, wrap)
            self._disc(f"{V}/MinusEnd", -h, 0.0, r, -1, steel)
            self._disc(f"{V}/PlusEnd", xb, 0.0, r, +1, steel)
            self._disc(f"{V}/Ring", xb + 0.0002, S.RING_R0, S.RING_R1, +1,
                       self.cell.MATS["ring_black"])
            self._side(f"{V}/ButtonSide", xb, h, S.BUTTON_R, steel)
            self._disc(f"{V}/ButtonTop", h, 0.0, S.BUTTON_R, +1, steel)
            if sticker:
                L2 = STICKER_L / 2
                xc = (xb - h) / 2                    # centred on the can
                self._side(f"{V}/Sticker", xc - L2, xc + L2, STICKER_R,
                           self._tex_mat("sticker_metal", rough=0.35))
        elif it.kind == "steel_pin":
            mat = self.cell.MATS[self._mat(it.colour, it.finish)]
            ch = 0.0005                              # small chamfer, flat ends
            self._side(f"{V}/Body", -h + ch, h - ch, r, mat)
            self._side(f"{V}/ChamferN", -h, -h + ch, r - ch, mat)
            self._side(f"{V}/ChamferP", h - ch, h, r - ch, mat)
            self._disc(f"{V}/EndN", -h, 0.0, r - ch, -1, mat)
            self._disc(f"{V}/EndP", h, 0.0, r - ch, +1, mat)
        else:                                        # plastic sleeve: a tube
            mat = self.cell.MATS[self._mat(it.colour, it.finish)]
            self._side(f"{V}/Outer", -h, h, r, mat)
            self._side(f"{V}/Bore", -h, h, S.SLEEVE_BORE, mat, inward=True)
            self._disc(f"{V}/EndN", -h, S.SLEEVE_BORE, r, -1, mat)
            self._disc(f"{V}/EndP", h, S.SLEEVE_BORE, r, +1, mat)
        return P

    # ---- runtime -------------------------------------------------------------------
    def finalize(self):
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.sensors.camera import Camera
        paths = {"top": (TOP_CAM_PATH, TOP_CAM), "overview": (OVERVIEW_CAM_PATH, OVERVIEW_CAM),
                 "wrist": (self.cell.WRIST_CAM_PATH, C.WRIST_CAM)}
        self.cams = {v: Camera(prim_path=paths[v][0], resolution=paths[v][1]["resolution"])
                     for v in self.views}
        self.cell.enable_ccd(self.world)
        self.world.reset()
        for v, c in self.cams.items():
            c.initialize()
            if v == "top":
                c.add_distance_to_image_plane_to_frame()
        art = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia_A")
        art.initialize()
        names = list(art.dof_names)
        self.art = art
        self.arm_idx = [names.index(f"J{i}") for i in range(1, 7)]
        self.idx = {n: names.index(n) for n in names}
        q = art.get_joint_positions()
        q[self.arm_idx] = LB.HOME_Q
        art.set_joint_positions(q)
        self.items = []
        for k, p in enumerate(self.item_paths):
            rp = SingleRigidPrim(prim_path=p, name=f"item{k}")
            rp.initialize()
            self.items.append(rp)
        self._after_reset()
        # settle the items, compile the materials on the last frames
        n = int(self.settle_s / PHYSICS_DT)
        for k in range(n):
            self.command(LB.HOME_Q, 0.0)
            self.world.step(render=k >= n - 30)

    def _after_reset(self):
        pass

    def command(self, q6, grip):
        from isaacsim.core.utils.types import ArticulationAction
        full = self.art.get_joint_positions().copy()
        full[self.arm_idx] = q6
        jaw = C.JAW_OPEN + (C.JAW_CLOSED - C.JAW_OPEN) * float(grip)
        for jn, v in self.cell.jaw_targets(jaw).items():
            full[self.idx[jn]] = v
        self.art.apply_action(ArticulationAction(joint_positions=full))

    def step(self, q6, grip, n, cb=None):
        self.command(q6, grip)
        for _ in range(n):
            self.world.step(render=False)
            self.t += self.dt
            if cb is not None:
                cb(self.t)

    def joint_positions(self):
        return np.asarray(self.art.get_joint_positions()[self.arm_idx], float)

    def jaw_gap(self):
        """Distance between the jaws' inner faces, m."""
        jp = self.art.get_joint_positions()
        return float(2 * C.JAW_OPEN - abs(jp[self.idx["jaw_l"]]) - abs(jp[self.idx["jaw_r"]]))

    def item_poses(self):
        """[(position in arm A's frame, quaternion wxyz)] -- for the graders."""
        out = []
        for rp in self.items:
            p, q = rp.get_world_pose()
            p = np.asarray(p, float) - [0.0, 0.0, C.TABLE_TOP_Z]
            out.append((p, np.asarray(q, float)))
        return out

    def render(self, views=None, depth=False, frames=1):
        """Images of the current state. world.render() / app.update() leave
        the camera annotators one request behind (stale images); the
        replicator orchestrator renders and waits for this frame's data."""
        import omni.replicator.core as rep
        views = self.views if views is None else views
        rep.orchestrator.step(rt_subframes=max(1, int(frames)), pause_timeline=False)
        out = {}
        for v in views:
            c = self.cams.get(v)
            if c is None:
                continue
            rgba = c.get_rgba()
            if rgba is not None and rgba.size:
                out[v] = np.asarray(rgba)[:, :, :3].astype(np.uint8)
            if depth and v == "top":
                d = c.get_depth()
                if d is not None:
                    out["top_depth"] = np.asarray(d, np.float32)
        return out

    def camera_model(self, view):
        """K (3x3) and T (4x4, camera -> arm A's frame; camera axes x right,
        y down, z forward) of a camera, for back-projecting its pixels."""
        c = self.cams[view]
        K = np.asarray(c.get_intrinsics_matrix(), float)
        p, q = c.get_world_pose(camera_axes="ros")
        w, x, y, z = (float(v) for v in q)
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(p, float) - [0.0, 0.0, C.TABLE_TOP_Z]
        return K, T

    def close(self):
        self.app.close()


def _bin_slot(name, lay):
    """Bins are named by position order in the prim tree, not by class."""
    order = sorted(lay.bins, key=lambda n: tuple(lay.bins[n][:2]))
    return order.index(name)
