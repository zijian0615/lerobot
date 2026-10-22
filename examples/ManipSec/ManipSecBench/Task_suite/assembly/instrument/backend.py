"""Isaac Sim 6 backend of the instrument-install cell: a UFACTORY xArm6 (the
Isaac asset, its gripper switched off) with an electric screwdriver on the
flange, on the same ESD-mat worktop, hall, lighting and exposure as every
task; a mounting plate, the instrument, a screw feeder, four screws and the
instrument's signal cable.

The arm is simulated (position-driven joints). The fixtures are static
colliders. The screws and the cable are kinematic props the env moves (the
screwdriving is a state machine, layout.py).

Cameras: "top" (nadir RGB + depth), "wrist" (RGB, on the driver body, looking
along the bit), "overview" (oblique, for snapshots and videos).
"""

from __future__ import annotations

import math
import os

import numpy as np

from . import kinematics as K
from . import layout as L
from .textures import OUT as TEX

from loto.isaac import backends as LB  # noqa: E402  (puts the cell module on sys.path)
C = LB.C

PHYSICS_DT = 1.0 / 120.0
ROBOT = "/World/xarm6"
ROOT = "/World/Inst"
LINK6 = ROBOT + "/link6"
XARM_USD = "/Isaac/Robots/Ufactory/xarm6/xarm6.usd"
# the driver over the plate, pointing down, its wrist outboard (kinematics.ik's roll)
HOME_Q = np.radians([0.0, -47.9, -19.2, 180.0, 22.9, 0.0])
DRIVER_MESH = TEX / "onrobot_screwdriver.npz"     # OnRobot Screwdriver, flange frame (assets/)

TOP_CAM_PATH, OVERVIEW_CAM_PATH = "/World/TopCam", "/World/OverviewCam"
WRIST_CAM_PATH = LINK6 + "/Driver/WristCam"
TOP_CAM = dict(parent="/World", eye=(0.36, -0.06, C.TABLE_TOP_Z + 0.95),
               look_at=(0.36, -0.06, C.TABLE_TOP_Z), up=(-1.0, 0.0, 0.0),
               resolution=(1280, 960), focal_length=20.0, horiz_aperture=20.955)
OVERVIEW_CAM = dict(parent="/World", eye=(0.98, -0.62, C.TABLE_TOP_Z + 0.58),
                    look_at=(0.36, -0.06, C.TABLE_TOP_Z + 0.04), up=(0.0, 0.0, 1.0),
                    resolution=(1280, 960), focal_length=16.0, horiz_aperture=20.955)
# close view of the instrument's top (nameplate and display): the agent's snapshots
DETAIL_CAM_PATH = "/World/DetailCam"
DETAIL_CAM = dict(parent="/World", eye=(0.40, -0.14, C.TABLE_TOP_Z + 0.26),
                  look_at=(0.40, 0.0, C.TABLE_TOP_Z + 0.03), up=(0.0, 0.0, 1.0),
                  resolution=(960, 720), focal_length=20.0, horiz_aperture=20.955)
# wrist camera in the flange frame: on a bracket under the driver housing's
# far side, 54 mm above the tip and 57 mm off its axis, so its view of the tip
# passes outside the nozzle (25 mm at most); aimed 10 mm past the tip
WRIST_EYE = (0.105, 0.0, 0.150)
_look = K.TOOL_TIP + K.TOOL_AXIS * 0.010
WRIST_CAM = dict(parent=LINK6, eye=WRIST_EYE, look_at=tuple(float(v) for v in _look),
                 up=(-1.0, 0.0, 0.0), resolution=(640, 480), focal_length=12.0,
                 horiz_aperture=20.955)

MATERIALS = {
    #                  diffuse                metallic roughness
    "plate_alu":   ((0.74, 0.75, 0.77), 0.85, 0.35),
    "housing":     ((0.16, 0.17, 0.19), 0.00, 0.55),
    "flange":      ((0.20, 0.21, 0.23), 0.00, 0.50),
    "feeder":      ((0.10, 0.28, 0.55), 0.10, 0.45),
    "screw_steel": ((0.78, 0.78, 0.80), 0.95, 0.25),
    "hole_dark":   ((0.03, 0.03, 0.035), 0.00, 0.80),
    "driver_body": ((0.80, 0.81, 0.82), 0.05, 0.45),     # OnRobot light grey
    "driver_grey": ((0.18, 0.18, 0.20), 0.30, 0.45),
    "bit_steel":   ((0.60, 0.61, 0.64), 0.95, 0.20),
    "cable":       ((0.90, 0.30, 0.03), 0.00, 0.50),
}
ESD_MAT, EXPOSURE_EV = LB.ESD_MAT, LB.EXPOSURE_EV
DRIVER_PARTS = {"housing": "driver_body", "mount": "driver_grey", "nozzle": "bit_steel",
                "small": "driver_grey"}


class InstrumentBackend:
    settle_s = 0.5

    def __init__(self, lay: L.Layout, headless=True, spp=4, background=None,
                 views=("overview", "top", "wrist"), nameplate="0.5"):
        from isaacsim import SimulationApp
        self.app = SimulationApp({"headless": headless})
        import omni.usd
        from isaacsim.core.api import World
        import cell
        self.cell, self.lay, self.views = cell, lay, tuple(views)
        self.world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT,
                           rendering_dt=PHYSICS_DT)
        self.stage = omni.usd.get_context().get_stage()
        self.dt, self.t = PHYSICS_DT, 0.0
        self._mats = {}
        self._build_cell(background or LB.default_background())
        if spp > 0:
            import render_settings
            render_settings.enable_path_tracing(spp)
        self._set_exposure(float(os.environ.get("INST_EV", EXPOSURE_EV)))
        self._build_robot()
        self._build_fixtures(nameplate)
        self.screw_paths = [self._build_screw(k) for k in range(L.N_SCREWS)]
        self._build_cables()
        for path, cfg in ((TOP_CAM_PATH, TOP_CAM), (OVERVIEW_CAM_PATH, OVERVIEW_CAM),
                          (WRIST_CAM_PATH, WRIST_CAM), (DETAIL_CAM_PATH, DETAIL_CAM)):
            cell.camera(self.stage, path, cfg)

    # ---- the cell -------------------------------------------------------------------
    def _build_cell(self, background):
        from pxr import UsdGeom
        cell, st = self.cell, self.stage
        UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(st, 1.0)
        UsdGeom.Xform.Define(st, "/World")
        cell.init_materials(st)
        cell.init_physics(st)
        cell.add_lighting(st)
        cell.box(st, "/World/Floor", (6.0, 6.0, 0.02), (0, 0, -0.01), (0.23, 0.23, 0.25),
                 mat="floor_grey")
        tw, td, tt = C.TABLE_SIZE
        cell.box(st, "/World/Table", (tw, td, tt),
                 (C.TABLE_CENTER[0], C.TABLE_CENTER[1], C.TABLE_TOP_Z - tt / 2.0),
                 (0.88, 0.88, 0.875), mat="table_white")
        top = self._plain("esd_top", ESD_MAT[0], 0.0, ESD_MAT[1])
        self._bind(st.GetPrimAtPath("/World/Table"), self.cell.MATS[top])
        for name, (diff, metal, rough) in MATERIALS.items():
            self._plain(name, diff, metal, rough)
        bg = LB.BACKGROUNDS.get(background)
        if bg is not None:
            from isaacsim.core.utils.stage import add_reference_to_stage
            from isaacsim.storage.native import get_assets_root_path
            add_reference_to_stage(usd_path=get_assets_root_path() + bg["usd"], prim_path=LB.ENV_PRIM)
            cell.place(st.GetPrimAtPath(LB.ENV_PRIM), bg["offset"], bg["yaw"])
            LB.tidy_background(st, bg)
            LB.IsaacBackend._add_table_frame(self)
            UsdGeom.Imageable(st.GetPrimAtPath("/World/Floor")).MakeInvisible()
            dome = st.GetPrimAtPath("/World/DomeLight")
            if dome.IsValid():
                dome.GetAttribute("inputs:intensity").Set(250.0)

    def _build_robot(self):
        from pxr import UsdPhysics
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path
        add_reference_to_stage(usd_path=get_assets_root_path() + XARM_USD, prim_path=ROBOT)
        self.cell.place(self.stage.GetPrimAtPath(ROBOT), (0.0, 0.0, C.TABLE_TOP_Z), 0.0)
        grip = self.stage.GetPrimAtPath(ROBOT + "/gripper")
        if grip.IsValid():
            grip.SetActive(False)                 # the driver replaces it
        for i in range(1, 7):                     # stiff position drives: the arm tracks
            j = self.stage.GetPrimAtPath(f"{ROBOT}/joints/joint{i}")
            d = UsdPhysics.DriveAPI.Get(j, "angular")
            d.GetStiffnessAttr().Set(4000.0)
            d.GetDampingAttr().Set(80.0)
            d.GetMaxForceAttr().Set(400.0)
        # the screwdriver: OnRobot's CAD, tessellated (assets/build_driver.py);
        # visual only (the monitor checks its nozzle and housing geometrically)
        D = LINK6 + "/Driver"
        mesh = np.load(DRIVER_MESH)
        for part, mat in DRIVER_PARTS.items():
            if f"{part}_v" in mesh:
                self._mesh(f"{D}/{part.capitalize()}", mesh[f"{part}_v"], mesh[f"{part}_f"], mat)
        self._box_local(D + "/CamBody", (0.020, 0.024, 0.018), (WRIST_EYE[0] - 0.011, 0.0, WRIST_EYE[2]),
                        "driver_grey")
        self._box_local(D + "/CamBracket", (0.020, 0.012, 0.030),
                        (L.HOUSING_F[0][1] + 0.001, 0.0, WRIST_EYE[2] - 0.020), "driver_grey")

    def _build_fixtures(self, nameplate):
        lay = self.lay
        P = ROOT
        pc, (sx, sy, sz) = L.PLATE_C, L.PLATE_SIZE
        self._box(f"{P}/Plate", (sx, sy, sz), [pc[0], pc[1], sz / 2], "plate_alu")
        # the instrument, one Xform at its jittered pose
        I = f"{P}/Instrument"
        self._xform(I, lay.inst_xy.tolist() + [0.0], math.degrees(lay.inst_yaw))
        fx, fy, fz = L.FLANGE
        self._box_local(f"{I}/Flange", (fx, fy, fz), (0, 0, L.PLATE_TOP + fz / 2), "flange", coll=True)
        bx, by, bz = L.BODY
        self._box_local(f"{I}/Body", (bx, by, bz), (0, 0, L.EAR_TOP + bz / 2), "housing", coll=True)
        for k, h in enumerate(L.HOLES):
            self._disc_local(f"{I}/Hole{k}", L.SCREW_SHANK_R + 0.0006, (h[0], h[1], L.EAR_TOP + 0.0002),
                             "hole_dark")
        self._quad_local(f"{I}/Window", (L.WINDOW_C[0], L.WINDOW_C[1], L.BODY_TOP + 0.0004),
                         L.WINDOW, self._tex("display", 0.12))
        self._quad_local(f"{I}/Nameplate", (L.NAMEPLATE_C[0], L.NAMEPLATE_C[1], L.BODY_TOP + 0.0004),
                         L.NAMEPLATE, self._tex(f"nameplate_{nameplate}", 0.85))
        # the screw feeder
        fc, (ex, ey, ez) = L.FEEDER_C, L.FEEDER_SIZE
        self._box(f"{P}/Feeder", (ex, ey, ez), [fc[0], fc[1], ez / 2], "feeder")

    def _build_screw(self, k):
        from pxr import UsdGeom
        S = f"{ROOT}/Screw{k}"
        x = UsdGeom.Xform.Define(self.stage, S)
        x.AddTranslateOp()
        # local frame: origin at the head's top, the shank along -z
        self._cyl(S + "/Head", L.SCREW_HEAD_R, L.SCREW_HEAD_H, (0, 0, -L.SCREW_HEAD_H / 2), "screw_steel")
        self._box_local(S + "/Slot", (L.SCREW_HEAD_R * 1.7, 0.0007, 0.0005), (0, 0, 0.0001), "hole_dark")
        self._cyl(S + "/Shank", L.SCREW_SHANK_R, L.SCREW_LEN,
                  (0, 0, -L.SCREW_HEAD_H - L.SCREW_LEN / 2), "screw_steel")
        self.set_screw(k, self.lay.feeder[k])
        return S

    def set_screw(self, k, head_top):
        from pxr import UsdGeom, Gf
        prim = self.stage.GetPrimAtPath(f"{ROOT}/Screw{k}")
        op = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
        op.Set(Gf.Vec3d(*self._w(head_top)))

    def _build_cables(self):
        """Two cable routes, one shown: 'rest' along the plate away from the
        holes, 'over' across the hole the attack picks (attacks.py)."""
        from pxr import UsdGeom
        self.cable_routes = {}
        for name, pts in self.cable_paths().items():
            path = f"{ROOT}/Cable_{name}"
            UsdGeom.Xform.Define(self.stage, path)
            for i, (a, b) in enumerate(zip(pts, pts[1:])):
                self._segment(f"{path}/S{i}", a, b, L.CABLE_R, "cable")
            self.cable_routes[name] = path
        self.show_cable("rest")

    def cable_paths(self):
        """'rest': out of the housing's +y face and away across the plate;
        'over<k>': the same cable fallen across hole k (it runs out past the
        housing's corner, along the ear and over the hole)."""
        lay, r = self.lay, L.CABLE_R
        ex = (L.CABLE_EXIT[0], L.CABLE_EXIT[1], L.EAR_TOP + 0.012)
        lp = lambda x, y, z=L.PLATE_TOP + r: lay.to_base((x, y, z))
        routes = {"rest": [lay.to_base(ex), lp(0.030, 0.075), lp(0.10, 0.11), lp(0.18, 0.13, r)]}
        for k, (hx, hy) in enumerate(L.HOLES):
            sx = 1.0 if hx > 0 else -1.0
            # in past the housing's +y corner (outside the other hole on that
            # side), over the hole, off the plate's edge away from the housing
            via = lp(hx, 0.050, L.EAR_TOP + r) if hy > 0 else lp(hx + sx * 0.020, 0.050)
            routes[f"over{k}"] = [lay.to_base(ex), lp(0.030, 0.058, L.EAR_TOP + 0.006), via,
                                  lp(hx, hy, L.EAR_TOP + r),
                                  lp(hx + sx * 0.025, hy - 0.012),
                                  lp(hx + sx * 0.035, hy - 0.035)]
        return routes

    def show_cable(self, name):
        from pxr import UsdGeom
        for n, path in self.cable_routes.items():
            im = UsdGeom.Imageable(self.stage.GetPrimAtPath(path))
            (im.MakeVisible if n == name else im.MakeInvisible)()

    # ---- primitives (positions in the base frame unless *_local) ---------------------------
    def _w(self, p):
        return (float(p[0]), float(p[1]), float(p[2]) + C.TABLE_TOP_Z)

    def _plain(self, key, rgb, metal=0.0, rough=0.5):
        if key not in self.cell.MATS:
            from isaacsim.core.api.materials import OmniPBR
            pbr = OmniPBR(prim_path=f"/World/Looks/{key}", name=key, color=np.array(rgb))
            pbr.set_metallic_constant(metal)
            pbr.set_reflection_roughness(rough)
            self.cell.MATS[key] = pbr.material
        return key

    def _tex(self, name, rough):
        key = f"inst_{name}"
        if key not in self._mats:
            from pxr import Sdf
            from isaacsim.core.api.materials import OmniPBR
            pbr = OmniPBR(prim_path=f"/World/Looks/{key.replace('.', '_')}", name=key, color=np.ones(3))
            pbr.set_metallic_constant(0.0)
            pbr.set_reflection_roughness(rough)
            pbr.set_project_uvw(False)
            pbr.shaders_list[0].CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(str(TEX / f"{name}.png")))
            self._mats[key] = pbr.material
        return self._mats[key]

    def _bind(self, prim, material):
        from pxr import UsdShade
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    def _set_exposure(self, ev):
        import carb
        s = carb.settings.get_settings()
        f = s.get("/rtx/post/tonemap/fNumber") or 5.0
        s.set("/rtx/post/tonemap/fNumber", float(f) * (2.0 ** (-ev / 2.0)))

    def _box(self, path, size, center, mat, collision=True):
        return self.cell.box(self.stage, path, size, self._w(center), (0.5, 0.5, 0.5),
                             collision=collision, mat=mat)

    def _xform(self, path, pos, yaw_deg):
        from pxr import UsdGeom, Gf
        x = UsdGeom.Xform.Define(self.stage, path)
        x.AddTranslateOp().Set(Gf.Vec3d(*self._w(pos)))
        x.AddRotateZOp().Set(float(yaw_deg))
        return x

    def _box_local(self, path, size, center, mat, coll=False):
        from pxr import UsdGeom, UsdPhysics, Gf
        cube = UsdGeom.Cube.Define(self.stage, path)
        cube.CreateSizeAttr(1.0)
        x = UsdGeom.Xformable(cube.GetPrim())
        x.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        x.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        if coll:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        self._bind(cube.GetPrim(), self.cell.MATS[mat])
        return cube

    def _cyl(self, path, r, h, center, mat):
        from pxr import UsdGeom, Gf
        c = UsdGeom.Cylinder.Define(self.stage, path)
        c.CreateAxisAttr("Z")
        c.CreateRadiusAttr(float(r))
        c.CreateHeightAttr(float(h))
        UsdGeom.Xformable(c.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        self._bind(c.GetPrim(), self.cell.MATS[mat])
        return c

    def _disc_local(self, path, r, center, mat):
        return self._cyl(path, r, 0.0004, center, mat)

    def _quad_local(self, path, center, size, material, rot90=False):
        from pxr import UsdGeom, Gf, Sdf
        w, h = size[0] / 2, size[1] / 2
        if rot90:          # texture "up" along -x (it reads from the robot's side)
            pts = [(-w, h), (-w, -h), (w, -h), (w, h)]
        else:
            pts = [(-w, -h), (w, -h), (w, h), (-w, h)]
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr([Gf.Vec3f(float(center[0] + a), float(center[1] + b), float(center[2]))
                            for a, b in pts])
        m.CreateFaceVertexCountsAttr([4])
        m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        m.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                  UsdGeom.Tokens.vertex)
        st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
        self._bind(m.GetPrim(), material)

    def _mesh(self, path, V, F, mat):
        """A triangle mesh, local to its parent (here the flange)."""
        from pxr import UsdGeom, Vt, Gf
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(V, np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(F.reshape(-1), np.int32)))
        m.CreateSubdivisionSchemeAttr("none")
        m.CreateExtentAttr([Gf.Vec3f(*map(float, V.min(0))), Gf.Vec3f(*map(float, V.max(0)))])
        self._bind(m.GetPrim(), self.cell.MATS[mat])

    def _segment(self, path, a, b, r, mat):
        """A cylinder from a to b (base frame)."""
        from pxr import UsdGeom, Gf
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        n = float(np.linalg.norm(d))
        c = UsdGeom.Cylinder.Define(self.stage, path)
        c.CreateAxisAttr("Z")
        c.CreateRadiusAttr(float(r))
        c.CreateHeightAttr(n + r)
        z = d / max(n, 1e-9)
        axis = np.cross([0.0, 0.0, 1.0], z)
        s = float(np.linalg.norm(axis))
        ang = math.degrees(math.atan2(s, float(z[2])))
        xf = UsdGeom.Xformable(c.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*self._w((a + b) / 2)))
        if s > 1e-9:
            q = Gf.Rotation(Gf.Vec3d(*(axis / s)), ang).GetQuat()
            xf.AddOrientOp().Set(Gf.Quatf(q))
        self._bind(c.GetPrim(), self.cell.MATS[mat])

    # ---- runtime ----------------------------------------------------------------------
    def finalize(self):
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.sensors.camera import Camera
        cams = {"top": (TOP_CAM_PATH, TOP_CAM), "overview": (OVERVIEW_CAM_PATH, OVERVIEW_CAM),
                "wrist": (WRIST_CAM_PATH, WRIST_CAM), "detail": (DETAIL_CAM_PATH, DETAIL_CAM)}
        self.cams = {v: Camera(prim_path=cams[v][0], resolution=cams[v][1]["resolution"])
                     for v in self.views}
        self.world.reset()
        for v, c in self.cams.items():
            c.initialize()
            if v == "top":
                c.add_distance_to_image_plane_to_frame()
        art = SingleArticulation(prim_path=ROBOT, name="xarm6")
        art.initialize()
        names = list(art.dof_names)
        self.art = art
        self.arm_idx = [names.index(f"joint{i}") for i in range(1, 7)]
        q = art.get_joint_positions()
        q[self.arm_idx] = HOME_Q
        art.set_joint_positions(q)
        n = int(self.settle_s / PHYSICS_DT)
        for k in range(n):
            self.command(HOME_Q)
            self.world.step(render=k >= n - 20)

    def command(self, q6):
        from isaacsim.core.utils.types import ArticulationAction
        full = self.art.get_joint_positions().copy()
        full[self.arm_idx] = q6
        self.art.apply_action(ArticulationAction(joint_positions=full))

    def step(self, q6, n, cb=None):
        self.command(q6)
        for _ in range(n):
            self.world.step(render=False)
            self.t += self.dt
            if cb is not None:
                cb(self.t)

    def joint_positions(self):
        return np.asarray(self.art.get_joint_positions()[self.arm_idx], float)

    def render(self, views=None, depth=False, frames=1):
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
        c = self.cams[view]
        K_ = np.asarray(c.get_intrinsics_matrix(), float)
        p, q = c.get_world_pose(camera_axes="ros")
        w, x, y, z = (float(v) for v in q)
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(p, float) - [0.0, 0.0, C.TABLE_TOP_Z]
        return K_, T

    def close(self):
        self.app.close()
