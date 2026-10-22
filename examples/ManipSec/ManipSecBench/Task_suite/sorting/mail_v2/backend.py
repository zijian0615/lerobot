"""Isaac Sim 6 scene of mail-sorting v2: the cell every task shares (table,
ESD worktop, hall, lighting, path tracing), the UFACTORY xArm6 with the lab
arm's finish (assembly/instrument/xarm_look.py) and a two-cup vacuum gripper,
the input tray (a tote), the two bins (letter trays with label cards) and the
envelopes -- kinematic, posed by mechanics.py through env.py.

The primitives, cameras and runtime calls are the instrument cell's
(assembly/instrument/backend.py); this class builds a different scene on them.
"""

from __future__ import annotations

import math
import os

import numpy as np

from assembly.instrument import xarm_look
from assembly.instrument.backend import (InstrumentBackend, LB, C, PHYSICS_DT, ROBOT, LINK6,
                                         XARM_USD, TOP_CAM_PATH, OVERVIEW_CAM_PATH, EXPOSURE_EV)
from . import layout as L
from .textures import OUT as TEX

ROOT = "/World/Mail"
HOME_Q = np.radians([0.0, -32.3, -46.7, 0.0, 79.0, -89.9])   # cups down over the tray's near edge
# the top camera covers the tray and both bins (y -0.45 .. 0.45); the overview
# is the approved preview's view
TOP_CAM = dict(parent="/World", eye=(0.33, 0.0, C.TABLE_TOP_Z + 1.00),
               look_at=(0.33, 0.0, C.TABLE_TOP_Z), up=(-1.0, 0.0, 0.0),
               resolution=(1920, 1440), focal_length=17.0, horiz_aperture=20.955)
OVERVIEW_CAM = dict(parent="/World", eye=(1.05, -0.78, C.TABLE_TOP_Z + 0.62),
                    look_at=(0.32, 0.0, C.TABLE_TOP_Z + 0.06), up=(0.0, 0.0, 1.0),
                    resolution=(1280, 800), focal_length=17.0, horiz_aperture=20.955)
# the showcase videos' view: steep and close over the tray and both bins, so
# the envelopes' markings and the cards read in the video (never given to programs)
SHOWCASE_CAM_PATH = "/World/ShowcaseCam"
SHOWCASE_CAM = dict(parent="/World", eye=(0.82, 0.0, C.TABLE_TOP_Z + 0.88),
                    look_at=(0.32, 0.0, C.TABLE_TOP_Z), up=(0.0, 0.0, 1.0),
                    resolution=(1600, 1000), focal_length=19.0, horiz_aperture=20.955)
WRIST_CAM_PATH = LINK6 + "/Vacuum/WristCam"
# wrist camera on a bracket beside the cup bar (flange +X), looking down past
# the bar's side and between the cups at what lies under them
WRIST_EYE = (0.075, 0.0, 0.100)
WRIST_CAM = dict(parent=LINK6, eye=WRIST_EYE, look_at=(0.0, 0.0, L.TOOL_LEN + 0.02),
                 up=(1.0, 0.0, 0.0), resolution=(960, 720), focal_length=10.0,
                 horiz_aperture=20.955)
MATERIALS = {
    "v_alu":    ((0.70, 0.71, 0.73), 0.85, 0.35),
    "v_dark":   ((0.14, 0.14, 0.16), 0.20, 0.50),
    "v_steel":  ((0.62, 0.63, 0.66), 0.95, 0.25),
    "v_rubber": ((0.10, 0.11, 0.13), 0.00, 0.75),
    "v_blue":   ((0.10, 0.32, 0.70), 0.10, 0.40),
    "v_hose":   ((0.15, 0.40, 0.85), 0.00, 0.45),
    "tote":     ((0.42, 0.45, 0.50), 0.00, 0.60),
    "bin_tray": ((0.86, 0.87, 0.88), 0.00, 0.55),
    "paper_edge": ((0.93, 0.92, 0.88), 0.00, 0.80),
}


class MailV2Backend(InstrumentBackend):
    settle_s = 0.5

    def __init__(self, episode: L.Episode, signs: dict, headless=True, spp=1, background=None,
                 views=("overview", "top", "wrist")):
        """signs: bin -> the word on its card (the scene attack swaps them)."""
        from isaacsim import SimulationApp
        self.app = SimulationApp({"headless": headless})
        import omni.usd
        from isaacsim.core.api import World
        import cell
        self.cell, self.views = cell, tuple(views)
        self.world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)
        self.stage = omni.usd.get_context().get_stage()
        self.dt, self.t = PHYSICS_DT, 0.0
        self._mats = {}
        self._build_cell(background or LB.default_background())
        for name, (diff, metal, rough) in MATERIALS.items():
            self._plain(name, diff, metal, rough)
        if spp > 0:
            import render_settings
            render_settings.enable_path_tracing(spp)
        self._set_exposure(float(os.environ.get("INST_EV", EXPOSURE_EV)))
        self._build_robot()
        self._build_fixtures(signs)
        self.env_paths = [self._build_envelope(k, e) for k, e in enumerate(episode.envelopes)]
        for path, cfg in ((TOP_CAM_PATH, TOP_CAM), (OVERVIEW_CAM_PATH, OVERVIEW_CAM),
                          (WRIST_CAM_PATH, WRIST_CAM), (SHOWCASE_CAM_PATH, SHOWCASE_CAM)):
            cell.camera(self.stage, path, cfg)

    def _tex(self, name, rough):
        key = f"mv2_{name}"
        if key not in self._mats:
            from pxr import Sdf
            from isaacsim.core.api.materials import OmniPBR
            pbr = OmniPBR(prim_path=f"/World/Looks/{key}", name=key, color=np.ones(3))
            pbr.set_metallic_constant(0.0)
            pbr.set_reflection_roughness(rough)
            pbr.set_project_uvw(False)
            pbr.shaders_list[0].CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(str(TEX / f"{name}.png")))
            self._mats[key] = pbr.material
        return self._mats[key]

    # ---- the robot -----------------------------------------------------------------
    def _build_robot(self):
        from pxr import UsdGeom, UsdPhysics
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path
        add_reference_to_stage(usd_path=get_assets_root_path() + XARM_USD, prim_path=ROBOT)
        self.cell.place(self.stage.GetPrimAtPath(ROBOT), (0.0, 0.0, C.TABLE_TOP_Z), 0.0)
        grip = self.stage.GetPrimAtPath(ROBOT + "/gripper")
        if grip.IsValid():
            grip.SetActive(False)                 # the vacuum gripper replaces it
        for i in range(1, 7):
            d = UsdPhysics.DriveAPI.Get(self.stage.GetPrimAtPath(f"{ROBOT}/joints/joint{i}"), "angular")
            d.GetStiffnessAttr().Set(4000.0)
            d.GetDampingAttr().Set(80.0)
            d.GetMaxForceAttr().Set(400.0)
        xarm_look.apply(self.stage, ROBOT)
        # the vacuum gripper along the flange +Z; the cups spread along the flange +Y
        V = LINK6 + "/Vacuum"
        UsdGeom.Xform.Define(self.stage, V)
        z = lambda a, b: (0.0, 0.0, (a + b) / 2)
        self._cyl(V + "/Adapter", 0.033, 0.010, z(0.0, 0.010), "v_alu")
        self._box_local(V + "/Ejector", (0.046, 0.052, 0.050), (0.0, 0.0, 0.035), "v_dark")
        self._box_local(V + "/EjectorPort", (0.016, 0.012, 0.016), (0.029, 0.0, 0.040), "v_blue")
        self._box_local(V + "/Switch", (0.012, 0.022, 0.012), (-0.029, 0.0, 0.028), "v_blue")
        self._cyl(V + "/Plunger", 0.007, 0.075, z(0.060, 0.135), "v_steel")
        self._cyl(V + "/Spring", 0.011, 0.040, z(0.070, 0.110), "v_steel")
        self._box_local(V + "/Bar", (0.020, L.CUP_SPAN + 0.030, 0.016), (0.0, 0.0, 0.142), "v_alu")
        for side, y in (("L", L.CUP_SPAN / 2), ("R", -L.CUP_SPAN / 2)):
            self._cyl(f"{V}/Stem{side}", 0.005, 0.010, (0.0, y, 0.155), "v_steel")
            for i in range(3):                     # bellows
                d0 = 0.160 + i * 0.012
                self._cyl(f"{V}/Bellow{side}{i}a", 0.013, 0.006, (0.0, y, d0 + 0.003), "v_rubber")
                self._cyl(f"{V}/Bellow{side}{i}b", 0.010, 0.006, (0.0, y, d0 + 0.009), "v_rubber")
            self._cyl(f"{V}/Lip{side}", L.CUP_R, 0.004, (0.0, y, L.TOOL_LEN - 0.002), "v_rubber")
            self._cyl(f"{V}/Hose{side}", 0.003, 0.090, (0.0, y * 0.7, 0.095), "v_hose")
        self._box_local(V + "/CamArm", (0.055, 0.012, 0.008), (0.050, 0.0, 0.088), "v_alu")
        self._box_local(V + "/CamBody", (0.018, 0.024, 0.020), (WRIST_EYE[0] + 0.004, 0.0, WRIST_EYE[2] - 0.012),
                        "v_dark")

    # ---- fixtures ------------------------------------------------------------------
    def _open_box(self, path, size, mat):
        """Floor and four walls, local to `path` (an Xform at the box's centre)."""
        sx, sy, h = size
        w = L.WALL
        self._box_local(f"{path}/Floor", (sx, sy, w), (0, 0, w / 2), mat, coll=False)
        self._box_local(f"{path}/WallXm", (w, sy, h), (-sx / 2 + w / 2, 0, h / 2), mat)
        self._box_local(f"{path}/WallXp", (w, sy, h), (sx / 2 - w / 2, 0, h / 2), mat)
        self._box_local(f"{path}/WallYm", (sx, w, h), (0, -sy / 2 + w / 2, h / 2), mat)
        self._box_local(f"{path}/WallYp", (sx, w, h), (0, sy / 2 - w / 2, h / 2), mat)

    def _build_fixtures(self, signs):
        from pxr import UsdGeom
        self._xform(f"{ROOT}/Tray", [L.TOTE_C[0], L.TOTE_C[1], 0.0], 0.0)
        self._open_box(f"{ROOT}/Tray", L.TOTE, "tote")
        self.bin_paths = {}
        for b in L.BINS:
            p = f"{ROOT}/Bin_{b}"                  # the prim is the physical bin
            xy = L.SLOTS[L.BUILT[b]]
            self._xform(p, [xy[0], xy[1], 0.0], 0.0)
            self._open_box(p, L.BIN, "bin_tray")
            h = UsdGeom.Xform.Define(self.stage, p + "/Holder")   # card holder on the far end
            h.AddTranslateOp().Set((L.BIN[0] / 2 + 0.006, 0.0, L.BIN[2] - 0.010))
            h.AddRotateYOp().Set(L.CARD_TILT_DEG)
            self._box_local(p + "/Holder/Plate", (0.042, 0.134, 0.003), (0, 0, -0.002), "bin_tray")
            cx = UsdGeom.Xform.Define(self.stage, p + "/Holder/Card")
            cx.AddRotateZOp().Set(90.0)
            self._quad_local(p + "/Holder/Card/Q", (0, 0, 0.0003), L.CARD,
                             self._tex(f"mailv2_card_{signs[b]}", 0.6))
            self.bin_paths[b] = p

    def set_bin(self, b, xy):
        from pxr import UsdGeom, Gf
        op = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.bin_paths[b])).GetOrderedXformOps()[0]
        op.Set(Gf.Vec3d(*self._w([xy[0], xy[1], 0.0])))

    def _build_envelope(self, k, e: L.Envelope):
        p = f"{ROOT}/Envelope{k}"
        self._xform(p, [e.xy[0], e.xy[1], e.z], math.degrees(e.yaw))   # origin on the face
        self._box_local(p + "/Body", L.ENV, (0, 0, -L.ENV[2] / 2), "paper_edge")
        self._quad_local(p + "/Face", (0, 0, 0.0002), L.ENV[:2],
                         self._tex(L.texture_name(e.label, e.paper, e.ink), 0.7))
        return p

    def set_envelope(self, k, xy, yaw, z):
        from pxr import UsdGeom, Gf
        ops = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.env_paths[k])).GetOrderedXformOps()
        ops[0].Set(Gf.Vec3d(*self._w([xy[0], xy[1], z])))
        ops[1].Set(float(math.degrees(yaw)))

    # ---- runtime ----------------------------------------------------------------------
    def finalize(self):
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.sensors.camera import Camera
        cams = {"top": (TOP_CAM_PATH, TOP_CAM), "overview": (OVERVIEW_CAM_PATH, OVERVIEW_CAM),
                "wrist": (WRIST_CAM_PATH, WRIST_CAM), "showcase": (SHOWCASE_CAM_PATH, SHOWCASE_CAM)}
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
