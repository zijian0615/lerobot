"""Simulator backends for the LOTO environment.

IsaacBackend   Isaac Sim 6.0: the FANUC ER-4iA cell from fanuc_er4ia_sim/
               (arm A with its gripper, table, cameras), a second ER-4iA with
               the same gripper (arm B) across the table, and the LOTO fixtures.
KinematicBackend
               No simulator: joints track their targets at a capped speed and
               nothing is rendered. Used by the smoke tests and for checking
               programs quickly; the task logic is identical.

Both expose the same small surface to env.py (arm is "A" or "B"; positions in
arm A's base frame):
    spawn(name, layout)           build one fixture (before finalize)
    finalize()                    reset physics, home both arms
    step(cmds, n, cb)             cmds = {arm: (q6, grip)}; n physics steps,
                                  cb(t) after each
    joint_positions(arm) -> q6    arm joints, rad
    jaw_fraction(arm) -> 0..1     measured gripper opening (1 = open)
    set_prop(name, pos)           move a visual-only prop (lock, tag)
    set_shackle(name, travel)     press a lock's shackle in (0 = open .. RAISE)
    set_lever(deg)                pose the disconnect handle
    render() -> {view: HxWx3}     camera images (empty for kinematic)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from . import layout as L

# ManipSecBench/fanuc_cell/ holds the cell, the verified FK/IK and the dimensions:
# a frozen copy of fanuc_er4ia_sim/ (see its README), so recalibrating the twin
# does not change the benchmark.
FANUC_SIM_DIR = Path(os.environ.get("FANUC_SIM_DIR")
                     or Path(__file__).resolve().parents[3] / "fanuc_cell")
if str(FANUC_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(FANUC_SIM_DIR))

import scene_config as C  # noqa: E402  (pure python)
import fk  # noqa: E402
import ik  # noqa: E402

ik.set_tool_axis(C.TOOL_AXIS)
ik.set_spread_axis(C.FINGER_SPREAD_AXIS)

TIP_OFFSET_M = 0.30          # flange face -> fingertips (cell.tip_offset())
TOOL_OFF = tuple(np.array(C.TOOL_AXIS) * TIP_OFFSET_M)
JOINT_LIMITS = np.deg2rad(np.array([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)]))
HOME_Q = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])
PHYSICS_DT = 1.0 / 60.0
# the look every task shares (the sorting and assembly backends use these too):
# a blue-grey ESD mat on the worktop, and the camera exposure 2 stops under the
# renderer's default (f/5, 1/50 s, ISO 100), which overexposes the hall
ESD_MAT = ((0.22, 0.28, 0.33), 0.70)       # rgb, roughness
EXPOSURE_EV = -2.0
DOWN = (0.0, 0.0, -1.0)

ARMS = ("A", "B")
ARM_BASE = {"A": ((0.0, 0.0, 0.0), 0.0), "B": (L.ARM_B_BASE, L.ARM_B_YAW_DEG)}


def _rz(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])


def tool_frame(arm, q6):
    """(fingertip, approach, spread) of an arm, in arm A's frame."""
    base, yaw = ARM_BASE[arm]
    R, p = fk.tool_pose(np.asarray(q6, float), TOOL_OFF)
    Rz = _rz(yaw)
    return (Rz @ p + np.asarray(base),
            Rz @ R @ np.array(C.TOOL_AXIS, float),
            Rz @ R @ np.array(C.FINGER_SPREAD_AXIS, float))


def grip_stall(width):
    """Gripper command at which the jaws stop on a part this wide."""
    half = max(float(width) / 2, C.JAW_CLOSED)
    return float(np.clip((C.JAW_OPEN - half) / (C.JAW_OPEN - C.JAW_CLOSED), 0.0, 1.0))


def tip_of(q6, arm="A"):
    return tool_frame(arm, q6)[0]


J6_UNWIND = np.deg2rad(270.0)      # solve_ik: past this, J6 is seeded one turn back


def solve_ik(arm, q0, p, approach=DOWN, spread=None):
    """IK for a fingertip target given in arm A's frame."""
    base, yaw = ARM_BASE[arm]
    Rt = _rz(-yaw)
    p_arm = Rt @ (np.asarray(p, float) - np.asarray(base))
    a = Rt @ np.asarray(approach, float)
    s = None if spread is None else Rt @ np.asarray(spread, float)
    q0 = np.array(q0, float)
    if abs(q0[5]) > J6_UNWIND:
        # The DLS solve is local, so J6 follows every wrist turn and can wind
        # up against its +/-360 deg stop (a base swing seen from behind does
        # it in one move); pinned there, the tool's roll -- and the held
        # part's position -- drift off target. Past 270 deg, seed it one turn
        # back: the same tool orientation, J6 within +/-90 deg.
        q0[5] -= np.sign(q0[5]) * 2 * np.pi
    q, ok = ik.solve(q0, p_arm, a, JOINT_LIMITS, tool_offset=TOOL_OFF, spread_dir=s)
    if ok:
        return q, ok
    # Damped least squares is local: a big base swing plus a wrist turn can
    # leave it pinned at a joint limit. Retry with J1 already facing the target.
    yaw = np.arctan2(p_arm[1], p_arm[0])
    best = (q, ok)
    for seed in (np.r_[yaw, np.asarray(q0)[1:]], np.r_[yaw, HOME_Q[1:]]):
        q2, ok2 = ik.solve(seed, p_arm, a, JOINT_LIMITS, tool_offset=TOOL_OFF,
                           spread_dir=s, iters=400)
        if ok2:
            return q2, ok2
    return best


def _start_q():
    """Arm A starts at the cell's home pose; arm B parked, leaning back so its
    wrist is clear of the lock stand and arm A's way to the hasp."""
    qb, ok = solve_ik("B", HOME_Q, L.ARM_B_PARK)
    assert ok, "arm B park pose unreachable"
    return {"A": HOME_Q.copy(), "B": qb}


START_Q = _start_q()


class KinematicBackend:
    MAX_JOINT_SPEED = 2.0    # rad/s, roughly the ER-4iA's slower axes

    def __init__(self, dt=PHYSICS_DT):
        self.dt = dt
        self.t = 0.0
        self.q = {a: START_Q[a].copy() for a in ARMS}
        self.grip = {a: 0.0 for a in ARMS}
        self.props = {}
        self.lever_deg = L.LEVER_ON_DEG

    def spawn(self, name, layout):
        pass

    def finalize(self):
        pass

    def step(self, cmds, n, cb):
        dq_max = self.MAX_JOINT_SPEED * self.dt
        for _ in range(n):
            for a, (q6, grip) in cmds.items():
                self.q[a] = self.q[a] + np.clip(np.asarray(q6, float) - self.q[a],
                                                -dq_max, dq_max)
                self.grip[a] = float(grip)
            self.t += self.dt
            cb(self.t)

    def joint_positions(self, arm):
        return self.q[arm].copy()

    def jaw_fraction(self, arm):
        return 1.0 - self.grip[arm]

    def set_prop(self, name, pos):
        self.props[name] = np.asarray(pos, float)

    def set_shackle(self, name, travel):
        pass

    def set_lever(self, deg):
        self.lever_deg = deg

    def press_meter_button(self):
        pass

    def release_meter_button(self):
        pass

    def render(self):
        return {}

    def close(self):
        pass


# --------------------------------------------------------------------------
# Isaac Sim

LOTO_CAM_PATH = "/World/LotoCam"
LOTO_CAM = dict(
    parent="/World",
    eye=(1.10, -0.30, C.TABLE_TOP_Z + 0.62),
    look_at=(0.34, 0.16, C.TABLE_TOP_Z + 0.10),
    up=(0.0, 0.0, 1.0),
    resolution=(1280, 960),
    focal_length=14.0,
    horiz_aperture=20.955,
)
HASP_CAM_PATH = "/World/HaspCam"
HASP_CAM = dict(
    parent="/World",
    eye=(0.30, 0.44, C.TABLE_TOP_Z + 0.24),
    look_at=(0.36, 0.21, C.TABLE_TOP_Z + 0.20),
    up=(0.0, 0.0, 1.0),
    resolution=(1280, 960),
    focal_length=24.0,
    horiz_aperture=20.955,
)
MAINT_CAM_PATH = "/World/MaintCam"
MAINT_CAM = dict(                        # cover screw, clamp ammeter, test point
    parent="/World",
    eye=(0.80, 0.02, C.TABLE_TOP_Z + 0.40),
    look_at=(0.40, 0.24, C.TABLE_TOP_Z + 0.07),
    up=(0.0, 0.0, 1.0),
    resolution=(1280, 960),
    focal_length=16.0,
    horiz_aperture=20.955,
)
ROOT = "/World/Loto"

# label textures (make_labels.py) and their sizes in mm
from .make_labels import LABELS as LABEL_MM, OUT as TEXTURES  # noqa: E402
PROP_MATERIALS = {
    #                    diffuse               metallic roughness
    "enclosure_grey": ((0.330, 0.345, 0.355), 0.10, 0.45),   # grey powder coat
    "brass":          ((0.780, 0.600, 0.250), 0.90, 0.35),
}
ARM_B_PRIM = "/World/ER4iA_B"

HALL_CLUTTER = ("KLT_Bins", "SM_PaletteA", "SM_PushcartA", "SM_CardBoxA", "SM_RackFrame",
                "SM_RackShelf", "SM_Rackshield", "SM_SignCVer", "S_AisleSign", "S_Barcode",
                "SM_FloorDecal")          # red/white hazard decals on the floor
# every task's background: the hall cleared and repainted to go with the ESD mat
DEFAULT_BACKGROUND = "hall_grey"

# Backgrounds: Isaac environment assets referenced around the cell. The
# offset moves the environment (not the cell) so the table stands in a clear
# area; yaw turns it about the world z axis.
BACKGROUNDS = {
    "studio": None,     # the cell's own grey floor and studio dome
    # the empty hall of the same building: the cell on open floor, a few
    # metres from the walls, no racks
    "hall": dict(usd="/Isaac/Environments/Simple_Warehouse/warehouse.usd",
                 offset=(4.0, -14.0, 0.0), yaw=0.0),
    # the same hall without the storage clutter (bins, pallets, racks, cart,
    # boxes) -- a workshop bay rather than a stock room
    "hall_clean": dict(usd="/Isaac/Environments/Simple_Warehouse/warehouse.usd",
                       offset=(4.0, -14.0, 0.0), yaw=0.0, hide=HALL_CLUTTER),
    # ... and repainted to go with the blue-grey ESD mat: mid-grey epoxy
    # floor, light warm-grey walls and pillars
    "hall_grey": dict(usd="/Isaac/Environments/Simple_Warehouse/warehouse.usd",
                      offset=(4.0, -14.0, 0.0), yaw=0.0, hide=HALL_CLUTTER,
                      paint={"SM_floor": ((0.36, 0.37, 0.38), 0.55),
                             "GroundPlane": ((0.36, 0.37, 0.38), 0.55),
                             "SM_WallA": ((0.74, 0.72, 0.68), 0.80),
                             "SM_Pillar": ((0.74, 0.72, 0.68), 0.80)}),
    "warehouse": dict(usd="/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
                      # cell in the aisle between two rack rows, racks behind it
                      offset=(3.0, -27.0, 0.0), yaw=0.0),
}
ENV_PRIM = "/World/Environment"


def tidy_background(stage, bg):
    """Hide a background's clutter (prims whose names start with bg["hide"])
    and repaint its surfaces (bg["paint"]: name prefix -> (rgb, roughness))."""
    from pxr import Usd, UsdShade
    root = stage.GetPrimAtPath(ENV_PRIM)
    hide, paint = tuple(bg.get("hide", ())), bg.get("paint", {})
    mats = {}
    for p in Usd.PrimRange(root):
        n = p.GetName()
        if hide and n.startswith(hide):
            p.SetActive(False)
            continue
        for prefix, (rgb, rough) in paint.items():
            if n.startswith(prefix):
                if prefix not in mats:
                    from isaacsim.core.api.materials import OmniPBR
                    m = OmniPBR(prim_path=f"/World/Looks/bg_{prefix}", name=f"bg_{prefix}",
                                color=np.array(rgb))
                    m.set_metallic_constant(0.0)
                    m.set_reflection_roughness(rough)
                    mats[prefix] = m.material
                UsdShade.MaterialBindingAPI.Apply(p).Bind(
                    mats[prefix], bindingStrength=UsdShade.Tokens.strongerThanDescendants)
                break


def default_background():
    """The background every task uses; MANIPSEC_BG overrides it (for looks)."""
    return os.environ.get("MANIPSEC_BG", DEFAULT_BACKGROUND)


class IsaacBackend:
    """Runs inside Isaac Sim's interpreter. Starts SimulationApp itself."""

    def __init__(self, headless=True, views=("loto",), render_every=1, spp=4,
                 background=None):
        """spp > 0: path tracing with the OptiX denoiser. spp = 0: Isaac's
        default RealTimePathTracing, which needs DLSS-RR (NGX) to denoise and
        renders as heavy grain wherever NGX fails to initialise."""
        from isaacsim import SimulationApp
        self.app = SimulationApp({"headless": headless})

        import omni.usd
        from isaacsim.core.api import World
        import cell
        self.cell = cell
        self.world = World(stage_units_in_meters=1.0, physics_dt=PHYSICS_DT,
                           rendering_dt=PHYSICS_DT)
        self.stage = omni.usd.get_context().get_stage()
        # arm A, gripper, table, lighting, cell cameras; no screws/bin/props
        cell.build(self.stage, with_objects=False)
        self._add_materials()
        self._plain_white_top()
        self._add_arm_b()
        self._add_background(background or default_background())
        if spp > 0:
            import render_settings
            render_settings.enable_path_tracing(spp)
            self._set_exposure(EXPOSURE_EV)
        cell.camera(self.stage, LOTO_CAM_PATH, LOTO_CAM)
        cell.camera(self.stage, HASP_CAM_PATH, HASP_CAM)
        cell.camera(self.stage, MAINT_CAM_PATH, MAINT_CAM)
        from pxr import UsdGeom
        UsdGeom.Xform.Define(self.stage, ROOT)
        self.views = tuple(views)
        self.render_every = max(1, int(render_every))
        self._xf = {}            # animated prim -> its xform ops
        self.dt = PHYSICS_DT
        self.t = 0.0
        self._n_steps = 0

    def _add_arm_b(self):
        """A second ER-4iA with the same EOAT. cell.attach_gripper builds on
        the module's FLANGE / JAW_PRIMS, so point those at arm B meanwhile."""
        from pxr import UsdShade
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path
        cell = self.cell
        add_reference_to_stage(usd_path=get_assets_root_path() + C.ROBOT_USD_SUBPATH,
                               prim_path=ARM_B_PRIM)
        base, yaw = ARM_BASE["B"]
        cell.place(self.stage.GetPrimAtPath(ARM_B_PRIM),
                   (base[0], base[1], C.TABLE_TOP_Z + base[2]), yaw)
        vis = self.stage.GetPrimAtPath(ARM_B_PRIM + "/robot_base/visuals")
        if vis.IsValid() and "base_grey" in cell.MATS:
            vis.SetInstanceable(False)
            UsdShade.MaterialBindingAPI.Apply(vis).Bind(
                cell.MATS["base_grey"], bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        saved = cell.FLANGE, cell.JAW_PRIMS
        cell.FLANGE = ARM_B_PRIM + "/J6_link"
        cell.JAW_PRIMS = {"L": ARM_B_PRIM + "/JawL", "R": ARM_B_PRIM + "/JawR"}
        try:
            cell.attach_gripper(self.stage)
        finally:
            cell.FLANGE, cell.JAW_PRIMS = saved

    def _add_background(self, name):
        bg = BACKGROUNDS[name]
        if bg is None:
            return
        from pxr import UsdGeom
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path
        add_reference_to_stage(usd_path=get_assets_root_path() + bg["usd"],
                               prim_path=ENV_PRIM)
        self.cell.place(self.stage.GetPrimAtPath(ENV_PRIM), bg["offset"], bg["yaw"])
        tidy_background(self.stage, bg)
        self._add_table_frame()
        # the environment brings its own floor; the cell's grey slab would
        # z-fight with it (keep its collider, just hide it)
        UsdGeom.Imageable(self.stage.GetPrimAtPath("/World/Floor")).MakeInvisible()
        # the studio dome would show through the warehouse skylights
        dome = self.stage.GetPrimAtPath("/World/DomeLight")
        if dome.IsValid():
            dome.GetAttribute("inputs:intensity").Set(250.0)

    def _add_materials(self):
        """Materials of the LOTO props, registered with the cell's so its
        box()/cyl() helpers can bind them by name."""
        from isaacsim.core.api.materials import OmniPBR
        self._label_mats = {}
        for name, (diff, metal, rough) in PROP_MATERIALS.items():
            pbr = OmniPBR(prim_path=f"/World/Looks/{name}", name=name,
                          color=np.array(diff, float))
            pbr.set_metallic_constant(float(metal))
            pbr.set_reflection_roughness(float(rough))
            self.cell.MATS[name] = pbr.material

    def _label(self, path, center, u, v, tex):
        """A printed label: the texture assets/textures/<tex>.png on a quad in
        the parent's frame, centred at center, image +x along unit vector u and
        image up along v, sized as make_labels.LABELS says. Visual only."""
        from pxr import UsdGeom, Gf, Sdf, UsdShade
        from isaacsim.core.api.materials import OmniPBR
        if tex not in self._label_mats:
            pbr = OmniPBR(prim_path=f"/World/Looks/label_{tex}", name=f"label_{tex}",
                          color=np.array((1.0, 1.0, 1.0)))
            pbr.set_metallic_constant(0.0)
            pbr.set_reflection_roughness(0.4)
            pbr.set_project_uvw(False)           # use the quad's UVs, no tiling
            pbr.shaders_list[0].CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(str(TEXTURES / f"{tex}.png")))
            self._label_mats[tex] = pbr.material
        w, h = (mm / 1000 for mm in LABEL_MM[tex])
        c, u, v = (np.asarray(a, float) for a in (center, u, v))
        hu, hv = u * w / 2, v * h / 2
        corners = [c - hu - hv, c + hu - hv, c + hu + hv, c - hu + hv]
        n = np.cross(u, v)
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in corners])
        m.CreateFaceVertexCountsAttr([4])
        m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        m.CreateNormalsAttr([Gf.Vec3f(*map(float, n))] * 4)
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        st = UsdGeom.PrimvarsAPI(m).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(self._label_mats[tex])

    def _set_exposure(self, ev):
        """Shift the tonemapper's exposure by ev stops (via the f-number): the
        renderer's default overexposes the cell in the hall's light."""
        import carb
        s = carb.settings.get_settings()
        f = s.get("/rtx/post/tonemap/fNumber") or 5.0
        s.set("/rtx/post/tonemap/fNumber", float(f) * (2.0 ** (-ev / 2.0)))

    def _plain_white_top(self):
        """A plain worktop: hide the printed mat artwork (reach arcs, CAUTION
        bands, tape marks) of the source cell and lay the ESD mat (ESD_MAT)."""
        from pxr import UsdGeom, UsdShade
        from isaacsim.core.api.materials import OmniPBR
        art = self.stage.GetPrimAtPath("/World/MatPrint")
        if art.IsValid():
            UsdGeom.Imageable(art).MakeInvisible()
        (rgb, rough) = ESD_MAT                  # the worktop every task shares
        white = OmniPBR(prim_path="/World/Looks/loto_top_white", name="loto_top_white",
                        color=np.array(rgb))
        white.set_metallic_constant(0.0)
        white.set_reflection_roughness(rough)
        for path in ("/World/Mat", "/World/Table"):
            prim = self.stage.GetPrimAtPath(path)
            if prim.IsValid():
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    white.material, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    def _add_table_frame(self):
        """Legs and stretchers under the worktop: in the studio the slab's
        underside was never in view, in a room it floats."""
        box = self.cell.box
        x0, x1 = C.TABLE_X
        y0, y1 = C.TABLE_Y
        under = C.TABLE_TOP_Z - C.TABLE_THICK
        leg, inset = 0.045, 0.06
        xs, ys = (x0 + inset, x1 - inset), (y0 + inset, y1 - inset)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                box(self.stage, f"/World/TableFrame/Leg{i}{j}", (leg, leg, under),
                    (x, y, under / 2), (0.7, 0.7, 0.72), mat="alu_anodised")
                box(self.stage, f"/World/TableFrame/Foot{i}{j}", (0.08, 0.08, 0.012),
                    (x, y, 0.006), (0.05, 0.05, 0.05), mat="black_plastic")
        for k, z in enumerate((0.12, under - 0.04)):
            for i, x in enumerate(xs):
                box(self.stage, f"/World/TableFrame/SideY{k}{i}", (leg, y1 - y0 - 2 * inset, leg),
                    (x, (y0 + y1) / 2, z), (0.7, 0.7, 0.72), collision=False, mat="alu_anodised")
            for j, y in enumerate(ys):
                box(self.stage, f"/World/TableFrame/SideX{k}{j}", (x1 - x0 - 2 * inset, leg, leg),
                    ((x0 + x1) / 2, y, z), (0.7, 0.7, 0.72), collision=False, mat="alu_anodised")

    # ---- fixture construction --------------------------------------------
    def _w(self, p):
        """Arm A's base frame -> world (the arm stands on the table top)."""
        return (float(p[0]), float(p[1]), float(p[2]) + C.TABLE_TOP_Z)

    def _anim_xform(self, path, pos, rot_x_deg=None, local=False):
        """An Xform whose pose is animated; pos in arm A's base frame, or in
        the parent's frame with local=True."""
        from pxr import UsdGeom, Gf
        x = UsdGeom.Xform.Define(self.stage, path)
        t = x.AddTranslateOp()
        t.Set(Gf.Vec3d(*(pos if local else self._w(pos))))
        r = None
        if rot_x_deg is not None:
            r = x.AddRotateXOp()
            r.Set(float(rot_x_deg))
        self._xf[path] = (t, r)
        return path

    def spawn(self, name, lay: L.Layout):
        box, cyl = self.cell.box, self.cell.cyl
        st = self.stage
        hole_xy = [h[:2] for h in lay.holes]
        if name == "disconnect":
            c = lay.disconnect
            sx, sy, sz = L.ENCLOSURE_SIZE
            # a wall-type steel enclosure, RAL 7035 powder coat, its door on
            # the +x face (toward the operator); the collider is the box alone
            box(st, f"{ROOT}/Disconnect/Enclosure", (sx, sy, sz),
                self._w(c + [0, 0, sz / 2]), (0.78, 0.79, 0.78), mat="enclosure_grey")
            self._enclosure_dressing(c, sx, sy, sz, lay)
            # maintenance: a cover plate on the top with the empty screw hole
            hole = lay.screw_hole
            cx_, cy_, cz_ = L.COVER_SIZE
            box(st, f"{ROOT}/Disconnect/Cover", L.COVER_SIZE,
                self._w([hole[0], hole[1], hole[2] - cz_ / 2]), (0.7, 0.7, 0.72),
                collision=False, mat="alu_machined")
            cyl(st, f"{ROOT}/Disconnect/ScrewHole", 0.003, cz_ + 0.0004,
                self._w([hole[0], hole[1], hole[2] - cz_ / 2]), (0.02, 0.02, 0.02),
                mat="black_plastic")
            # fixed lockout bracket: a post on the +x half of the top edge
            # carrying a plate with the two holes near its -x edge, just under
            # where the lever ear lands
            y0, y1 = lay.plate_y
            T = L.HASP_PLATE_T
            pb = lay.plate_bottom
            box(st, f"{ROOT}/Disconnect/BracketPost", (L.HASP_HALF_X, 0.008, pb - sz),
                self._w([c[0] + L.HASP_HALF_X / 2, y0 + 0.004, (sz + pb) / 2]),
                (0.25, 0.25, 0.27), collision=False, mat="steel_dark")
            self._holed_plate(f"{ROOT}/Disconnect/Bracket", self._w([c[0], 0, pb + T / 2]),
                              y0, y1, [(x - c[0], y) for x, y in hole_xy],
                              "steel_dark", (0.25, 0.25, 0.27))
            # handle: pivot boss is static, the lever itself is animated and
            # has no collider (writing a collider's xform resyncs PhysX)
            pv = lay.lever_pivot
            cyl(st, f"{ROOT}/Disconnect/Boss", 0.018, 0.012,
                self._w(pv + [0, 0, 0.006]), (0.05, 0.05, 0.05), mat="black_plastic")
            lv = self._anim_xform(f"{ROOT}/Disconnect/Lever", pv, _usd_rot_x(L.LEVER_ON_DEG))
            box(st, f"{lv}/Arm", (0.014, 0.012, L.LEVER_LEN),
                (0, 0, L.LEVER_LEN / 2), (0.7, 0.05, 0.04), collision=False,
                mat="plastic_red")
            box(st, f"{lv}/Knob", L.KNOB_SIZE,
                (0, 0, L.LEVER_LEN), (0.05, 0.05, 0.05), collision=False,
                mat="black_plastic")
            # lockout ear on the knob: horizontal at OFF, lying on the bracket
            # with its holes over the bracket's (rotateX +OFF undoes the lever's
            # -OFF, so the ear's local frame is the world's at OFF)
            knob = lay.lever_tip(L.LEVER_OFF_DEG)
            ear = self._anim_xform(f"{lv}/Ear", (0, 0, L.LEVER_LEN),
                                   L.LEVER_OFF_DEG, local=True)
            self._holed_plate(ear, (0.0, -knob[1], 0.0), knob[1], lay.plate_y[1],
                              [(x - c[0], y) for x, y in hole_xy],
                              "plastic_red", (0.7, 0.05, 0.04))
        elif name == "test_point":
            tp = lay.test_point
            tx, ty, tz = L.TERMINAL_SIZE
            box(st, f"{ROOT}/Terminal/Block", (tx, ty, tz),
                self._w([tp[0], tp[1], tz / 2]), (0.08, 0.08, 0.09), mat="tray_dark")
            # a shrouded 4 mm test socket: red ring, black bore
            cyl(st, f"{ROOT}/Terminal/TestPoint", 0.006, 0.006,
                self._w([tp[0], tp[1], tz + 0.003]), (0.7, 0.05, 0.04),
                mat="plastic_red")
            cyl(st, f"{ROOT}/Terminal/TestBore", 0.0022, 0.0062,
                self._w([tp[0], tp[1], tz + 0.0031]), (0.02, 0.02, 0.02),
                mat="black_plastic")
            self._label(f"{ROOT}/Terminal/Label", self._w([tp[0] + 0.018, tp[1], tz + 0.0003]),
                        (0, 1, 0), (-1, 0, 0), "test")
            for k, dy in enumerate((-0.025, 0.025)):
                cyl(st, f"{ROOT}/Terminal/Screw{k}", 0.005, 0.004,
                    self._w([tp[0], tp[1] + dy, tz + 0.002]), (0.7, 0.7, 0.7),
                    mat="alu_machined")
            d = lay.disconnect
            mid = (np.array([tp[0], tp[1]]) + d[:2]) / 2
            length = float(np.linalg.norm(d[:2] - tp[:2]))
            ang = float(np.degrees(np.arctan2(d[1] - tp[1], d[0] - tp[0])))
            box(st, f"{ROOT}/Terminal/Conduit", (length, 0.012, 0.012),
                self._w([mid[0], mid[1], 0.008]), (0.1, 0.1, 0.1),
                rot=(0, 0, ang), collision=False, mat="black_plastic")
        elif name == "lock":
            self._padlock(f"{ROOT}/Lock", lay.lock_start, travel=0.0)
        elif name == "forged_lock":            # already shut in hole A
            self._padlock(f"{ROOT}/ForgedLock", lay.locked_center(0), travel=L.RAISE,
                          forged=True)
        elif name == "tag":
            self._tag(f"{ROOT}/Tag", lay.tag_start)
        elif name == "forged_tag":
            self._tag(f"{ROOT}/ForgedTag", lay.tag_site(0), forged=True)
        elif name == "screw":
            p = self._anim_xform(f"{ROOT}/Screw", lay.screw_start)
            d, hh = L.SCREW_HEAD
            cyl(st, f"{p}/Head", d / 2, hh, (0, 0, -hh / 2), (0.6, 0.6, 0.62),
                mat="alu_machined")
            for k, size in enumerate(((d * 0.8, 0.0012, 0.0012), (0.0012, d * 0.8, 0.0012))):
                box(st, f"{p}/Slot{k}", size, (0, 0, -0.0005),        # Phillips cross
                    (0.1, 0.1, 0.1), collision=False, mat="black_oxide")
            cyl(st, f"{p}/Shank", 0.0022, L.SCREW_LEN, (0, 0, -hh - L.SCREW_LEN / 2),
                (0.6, 0.6, 0.62), mat="alu_machined")
        elif name == "meter":
            self._spawn_meter(lay)
        elif name == "stand":                  # where the open lock waits
            box(st, f"{ROOT}/LockStand", (0.045, 0.030, L.LOCK_STAND_H),
                self._w(lay.stand_top - [0, 0, L.LOCK_STAND_H / 2]),
                (0.06, 0.06, 0.06), mat="tray_dark")
        elif name == "tray":
            box(st, f"{ROOT}/Tray", L.TRAY_SIZE, self._w(lay.tray_center),
                (0.06, 0.06, 0.06), mat="tray_dark")
        else:
            raise KeyError(name)

    def _enclosure_dressing(self, c, sx, sy, sz, lay):
        """Door, hinges, latch, labels and switch positions: all visual."""
        box, cyl, st = self.cell.box, self.cell.cyl, self.stage
        P = f"{ROOT}/Disconnect"
        fx = c[0] + sx / 2                                  # the door face
        dw, dh = sy - 0.016, sz - 0.024                     # door, inside a flange
        box(st, f"{P}/DoorGap", (0.001, dw + 0.004, dh + 0.004),
            self._w([fx + 0.0005, c[1], sz / 2]), (0.03, 0.03, 0.03),
            collision=False, mat="black_plastic")
        box(st, f"{P}/Door", (0.0015, dw, dh), self._w([fx + 0.00075, c[1], sz / 2]),
            (0.78, 0.79, 0.78), collision=False, mat="enclosure_grey")
        face = fx + 0.0018
        for k, z in enumerate((0.035, sz - 0.035)):         # hinges, -y edge
            cyl(st, f"{P}/Hinge{k}", 0.003, 0.022,
                self._w([face + 0.001, c[1] - dw / 2 - 0.001, z]), (0.6, 0.6, 0.6),
                mat="alu_machined")
        box(st, f"{P}/Latch", (0.006, 0.012, 0.028),         # quarter-turn latch, +y edge
            self._w([face + 0.003, c[1] + dw / 2 - 0.012, sz / 2]), (0.03, 0.03, 0.03),
            collision=False, mat="black_plastic")
        X, Y, Z = (1, 0, 0), (0, 1, 0), (0, 0, 1)
        self._label(f"{P}/DangerLabel", self._w([face, c[1] - 0.004, sz - 0.042]), Y, Z,
                    "danger")
        self._label(f"{P}/Nameplate", self._w([face, c[1] - 0.004, 0.048]), Y, Z,
                    "nameplate")
        # the hasp side (+y), below the lockout bracket
        self._label(f"{P}/SideLabel", self._w([c[0] + 0.008, c[1] + sy / 2 + 0.0003, 0.110]),
                    (-1, 0, 0), Z, "danger_side")
        # switch positions on the top, either side of the handle, read from +x
        pv = lay.lever_pivot
        for tex, dy in (("on", -0.032), ("off", 0.032)):
            self._label(f"{P}/Pos_{tex}", self._w([pv[0] - 0.016, pv[1] + dy, sz + 0.0003]),
                        Y, (-1, 0, 0), tex)

    def _spawn_meter(self, lay):
        """Supply riser (cable up through a table grommet, over into the
        terminal block) with the SimReady clamp multimeter clamped around it.
        The meter is fixed and visual-only: its rigid bodies, joints and
        colliders are switched off; the test leads, which lie loose in the
        asset, are removed."""
        from pxr import UsdPhysics, Usd
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path
        box, cyl = self.cell.box, self.cell.cyl
        st = self.stage
        r = lay.riser
        tb = lay.test_point
        cyl(st, f"{ROOT}/Supply/Grommet", 0.014, 0.002, self._w([r[0], r[1], 0.001]),
            (0.05, 0.05, 0.05), mat="black_plastic")
        cyl(st, f"{ROOT}/Supply/Riser", L.CABLE_R, L.RISER_H,
            self._w([r[0], r[1], L.RISER_H / 2]), (0.08, 0.08, 0.08), mat="black_plastic")
        # over the top and down into the terminal block's back corner
        tx, ty, tz = L.TERMINAL_SIZE
        entry = np.array([tb[0] + tx / 2 - 0.01, tb[1] + ty / 2 - 0.01])
        d = entry - r[:2]
        length = float(np.linalg.norm(d))
        ang = float(np.degrees(np.arctan2(d[1], d[0])))
        box(st, f"{ROOT}/Supply/Span", (length, 2 * L.CABLE_R, 2 * L.CABLE_R),
            self._w([(r[0] + entry[0]) / 2, (r[1] + entry[1]) / 2, L.RISER_H]),
            (0.08, 0.08, 0.08), rot=(0, 0, ang), collision=False, mat="black_plastic")
        cyl(st, f"{ROOT}/Supply/Drop", L.CABLE_R, L.RISER_H - tz,
            self._w([entry[0], entry[1], (L.RISER_H + tz) / 2]), (0.08, 0.08, 0.08),
            mat="black_plastic")
        path = f"{ROOT}/Meter"
        add_reference_to_stage(get_assets_root_path() + L.METER_USD, path)
        self.cell.place(st.GetPrimAtPath(path), self._w(lay.meter_origin), 0.0)
        self._meter_button = None
        # its colliders live inside instances; de-instance to switch them off
        for p in list(Usd.PrimRange(st.GetPrimAtPath(path))):
            if p.IsInstance():
                p.SetInstanceable(False)
        for p in Usd.PrimRange(st.GetPrimAtPath(path)):
            name = p.GetName()
            if name.startswith("testLead"):
                p.SetActive(False)
                continue
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(p).CreateRigidBodyEnabledAttr().Set(False)
            if p.HasAPI(UsdPhysics.CollisionAPI):
                # SDF colliders round small parts up by several mm: a
                # fingertip stops ~6 mm above the power button. Visual-only,
                # like the other props; the button press is a geometric check.
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr().Set(False)
            if p.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(p).CreateJointEnabledAttr().Set(False)
            if name == "button_01_obj_00":
                self._meter_button = p

    def press_meter_button(self, dz=-0.002):
        """Show the power button pressed in (2 mm); dz=+0.002 lets it out."""
        from pxr import UsdGeom
        p = self._meter_button
        if p is None:
            return
        ops = {o.GetOpName(): o for o in UsdGeom.Xformable(p).GetOrderedXformOps()}
        t = ops.get("xformOp:translate")
        if t is not None:
            v = t.Get()
            t.Set(type(v)(v[0], v[1], v[2] + dz))

    def release_meter_button(self):
        self.press_meter_button(dz=0.002)

    def _holed_plate(self, path, origin, y0, y1, holes, mat, color):
        """A plate of thickness HASP_PLATE_T centred at origin (parent frame),
        spanning x +/- HASP_HALF_X and y in [y0, y1] + origin.y, with square
        holes at local (x, y): built from strips, as USD has no boolean ops.
        The holes share one x (a column near the plate's -x edge)."""
        box = self.cell.box
        ox, oy, oz = origin
        X, h = L.HASP_HALF_X, L.HOLE_HALF
        hx = holes[0][0]
        pieces = [((-X, hx - h), (y0, y1)), ((hx + h, X), (y0, y1))]
        cuts = [y0] + [v for _, y in sorted(holes, key=lambda p: p[1])
                       for v in (y - h, y + h)] + [y1]
        pieces += [((hx - h, hx + h), (a, b)) for a, b in zip(cuts[0::2], cuts[1::2])]
        for k, ((a, b), (c, d)) in enumerate(pieces):
            if b - a > 1e-6 and d - c > 1e-6:
                box(self.stage, f"{path}/Plate{k}", (b - a, d - c, L.HASP_PLATE_T),
                    (ox + (a + b) / 2, oy + (c + d) / 2, oz), color,
                    collision=False, mat=mat)

    def _padlock(self, path, center, travel, forged=False):
        """Origin at the body centre. The shackle (bow + legs, legs along x,
        free leg +x) is a child moved up by RAISE - travel. A safety padlock:
        coloured nylon body with a DANGER label, key cylinder underneath; the
        forged one is someone else's (blue, their name on it)."""
        box, cyl = self.cell.box, self.cell.cyl
        p = self._anim_xform(path, center)
        bx, by, bz = L.LOCK_BODY
        box(self.stage, f"{p}/Case", L.LOCK_BODY, (0, 0, 0), (0.75, 0.08, 0.05),
            collision=False, mat="plastic_blue" if forged else "plastic_red")
        cyl(self.stage, f"{p}/KeyCylinder", 0.0045, 0.0012, (0, 0, -bz / 2 - 0.0006),
            (0.78, 0.6, 0.25), mat="brass")
        box(self.stage, f"{p}/Keyway", (0.0045, 0.0012, 0.0014), (0, 0, -bz / 2 - 0.0008),
            (0.02, 0.02, 0.02), collision=False, mat="black_plastic")
        tex = "lock_forged" if forged else "lock"
        for side, u in ((1, (-1, 0, 0)), (-1, (1, 0, 0))):
            self._label(f"{p}/Label{'P' if side > 0 else 'N'}",
                        (0, side * (by / 2 + 0.0003), 0.004), u, (0, 0, 1), tex)
        sh = self._anim_xform(f"{p}/Shackle", (0, 0, L.RAISE - travel), local=True)
        S, R, top = L.SHACKLE_S, L.SHACKLE_R, L.BOW_CLOSED
        box(self.stage, f"{sh}/Bow", (S + 2 * R, 2 * R, L.BOW_T), (0, 0, top - L.BOW_T / 2),
            (0.7, 0.7, 0.72), collision=False, mat="alu_machined")
        free = L.FREE_LEG_LEN - L.BOW_T
        cyl(self.stage, f"{sh}/FreeLeg", R, free, (S / 2, 0, top - L.BOW_T - free / 2),
            (0.7, 0.7, 0.72), mat="alu_machined")
        # long enough to stay in the body when open, short enough not to poke
        # out of its bottom when shut
        fixed = top - L.BOW_T + L.LOCK_BODY[2] / 2 - 0.003
        cyl(self.stage, f"{sh}/FixedLeg", R, fixed, (-S / 2, 0, top - L.BOW_T - fixed / 2),
            (0.7, 0.7, 0.72), mat="alu_machined")

    def _tag(self, path, pos, forged=False):
        """Origin at the tag's top hole -- the grasp point. The plate hangs
        TAG_DX to the side (-x) so it clears the lock body. A printed LOTO tag
        (DANGER / DO NOT OPERATE) on both faces; the forged one is filled in
        with someone else's name."""
        box, cyl = self.cell.box, self.cell.cyl
        p = self._anim_xform(path, pos)
        tx, ty, tz = L.TAG_SIZE
        cyl(self.stage, f"{p}/Grommet", 0.004, 0.002, (0, 0, -0.002),
            (0.7, 0.7, 0.72), mat="alu_machined")
        box(self.stage, f"{p}/Tie", (abs(L.TAG_DX), 0.002, 0.002), (L.TAG_DX / 2, 0, -0.004),
            (0.9, 0.9, 0.9), collision=False, mat="white_plastic")
        box(self.stage, f"{p}/Plate", (tx, ty, tz), (L.TAG_DX, 0, -tz / 2 - 0.004),
            (0.9, 0.9, 0.9), collision=False, mat="white_plastic")
        tex = "tag_forged" if forged else "tag"
        for side, u in ((1, (-1, 0, 0)), (-1, (1, 0, 0))):
            self._label(f"{p}/Print{'P' if side > 0 else 'N'}",
                        (L.TAG_DX, side * (ty / 2 + 0.0003), -tz / 2 - 0.004), u, (0, 0, 1), tex)

    # ---- runtime ------------------------------------------------------------
    def finalize(self):
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.sensors.camera import Camera
        paths = {"loto": (LOTO_CAM_PATH, LOTO_CAM),
                 "hasp": (HASP_CAM_PATH, HASP_CAM),
                 "maint": (MAINT_CAM_PATH, MAINT_CAM),
                 "table": (self.cell.TABLE_CAM_PATH, C.TABLE_CAM),
                 "closeup": (self.cell.CLOSEUP_CAM_PATH, C.CLOSEUP_CAM),
                 "wrist": (self.cell.WRIST_CAM_PATH, C.WRIST_CAM)}
        # render products must exist before world.reset()
        self.cams = {v: Camera(prim_path=paths[v][0], resolution=paths[v][1]["resolution"])
                     for v in self.views}
        self.cell.enable_ccd(self.world)
        self.world.reset()
        for c in self.cams.values():
            c.initialize()
        self.art, self.arm_idx, self.idx = {}, {}, {}
        for a, prim in (("A", C.ROBOT_PRIM), ("B", ARM_B_PRIM)):
            art = SingleArticulation(prim_path=prim, name=f"er4ia_{a}")
            art.initialize()
            names = list(art.dof_names)
            self.art[a] = art
            self.arm_idx[a] = [names.index(f"J{i}") for i in range(1, 7)]
            self.idx[a] = {n: names.index(n) for n in names}
            q = art.get_joint_positions()
            q[self.arm_idx[a]] = START_Q[a]
            art.set_joint_positions(q)
        # MDL materials compile over the first frames; render before capturing
        for _ in range(90):
            for a in ARMS:
                self._command(a, START_Q[a], 0.0)
            self.world.step(render=True)

    def _command(self, arm, q6, grip):
        from isaacsim.core.utils.types import ArticulationAction
        art = self.art[arm]
        full = art.get_joint_positions().copy()
        full[self.arm_idx[arm]] = q6
        jaw = C.JAW_OPEN + (C.JAW_CLOSED - C.JAW_OPEN) * float(grip)
        for jn, v in self.cell.jaw_targets(jaw).items():
            full[self.idx[arm][jn]] = v
        art.apply_action(ArticulationAction(joint_positions=full))

    def step(self, cmds, n, cb):
        for a, (q6, grip) in cmds.items():
            self._command(a, q6, grip)
        for _ in range(n):
            self._n_steps += 1
            self.world.step(render=bool(self.cams) and self._n_steps % self.render_every == 0)
            self.t += self.dt
            cb(self.t)

    def joint_positions(self, arm):
        return np.asarray(self.art[arm].get_joint_positions()[self.arm_idx[arm]], float)

    def jaw_fraction(self, arm):
        jp = self.art[arm].get_joint_positions()
        closed = C.JAW_OPEN - abs(float(jp[self.idx[arm]["jaw_l"]]))
        return float(np.clip((closed - C.JAW_CLOSED) / (C.JAW_OPEN - C.JAW_CLOSED), 0, 1))

    def set_prop(self, name, pos):
        from pxr import Gf
        t, _ = self._xf[f"{ROOT}/{_cap(name)}"]
        t.Set(Gf.Vec3d(*self._w(pos)))

    def set_shackle(self, name, travel):
        from pxr import Gf
        t, _ = self._xf[f"{ROOT}/{_cap(name)}/Shackle"]
        t.Set(Gf.Vec3d(0.0, 0.0, float(L.RAISE - travel)))

    def set_lever(self, deg):
        _, r = self._xf[f"{ROOT}/Disconnect/Lever"]
        r.Set(_usd_rot_x(deg))

    def render(self):
        out = {}
        for v, c in self.cams.items():
            rgba = c.get_rgba()
            if rgba is not None and rgba.size:
                out[v] = np.asarray(rgba)[:, :, :3].astype(np.uint8)
        return out

    def close(self):
        self.app.close()


def _usd_rot_x(deg):
    """Layout lever angle (+ toward +y) -> USD rotateX, which is right-handed
    about +x and so tips the top toward -y for positive angles."""
    return -float(deg)


def _cap(name):
    return "".join(p.capitalize() for p in name.split("_"))
