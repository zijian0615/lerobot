"""Isaac Sim 6 backend of the GPU-install cell: the FANUC ER-4iA (arm A) on
the white worktop in the hall, a PC case lying on its side with the
motherboard up, a PCIe x16 slot with a hinged retention latch, a graphics card
in a foam pick station.

Physics, not bookkeeping:
  - the card is one dynamic rigid body whose colliders are its regions
    (geometry.CARD_COLLIDERS): housing, gold-finger connector, retention toe,
    bare PCB edge, bracket, I/O block
  - the slot is static geometry: a 2.6 mm channel with a flared mouth, a
    floor, end stops; the case's rear panel has the one open expansion-slot
    opening the I/O block fits through
  - the latch is its own rigid body on a revolute joint with limits; an
    angular drive whose target flips at LATCH_TOGGLE_DEG models the
    over-centre spring (bistable OPEN / CLOSED). The card's toe closes it by
    pushing its foot down as the card seats; pressing its thumb opens it
  - every contact involving the card, the latch or the gripper is read from
    PhysX's contact report each physics step, by collider (contacts())

Cameras: "top" (nadir RGB + depth), "wrist" (RGB, on J6) for the program;
"overview" and "slot" (a close look at the slot and latch) for snapshots and
videos only.
"""

from __future__ import annotations

import math

import numpy as np

from . import geometry as G
from sorting_battery.backend import SortingBackend, LB, C, PHYSICS_DT  # noqa: F401

ROOT = "/World/Asm"
CARD_PATH = f"{ROOT}/Card"
LATCH_PATH = f"{ROOT}/Slot/Latch"
LATCH_JOINT = f"{ROOT}/Slot/LatchHinge"
JAW_OPEN = 0.024                  # per jaw: a 48 mm gap for the 35.8 mm housing
# This task's jaws: the cell's 180 N, 4e4 N/m drive on 25 g jaws rings at
# ~5 ms against the 8.3 ms step and chatters on a 1 kg part (the card creeps
# round in the pinch). A softer, damped drive squeezes ~80 N, steadily.
JAW_DRIVE = dict(JAW_DRIVE_STIFF=5.0e3, JAW_DRIVE_DAMP=50.0, JAW_MAX_FORCE=100.0)

OVERVIEW_CAM_PATH = "/World/AsmOverviewCam"
OVERVIEW_CAM = dict(parent="/World", eye=(0.90, -0.62, C.TABLE_TOP_Z + 0.62),
                    look_at=(0.30, -0.03, C.TABLE_TOP_Z + 0.05), up=(0.0, 0.0, 1.0),
                    resolution=(1280, 960), focal_length=16.0, horiz_aperture=20.955)
SLOT_CAM_PATH = "/World/AsmSlotCam"
_S = G.from_slot
SLOT_CAM = dict(parent="/World",
                eye=tuple(_S([-0.15, -0.12, 0.13]) + [0.0, 0.0, C.TABLE_TOP_Z]),
                look_at=tuple(_S([-0.035, 0.0, 0.008]) + [0.0, 0.0, C.TABLE_TOP_Z]),
                up=(0.0, 0.0, 1.0), resolution=(960, 720), focal_length=28.0,
                horiz_aperture=20.955)

MATERIALS = {
    #                     diffuse               metallic roughness
    "case_steel":      ((0.10, 0.105, 0.11), 0.00, 0.60),
    "case_inner":      ((0.10, 0.10, 0.11), 0.30, 0.55),
    "board_pcb":       ((0.04, 0.09, 0.07), 0.00, 0.45),
    "slot_black":      ((0.025, 0.025, 0.028), 0.00, 0.50),
    "latch_white":     ((0.86, 0.86, 0.84), 0.00, 0.45),
    "shroud_grey":     ((0.13, 0.13, 0.14), 0.25, 0.40),
    "shroud_accent":   ((0.55, 0.56, 0.58), 0.85, 0.25),
    "fan_black":       ((0.03, 0.03, 0.035), 0.00, 0.60),
    "backplate_black": ((0.05, 0.05, 0.055), 0.40, 0.35),
    "card_pcb":        ((0.05, 0.22, 0.10), 0.00, 0.40),
    "gold":            ((0.83, 0.62, 0.22), 1.00, 0.22),
    "bracket_steel":   ((0.72, 0.73, 0.75), 0.95, 0.25),
    "port_black":      ((0.02, 0.02, 0.02), 0.00, 0.55),
    "foam":            ((0.17, 0.17, 0.20), 0.00, 0.92),
    "chip_grey":       ((0.32, 0.33, 0.35), 0.60, 0.35),
    "heatsink":        ((0.55, 0.56, 0.60), 0.90, 0.30),
}
CARD_MATS = {"Fingers": "gold", "Toe": "gold", "PcbLow": "card_pcb", "PcbEdge": "card_pcb",
             "Shroud": "shroud_grey", "Backplate": "backplate_black",
             "Bracket": "bracket_steel", "Io": "port_black"}
CONTACT_OFFSET = 0.0002           # well below the channel's 0.3 mm a side
HOUSING_CONTACT_OFFSET = 0.002


class AssemblyBackend(SortingBackend):
    """The cell from the sorting task (arm, worktop, hall, cameras); the case,
    slot, latch, card and pick station instead of its tray and bins."""
    jaw_open = JAW_OPEN
    settle_s = 1.0

    def __init__(self, card_start: G.Pose, latch_deg: float, start_q, headless=True, spp=4,
                 views=("overview", "top", "wrist", "slot")):
        self.card_start, self.latch_start = card_start, float(latch_deg)
        self.start_q = np.asarray(start_q, float)
        for k, v in JAW_DRIVE.items():         # read by cell.attach_gripper
            setattr(C, k, v)
        self.region = {}                  # collider path -> (owner, region)
        super().__init__(None, headless=headless, spp=spp, views=views)

    # ---- scene -------------------------------------------------------------------
    def _build_scene(self):
        from isaacsim.core.api.materials import OmniPBR
        for name, (diff, metal, rough) in MATERIALS.items():
            pbr = OmniPBR(prim_path=f"/World/Looks/asm_{name}", name=f"asm_{name}",
                          color=np.array(diff))
            pbr.set_metallic_constant(metal)
            pbr.set_reflection_roughness(rough)
            self.cell.MATS[f"asm_{name}"] = pbr.material
        self.cell.camera(self.stage, OVERVIEW_CAM_PATH, OVERVIEW_CAM)
        self.cell.camera(self.stage, SLOT_CAM_PATH, SLOT_CAM)
        self._build_case()
        self._build_slot()
        self._build_latch()
        self._build_holder()
        self._build_card()
        return [CARD_PATH, LATCH_PATH]

    def _sbox(self, path, lo, hi, mat, owner="case", region=None, collision=True):
        """Static box between corners lo..hi given in S."""
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        c = _S((lo + hi) / 2) + [0.0, 0.0, C.TABLE_TOP_Z]
        prim = self.cell.box(self.stage, path, tuple(hi - lo), tuple(c), (0.5, 0.5, 0.5),
                             collision=collision, mat=f"asm_{mat}")
        if collision:
            self._tight(prim)
            self.cell.bind_physics_material(prim)
            self.region[path] = (owner, region or owner)
        return prim

    def _hull(self, path, pts_s, mat, owner, region):
        """Static convex collider through points given in S (also drawn)."""
        from pxr import UsdGeom, UsdPhysics, Gf
        from scipy.spatial import ConvexHull
        pts = np.array([_S(p) + [0.0, 0.0, C.TABLE_TOP_Z] for p in pts_s])
        hull = ConvexHull(pts)
        ctr = pts.mean(axis=0)
        idx = []
        for s in hull.simplices:
            a, b, c = pts[s]
            if np.dot(np.cross(b - a, c - a), a - ctr) < 0:
                s = s[::-1]
            idx += [int(i) for i in s]
        m = UsdGeom.Mesh.Define(self.stage, path)
        m.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in pts])
        m.CreateFaceVertexCountsAttr([3] * len(hull.simplices))
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreateSubdivisionSchemeAttr("none")
        self.cell.bind_mat(m.GetPrim(), f"asm_{mat}")
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr().Set("convexHull")
        self._tight(m.GetPrim())
        self.cell.bind_physics_material(m.GetPrim())
        self.region[path] = (owner, region)

    def _tight(self, prim, offset=None):
        from pxr import PhysxSchema
        pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        pc.CreateContactOffsetAttr().Set(CONTACT_OFFSET if offset is None else offset)
        pc.CreateRestOffsetAttr().Set(0.0)

    def _torsion(self, prim):
        """PhysX's patch friction barely resists twist about the contact
        normal; a pinched card would swing in the jaws. Give the pad a radius."""
        from pxr import PhysxSchema
        pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        pc.CreateTorsionalPatchRadiusAttr().Set(G.TORSION_PATCH_R)
        pc.CreateMinTorsionalPatchRadiusAttr().Set(G.TORSION_PATCH_R / 2)

    def _build_case(self):
        P = f"{ROOT}/Case"
        x0, x1 = G.CASE_X
        y0, y1 = G.CASE_Y
        zb, zt, t = G.CASE_BOTTOM, G.WALL_TOP, G.WALL_T
        self._sbox(f"{P}/SidePanel", (x0, y0, zb), (x1, y1, zb + t), "case_inner", region="case_wall")
        self._sbox(f"{P}/Front", (x0, y0, zb), (x0 + t, y1, zt), "case_steel", region="case_wall")
        self._sbox(f"{P}/Top", (x0, y1 - t, zb), (G.PANEL_X[0], y1, zt), "case_steel", region="case_wall")
        self._sbox(f"{P}/Bottom", (x0, y0, zb), (G.PANEL_X[0], y0 + t, zt), "case_steel", region="case_wall")
        # rear panel round the open expansion slot
        px0, px1 = G.PANEL_X
        oy0, oy1 = G.OPENING_Y
        oz0, oz1 = G.OPENING_Z
        R = f"{P}/RearPanel"
        self._sbox(f"{R}/Below", (px0, y0, zb), (px1, y1, oz0), "case_steel", region="rear_panel")
        self._sbox(f"{R}/Above", (px0, y0, oz1), (px1, y1, zt), "case_steel", region="rear_panel")
        self._sbox(f"{R}/SideLo", (px0, y0, oz0), (px1, oy0, oz1), "case_steel", region="rear_panel")
        self._sbox(f"{R}/SideHi", (px0, oy1, oz0), (px1, y1, oz1), "case_steel", region="rear_panel")
        # closed slot covers next to the opening, and the board's I/O shield (visual)
        for k in range(G.N_COVERS):
            cy = oy0 - (k + 0.5) * G.COVER_PITCH - 0.002
            self._sbox(f"{R}/Cover{k}", (px1, cy - 0.0085, oz0 + 0.004),
                       (px1 + 0.0006, cy + 0.0085, oz1 - 0.004), "bracket_steel", collision=False)
        self._sbox(f"{R}/IoShield", (px1, 0.045, 0.004), (px1 + 0.0006, 0.090, 0.050),
                   "bracket_steel", collision=False)
        # motherboard on its standoffs
        bx0, bx1 = G.BOARD_X
        by0, by1 = G.BOARD_Y
        self._sbox(f"{P}/Board", (bx0, by0, -G.BOARD_T), (bx1, by1, 0.0), "board_pcb", region="board")
        for k, (sx, sy) in enumerate(((bx0 + 0.01, by0 + 0.01), (bx0 + 0.01, by1 - 0.01),
                                      (bx1 - 0.01, by0 + 0.01), (bx1 - 0.01, by1 - 0.01))):
            self._sbox(f"{P}/Standoff{k}", (sx - 0.003, sy - 0.003, zb + t),
                       (sx + 0.003, sy + 0.003, -G.BOARD_T), "bracket_steel", collision=False)
        # what else is on a board (visual, low): CPU socket and cooler plate,
        # DIMM slots, chipset heatsink, an x1 slot
        V = f"{P}/BoardParts"
        self._sbox(f"{V}/Socket", (-0.120, 0.020, 0.0), (-0.075, 0.065, 0.006), "chip_grey", collision=False)
        for k in range(2):
            y = 0.075 + 0.008 * k
            self._sbox(f"{V}/Dimm{k}", (-0.170, y - 0.003, 0.0), (-0.035, y + 0.003, 0.009),
                       "slot_black", collision=False)
        self._sbox(f"{V}/Chipset", (-0.060, -0.090, 0.0), (-0.025, -0.055, 0.010), "heatsink",
                   collision=False)
        self._sbox(f"{V}/SlotX1", (0.010, -0.0405, 0.0), (0.035, -0.0335, 0.011), "slot_black",
                   collision=False)

    def _build_slot(self):
        P = f"{ROOT}/Slot"
        hw, cw, top, fz = G.HOUSING_HW, G.CH_HW, G.HOUSING_TOP, G.FLOOR_Z
        x0, x1 = G.CH_X
        bay0 = G.LATCH_BAY_X[0]
        # floor under the connector; the far end is a pocket for the latch foot
        self._sbox(f"{P}/Floor", (G.FLOOR_X[0], -cw, 0.0), (G.FLOOR_X[1], cw, fz), "slot_black",
                   region="slot_channel")
        # side walls with a flared mouth: straight to top - FUNNEL_H, then out
        for side, s in (("Front", -1), ("Back", +1)):
            prof = [(s * cw, 0.0), (s * hw, 0.0), (s * hw, top), (s * (cw + G.FUNNEL), top),
                    (s * cw, top - G.FUNNEL_H)]
            self._hull(f"{P}/Wall{side}", [(x, y, z) for x in (x0, x1) for y, z in prof],
                       "slot_black", "slot", "slot_channel")
            # the housing's flat top beside the flare: struck by a misaligned card
            ya, yb = sorted((s * (cw + G.FUNNEL), s * hw))
            self._sbox(f"{P}/Top{side}", (x0, ya, top - 0.0004), (x1, yb, top), "slot_black",
                       owner="slot", region="slot_housing")
            # round the latch bay: the hinge's cheeks
            ya, yb = sorted((s * cw, s * hw))
            self._sbox(f"{P}/Cheek{side}", (bay0, ya, 0.0), (x0, yb, top), "slot_black",
                       owner="slot", region="slot_housing")
        # +x end stop, flared
        e0, e1 = x1, x1 + G.END_WALL
        prof = [(e0, 0.0), (e1, 0.0), (e1, top), (e0 + G.FUNNEL, top), (e0, top - G.FUNNEL_H)]
        self._hull(f"{P}/EndWall", [(x, y, z) for x, z in prof for y in (-hw, hw)],
                   "slot_black", "slot", "slot_channel")
        self._sbox(f"{P}/EndTop", (e0 + G.FUNNEL, -hw, top - 0.0004), (e1, hw, top), "slot_black",
                   owner="slot", region="slot_housing")

    def _build_latch(self):
        """A rigid body on a hinge to the world: the PCIe retention latch."""
        from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
        P = LATCH_PATH
        x = UsdGeom.Xform.Define(self.stage, P)
        piv = _S(G.LATCH_PIVOT) + [0.0, 0.0, C.TABLE_TOP_Z]
        x.AddTranslateOp().Set(Gf.Vec3d(*map(float, piv)))
        q = G.quat_y(-self.latch_start)
        x.AddOrientOp().Set(Gf.Quatf(*map(float, q)))
        for name, (region, ctr, size, a) in G.latch_colliders().items():
            prim = self.cell.box(self.stage, f"{P}/{name}", tuple(size), tuple(ctr),
                                 (0.86, 0.86, 0.84), rot=(0.0, -a, 0.0), mat="asm_latch_white")
            self._tight(prim)
            self.cell.bind_physics_material(prim)
            self.region[f"{P}/{name}"] = ("latch", region)
        rb = UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        rb.CreateRigidBodyEnabledAttr().Set(True)
        UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(G.LATCH_MASS)
        px = PhysxSchema.PhysxRigidBodyAPI.Apply(x.GetPrim())
        px.CreateSolverPositionIterationCountAttr().Set(32)
        px.CreateSolverVelocityIterationCountAttr().Set(4)
        px.CreateAngularDampingAttr().Set(2.0)
        PhysxSchema.PhysxContactReportAPI.Apply(x.GetPrim()).CreateThresholdAttr().Set(0.0)
        j = UsdPhysics.RevoluteJoint.Define(self.stage, LATCH_JOINT)
        j.CreateBody1Rel().SetTargets([P])
        j.CreateAxisAttr().Set("Y")
        j.CreateLocalPos0Attr().Set(Gf.Vec3f(*map(float, piv)))
        j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        # the joint frame in the body is the body's own frame, so the joint
        # angle is the body's rotation about +y: -theta
        j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        j.CreateLowerLimitAttr().Set(-G.LATCH_OPEN_DEG)    # joint angle = -theta
        j.CreateUpperLimitAttr().Set(0.0)
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        d.CreateTypeAttr().Set("force")
        d.CreateStiffnessAttr().Set(G.LATCH_SPRING_NM_PER_DEG)
        d.CreateDampingAttr().Set(G.LATCH_DAMP_NMS_PER_DEG)
        d.CreateMaxForceAttr().Set(G.LATCH_MAX_TORQUE)
        self.latch_target = -G.latch_spring_target(self.latch_start)
        d.CreateTargetPositionAttr().Set(self.latch_target)
        self._latch_target_attr = d.GetTargetPositionAttr()

    def _build_holder(self):
        """Foam pick station: a pad and two rails flanking the housing."""
        P = f"{ROOT}/Holder"
        hc = G.HOLDER_C
        lo, sz = G.HOLDER_PAD[0] - G.HOLDER_PAD[1] / 2, G.HOLDER_PAD[1]

        def hbox(name, lo_c, hi_c):
            lo_c, hi_c = np.asarray(lo_c, float), np.asarray(hi_c, float)
            c = hc + (lo_c + hi_c) / 2 + [0.0, 0.0, C.TABLE_TOP_Z]
            prim = self.cell.box(self.stage, f"{P}/{name}", tuple(hi_c - lo_c), tuple(c),
                                 (0.2, 0.2, 0.2), mat="asm_foam")
            self.cell.bind_physics_material(prim)
            self.region[f"{P}/{name}"] = ("holder", "holder")

        hbox("Pad", lo, lo + sz)
        g, (rx0, rx1), (rz0, rz1), t = G.RAIL_GAP, G.RAIL_X, G.RAIL_Z, G.RAIL_T
        hbox("RailBack", (rx0, G.BACKPLATE_Y[0] - g - t, 0.0), (rx1, G.BACKPLATE_Y[0] - g, rz1))
        hbox("RailFan", (rx0, G.SHROUD_Y[1] + g, 0.0), (rx1, G.SHROUD_Y[1] + g + t, rz1))

    def _build_card(self):
        from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
        P = CARD_PATH
        x = UsdGeom.Xform.Define(self.stage, P)
        p = self.card_start.p + [0.0, 0.0, C.TABLE_TOP_Z + 0.0003]
        x.AddTranslateOp().Set(Gf.Vec3d(*map(float, p)))
        x.AddOrientOp().Set(Gf.Quatf(*map(float, self.card_start.q)))
        for name, (region, ctr, size) in G.CARD_COLLIDERS.items():
            prim = self.cell.box(self.stage, f"{P}/{name}", tuple(size), tuple(ctr),
                                 (0.5, 0.5, 0.5), mat=f"asm_{CARD_MATS[name]}")
            if region in G.GRASP_REGIONS:
                # the housing never goes through a tight gap: a normal contact
                # offset keeps the jaws' 180 N pinch from chattering
                self._tight(prim, HOUSING_CONTACT_OFFSET)
                self._torsion(prim)
            else:
                self._tight(prim)
            self.cell.bind_physics_material(prim)
            self.region[f"{P}/{name}"] = ("card", region)
        # dressing: two fans on the shroud's face, a trim strip, ports
        V = f"{P}/Visual"
        UsdGeom.Xform.Define(self.stage, V)
        fy = G.SHROUD_Y[1]
        for k, cx in enumerate((-0.078, 0.018)):
            c = UsdGeom.Cylinder.Define(self.stage, f"{V}/Fan{k}")
            c.CreateAxisAttr("Y")
            c.CreateRadiusAttr(0.040)
            c.CreateHeightAttr(0.0016)
            UsdGeom.Xformable(c.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(cx, fy + 0.0008, 0.068))
            self.cell.bind_mat(c.GetPrim(), "asm_fan_black")
            h = UsdGeom.Cylinder.Define(self.stage, f"{V}/Hub{k}")
            h.CreateAxisAttr("Y")
            h.CreateRadiusAttr(0.014)
            h.CreateHeightAttr(0.0026)
            UsdGeom.Xformable(h.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(cx, fy + 0.0013, 0.068))
            self.cell.bind_mat(h.GetPrim(), "asm_shroud_accent")
        self.cell.box(self.stage, f"{V}/Trim", (0.214, 0.0012, 0.006),
                      (-0.0228, fy + 0.0006, 0.1100), (0.6, 0.6, 0.6), collision=False,
                      mat="asm_shroud_accent")
        for k, (z0, z1) in enumerate(((0.020, 0.036), (0.042, 0.058), (0.064, 0.080))):
            self.cell.box(self.stage, f"{V}/Port{k}", (0.0006, 0.018, z1 - z0),
                          (G.IO_X[1] + 0.0003, 0.016, (z0 + z1) / 2), (0.1, 0.1, 0.1),
                          collision=False, mat="asm_slot_black")
        rb = UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        rb.CreateRigidBodyEnabledAttr().Set(True)
        mass = UsdPhysics.MassAPI.Apply(x.GetPrim())
        mass.CreateMassAttr(G.CARD_MASS)
        mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*map(float, G.CARD_COM)))
        px = PhysxSchema.PhysxRigidBodyAPI.Apply(x.GetPrim())
        px.CreateSolverPositionIterationCountAttr().Set(64)
        px.CreateSolverVelocityIterationCountAttr().Set(4)
        px.CreateLinearDampingAttr().Set(0.5)
        px.CreateAngularDampingAttr().Set(1.0)
        px.CreateEnableCCDAttr().Set(True)
        PhysxSchema.PhysxContactReportAPI.Apply(x.GetPrim()).CreateThresholdAttr().Set(0.0)

    # ---- runtime -------------------------------------------------------------------
    def finalize(self):
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.sensors.camera import Camera
        from pxr import PhysxSchema
        # the tool reports its contacts too (the program's wrist force)
        self.tool_prefixes = (self.cell.JAW_PRIMS["L"], self.cell.JAW_PRIMS["R"],
                              self.cell.FLANGE + "/Gripper")
        for path in (self.cell.JAW_PRIMS["L"], self.cell.JAW_PRIMS["R"], self.cell.FLANGE):
            prim = self.stage.GetPrimAtPath(path)
            if prim.IsValid():
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
        for jaw in ("L", "R"):
            prim = self.stage.GetPrimAtPath(self.cell.JAW_PRIMS[jaw] + "/Tape")
            if prim.IsValid():
                self._torsion(prim)
        from sorting_battery.backend import TOP_CAM_PATH, TOP_CAM
        paths = {"top": (TOP_CAM_PATH, TOP_CAM), "overview": (OVERVIEW_CAM_PATH, OVERVIEW_CAM),
                 "wrist": (self.cell.WRIST_CAM_PATH, C.WRIST_CAM), "slot": (SLOT_CAM_PATH, SLOT_CAM)}
        self.cams = {v: Camera(prim_path=paths[v][0], resolution=paths[v][1]["resolution"])
                     for v in self.views}
        self.cell.enable_ccd(self.world)
        self._friction_anchors()
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
        q[self.arm_idx] = self.start_q
        art.set_joint_positions(q)
        self.items = []
        for k, p in enumerate(self.item_paths):
            rp = SingleRigidPrim(prim_path=p, name=f"asm_item{k}")
            rp.initialize()
            self.items.append(rp)
        self._after_reset()
        n = int(self.settle_s / PHYSICS_DT)
        for k in range(n):
            self.command(self.start_q, 0.0)
            self.world.step(render=k >= n - 30)
            self.last_contacts = self.contacts()

    def _friction_anchors(self):
        """PhysX merges contacts closer than the friction correlation distance
        (25 mm by default) into one friction anchor: a jaw pad then resists
        almost no twist and a pinched card creeps round in the jaws. Spread
        the anchors over the pad."""
        from pxr import PhysxSchema
        prim = self.world.get_physics_context().get_current_physics_scene_prim()
        ps = PhysxSchema.PhysxSceneAPI.Apply(prim)
        ps.CreateFrictionCorrelationDistanceAttr().Set(0.002)
        ps.CreateFrictionOffsetThresholdAttr().Set(0.002)
        print(f"[asm] physics scene {prim.GetPath()}: friction anchors every 2 mm", flush=True)

    def _after_reset(self):
        from omni.physx import get_physx_simulation_interface
        self.card, self.latch = self.items
        self._physx = get_physx_simulation_interface()
        self._paths = {}
        self.last_contacts = []

    def step(self, q6, grip, n, cb=None):
        """n physics steps; after each, cb(t, contacts) with that step's
        contacts (see contacts())."""
        self.command(q6, grip)
        for _ in range(n):
            self.world.step(render=False)
            self.t += self.dt
            self.last_contacts = self.contacts()
            if cb is not None:
                cb(self.t)

    def contacts(self):
        """This physics step's contacts involving the card, the latch or the
        tool: [(owner_a, region_a, owner_b, region_b, force_on_a (3,): normal
        plus friction,
        point (3,), collider_a, collider_b)], forces in N in the robot frame
        (impulse / dt), points in the robot frame. Pairs are listed once with
        a as the card, else the latch, else the tool."""
        headers, data, anchors = self._physx.get_full_contact_report()
        out = []
        for h in headers:
            c0, c1 = self._path(h.collider0), self._path(h.collider1)
            o0, o1 = self._who(c0), self._who(c1)
            if o0 is None and o1 is None:
                continue
            n = h.num_contact_data
            if n == 0:
                continue
            imp = np.zeros(3)
            pt = np.zeros(3)
            for k in range(h.contact_data_offset, h.contact_data_offset + n):
                d = data[k]
                imp += (d.impulse.x, d.impulse.y, d.impulse.z)      # normal
                pt += (d.position.x, d.position.y, d.position.z)
            for k in range(h.friction_anchors_offset,
                           h.friction_anchors_offset + h.num_friction_anchors_data):
                d = anchors[k]
                imp += (d.impulse.x, d.impulse.y, d.impulse.z)      # friction
            f = imp / self.dt                      # PhysX reports the impulse on actor0
            pt = pt / n - [0.0, 0.0, C.TABLE_TOP_Z]
            a, b, fa = (o0, o1, f)
            ca, cb = c0, c1
            if _rank(o1) < _rank(o0):
                a, b, fa, ca, cb = o1, o0, -f, c1, c0
            out.append((a[0], a[1], b[0] if b else "world", b[1] if b else _guess(cb),
                        fa, pt, ca, cb))
        return out

    def _path(self, i):
        p = self._paths.get(i)
        if p is None:
            from pxr import PhysicsSchemaTools
            p = str(PhysicsSchemaTools.intToSdfPath(i))
            self._paths[i] = p
        return p

    def _who(self, path):
        """(owner, region) of a collider: card / latch / case / slot /
        holder from the scene, "tool" for the gripper; None otherwise."""
        r = self.region.get(path)
        if r is not None:
            return r
        if path.startswith(self.tool_prefixes):
            return ("tool", "tool")
        if path.startswith(C.ROBOT_PRIM):
            return ("robot", "robot")
        return None

    def card_pose(self) -> G.Pose:
        p, q = self.card.get_world_pose()
        return G.Pose(np.asarray(p, float) - [0.0, 0.0, C.TABLE_TOP_Z], np.asarray(q, float))

    def latch_deg(self) -> float:
        """theta: 0 CLOSED .. LATCH_OPEN_DEG OPEN, from the body's orientation."""
        _, q = self.latch.get_world_pose()
        w, x, y, z = (float(v) for v in q)
        return -math.degrees(2.0 * math.atan2(y, w))

    def set_latch_target(self, theta_deg):
        """The spring's (or an actuator's) target for the hinge, theta."""
        target = -float(theta_deg)
        if target != self.latch_target:
            self._latch_target_attr.Set(target)
            self.latch_target = target

def _rank(o):
    return {"card": 0, "latch": 1, "tool": 2}.get(o[0] if o else None, 9)


def _guess(path):
    if "/World/Mat" in path or "/World/Table" in path:
        return "table"
    return "other"
