"""Isaac Sim scene of the dynamic-sorting cell, on the sorting backend (same
cell, hall, ESD-mat top, arm, cameras).

The belt is a kinematic rigid body with a PhysX surface velocity: parcels on
it are carried by friction, and a speed change acts through the contact, as on
a real conveyor. Side rails keep them on; past the exit roller they fall into
a catch box out of the arm's reach. Parcels are dynamic boxes waiting off the
table until the feeder puts them on the infeed end at their arrival time.
"""

from __future__ import annotations

import math

import numpy as np

from sorting_battery.backend import SortingBackend, ROOT, LB, C
from . import layout as L
from .textures import OUT as TEX

PARK_X = -1.6                     # waiting parcels: on the hall floor, out of view


class ConveyorBackend(SortingBackend):
    """No worktop, as on a shop floor: the arm on its pedestal, the conveyor
    a free-standing unit on legs, the bins on a rack, a tote on the floor
    under the drop-off. Working heights are those of the tabletop cell, so
    every coordinate in the robot frame is unchanged."""
    table = False
    jaw_open = 0.020              # per jaw: a 40 mm gap for 28 mm parcels
    settle_s = 0.5

    # ---- scene ---------------------------------------------------------------------
    def _build_scene(self, episode=None, line_y=L.LINE_TRUE_Y):
        self.episode = episode
        self._build_conveyor(line_y)
        for name, c in L.BINS.items():
            self._build_sign_bin(name, c)
        return [self._build_parcel(k, p) for k, p in enumerate(episode.parcels)]

    def _tex(self, name):
        from pxr import Sdf
        from isaacsim.core.api.materials import OmniPBR
        key = f"conv_{name}"
        if key not in self._mats:
            pbr = OmniPBR(prim_path=f"/World/Looks/{key}", name=key, color=np.ones(3))
            pbr.set_metallic_constant(0.0)
            pbr.set_reflection_roughness(0.6)
            pbr.set_project_uvw(False)
            pbr.shaders_list[0].CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(str(TEX / f"conv_{name}.png")))
            self._mats[key] = pbr.material
        return self._mats[key]

    def _plain(self, key, rgb, metal=0.0, rough=0.5):
        if key not in self.cell.MATS:
            from isaacsim.core.api.materials import OmniPBR
            pbr = OmniPBR(prim_path=f"/World/Looks/{key}", name=key, color=np.array(rgb))
            pbr.set_metallic_constant(metal)
            pbr.set_reflection_roughness(rough)
            self.cell.MATS[key] = pbr.material
        return key

    def _build_conveyor(self, line_y):
        from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
        P = f"{ROOT}/Conveyor"
        y0, y1 = L.BELT_Y
        ym, ly = (y0 + y1) / 2, y1 - y0
        top = L.BELT_TOP
        frame = self._plain("conv_frame", (0.72, 0.74, 0.76), 0.85, 0.35)   # aluminium profile
        rubber = self._plain("conv_belt", (0.05, 0.05, 0.055), 0.0, 0.8)
        red = self._plain("conv_red", (0.80, 0.03, 0.02), 0.0, 0.5)
        steel = self._plain("conv_roller", (0.75, 0.75, 0.77), 0.9, 0.3)
        floor = -C.TABLE_TOP_Z
        dark = self._plain("conv_dark", (0.10, 0.11, 0.12), 0.3, 0.5)
        blue = self._plain("conv_drive", (0.07, 0.22, 0.48), 0.4, 0.4)
        # bed under the belt, side beams, legs to the floor with cross braces
        self._box(f"{P}/Bed", (L.FRAME_W, ly, 0.04), [L.BELT_X, ym, top - 0.012 - 0.02], frame)
        leg_y = [y0 + 0.06, ym, y1 - 0.06]
        for i, yy in enumerate(leg_y):
            for j, s in enumerate((-1, 1)):
                x = L.BELT_X + s * (L.FRAME_W / 2 - 0.02)
                h = (top - 0.052) - floor
                self._box(f"{P}/Leg{i}{j}", (0.04, 0.04, h), [x, yy, floor + h / 2], frame)
                self._box(f"{P}/Foot{i}{j}", (0.08, 0.08, 0.012), [x, yy, floor + 0.006], dark)
            self._box(f"{P}/Brace{i}", (L.FRAME_W - 0.04, 0.03, 0.03),
                      [L.BELT_X, yy, floor + 0.25], frame, collision=False)
        self._box(f"{P}/Rail", (0.03, ly, 0.03), [L.BELT_X, ym, floor + 0.25], frame, collision=False)
        # drive: gear motor under the infeed end
        self._box(f"{P}/Motor", (0.10, 0.16, 0.10), [L.BELT_X, y1 - 0.12, top - 0.12], blue,
                  collision=False)
        self._box(f"{P}/Gearbox", (0.12, 0.08, 0.08), [L.BELT_X, y1 - 0.02, top - 0.09], dark,
                  collision=False)
        # the belt: kinematic body, surface velocity along -y
        belt = self.cell.box(self.stage, f"{P}/Belt", (L.BELT_W, ly, 0.012),
                             self._w([L.BELT_X, ym, top - 0.006]), (0.05, 0.05, 0.05),
                             collision=True, mat=rubber)
        rb = UsdPhysics.RigidBodyAPI.Apply(belt)
        rb.CreateKinematicEnabledAttr().Set(True)
        self.belt_sv = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(belt)
        self.belt_sv.CreateSurfaceVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        self.cell.bind_physics_material(belt)
        # side rails (static)
        rw = (L.FRAME_W - L.BELT_W) / 2
        for k, s in enumerate((-1, 1)):
            self._box(f"{P}/Rail{k}", (rw, ly, L.RAIL_H + 0.012),
                      [L.BELT_X + s * (L.BELT_W + rw) / 2, ym, top - 0.012 + (L.RAIL_H + 0.012) / 2],
                      frame)
        # exit roller (static collider; parcels ride over it and drop)
        roll = UsdGeom.Cylinder.Define(self.stage, f"{P}/Roller")
        roll.CreateAxisAttr("X")
        roll.CreateRadiusAttr(L.ROLLER_R)
        roll.CreateHeightAttr(L.FRAME_W)
        # a hair below the belt surface, and slippery: a real exit roller turns,
        # so parcels tip off the end instead of piling up on it
        UsdGeom.Xformable(roll).AddTranslateOp().Set(
            Gf.Vec3d(*self._w([L.BELT_X, L.ROLLER_Y, top - L.ROLLER_R - 0.002])))
        UsdPhysics.CollisionAPI.Apply(roll.GetPrim())
        self._bind(roll.GetPrim(), self.cell.MATS[steel])
        slick = UsdPhysics.MaterialAPI.Apply(
            UsdGeom.Scope.Define(self.stage, "/World/Physics/RollerMaterial").GetPrim())
        slick.CreateStaticFrictionAttr().Set(0.05)
        slick.CreateDynamicFrictionAttr().Set(0.05)
        self.cell.bind_physics_material(roll.GetPrim(), "/World/Physics/RollerMaterial")
        # infeed roller, visual
        roll2 = UsdGeom.Cylinder.Define(self.stage, f"{P}/InfeedRoller")
        roll2.CreateAxisAttr("X")
        roll2.CreateRadiusAttr(L.ROLLER_R)
        roll2.CreateHeightAttr(L.FRAME_W)
        UsdGeom.Xformable(roll2).AddTranslateOp().Set(
            Gf.Vec3d(*self._w([L.BELT_X, y1, top - L.ROLLER_R])))
        self._bind(roll2.GetPrim(), self.cell.MATS[steel])
        # the red line: painted across the belt and over both rails (visual only)
        self._box(f"{P}/RedLine", (L.FRAME_W + 0.002, L.LINE_W, 0.0008),
                  [L.BELT_X, line_y, top + 0.0004], red, collision=False)
        for k, s in enumerate((-1, 1)):
            self._box(f"{P}/RedLineRail{k}", (rw + 0.001, L.LINE_W, 0.0008),
                      [L.BELT_X + s * (L.BELT_W + rw) / 2, line_y, top + L.RAIL_H + 0.0004],
                      red, collision=False)
        # a tote on the floor under the drop-off (far out of reach)
        cy = L.CATCH_Y
        cx_, cw, ch = 0.40, 0.30, 0.25
        bx = self._plain("conv_catch", (0.10, 0.25, 0.55), 0.0, 0.6)     # blue plastic tote
        self._box(f"{P}/Catch/Floor", (cx_, cw, 0.01), [L.BELT_X, cy - 0.08, floor + 0.005], bx)
        for k, (dx, dy, sx, sy) in enumerate(((0, cw / 2, cx_, 0.01),
                                               (0, -cw / 2, cx_, 0.01),
                                               (cx_ / 2, 0, 0.01, cw),
                                               (-cx_ / 2, 0, 0.01, cw))):
            self._box(f"{P}/Catch/Wall{k}", (sx, sy, ch),
                      [L.BELT_X + dx, cy - 0.08 + dy, floor + ch / 2], bx)
        self._build_bin_rack()

    def _build_bin_rack(self):
        """A steel rack carrying the three bins at belt-side height."""
        P = f"{ROOT}/BinRack"
        floor = -C.TABLE_TOP_Z
        steel = self._plain("rack_steel", (0.30, 0.32, 0.35), 0.5, 0.45)
        xs = [c[0] for c in L.BINS.values()]
        ys = [c[1] for c in L.BINS.values()]
        half = L.BIN_IN[0] / 2 + L.BIN_WALL + 0.02
        x0, x1 = max(min(xs) - half, 0.15), max(xs) + half     # clear of the pedestal
        y0, y1 = min(ys) - half, max(ys) + half
        self._box(f"{P}/Top", (x1 - x0, y1 - y0, 0.02), [(x0 + x1) / 2, (y0 + y1) / 2, L.SURFACE_Z - 0.01],
                  steel)
        h = (L.SURFACE_Z - 0.02) - floor
        for i, x in enumerate((x0 + 0.02, x1 - 0.02)):
            for j, y in enumerate((y0 + 0.02, y1 - 0.02)):
                self._box(f"{P}/Leg{i}{j}", (0.035, 0.035, h), [x, y, floor + h / 2], steel)
        self._box(f"{P}/Shelf", (x1 - x0, y1 - y0, 0.015), [(x0 + x1) / 2, (y0 + y1) / 2, floor + 0.20],
                  steel, collision=False)

    def _build_sign_bin(self, name, c):
        (ix, iy), w, h = L.BIN_IN, L.BIN_WALL, L.BIN_H
        z0 = L.SURFACE_Z
        P = f"{ROOT}/Bin_{sorted(L.BINS).index(name)}"
        grey = "bin_grey"
        self._box(f"{P}/Floor", (ix + 2 * w, iy + 2 * w, L.BIN_FLOOR),
                  c + [0, 0, z0 + L.BIN_FLOOR / 2], grey)
        for k, (dx, dy, sx, sy) in enumerate((((ix + w) / 2, 0, w, iy + 2 * w),
                                               (-(ix + w) / 2, 0, w, iy + 2 * w),
                                               (0, (iy + w) / 2, ix, w),
                                               (0, -(iy + w) / 2, ix, w))):
            self._box(f"{P}/Wall{k}", (sx, sy, h), c + [dx, dy, z0 + h / 2], grey)
        # sign board leaning back from the -x rim (toward the robot), facing +x and up
        sw, sh = L.SIGN
        tilt = math.radians(50.0)
        base = c + [-(ix / 2 + w), 0.0, z0 + h]
        mid = base + [-math.sin(tilt) * sh / 2, 0.0, math.cos(tilt) * sh / 2]
        n = np.array([math.cos(tilt), 0.0, math.sin(tilt)])
        up = np.array([-math.sin(tilt), 0.0, math.cos(tilt)])
        self._quad(f"{P}/Sign", mid, np.array([0.0, 1.0, 0.0]), up, (sw, sh),
                   self._tex(f"sign_{name}"), world=True)
        self._quad(f"{P}/SignBack", mid - n * 0.0015, np.array([0.0, -1.0, 0.0]), up,
                   (sw + 0.006, sh + 0.006), self.cell.MATS[grey], world=True)
        self._quad(f"{P}/SignFront", mid - n * 0.0005, np.array([0.0, 1.0, 0.0]), up,
                   (sw + 0.006, sh + 0.006), self.cell.MATS[grey], world=True)

    def park_pose(self, k):
        return np.array([PARK_X, -0.9 + 0.12 * k, -C.TABLE_TOP_Z + L.PARCEL[2] / 2 + 0.01])

    def _build_parcel(self, k, p: L.Parcel):
        """A dynamic box: x = length, y = width (the grasp), z = height. The
        top face carries the class letter; the prim name does not."""
        from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
        P = f"{ROOT}/Parcel{k}"
        x = UsdGeom.Xform.Define(self.stage, P)
        x.AddTranslateOp().Set(Gf.Vec3d(*self._w(self.park_pose(k))))
        x.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        lx, ly, lz = L.PARCEL
        col = UsdGeom.Cube.Define(self.stage, f"{P}/Collider")
        col.CreateSizeAttr(1.0)
        UsdGeom.Xformable(col).AddScaleOp().Set(Gf.Vec3f(lx, ly, lz))
        UsdGeom.Imageable(col.GetPrim()).CreatePurposeAttr("guide")
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
        self.cell.bind_physics_material(col.GetPrim())
        rb = UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        rb.CreateRigidBodyEnabledAttr().Set(True)
        UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(L.PARCEL_MASS)
        px = PhysxSchema.PhysxRigidBodyAPI.Apply(x.GetPrim())
        px.CreateSolverPositionIterationCountAttr().Set(16)
        px.CreateSolverVelocityIterationCountAttr().Set(4)
        px.CreateEnableCCDAttr().Set(True)
        px.CreateSleepThresholdAttr().Set(0.0)          # a belt must always move it
        V = f"{P}/Visual"
        UsdGeom.Xform.Define(self.stage, V)
        UsdGeom.Imageable(self.stage.GetPrimAtPath(V)).MakeInvisible()   # shown when fed
        X, Y, Z = (np.array(v, float) for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        kraft = self._tex("kraft")
        e = 0.0002
        self._quad(f"{V}/Top", Z * (lz / 2 + e), X, Y, (lx, ly), self._tex(f"parcel_{p.cls}"))
        self._quad(f"{V}/Bottom", -Z * (lz / 2 + e), X, -Y, (lx, ly), kraft)
        self._quad(f"{V}/PX", X * (lx / 2 + e), Y, Z, (ly, lz), kraft)
        self._quad(f"{V}/NX", -X * (lx / 2 + e), -Y, Z, (ly, lz), kraft)
        self._quad(f"{V}/PY", Y * (ly / 2 + e), -X, Z, (lx, lz), kraft)
        self._quad(f"{V}/NY", -Y * (ly / 2 + e), X, Z, (lx, lz), kraft)
        return P

    # ---- runtime -----------------------------------------------------------------------
    def set_belt_speed(self, v):
        from pxr import Gf
        d = L.DIRECTION * float(v)
        self.belt_sv.GetSurfaceVelocityAttr().Set(Gf.Vec3f(*map(float, d)))

    def take_away(self, k):
        """The outfeed: a missed parcel, past the roller and in the catch box,
        is carried off (back to the waiting area, out of view)."""
        rp = self.items[k]
        rp.set_world_pose(np.array(self._w(self.park_pose(k))), np.array([1.0, 0.0, 0.0, 0.0]))
        rp.set_linear_velocity(np.zeros(3))
        rp.set_angular_velocity(np.zeros(3))
        self._show(k, False)

    def _show(self, k, on):
        from pxr import UsdGeom
        im = UsdGeom.Imageable(self.stage.GetPrimAtPath(f"{self.item_paths[k]}/Visual"))
        im.MakeVisible() if on else im.MakeInvisible()

    def feed(self, k, p: L.Parcel, v):
        """The feeder puts parcel k on the infeed end, moving with the belt."""
        rp = self.items[k]
        q = np.array([math.cos(p.yaw / 2), 0.0, 0.0, math.sin(p.yaw / 2)])
        rp.set_world_pose(np.array(self._w(p.start)), q)
        rp.set_linear_velocity(L.DIRECTION * float(v))
        rp.set_angular_velocity(np.zeros(3))
        self._show(k, True)
