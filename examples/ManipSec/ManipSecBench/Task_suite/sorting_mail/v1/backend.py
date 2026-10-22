"""Isaac Sim scene of the mail-sorting cell, on the sorting backend (same
cell, hall, ESD-mat worktop, arm, cameras).

The tray is static. Each bin is one kinematic rigid body (floor, walls and its
sign board), so the environment attack can move it mid-run through the
physics API -- moving a static collider by its USD transform would resync
PhysX and break the articulation. Envelopes are dynamic boxes lying face up.
Prims carry no class names: bins are Bin0 (built in the left slot) and Bin1,
envelopes Envelope<k>.
"""

from __future__ import annotations

import math

import numpy as np

from sorting_battery.backend import SortingBackend, ROOT
from . import layout as L
from .textures import path as tex_path


class MailBackend(SortingBackend):
    jaw_open = L.JAW_GAP / 2          # per jaw: a 64 mm gap for 50 mm envelopes
    # gripper force, N: the cell's 180 N (for steel parts) squeezes a 25 g, 10 mm
    # envelope out from between the jaws -- one jaw shoves it against the other
    # and climbs over its edge. A mail gripper runs at low pressure.
    jaw_force = 40.0
    settle_s = 1.0

    def __init__(self, *a, **kw):
        from sorting_battery.backend import C
        C.JAW_MAX_FORCE = self.jaw_force          # read by cell.attach_gripper
        super().__init__(*a, **kw)

    # ---- scene ---------------------------------------------------------------------
    def _build_scene(self, episode=None, signs=None):
        """signs: bin -> the word on its sign (attacks.signs)."""
        self.episode = episode
        self._build_tray()
        self.bin_names = list(L.BINS)                 # Bin<k> holds bin_names[k]
        self.bin_paths = [self._build_bin(k, b, L.SLOTS[L.HOME_SLOT[b]], signs[b])
                          for k, b in enumerate(self.bin_names)]
        return [self._build_envelope(k, e) for k, e in enumerate(episode.envelopes)]

    def _tex(self, name, rough=0.55):
        key = f"tex_{name}"
        if key not in self._mats:
            from pxr import Sdf
            from isaacsim.core.api.materials import OmniPBR
            pbr = OmniPBR(prim_path=f"/World/Looks/{key}", name=key, color=np.ones(3))
            pbr.set_metallic_constant(0.0)
            pbr.set_reflection_roughness(rough)
            pbr.set_project_uvw(False)
            pbr.shaders_list[0].CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(str(tex_path(name))))
            self._mats[key] = pbr.material
        return self._mats[key]

    def _build_tray(self):
        c, (ix, iy), w = L.TRAY, L.TRAY_IN, L.TRAY_WALL
        z0 = L.SURFACE_Z
        P = f"{ROOT}/Tray"
        self._box(f"{P}/Floor", (ix + 2 * w, iy + 2 * w, L.TRAY_FLOOR),
                  c + [0, 0, z0 + L.TRAY_FLOOR / 2], "tray_grey")
        for k, (dx, dy, sx, sy) in enumerate((((ix + w) / 2, 0, w, iy + 2 * w),
                                               (-(ix + w) / 2, 0, w, iy + 2 * w),
                                               (0, (iy + w) / 2, ix, w),
                                               (0, -(iy + w) / 2, ix, w))):
            self._box(f"{P}/Wall{k}", (sx, sy, L.TRAY_H), c + [dx, dy, z0 + L.TRAY_H / 2],
                      "tray_grey")

    def _build_bin(self, k, name, c, word):
        """A kinematic body at c (its floor's centre on the mat); children in
        its local frame. The sign leans back from the -x rim (toward the
        robot), facing +x and up."""
        from pxr import UsdGeom, UsdPhysics, Gf
        P = f"{ROOT}/Bin{k}"
        x = UsdGeom.Xform.Define(self.stage, P)
        x.AddTranslateOp().Set(Gf.Vec3d(*self._w(c)))
        x.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        rb = UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        rb.CreateKinematicEnabledAttr().Set(True)
        (ix, iy), w, h = L.BIN_IN, L.BIN_WALL, L.BIN_H
        z0 = L.SURFACE_Z
        grey = self.cell.MATS["bin_grey"]

        def part(sub, size, centre):
            prim = self.cell.box(self.stage, f"{P}/{sub}", size, tuple(map(float, centre)),
                                 (0.2, 0.2, 0.2), collision=True, mat="bin_grey")
            self.cell.bind_physics_material(prim)
        part("Floor", (ix + 2 * w, iy + 2 * w, L.BIN_FLOOR), [0, 0, z0 + L.BIN_FLOOR / 2])
        for j, (dx, dy, sx, sy) in enumerate((((ix + w) / 2, 0, w, iy + 2 * w),
                                               (-(ix + w) / 2, 0, w, iy + 2 * w),
                                               (0, (iy + w) / 2, ix, w),
                                               (0, -(iy + w) / 2, ix, w))):
            part(f"Wall{j}", (sx, sy, h), [dx, dy, z0 + h / 2])
        sw, sh = L.SIGN
        tilt = math.radians(L.SIGN_TILT_DEG)
        base = np.array([-(ix / 2 + w), 0.0, z0 + h])
        mid = base + [-math.sin(tilt) * sh / 2, 0.0, math.cos(tilt) * sh / 2]
        n = np.array([math.cos(tilt), 0.0, math.sin(tilt)])
        up = np.array([-math.sin(tilt), 0.0, math.cos(tilt)])
        Y = np.array([0.0, 1.0, 0.0])
        self._quad(f"{P}/Sign", mid, Y, up, (sw, sh), self._tex(f"mail_sign_{word}", 0.9))   # matte paper: 0.5 glared
        self._quad(f"{P}/SignBack", mid - n * 0.0015, -Y, up, (sw + 0.006, sh + 0.006), grey)
        self._quad(f"{P}/SignFront", mid - n * 0.0005, Y, up, (sw + 0.006, sh + 0.006), grey)
        return P

    def _build_envelope(self, k, e: L.Envelope):
        """A dynamic box: x = length, y = width (the grasp), z = thickness;
        the marking on its top face."""
        from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
        P = f"{ROOT}/Envelope{k}"
        x = UsdGeom.Xform.Define(self.stage, P)
        x.AddTranslateOp().Set(Gf.Vec3d(*self._w(e.center + [0.0, 0.0, 0.0005])))
        x.AddOrientOp().Set(Gf.Quatf(math.cos(e.yaw / 2), 0.0, 0.0, math.sin(e.yaw / 2)))
        lx, ly, lz = L.ENVELOPE
        col = UsdGeom.Cube.Define(self.stage, f"{P}/Collider")
        col.CreateSizeAttr(1.0)
        UsdGeom.Xformable(col).AddScaleOp().Set(Gf.Vec3f(lx, ly, lz))
        UsdGeom.Imageable(col.GetPrim()).CreatePurposeAttr("guide")
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
        self.cell.bind_physics_material(col.GetPrim())
        rb = UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        rb.CreateRigidBodyEnabledAttr().Set(True)
        UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(L.ENVELOPE_MASS)
        px = PhysxSchema.PhysxRigidBodyAPI.Apply(x.GetPrim())
        px.CreateSolverPositionIterationCountAttr().Set(16)
        px.CreateSolverVelocityIterationCountAttr().Set(4)
        px.CreateEnableCCDAttr().Set(True)
        px.CreateLinearDampingAttr().Set(0.05)
        V = f"{P}/Visual"
        UsdGeom.Xform.Define(self.stage, V)
        X, Y, Z = (np.array(v, float) for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        side = self._plain_mat(f"mail_paper_{e.paper}", tuple(c / 255 for c in L.PAPERS[e.paper]),
                               0.0, 0.7)
        side = self.cell.MATS[side]
        eps = 0.0002
        self._quad(f"{V}/Top", Z * (lz / 2 + eps), X, Y, (lx, ly), self._tex(e.texture))
        self._quad(f"{V}/Bottom", -Z * (lz / 2 + eps), X, -Y, (lx, ly),
                   self._tex(f"mail_back_{e.paper}"))
        self._quad(f"{V}/PX", X * (lx / 2 + eps), Y, Z, (ly, lz), side)
        self._quad(f"{V}/NX", -X * (lx / 2 + eps), -Y, Z, (ly, lz), side)
        self._quad(f"{V}/PY", Y * (ly / 2 + eps), -X, Z, (lx, lz), side)
        self._quad(f"{V}/NY", -Y * (ly / 2 + eps), X, Z, (lx, lz), side)
        return P

    def _after_reset(self):
        from isaacsim.core.prims import SingleRigidPrim
        self.bins = []
        for k, p in enumerate(self.bin_paths):
            rp = SingleRigidPrim(prim_path=p, name=f"bin{k}")
            rp.initialize()
            self.bins.append(rp)

    # ---- runtime -----------------------------------------------------------------------
    def bin_centres(self) -> dict:
        """Bin name -> its centre now (robot frame, z = 0)."""
        out = {}
        for name, rp in zip(self.bin_names, self.bins):
            p, _ = rp.get_world_pose()
            out[name] = np.asarray(p, float) - [0.0, 0.0, self._w((0, 0, 0))[2]]
        return out

    def move_bin(self, name, c):
        """The physics pose (at once) and the prim's transform: the renderer
        reads the latter, and for a kinematic body it is also the target
        PhysX keeps it at."""
        from pxr import UsdGeom, Gf
        k = self.bin_names.index(name)
        self.bins[k].set_world_pose(np.array(self._w(c)), np.array([1.0, 0.0, 0.0, 0.0]))
        ops = {o.GetOpName(): o for o in
               UsdGeom.Xformable(self.stage.GetPrimAtPath(self.bin_paths[k])).GetOrderedXformOps()}
        ops["xformOp:translate"].Set(Gf.Vec3d(*self._w(c)))

    def shift_item(self, k, d):
        """Carry envelope k by d (it rides in a bin that is moved)."""
        rp = self.items[k]
        p, q = rp.get_world_pose()
        rp.set_world_pose(np.asarray(p, float) + np.asarray(d, float), np.asarray(q, float))
        rp.set_linear_velocity(np.zeros(3))
        rp.set_angular_velocity(np.zeros(3))
