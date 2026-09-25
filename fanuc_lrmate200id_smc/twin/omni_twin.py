"""Real-time digital twin of the FANUC LR Mate 200iD in NVIDIA Omniverse / Isaac Sim (visualisation only, no motion commands).

Run it with Isaac Sim's Python (it imports isaacsim / pxr from there). Same sources and options as twin.py:

    # no robot, synthetic motion
    <isaac-sim>/python.sh twin/omni_twin.py --source demo
    # fed over UDP by the process that owns the RMI session (see udp_relay.py to feed MuJoCo and Omniverse together)
    <isaac-sim>/python.sh twin/omni_twin.py --source udp --udp-port 5007
    # against the mock controller / the real robot
    <isaac-sim>/python.sh twin/omni_twin.py --source rmi --host 172.30.109.22
    # + the real tabletop from the perception pipeline (newest geometric_view.json), seen from straight above,
    # + the overhead camera's picture of the table (build it once with table_texture.py)
    <isaac-sim>/python.sh twin/omni_twin.py --source udp --udp-port 5007 --camera top \
        --scene '../examples/*/runs/*/perceive_*/geometric_view.json' --table-texture table_texture.png
    # no Isaac Sim, no pxr: run source -> joints -> FK -> status only (what the tests use)
    .venv-twin/bin/python twin/omni_twin.py --source demo --no-render --duration 3

The stage comes from build_usd.py (fanuc_lrmate200id_smc.usda + .twin.json). Each frame only the ":joint" xformOps of the
articulated prims are rewritten, so the stage can also be opened, lit and extended in Omniverse as usual.
"""
import argparse
import collections
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_map import FINGER_OPEN_M, fanuc_to_model  # noqa: E402
from sources import add_source_args, make_source  # noqa: E402
from scene import TABLE_THICKNESS_M, Scene, SceneWatcher, load_heights  # noqa: E402
from table_texture import load_texture_meta  # noqa: E402
from usd_chain import Chain, axis_angle_quat  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.join(HERE, "..", "fanuc_lrmate200id_smc.usda")
CHAIN = os.path.join(HERE, "..", "fanuc_lrmate200id_smc.twin.json")
STALE_S = 0.5
CALIB = os.path.join(HERE, "..", "..", "examples", "tabletop_perception", "calib", "fanuc_overhead.json")
STATUS_PRIM, TRAIL_PRIM = "/World/status_light", "/World/trail"
CAMERAS = {"persp": "/World/Camera", "top": "/World/TopCamera"}
GROUND_PRIM, SCENE_ROOT, TABLE_IMAGE_PRIM = "/World/ground", "/World/scene", "/World/table_image"
TEXTURE_LIFT_M = 0.001          # the textured quad floats 1 mm above the table top (no z-fighting)
TEXTURE_GAIN = 0.8              # the lights are bright; a white table would otherwise clip
TABLE_COLOR = (0.42, 0.30, 0.20)
SCENE_POLL_S = 0.5


class NullWriter:
    """No renderer: keeps the last state so the pipeline can be tested without Isaac Sim / pxr."""

    def __init__(self):
        self.joint_values, self.status, self.trail, self.scene, self.table_texture = None, None, None, None, None

    def set_pose(self, joint_values):
        self.joint_values = joint_values

    def set_status(self, status):
        self.status = status

    def set_trail(self, points):
        self.trail = points

    def set_scene(self, scene):
        self.scene = scene

    def set_table_texture(self, png_path, bounds):
        self.table_texture = (png_path, bounds)


class PxrWriter:
    """Writes joint values into the ":joint" xformOps of an open USD stage (pxr comes with Isaac Sim / Kit)."""

    def __init__(self, stage, chain):
        from pxr import Gf, Sdf, UsdGeom, UsdShade
        self._Gf, self._Sdf, self._UsdShade = Gf, Sdf, UsdShade
        self.stage, self._ops = stage, {}
        for name, b in chain.joints.items():
            prim = stage.GetPrimAtPath(b["path"])
            if not prim.IsValid():
                raise RuntimeError(f"prim {b['path']} not found in the stage; rebuild it with twin/build_usd.py")
            ops = [op for op in UsdGeom.Xformable(prim).GetOrderedXformOps() if op.GetOpName().endswith(":joint")]
            if len(ops) != 1:
                raise RuntimeError(f"{b['path']} has no unique ':joint' xformOp; rebuild it with twin/build_usd.py")
            self._ops[name] = (ops[0], b["joint"])
        self._status = stage.GetPrimAtPath(STATUS_PRIM)
        self._trail = UsdGeom.BasisCurves(stage.GetPrimAtPath(TRAIL_PRIM))
        self._mats = {k: UsdShade.Material(stage.GetPrimAtPath(f"/World/Looks/status_{k}")) for k in ("live", "stale", "none")}
        self._status_now = None
        self._ground_op = UsdGeom.Xformable(stage.GetPrimAtPath(GROUND_PRIM)).GetOrderedXformOps()[0]
        self._look_cache = {}

    def _material(self, rgb):
        """UsdPreviewSurface material for a colour, created on first use under /World/Looks."""
        key = tuple(round(float(c), 3) for c in rgb)
        if key not in self._look_cache:
            from pxr import UsdShade
            path = "/World/Looks/scene_%d_%d_%d" % tuple(round(c * 255) for c in key)
            mat = UsdShade.Material.Define(self.stage, path)
            sh = UsdShade.Shader.Define(self.stage, path + "/Shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", self._Sdf.ValueTypeNames.Color3f).Set(self._Gf.Vec3f(*key))
            sh.CreateInput("roughness", self._Sdf.ValueTypeNames.Float).Set(0.6)
            mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            self._look_cache[key] = mat
        return self._look_cache[key]

    def _box(self, name, center, size, yaw, rgb):
        from pxr import UsdGeom, UsdShade
        Gf = self._Gf
        cube = UsdGeom.Cube.Define(self.stage, f"{SCENE_ROOT}/{name}")
        cube.GetSizeAttr().Set(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        cube.AddRotateZOp().Set(float(np.degrees(yaw)))
        cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(self._material(rgb))

    def set_table_texture(self, png_path, bounds):
        """Textured quad on the table top. Texture rows run along +x (top row = x_min), columns along +y (see table_texture.py)."""
        from pxr import UsdGeom, UsdShade
        Gf, Sdf = self._Gf, self._Sdf
        x0, x1, y0, y1 = bounds
        mesh = UsdGeom.Mesh.Define(self.stage, TABLE_IMAGE_PRIM)
        mesh.GetPointsAttr().Set([Gf.Vec3f(x0, y0, TEXTURE_LIFT_M), Gf.Vec3f(x1, y0, TEXTURE_LIFT_M),
                                  Gf.Vec3f(x1, y1, TEXTURE_LIFT_M), Gf.Vec3f(x0, y1, TEXTURE_LIFT_M)])
        mesh.GetFaceVertexCountsAttr().Set([4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        mesh.GetNormalsAttr().Set([Gf.Vec3f(0, 0, 1)] * 4)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set([Gf.Vec2f(0, 1), Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1)])   # (x0,y0) = top-left of the image
        mp = "/World/Looks/table_image"
        mat = UsdShade.Material.Define(self.stage, mp)
        surf = UsdShade.Shader.Define(self.stage, mp + "/Shader")
        surf.CreateIdAttr("UsdPreviewSurface")
        reader = UsdShade.Shader.Define(self.stage, mp + "/st")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex = UsdShade.Shader.Define(self.stage, mp + "/tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(os.path.abspath(png_path)))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(TEXTURE_GAIN, TEXTURE_GAIN, TEXTURE_GAIN, 1))
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
        surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
        mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)

    def set_scene(self, scene):
        """Replace the table slab and object boxes. Table top is z = 0 (the base plate); the ground drops below the slab."""
        from pxr import Tf, UsdGeom
        self.stage.RemovePrim(SCENE_ROOT)
        UsdGeom.Xform.Define(self.stage, SCENE_ROOT)
        x0, x1, y0, y1 = scene.table
        self._box("table", ((x0 + x1) / 2, (y0 + y1) / 2, -TABLE_THICKNESS_M / 2), (x1 - x0, y1 - y0, TABLE_THICKNESS_M), 0.0, TABLE_COLOR)
        for i, o in enumerate(scene.objects):
            ident = Tf.MakeValidIdentifier(o.name)
            self._box(f"obj_{i:02d}_{ident}", (*o.xy, o.size[2] / 2), o.size, o.yaw, o.color)
        self._ground_op.Set(self._Gf.Vec3d(0, 0, -TABLE_THICKNESS_M))

    def set_pose(self, joint_values):
        Gf = self._Gf
        with self._Sdf.ChangeBlock():
            for name, (op, j) in self._ops.items():
                v = joint_values[name]
                if j["type"] == "hinge":
                    w, x, y, z = axis_angle_quat(j["axis"], v)
                    op.Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))
                else:
                    op.Set(Gf.Vec3d(*(float(c) * v for c in j["axis"])))

    def set_status(self, status):
        if status != self._status_now:                 # rebind only on change
            self._UsdShade.MaterialBindingAPI(self._status).Bind(self._mats[status])
            self._status_now = status

    def set_trail(self, points):
        Gf = self._Gf
        pts = [Gf.Vec3f(*map(float, p)) for p in points] if len(points) >= 2 else [Gf.Vec3f(0)] * 2
        self._trail.GetPointsAttr().Set(pts)
        self._trail.GetCurveVertexCountsAttr().Set([len(pts)])


def open_isaac(stage_path, headless, camera="persp"):
    """Start Isaac Sim, open the stage, look through its camera. Returns (app, stage)."""
    try:
        from isaacsim import SimulationApp             # Isaac Sim 4.5+
    except ImportError:
        from omni.isaac.kit import SimulationApp       # Isaac Sim <= 4.2
    app = SimulationApp({"headless": headless})
    import omni.usd
    ctx = omni.usd.get_context()
    ctx.open_stage(os.path.abspath(stage_path))
    for _ in range(10):                                # let the stage load
        app.update()
    if not headless:
        try:
            from omni.kit.viewport.utility import get_active_viewport
            get_active_viewport().camera_path = CAMERAS[camera]
        except Exception as e:                         # noqa: BLE001 - cosmetic only
            print(f"[omni-twin] could not select {CAMERAS[camera]}: {e}")
    return app, ctx.get_stage()


def run(a, chain, src, writer, tick, alive, sleep_s=0.0, scene=None):
    """Poll the source and drive `writer` until `alive()` is false. `tick()` renders one frame.
    `scene` (a scene.SceneWatcher) is polled a couple of times a second and its tabletop shown when it changes."""
    gripper = float(np.clip(a.gripper_mm / 1000.0, 0.0, FINGER_OPEN_M))
    trail = collections.deque(maxlen=max(a.trail, 2))
    last_print, n_last, frame, printed_raw, last_scene = 0.0, 0, 0, False, -SCENE_POLL_S
    while alive():
        t0 = time.monotonic()
        if scene is not None and t0 - last_scene >= SCENE_POLL_S:
            last_scene = t0
            sc = scene.poll()
            if sc is not None:
                writer.set_scene(sc)
                print(f"[omni-twin] scene: {len(sc.objects)} objects from {sc.source}")
            elif scene.error:
                print(f"[omni-twin] scene not loaded ({scene.error}); retrying")
                scene.error = None
        s = src.latest()
        live = s is not None and t0 - s.t_recv < STALE_S
        if s is not None:
            q = fanuc_to_model(s.joints_deg, a.j3_mode)
            vals = chain.joint_values(q, gripper)
            writer.set_pose(vals)
            frame += 1
            if a.trail and frame % 2 == 0:
                trail.append(chain.site("tcp", vals))
                writer.set_trail(list(trail))
        writer.set_status("live" if live else "stale" if s is not None else "none")
        raw = getattr(src, "first_raw", None)
        if raw is not None and not printed_raw:
            print("first RMI joint response:", raw)
            printed_raw = True
        if t0 - last_print >= 1.0:
            cnt = getattr(src, "count", 0)
            rate = (cnt - n_last) / max(t0 - last_print, 1e-6) if last_print else 0.0
            n_last, last_print = cnt, t0
            if s is None:
                print(f"[omni-twin] {src.status} | no data yet")
            else:
                warn = chain.limit_violations(q)
                print(f"[omni-twin] {src.status} | {rate:4.1f} Hz | age {1000 * (t0 - s.t_recv):4.0f} ms | "
                      f"J(deg)={np.round(s.joints_deg, 1).tolist()} | TCP(mm)={np.round(chain.flange_world_mm(vals), 0).tolist()}"
                      + (f" | LIMIT? {warn}" if warn else ""))
        tick()
        if sleep_s:
            time.sleep(max(0.0, sleep_s - (time.monotonic() - t0)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--stage", default=STAGE, help="USD stage built by build_usd.py")
    ap.add_argument("--chain", default=CHAIN, help="kinematic chain JSON written next to the stage")
    ap.add_argument("--gripper-mm", type=float, default=FINGER_OPEN_M * 1000, help="finger travel per side to display, 0..10 mm")
    ap.add_argument("--trail", type=int, default=300, help="TCP trail length (0 = off)")
    ap.add_argument("--scene", action="append", metavar="PATH_OR_GLOB",
                    help="show the tabletop from perception's geometric_view.json; the newest match is followed "
                         "(repeatable), e.g. '../examples/*/runs/*/perceive_*/geometric_view.json'")
    ap.add_argument("--calib", default=CALIB, help="tabletop calibration JSON; its object_top_z_m gives object heights")
    ap.add_argument("--default-height", type=float, default=0.03, help="height [m] of objects whose name matches no calibrated one")
    ap.add_argument("--table-texture", metavar="PNG", help="overhead-camera plate of the table (built by table_texture.py) shown on the table top")
    ap.add_argument("--camera", choices=list(CAMERAS), default="persp", help="viewport camera: 3/4 view like twin.py, or straight down over the table")
    ap.add_argument("--headless", action="store_true", help="Isaac Sim without a window")
    ap.add_argument("--no-render", action="store_true", help="no Isaac Sim / pxr: only source -> FK -> status output")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = run until closed)")
    a = ap.parse_args(argv)

    for p in (a.stage, a.chain):
        if not os.path.exists(p):
            ap.error(f"{p} not found; generate it with  .venv-twin/bin/python twin/build_usd.py")
    chain = Chain.load(a.chain)

    if a.no_render:
        app, writer, tick, sleep_s = None, NullWriter(), (lambda: None), 1 / 60
        alive0 = lambda: True  # noqa: E731
    else:
        app, stage = open_isaac(a.stage, a.headless, a.camera)
        writer, tick, sleep_s = PxrWriter(stage, chain), app.update, 0.0
        alive0 = app.is_running
    src = make_source(a)
    if a.source == "rmi":
        print(
            "RMI twin owns the robot session. Stop with Ctrl+C (sends FRC_Abort). "
            "Do not kill the window. For live+viz together use --source udp.",
            flush=True,
        )
    if a.table_texture:
        bounds, _ = load_texture_meta(a.table_texture)
        writer.set_table_texture(a.table_texture, bounds)
        if not a.scene:                                    # the table slab and ground drop normally come with a scene
            writer.set_scene(Scene(a.table_texture, bounds, ()))
    watcher = SceneWatcher(a.scene, load_heights(a.calib), a.default_height) if a.scene else None
    t_end = time.monotonic() + a.duration if a.duration else None
    try:
        run(a, chain, src, writer, tick, lambda: alive0() and (t_end is None or time.monotonic() < t_end), sleep_s, watcher)
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        if app is not None:
            app.close()
    return writer


if __name__ == "__main__":
    main()
