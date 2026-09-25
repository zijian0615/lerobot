"""Real-time digital twin of the EMS885 dual-xArm tabletop in NVIDIA Omniverse / Isaac Sim (visualisation only).

Nothing here sends a motion command. Run it with Isaac Sim's Python (it imports isaacsim / pxr from there):

    # no robot: synthetic motion of both arms
    <isaac-sim-python> xarm_ems885_twin/twin/omni_twin.py --source demo
    # fed over UDP by publish_joints.py on the machine that can reach the arms (the Jetson)
    <isaac-sim-python> xarm_ems885_twin/twin/omni_twin.py --source udp --udp-port 5015 --camera top
    # read the arms directly (only where 192.168.1.x / 192.168.2.x are reachable; needs xarm-python-sdk)
    <isaac-sim-python> xarm_ems885_twin/twin/omni_twin.py --source xarm --read-gripper
    # no Isaac Sim, no pxr: source -> joints -> FK -> status only (what the tests use)
    uv run python xarm_ems885_twin/twin/omni_twin.py --source demo --no-render --duration 3

The stage comes from build_usd.py (xarm_ems885.usda + xarm_ems885.twin.json). Each frame only the ":joint" xformOps of the
articulated prims are rewritten (one Sdf.ChangeBlock), so the stage stays a plain USD you can also open in Composer / usdview.
Per arm: a status light above the base (green live, orange data older than 0.5 s, red no data) and a TCP trail.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import add_source_args, make_source  # noqa: E402
from xarm_model import Chain, axis_angle_quat  # noqa: E402

TWIN = Path(__file__).resolve().parent.parent
STAGE = TWIN / "xarm_ems885.usda"
CHAIN = TWIN / "xarm_ems885.twin.json"
STALE_S = 0.5


class NullWriter:
    """No renderer: keeps the last state so the pipeline can be tested without Isaac Sim / pxr."""

    def __init__(self):
        self.joint_values, self.status, self.trails = None, {}, {}

    def set_pose(self, joint_values):
        self.joint_values = joint_values

    def set_status(self, arm, status):
        self.status[arm] = status

    def set_trail(self, arm, points):
        self.trails[arm] = points


class PxrWriter:
    """Writes joint values into the ':joint' xformOps of the open stage (pxr comes with Isaac Sim / Kit)."""

    def __init__(self, stage, chain):
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        self._Gf, self._Sdf, self._UsdShade = Gf, Sdf, UsdShade
        self.stage, self._ops = stage, {}
        for b in chain.bodies:
            if not b["joint"]:
                continue
            prim = stage.GetPrimAtPath(b["path"])
            if not prim.IsValid():
                raise RuntimeError(f"prim {b['path']} not found; rebuild the stage with twin/build_usd.py")
            ops = [op for op in UsdGeom.Xformable(prim).GetOrderedXformOps() if op.GetOpName().endswith(":joint")]
            if len(ops) != 1:
                raise RuntimeError(f"{b['path']} has no unique ':joint' xformOp; rebuild the stage with twin/build_usd.py")
            self._ops[b["joint"]["name"]] = (ops[0], b["joint"]["axis"])
        prims = chain.spec["prims"]
        self._status = {arm: stage.GetPrimAtPath(p) for arm, p in prims["status"].items()}
        self._trail = {arm: UsdGeom.BasisCurves(stage.GetPrimAtPath(p)) for arm, p in prims["trail"].items()}
        self._mats = {(arm, k): UsdShade.Material(stage.GetPrimAtPath(f"/World/Looks/status_{arm}_{k}"))
                      for arm in self._status for k in ("live", "stale", "none")}
        self._status_now = {}

    def set_pose(self, joint_values):
        Gf = self._Gf
        with self._Sdf.ChangeBlock():
            for name, (op, axis) in self._ops.items():
                w, x, y, z = axis_angle_quat(axis, joint_values.get(name, 0.0))
                op.Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))

    def set_status(self, arm, status):
        if self._status_now.get(arm) != status and arm in self._status:
            self._UsdShade.MaterialBindingAPI(self._status[arm]).Bind(self._mats[(arm, status)])
            self._status_now[arm] = status

    def set_trail(self, arm, points):
        if arm not in self._trail:
            return
        Gf = self._Gf
        pts = [Gf.Vec3f(*map(float, p)) for p in points] if len(points) >= 2 else [Gf.Vec3f(0)] * 2
        self._trail[arm].GetPointsAttr().Set(pts)
        self._trail[arm].GetCurveVertexCountsAttr().Set([len(pts)])


def open_isaac(stage_path, headless, camera_path):
    """Start Isaac Sim, open the stage, look through `camera_path`. Returns (app, stage)."""
    try:
        from isaacsim import SimulationApp              # Isaac Sim 4.5+
    except ImportError:
        from omni.isaac.kit import SimulationApp        # Isaac Sim <= 4.2
    app = SimulationApp({"headless": headless})
    import omni.usd

    ctx = omni.usd.get_context()
    ctx.open_stage(str(Path(stage_path).resolve()))
    for _ in range(10):                                 # let the stage load
        app.update()
    if not headless:
        try:
            from omni.kit.viewport.utility import get_active_viewport

            get_active_viewport().camera_path = camera_path
        except Exception as e:  # noqa: BLE001 - cosmetic only
            print(f"[xarm-twin] could not select {camera_path}: {e}")
    return app, ctx.get_stage()


def run(a, chain, src, writer, tick, alive, sleep_s=0.0):
    """Poll the source and drive `writer` until alive() is false. tick() renders one frame."""
    trails = {arm: collections.deque(maxlen=max(a.trail, 2)) for arm in chain.arms}
    state = {arm: (info.get("home_joints") or [0.0] * 6, 0.0) for arm, info in chain.arms.items()}
    seen = set()
    last_print, n_last, frame = 0.0, 0, 0
    while alive():
        t0 = time.monotonic()
        samples = src.latest()
        for arm, s in samples.items():
            if arm in state:
                state[arm] = (s.q, s.drive(state[arm][1]))
                seen.add(arm)
        vals = chain.joint_values(state)
        writer.set_pose(vals)
        frame += 1
        frames = chain.forward(vals) if a.trail and frame % 2 == 0 else None
        for arm in chain.arms:
            s = samples.get(arm)
            status = "none" if s is None else "live" if t0 - s.t_recv < STALE_S else "stale"
            writer.set_status(arm, status)
            if frames is not None and arm in seen:
                trails[arm].append(chain.site(f"{arm}/tcp", vals, frames))
                writer.set_trail(arm, list(trails[arm]))
        if t0 - last_print >= 1.0:
            cnt = getattr(src, "count", 0)
            rate = (cnt - n_last) / max(t0 - last_print, 1e-6) if last_print else 0.0
            n_last, last_print = cnt, t0
            parts = []
            for arm in chain.arms:
                s = samples.get(arm)
                if s is None:
                    parts.append(f"{arm}: no data")
                    continue
                warn = chain.limit_violations(arm, s.q)
                parts.append(f"{arm}: age {1000 * (t0 - s.t_recv):4.0f} ms J(deg)={np.round(np.degrees(s.q), 1).tolist()} "
                             f"flange(mm)={np.round(chain.flange_in_base_mm(arm, vals), 0).tolist()}"
                             + (f" LIMIT? {warn}" if warn else ""))
            print(f"[xarm-twin] {src.status} | {rate:4.1f} upd/s | " + " | ".join(parts), flush=True)
        tick()
        if sleep_s:
            time.sleep(max(0.0, sleep_s - (time.monotonic() - t0)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--stage", type=Path, default=STAGE, help="USD stage built by build_usd.py")
    ap.add_argument("--chain", type=Path, default=CHAIN, help="kinematic chain JSON written next to the stage")
    ap.add_argument("--trail", type=int, default=300, help="TCP trail length per arm (0 = off)")
    ap.add_argument("--camera", choices=("persp", "top"), default="persp",
                    help="viewport camera: 3/4 view from behind the robots, or straight down like the overhead camera")
    ap.add_argument("--headless", action="store_true", help="Isaac Sim without a window")
    ap.add_argument("--no-render", action="store_true", help="no Isaac Sim / pxr: only source -> FK -> status output")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = run until closed)")
    a = ap.parse_args(argv)

    for p in (a.stage, a.chain):
        if not p.exists():
            ap.error(f"{p} not found; generate it with  uv run python xarm_ems885_twin/twin/build_usd.py")
    chain = Chain.load(a.chain)
    if a.no_render:
        app, writer, tick, sleep_s = None, NullWriter(), (lambda: None), 1 / 60
        alive0 = lambda: True  # noqa: E731
    else:
        app, stage = open_isaac(a.stage, a.headless, chain.spec["prims"]["cameras"][a.camera])
        writer, tick, sleep_s = PxrWriter(stage, chain), app.update, 0.0
        alive0 = app.is_running
    src = make_source(a, chain)
    t_end = time.monotonic() + a.duration if a.duration else None
    try:
        run(a, chain, src, writer, tick, lambda: alive0() and (t_end is None or time.monotonic() < t_end), sleep_s)
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        if app is not None:
            app.close()
    return writer


if __name__ == "__main__":
    main()
