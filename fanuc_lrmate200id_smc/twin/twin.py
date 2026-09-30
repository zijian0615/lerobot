"""Real-time digital twin of the FANUC LR Mate 200iD in MuJoCo (visualisation only, no motion commands).

    # no robot, synthetic motion (macOS: use mjpython for the GUI viewer)
    uv run --no-project --with mujoco mjpython twin/twin.py --source demo
    # against the mock controller / the real robot
    uv run --no-project --with mujoco mjpython twin/twin.py --source rmi --host 172.30.109.22
    # fed by another process that owns the RMI session
    uv run --no-project --with mujoco mjpython twin/twin.py --source udp --udp-port 5005
    # first-time check of the J2/J3 convention against the controller's Cartesian position
    uv run --no-project --with mujoco python twin/twin.py --source rmi --host 172.30.109.22 --check-cartesian
"""
import argparse
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_map import FINGER_OPEN_M, J3_MODES, fanuc_to_model, flange_world_mm, limit_violations, model_to_fanuc  # noqa: E402
from sources import RmiJointSource, add_source_args, make_source  # noqa: E402

SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scene.xml")
STALE_S = 0.5


def fk_check(a, model):
    """Compare controller XYZ with our forward kinematics for both J3 conventions (assumes UTool 0 / UFrame 0)."""
    src = RmiJointSource(a.host, a.port, a.group, init=not a.no_init)
    joints, cart = src.read_cartesian_once()
    print(f"controller joints (deg): {np.round(joints, 3).tolist()}")
    print(f"controller Cartesian   : X={cart[0]:.1f} Y={cart[1]:.1f} Z={cart[2]:.1f} mm  W/P/R={np.round(cart[3:], 2).tolist()}")
    d = mujoco.MjData(model)
    best = None
    for mode in J3_MODES:
        d.qpos[:6] = fanuc_to_model(joints, mode)
        mujoco.mj_forward(model, d)
        p = flange_world_mm(model, d)
        err = float(np.linalg.norm(p - np.array(cart[:3])))
        print(f"  j3-mode {mode:8s}: model flange = ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) mm   error {err:.1f} mm")
        best = min(best or (err, mode), (err, mode))
    print(f"-> best match: --j3-mode {best[1]} (error {best[0]:.1f} mm). Only meaningful if UTool=0 and UFrame=0 are active,")
    print("   and for a pose where J2 != 0 (otherwise both conventions agree). Try a pose with J2 = 30-40 deg.")


def apply(model, data, sample, a, gripper):
    q = fanuc_to_model(sample.joints_deg, a.j3_mode)
    data.qpos[:6] = q
    data.qpos[6:8] = gripper
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return q


def status_light(scn, color):
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.03, 0, 0]), np.array([-0.30, 0.0, 0.95]),
                        np.eye(3).ravel(), np.array(color, dtype=np.float32))
    scn.ngeom += 1


def draw_overlay(viewer, trail, live, stale):
    scn = viewer.user_scn
    scn.ngeom = 0
    status_light(scn, (0.1, 0.9, 0.2, 1) if live else (1.0, 0.6, 0.0, 1) if stale else (0.9, 0.1, 0.1, 1))
    n = len(trail)
    for i, p in enumerate(trail):
        if scn.ngeom >= scn.maxgeom - 1:
            break
        a = 0.15 + 0.85 * (i + 1) / n
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.004, 0, 0]), p, np.eye(3).ravel(),
                            np.array([0.1, 0.6, 1.0, a], dtype=np.float32))
        scn.ngeom += 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_args(ap)
    ap.add_argument("--gripper-mm", type=float, default=FINGER_OPEN_M * 1000, help="finger travel per side to display, 0..10 mm")
    ap.add_argument("--trail", type=int, default=300, help="TCP trail length (0 = off)")
    ap.add_argument("--check-cartesian", action="store_true")
    ap.add_argument("--headless", action="store_true", help="no viewer; print status only")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = run until closed)")
    a = ap.parse_args(argv)

    model = mujoco.MjModel.from_xml_path(SCENE)
    if a.check_cartesian:
        return fk_check(a, model)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    gripper = np.clip(a.gripper_mm / 1000.0, 0.0, FINGER_OPEN_M)

    src = make_source(a)
    if a.source == "rmi":
        print(
            "RMI twin owns the robot session. Stop with Ctrl+C (sends FRC_Abort). "
            "Do not kill the window. For live+viz together use --source udp.",
            flush=True,
        )
    trail, last_print, n_last, frame = [], 0.0, 0, 0
    t_end = time.monotonic() + a.duration if a.duration else None
    printed_raw = False

    def step(viewer=None):
        nonlocal last_print, n_last, frame, printed_raw
        s = src.latest()
        now = time.monotonic()
        live = s is not None and now - s.t_recv < STALE_S
        if s is not None:
            if viewer is not None:
                with viewer.lock():
                    q = apply(model, data, s, a, gripper)
            else:
                q = apply(model, data, s, a, gripper)
            frame += 1
            if a.trail and frame % 2 == 0:
                trail.append(data.site("tcp").xpos.copy())
                del trail[:-a.trail]
        if viewer is not None:
            with viewer.lock():
                draw_overlay(viewer, trail, live, s is not None)
            viewer.sync()
        raw = getattr(src, "first_raw", None)
        if raw is not None and not printed_raw:
            print("first RMI joint response:", raw)
            printed_raw = True
        if now - last_print >= 1.0:
            cnt = getattr(src, "count", 0)
            rate = (cnt - n_last) / max(now - last_print, 1e-6) if last_print else 0.0
            n_last, last_print = cnt, now
            if s is None:
                print(f"[twin] {src.status} | no data yet")
            else:
                warn = limit_violations(model, fanuc_to_model(s.joints_deg, a.j3_mode))
                print(f"[twin] {src.status} | {rate:4.1f} Hz | age {1000 * (now - s.t_recv):4.0f} ms | "
                      f"J(deg)={np.round(s.joints_deg, 1).tolist()} | TCP(mm)={np.round(flange_world_mm(model, data), 0).tolist()}"
                      + (f" | LIMIT? {warn}" if warn else ""))

    try:
        if a.headless:
            while t_end is None or time.monotonic() < t_end:
                step()
                time.sleep(1 / 60)
        else:
            from mujoco import viewer as mj_viewer   # not "import mujoco.viewer": that would shadow `mujoco` locally
            with mj_viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
                v.cam.azimuth, v.cam.elevation, v.cam.distance = 140, -20, 1.9
                v.cam.lookat[:] = (0.25, 0.0, 0.4)
                while v.is_running() and (t_end is None or time.monotonic() < t_end):
                    t0 = time.monotonic()
                    step(v)
                    time.sleep(max(0.0, 1 / 60 - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
    return data


if __name__ == "__main__":
    main()
