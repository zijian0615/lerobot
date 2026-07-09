"""Step-by-step diagnostics for the raprober pick-and-place pipeline.

Run each check to isolate WHERE the error is, instead of guessing. Every check
prints ground-truth measurements.

Run:  uv run python -m raprober.diagnose --port /dev/ttyACM0 --camera /dev/video4

Checks
------
1. Joints + FK        : read joints, print tip (x,y,z). Sanity-check against reality.
2. Motion accuracy    : command the observe pose, read back ACTUAL joints, report
                        per-joint tracking error (reveals max_relative_target clamping
                        / undershoot -> the arm not reaching commanded poses).
3. Detection          : capture a frame, detect the cube, save annotated image.
4. Localization check : compare cube position from VISION (pixel->homography) vs from
                        KINEMATICS (hand-guide tip onto cube -> FK). If they disagree,
                        the hand-eye homography is the problem.
"""

from __future__ import annotations

import argparse
import logging

import cv2
import numpy as np

from .arm import Arm
from .calibration import load_homography, pixel_to_table
from .config import DEFAULT_CONFIG, RaproberConfig
from .kinematics_ik import ArmIK
from .perception import detect_cube, draw_detection

logger = logging.getLogger(__name__)


def check_joints_fk(arm: Arm, ik: ArmIK) -> None:
    print("\n=== CHECK 1: joints + forward kinematics ===")
    joints = arm.get_joints()
    print("  joints (deg):", {k: round(v, 1) for k, v in joints.items()})
    x, y, z = ik.tip_position(joints)
    print(f"  FK tip in base frame: x={x:.4f}  y={y:.4f}  z={z:.4f}  (meters)")
    print("  -> Physically eyeball: is the gripper roughly at that (x,y,z) from the base?")
    print("     (x=+forward reach is -y in this URDF; z=height above base plane.)")


def check_motion_accuracy(arm: Arm, config: RaproberConfig) -> None:
    print("\n=== CHECK 2: motion accuracy (commanded vs actual) ===")
    print("  Commanding the OBSERVE pose, then reading back where the arm actually is.")
    target = {name: config.observe_pose[i] for i, name in enumerate(config.motor_names)}
    arm.move_to_joints(target)
    actual = arm.get_joints()
    print(f"  {'joint':<14}{'commanded':>10}{'actual':>10}{'error':>10}")
    max_err = 0.0
    for name in config.motor_names:
        c = float(target[name])
        a = float(actual[name])
        e = a - c
        max_err = max(max_err, abs(e))
        print(f"  {name:<14}{c:>10.1f}{a:>10.1f}{e:>10.1f}")
    print(f"  --> max tracking error: {max_err:.1f} deg")
    if max_err <= 4.0:
        print("  OK: arm tracks commanded joints well.")
    elif max_err <= 6.0:
        print(
            "  OK (marginal): small residual on one joint (often elbow_flex under "
            "gravity). Safe to continue calibration / pick-place."
        )
    else:
        print(
            "  PROBLEM: large tracking error (>6 deg). The arm did not reach the "
            "commanded pose — check collisions, reachability, or increase timeout."
        )


def check_detection(arm: Arm, config: RaproberConfig) -> None:
    print("\n=== CHECK 3: cube detection ===")
    import os

    arm.go_observe()
    img = arm.read_image()
    det = detect_cube(img, config.cube)
    out = os.path.join(os.path.dirname(config.handeye_path), "diag_detection.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, cv2.cvtColor(draw_detection(img, det), cv2.COLOR_RGB2BGR))
    if det is None:
        print(f"  NO cube detected. Annotated frame: {out}")
        print("  -> Check lighting, cube.h_min/h_max/s_min/v_min, and cube.roi (may be cropping the cube).")
    else:
        print(f"  cube pixel (u={det.u:.1f}, v={det.v:.1f}) area={det.area:.0f}")
        print(f"  Annotated frame saved: {out}  -> confirm the box/dot is ON the cube.")


def check_localization(arm: Arm, ik: ArmIK, config: RaproberConfig) -> None:
    print("\n=== CHECK 4: localization (vision vs kinematics) ===")
    try:
        H = load_homography(config.handeye_path)
    except FileNotFoundError as e:
        print(f"  No homography yet: {e}")
        return

    arm.go_observe()
    img = arm.read_image()
    det = detect_cube(img, config.cube)
    if det is None:
        print("  No cube detected — place the cube in view and retry.")
        return
    vx, vy = pixel_to_table(det.u, det.v, H)
    print(f"  VISION says cube is at:      x={vx:.4f}  y={vy:.4f}  (m)")

    print("  Now verify with kinematics: releasing torque.")
    arm.disable_torque()
    input("  Hand-guide the gripper TIP onto the cube top-center, then press ENTER...")
    joints = arm.get_joints()
    kx, ky, kz = ik.tip_position(joints)
    arm.enable_torque()
    print(f"  KINEMATICS says tip is at:   x={kx:.4f}  y={ky:.4f}  z={kz:.4f}  (m)")

    err = float(np.hypot(vx - kx, vy - ky))
    print(f"  --> vision-vs-kinematics disagreement: {err * 1000:.0f} mm")
    if err > 0.020:
        print(
            "  PROBLEM: vision and kinematics disagree (>20 mm). Redo calibration "
            "(active mode) and make sure detection lands on the cube."
        )
    elif err > 0.015:
        print("  OK (marginal): ~15-20 mm — often hand-placement error; pick-place may work.")
    else:
        print("  OK: vision and kinematics agree. Localization is trustworthy.")


def check_grasp_height(arm: Arm, ik: ArmIK, config: RaproberConfig) -> None:
    print("\n=== CHECK 5: grasp-height calibration (table_z) ===")
    print(
        "  table_z is the jaw-frame z (base frame) at grasp height, NOT a physical\n"
        "  floor/table measurement. The 'jaw' tip frame sits well above the fingertips,\n"
        "  so we must READ it via FK in the real grasp pose."
    )
    g = config.grasp
    print(f"  current config: table_z={g.table_z:.4f}  grasp_z_offset={g.grasp_z_offset:.4f}")
    arm.disable_torque()
    input(
        "  Releasing torque. Move the gripper to the REAL grasp pose: fingers straddling\n"
        "  the cube on the table as if grasping it, gripper pointing straight down.\n"
        "  Then press ENTER..."
    )
    joints = arm.get_joints()
    x, y, z = ik.tip_position(joints)
    arm.enable_torque()
    print(f"  FK jaw-frame position: x={x:.4f}  y={y:.4f}  z={z:.4f} (m)")
    suggested_table_z = z - g.grasp_z_offset
    print("  --> at grasp, move_to_xyz(..., grasp_z) uses grasp_z = table_z + offset")
    print(f"      so recommended  table_z = z - grasp_z_offset = {suggested_table_z:.4f}")
    print(f"      (current table_z is {g.table_z:.4f}; delta = {(suggested_table_z - g.table_z) * 1000:.0f} mm)")
    print("  Set this in raprober/config.py -> GraspParams.table_z, then dry-run to verify.")


def _cli() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description="Diagnose the raprober pipeline stage by stage.")
    parser.add_argument("--port", default=DEFAULT_CONFIG.arm_port)
    parser.add_argument("--camera", default=DEFAULT_CONFIG.camera.index_or_path)
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    config.arm_port = args.port
    config.camera.index_or_path = args.camera

    arm = Arm(config)
    ik = ArmIK(config)
    arm.connect()
    try:
        while True:
            print(
                "\n=== raprober diagnostics ===\n"
                "  1) joints + FK\n"
                "  2) motion accuracy (commanded vs actual)\n"
                "  3) cube detection\n"
                "  4) localization: vision vs kinematics\n"
                "  5) grasp-height calibration (table_z)\n"
                "  a) run 1->2->3->4 in order\n"
                "  q) quit"
            )
            c = input("select> ").strip().lower()
            if c == "1":
                check_joints_fk(arm, ik)
            elif c == "2":
                check_motion_accuracy(arm, config)
            elif c == "3":
                check_detection(arm, config)
            elif c == "4":
                check_localization(arm, ik, config)
            elif c == "5":
                check_grasp_height(arm, ik, config)
            elif c == "a":
                check_joints_fk(arm, ik)
                check_motion_accuracy(arm, config)
                check_detection(arm, config)
                check_localization(arm, ik, config)
            elif c == "q":
                break
            else:
                print("  unknown option")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    _cli()
