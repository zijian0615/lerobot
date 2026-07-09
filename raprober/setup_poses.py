"""Interactive helper to set ``observe_pose`` / ``home_pose`` / ``grasp.table_z``.

These three values are hardware-specific and must be read from the physical arm.
This tool releases motor torque, lets you hand-guide the arm into position, reads
the joint angles (and the tip height via forward kinematics), then prints the
values — and optionally patches ``config.py`` for you.

Run:  ``python -m raprober.setup_poses --port /dev/ttyACM0 --camera /dev/video0``

Safety: torque is released while you hand-guide the arm; **support the arm** so
it does not fall. Torque is re-engaged (holding the current pose) as soon as you
press ENTER.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import cv2
import numpy as np

from .arm import Arm
from .config import DEFAULT_CONFIG, RaproberConfig
from .kinematics_ik import ArmIK
from .perception import detect_cube, draw_detection

logger = logging.getLogger(__name__)
CONFIG_PATH = Path(__file__).resolve().parent / "config.py"


def _hand_guide_read(arm: Arm, prompt: str) -> dict[str, float]:
    """Release torque, wait for the user to pose the arm, then read joints."""
    print(f"\n{prompt}")
    arm.disable_torque()
    input("  Torque released. Hand-guide the arm, then press ENTER to read...")
    joints = arm.get_joints()
    arm.enable_torque()  # holds the current pose
    return joints


def _pose_list(joints: dict[str, float], motor_names: list[str]) -> list[float]:
    return [round(float(joints[m]), 1) for m in motor_names]


def record_observe(arm: Arm, config: RaproberConfig) -> list[float]:
    joints = _hand_guide_read(
        arm,
        "[observe_pose] Pose the arm so the wrist camera looks DOWN over the whole "
        "cube work area.",
    )
    pose = _pose_list(joints, config.motor_names)
    print(f"  observe_pose = {pose}")
    # Show what the camera sees + whether the cube is detected.
    try:
        img = arm.read_image()
        det = detect_cube(img, config.cube)
        out = CONFIG_PATH.parent / "calib_out" / "observe_view.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), cv2.cvtColor(draw_detection(img, det), cv2.COLOR_RGB2BGR))
        msg = "cube DETECTED" if det else "NO cube detected (that's ok if none is placed)"
        print(f"  Camera view saved to {out} ({msg}).")
    except Exception as e:  # noqa: BLE001 - camera is optional here
        print(f"  (Could not grab camera frame: {e})")
    return pose


def record_grasp_wrist_roll(arm: Arm) -> float:
    joints = _hand_guide_read(
        arm,
        "[grasp_wrist_roll] Pose for a TOP-DOWN grasp: gripper pointing DOWN, "
        "both fingers ready to straddle the cube from the SIDES (not the thin end).",
    )
    roll = round(float(joints["wrist_roll"]), 1)
    print(f"  grasp_wrist_roll_deg = {roll}")
    return roll


def record_home(arm: Arm, config: RaproberConfig) -> list[float]:
    joints = _hand_guide_read(
        arm,
        "[home_pose] Pose the arm into a SAFE folded rest position (out of the "
        "camera's way, won't fall onto the table).",
    )
    pose = _pose_list(joints, config.motor_names)
    print(f"  home_pose = {pose}")
    return pose


def measure_table_z(arm: Arm, config: RaproberConfig) -> float:
    ik = ArmIK(config)
    joints = _hand_guide_read(
        arm,
        "[table_z] Lower the arm so the gripper TIP just touches the table surface "
        "in the grasp area.",
    )
    x, y, z = ik.tip_position(joints)
    print(f"  Tip at base-frame (x={x:.4f}, y={y:.4f}, z={z:.4f}) m  ->  table_z = {z:.4f}")
    return round(float(z), 4)


# --------------------------------------------------------------------------- #
# Patching config.py
# --------------------------------------------------------------------------- #
def _patch_list_field(text: str, field_name: str, values: list[float]) -> str:
    new_list = "[" + ", ".join(str(v) for v in values) + "]"
    pattern = rf"({field_name}: list\[float\] = field\(\s*\n?\s*default_factory=lambda: )\[[^\]]*\]"
    new, n = re.subn(pattern, lambda m: m.group(1) + new_list, text)
    if n == 0:
        raise RuntimeError(f"Could not find field '{field_name}' to patch.")
    return new


def _patch_table_z(text: str, value: float) -> str:
    pattern = r"(table_z: float = )[-+0-9.eE]+"
    new, n = re.subn(pattern, rf"\g<1>{value}", text)
    if n == 0:
        raise RuntimeError("Could not find 'table_z' to patch.")
    return new


def _patch_grasp_wrist_roll(text: str, value: float) -> str:
    pattern = r"(grasp_wrist_roll_deg: float \| None = )[-+0-9.eE]+"
    new, n = re.subn(pattern, rf"\g<1>{value}", text)
    if n == 0:
        raise RuntimeError("Could not find 'grasp_wrist_roll_deg' to patch.")
    return new


def apply_to_config(
    observe: list[float] | None,
    home: list[float] | None,
    table_z: float | None,
    grasp_wrist_roll: float | None = None,
) -> None:
    text = CONFIG_PATH.read_text()
    if observe is not None:
        text = _patch_list_field(text, "observe_pose", observe)
    if home is not None:
        text = _patch_list_field(text, "home_pose", home)
    if table_z is not None:
        text = _patch_table_z(text, table_z)
    if grasp_wrist_roll is not None:
        text = _patch_grasp_wrist_roll(text, grasp_wrist_roll)
    CONFIG_PATH.write_text(text)
    print(f"\nPatched {CONFIG_PATH}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cli() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Set observe/home poses and table_z from the real arm.")
    parser.add_argument("--port", default=DEFAULT_CONFIG.arm_port)
    parser.add_argument("--camera", default=DEFAULT_CONFIG.camera.index_or_path)
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="connect arm only (no OpenCV camera). Use for home/grasp_wrist_roll setup.",
    )
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    config.arm_port = args.port
    if not args.no_camera:
        config.camera.index_or_path = args.camera

    arm = Arm(config, use_camera=not args.no_camera)
    observe = home = table_z = grasp_wrist_roll = None
    arm.connect()
    try:
        while True:
            print(
                "\n=== raprober pose setup ===\n"
                "  1) record observe_pose (needs camera)\n"
                "  2) record home_pose\n"
                "  3) measure table_z\n"
                "  4) record grasp_wrist_roll_deg (top-down, jaws on cube sides)\n"
                "  w) write recorded values into config.py\n"
                "  q) quit"
            )
            choice = input("select> ").strip().lower()
            if choice == "1":
                if not arm.use_camera:
                    print("  observe_pose needs a camera — rerun without --no-camera "
                          f"and pass --camera /dev/video0")
                    continue
                observe = record_observe(arm, config)
            elif choice == "2":
                home = record_home(arm, config)
            elif choice == "3":
                table_z = measure_table_z(arm, config)
            elif choice == "4":
                grasp_wrist_roll = record_grasp_wrist_roll(arm)
            elif choice == "w":
                if observe is None and home is None and table_z is None and grasp_wrist_roll is None:
                    print("  Nothing recorded yet.")
                    continue
                apply_to_config(observe, home, table_z, grasp_wrist_roll)
            elif choice == "q":
                break
            else:
                print("  Unknown option.")
    finally:
        arm.disconnect()

    print("\nSummary:")
    print(f"  observe_pose          = {observe}")
    print(f"  home_pose             = {home}")
    print(f"  table_z               = {table_z}")
    print(f"  grasp_wrist_roll_deg  = {grasp_wrist_roll}")


if __name__ == "__main__":
    _cli()
