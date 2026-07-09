"""Hand-eye calibration: pixel (u, v) -> table (x, y) in the base frame.

The camera is a FIXED overhead camera (eye-to-hand), so a single planar
homography is valid globally for the table plane. The procedure collects >= 4
correspondences between:

    * the cube's pixel centroid seen from the observe pose, and
    * the base-frame (x, y) of the tip when it is placed on that same cube
      (obtained via forward kinematics),

then fits a planar homography H such that

    [x, y, 1]^T ~ H @ [u, v, 1]^T

The homography is saved to ``config.handeye_path`` and consumed at run time by
``pixel_to_table``.

Run:  ``python -m raprober.calibration``  (arm + camera must be connected).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from .arm import Arm
from .config import DEFAULT_CONFIG, RaproberConfig
from .kinematics_ik import ArmIK
from .perception import detect_cube, draw_detection

logger = logging.getLogger(__name__)


def _detect_cube_calib(image_rgb, config: RaproberConfig):
    """Detect cube during calibration — full frame, no ``cube.roi`` crop.

    ``cube.roi`` (default y≥150) drops table regions that project to the upper
    part of the image (negative-y grid corners). At ``observe_pose`` the arm is
    already out of the way, so we can search the whole frame here. Pick-place
    still uses ``cube.roi`` to ignore the arm.
    """
    return detect_cube(image_rgb, replace(config.cube, roi=None))


def _save_calib_sample_image(
    debug_dir: Path,
    tag: str,
    img: np.ndarray,
    det,
    *,
    caption_lines: list[str],
) -> Path:
    """Save an annotated observe-frame image for one calibration point."""
    vis = draw_detection(img, det)
    y = 58 if det is not None else 30
    for line in caption_lines:
        cv2.putText(
            vis,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 22
    out = debug_dir / f"{tag}.png"
    cv2.imwrite(str(out), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return out


def _retract_for_calib_view(
    arm: Arm,
    ik: ArmIK,
    x: float,
    y: float,
    config: RaproberConfig,
    *,
    mode: str | None = None,
) -> None:
    """Move the arm out of the overhead camera FOV so the cube on the table is visible."""
    mode = mode or config.calib_grid.calib_detect_retract
    g = config.grasp
    move_kw = {"tol_deg": 6.0, "settle_s": 0.12, "timeout_s": 20.0}
    if mode == "lift_home":
        print("   Retracting: lift above sample, then home (clear camera view)...")
        current = arm.get_joints()
        target = ik.solve(current, (x, y, g.hover_z), align_down=False)
        target["gripper"] = g.gripper_open
        arm.move_to_joints(target, **move_kw)
        arm.go_home(via_observe=False, open_gripper=True, **move_kw)
    elif mode == "home":
        print("   Retracting to home (clear camera view)...")
        arm.go_home(via_observe=False, open_gripper=True, **move_kw)
    elif mode == "observe":
        print("   Retracting to observe pose...")
        arm.go_observe(**move_kw)
    else:
        raise ValueError(f"Unknown calib_detect_retract mode: {mode!r}")


def _prompt_calib_detect_retry(detect_mode: str) -> str:
    """Return 'retry', 'skip', 'observe', or 'home'."""
    print(
        "   Cube not visible — the arm may still block this table region.\n"
        "   [Enter/r] reposition cube and retry\n"
        "   [o] retry using observe pose (legacy)\n"
        "   [h] retry using home-only retract\n"
        "   [s] skip this grid point"
    )
    choice = input("   > ").strip().lower()
    if choice in ("s", "skip"):
        return "skip"
    if choice in ("o", "observe"):
        return "observe"
    if choice in ("h", "home"):
        return "home"
    return "retry"


def _prompt_retry_skip(prompt: str) -> str:
    """Return ``retry``, ``skip``, or ``abort`` (give up on this point)."""
    choice = input(prompt).strip().lower()
    if choice in ("s", "skip"):
        return "skip"
    if choice in ("n", "no"):
        return "abort"
    return "retry"


# --------------------------------------------------------------------------- #
# Runtime use
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HandEyeCalib:
    """Saved pixel->table homography plus the region where it is trustworthy."""

    H: np.ndarray
    pixel_u_min: float
    pixel_u_max: float
    pixel_v_min: float
    pixel_v_max: float
    table_x_min: float
    table_x_max: float
    table_y_min: float
    table_y_max: float

    @classmethod
    def from_json(cls, data: dict) -> "HandEyeCalib":
        H = np.array(data["H"], dtype=float)
        samples = data.get("samples") or []
        if not samples:
            # Legacy file without samples — assume the whole frame (weak guard).
            return cls(H, 0.0, 640.0, 0.0, 480.0, 0.05, 0.40, -0.15, 0.15)
        us = [float(s["u"]) for s in samples]
        vs = [float(s["v"]) for s in samples]
        xs = [float(s["x"]) for s in samples]
        ys = [float(s["y"]) for s in samples]
        # Small margin inside the convex hull of calibration points.
        pu_margin, pv_margin = 35.0, 30.0
        tx_margin, ty_margin = 0.040, 0.040
        return cls(
            H,
            min(us) - pu_margin,
            max(us) + pu_margin,
            min(vs) - pv_margin,
            max(vs) + pv_margin,
            min(xs) - tx_margin,
            max(xs) + tx_margin,
            min(ys) - ty_margin,
            max(ys) + ty_margin,
        )

    def pixel_in_bounds(self, u: float, v: float) -> bool:
        return (
            self.pixel_u_min <= u <= self.pixel_u_max
            and self.pixel_v_min <= v <= self.pixel_v_max
        )

    def table_in_bounds(self, x: float, y: float) -> bool:
        return (
            self.table_x_min <= x <= self.table_x_max
            and self.table_y_min <= y <= self.table_y_max
        )


def _load_calib_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Hand-eye calibration not found at {p}.\n"
            "Run it first:  uv run python -m raprober.calibration "
            "--port <ARM_PORT> --camera <CAMERA>\n"
            "(spread the cube across the whole work area; press the tip onto the "
            "cube top-center each time.)"
        )
    return json.loads(p.read_text())


def load_handeye(path: str | Path) -> HandEyeCalib:
    """Load homography and the calibrated pixel/table validity region."""
    return HandEyeCalib.from_json(_load_calib_json(path))


def load_homography(path: str | Path) -> np.ndarray:
    """Load the saved 3x3 pixel->table homography."""
    return load_handeye(path).H


def pixel_to_table(u: float, v: float, H: np.ndarray) -> tuple[float, float]:
    """Map a pixel centroid to a base-frame table (x, y) in meters."""
    p = H @ np.array([u, v, 1.0], dtype=float)
    return float(p[0] / p[2]), float(p[1] / p[2])


def save_homography(path: str | Path, H: np.ndarray, observe_pose: list[float], samples: list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"H": H.tolist(), "observe_pose": observe_pose, "samples": samples},
            indent=2,
        )
    )


# --------------------------------------------------------------------------- #
# Interactive calibration
# --------------------------------------------------------------------------- #
def run_calibration(config: RaproberConfig, num_samples: int = 6) -> np.ndarray:
    """Interactive hand-eye calibration. Returns the fitted homography."""
    arm = Arm(config)
    ik = ArmIK(config)

    pixels: list[tuple[float, float]] = []
    world: list[tuple[float, float]] = []
    samples: list[dict] = []
    debug_dir = Path(config.handeye_path).parent / "calib_samples"
    debug_dir.mkdir(parents=True, exist_ok=True)

    arm.connect()
    try:
        print("\n=== raprober hand-eye calibration ===")
        print(f"Collecting {num_samples} correspondences. Vary the cube position each time.")
        print(f"Detection images -> {debug_dir}/\n")
        arm.go_observe()

        i = 0
        while i < num_samples:
            print(f"\n--- sample {i + 1}/{num_samples} ---")
            input("1) Place the black cube in view on the table, then press ENTER to detect...")

            arm.go_observe()  # ensure we detect from the canonical pose
            img = arm.read_image()
            det = _detect_cube_calib(img, config)
            tag = f"manual_{i + 1:02d}"
            status = "KEPT" if det is not None else "NO DETECT"
            if det is not None:
                cap = [
                    f"sample {i + 1}/{num_samples}  [{status}]",
                    f"pixel u={det.u:.0f} v={det.v:.0f} area={det.area:.0f}",
                ]
            else:
                cap = [f"sample {i + 1}/{num_samples}  [{status}]"]
            out = _save_calib_sample_image(debug_dir, tag, img, det, caption_lines=cap)
            print(f"   Saved detection image: {out}")
            if det is None:
                print("   No cube detected — adjust position/lighting and retry this sample.")
                continue
            print(f"   Detected pixel: (u={det.u:.1f}, v={det.v:.1f}), area={det.area:.0f}")

            print(
                "2) Releasing torque. Press the gripper TIP straight down onto the "
                "TOP-CENTER of the cube (same cube every time, tip vertical)."
            )
            arm.disable_torque()
            input("   Press ENTER when the tip is exactly on the cube top-center...")
            joints = arm.get_joints()
            x, y, z = ik.tip_position(joints)
            arm.enable_torque()
            print(f"   Tip base-frame position: x={x:.4f} y={y:.4f} z={z:.4f} (m)")

            keep = input("   Keep this sample? [Y/n] ").strip().lower()
            if keep == "n":
                arm.go_observe()
                continue

            pixels.append((det.u, det.v))
            world.append((x, y))
            samples.append(
                {"u": det.u, "v": det.v, "x": x, "y": y, "z": z, "joints": joints}
            )
            kept = _save_calib_sample_image(
                debug_dir,
                f"manual_{i + 1:02d}_kept",
                img,
                det,
                caption_lines=[
                    f"sample {i + 1}/{num_samples}  [KEPT]",
                    f"pixel u={det.u:.0f} v={det.v:.0f}",
                    f"FK x={x:.4f} y={y:.4f} z={z:.4f}",
                ],
            )
            print(f"   Kept sample image: {kept}")
            i += 1
            arm.go_observe()

        return _fit_validate_save(config, pixels, world, samples, check_z=True)
    finally:
        arm.go_home()
        arm.disconnect()


def _fit_validate_save(
    config: RaproberConfig,
    pixels: list[tuple[float, float]],
    world: list[tuple[float, float]],
    samples: list[dict],
    check_z: bool,
) -> np.ndarray:
    """Validate spread/height, fit the homography, report residuals, save."""
    if len(pixels) < 4:
        raise RuntimeError(f"Need >= 4 valid correspondences, got {len(pixels)}.")

    wx = np.array([w[0] for w in world])
    wy = np.array([w[1] for w in world])
    span_x, span_y = float(np.ptp(wx)), float(np.ptp(wy))
    print("\n--- calibration data quality ---")
    print(f"  world X span: {span_x * 100:.1f} cm    world Y span: {span_y * 100:.1f} cm")
    bad = False
    if span_x < 0.08 or span_y < 0.08:
        print(
            "  WARNING: points are clustered in a tiny area. Spread the cube across "
            "the WHOLE reachable work zone (aim for >=10 cm span in X and Y)."
        )
        bad = True
    if check_z:
        wz = np.array([s["z"] for s in samples])
        span_z = float(np.ptp(wz))
        print(f"  touch Z span: {span_z * 100:.1f} cm (should be small: consistent touch height)")
        if span_z > 0.02:
            print(
                "  WARNING: touch heights vary a lot -> inconsistent pressing. Press the "
                "tip onto the TOP-CENTER of the SAME cube every time (tip vertical)."
            )
            bad = True
    if bad:
        cont = input("  Data looks poor. Save anyway? [y/N] ").strip().lower()
        if cont != "y":
            raise RuntimeError("Calibration aborted due to poor sample quality. Please redo.")

    src = np.array(pixels, dtype=np.float32)
    dst = np.array(world, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, method=0)  # exact/least-squares
    if H is None:
        raise RuntimeError("findHomography failed — collect more/less-collinear points.")

    errs = []
    for (u, v), (gx, gy) in zip(pixels, world, strict=True):
        px, py = pixel_to_table(u, v, H)
        errs.append(float(np.hypot(px - gx, py - gy)))
    print(f"\nFit residual (m): mean={np.mean(errs):.4f} max={np.max(errs):.4f}")
    if np.max(errs) > 0.015:
        print(
            "WARNING: max residual > 15 mm — calibration is not trustworthy. "
            "Redo: move the cube to a NEW spot under the tip at each point; "
            "check diag_detection (green box on cube)."
        )
    elif np.max(errs) > 0.01:
        print("WARNING: max residual > 1 cm. Consider recollecting with more spread-out points.")

    save_homography(config.handeye_path, H, config.observe_pose, samples)
    print(f"Saved homography to {config.handeye_path}")
    return H


def default_grid(config: RaproberConfig) -> list[tuple[float, float]]:
    """Base-frame (x, y) grid for active calibration (see ``config.calib_grid``)."""
    g = config.calib_grid
    return [(x, y) for y in g.ys for x in g.xs]


def run_active_calibration(
    config: RaproberConfig,
    cube_height_m: float = 0.025,
    calib_clearance_m: float | None = None,
) -> np.ndarray:
    """IK-driven hand-eye calibration (guide → retract → detect → return → FK).

    For each grid point:

    1. Arm moves to the grid target (shows you where to put the cube).
    2. You slide the cube under the gripper; **do not move it** after this.
    3. Arm retracts (lift + home) → camera detects cube pixel ``(u, v)``.
    4. Arm returns to the target → FK records tip ``(x, y)``.

    Pixel and FK refer to the same cube position as long as the cube stays put
    from step 2 through step 4.
    """
    arm = Arm(config)
    ik = ArmIK(config)
    grid_cfg = config.calib_grid
    clearance = (
        config.grasp.calib_clearance_m if calib_clearance_m is None else calib_clearance_m
    )
    cube_top_z = config.grasp.table_z + cube_height_m
    approach_z = cube_top_z + clearance
    debug_dir = Path(config.handeye_path).parent / "calib_samples"
    debug_dir.mkdir(parents=True, exist_ok=True)

    pixels: list[tuple[float, float]] = []
    world: list[tuple[float, float]] = []
    samples: list[dict] = []

    grid = default_grid(config)
    arm.connect()
    try:
        wx = [p[0] for p in grid]
        wy = [p[1] for p in grid]
        print("\n=== raprober ACTIVE hand-eye calibration ===")
        print(
            f"{len(grid)} grid points "
            f"(x={min(wx):.2f}..{max(wx):.2f} m, y={min(wy):.2f}..{max(wy):.2f} m), "
            f"approach z={approach_z:.3f} m "
            f"(cube top {cube_top_z:.3f} + clearance {clearance:.3f})."
        )
        print(f"Detection images -> {debug_dir}/")
        print(
            "Per point: arm -> target (guide) -> you place cube -> arm retracts -> "
            "detect -> arm returns -> FK.\n"
            "Do NOT move the cube after step 2 until FK is recorded.\n"
        )
        arm.open_gripper()
        detect_retract = grid_cfg.calib_detect_retract

        for idx, (x, y) in enumerate(grid):
            print(f"\n--- point {idx + 1}/{len(grid)}: target (x={x:.3f}, y={y:.3f}) ---")
            current = arm.get_joints()
            target = ik.solve(current, (x, y, approach_z), align_down=True)
            err = ik.position_error_m(target, (x, y, approach_z))
            if err > 0.02:
                print(f"   SKIP: unreachable (IK residual {err * 1000:.0f} mm).")
                continue

            target["gripper"] = config.grasp.gripper_closed
            tag = f"pt{idx + 1:02d}_x{x:.2f}_y{y:+.2f}"

            while True:
                print("\n   Step 1/4: Moving arm to grid target (guide for cube placement)...")
                reach_err = arm.move_to_joints(target)
                if reach_err > 6.0:
                    print(
                        f"   SKIP: arm did not reach target (max joint error {reach_err:.1f} deg)."
                    )
                    action = _prompt_retry_skip("   Retry? [Y/n/s] ")
                    if action != "retry":
                        break
                    continue

                choice = input(
                    "\n   Step 2/4: Slide cube TOP-CENTER under the gripper, then press ENTER\n"
                    "   (type s + ENTER to skip this point; do NOT move cube after this)..."
                ).strip().lower()
                if choice in ("s", "skip"):
                    print("   Skipping this grid point.")
                    arm.go_home(via_observe=False, open_gripper=True)
                    break

                print("\n   Step 3/4: Retracting arm and detecting cube...")
                _retract_for_calib_view(arm, ik, x, y, config, mode=detect_retract)
                img = arm.read_image()
                det = _detect_cube_calib(img, config)
                if det is not None:
                    cap = [
                        f"pt {idx + 1}/{len(grid)}  STEP3 detect  cmd ({x:.2f},{y:.2f})",
                        f"pixel u={det.u:.0f} v={det.v:.0f} area={det.area:.0f}",
                    ]
                else:
                    cap = [
                        f"pt {idx + 1}/{len(grid)}  STEP3  [NO DETECT]  cmd ({x:.2f},{y:.2f})",
                    ]
                out = _save_calib_sample_image(debug_dir, f"{tag}_detect", img, det, caption_lines=cap)
                print(f"   Saved detect image: {out}")

                if det is None:
                    print("   No cube detected — cube may have moved or arm still blocks view.")
                    action = _prompt_calib_detect_retry(detect_retract)
                    if action == "skip":
                        print("   Skipping this grid point.")
                        break
                    if action in ("observe", "home"):
                        detect_retract = action
                    continue

                print(f"   Detected pixel (u={det.u:.1f}, v={det.v:.1f}), area={det.area:.0f}")
                if det.area < grid_cfg.min_detect_area_px:
                    print(
                        f"   WARNING: area {det.area:.0f} < {grid_cfg.min_detect_area_px} px "
                        "(partial blob or wrong object?)."
                    )
                    action = _prompt_retry_skip("   Retry from step 1? [Y/n/s] ")
                    if action != "retry":
                        if action == "skip":
                            print("   Skipping this grid point.")
                        break
                    continue

                too_close = any(
                    float(np.hypot(det.u - pu, det.v - pv)) < grid_cfg.min_pixel_sep
                    for pu, pv in pixels
                )
                if too_close:
                    print(
                        f"   Cube pixel too close (<{grid_cfg.min_pixel_sep:.0f} px) to a prior "
                        "sample — cube was probably left at the previous spot."
                    )
                    action = _prompt_retry_skip("   Retry from step 1? [Y/n/s] ")
                    if action != "retry":
                        if action == "skip":
                            print("   Skipping this grid point.")
                        break
                    continue

                print("\n   Step 4/4: Returning arm to target and recording FK...")
                reach_err = arm.move_to_joints(target)
                if reach_err > 6.0:
                    print(f"   Arm did not return to target (max joint error {reach_err:.1f} deg).")
                    arm.go_home(via_observe=False, open_gripper=True)
                    action = _prompt_retry_skip("   Retry from step 1? [Y/n/s] ")
                    if action != "retry":
                        if action == "skip":
                            print("   Skipping this grid point.")
                        break
                    continue

                joints = arm.get_joints()
                x_act, y_act, z_act = ik.tip_position(joints)
                cmd_err = float(np.hypot(x_act - x, y_act - y))
                print(
                    f"   FK tip: x={x_act:.4f} y={y_act:.4f} z={z_act:.4f} "
                    f"(cmd {x:.3f},{y:.3f}, delta {cmd_err * 1000:.0f} mm)"
                )
                if cmd_err > 0.035:
                    print("   WARNING: large FK vs command error — check arm tracking.")
                    choice = input("   Keep this sample? [y/N/s] ").strip().lower()
                    if choice in ("s", "skip"):
                        print("   Skipping this grid point.")
                        arm.go_home(via_observe=False, open_gripper=True)
                        break
                    if choice != "y":
                        arm.go_home(via_observe=False, open_gripper=True)
                        continue

                pixels.append((det.u, det.v))
                world.append((x_act, y_act))
                samples.append(
                    {
                        "u": det.u,
                        "v": det.v,
                        "x": x_act,
                        "y": y_act,
                        "z": z_act,
                        "x_cmd": x,
                        "y_cmd": y,
                    }
                )
                kept = _save_calib_sample_image(
                    debug_dir,
                    f"{tag}_kept",
                    img,
                    det,
                    caption_lines=[
                        f"pt {idx + 1}/{len(grid)}  [KEPT]",
                        f"cmd ({x:.2f},{y:.2f})  FK ({x_act:.4f},{y_act:.4f})",
                        f"pixel u={det.u:.0f} v={det.v:.0f}  FK-cmd {cmd_err * 1000:.0f}mm",
                    ],
                )
                print(f"   Kept sample image: {kept}")

                print("   Clearing arm for next point...")
                _retract_for_calib_view(arm, ik, x_act, y_act, config, mode=detect_retract)
                break

        return _fit_validate_save(config, pixels, world, samples, check_z=False)
    finally:
        arm.go_home()
        arm.disconnect()


def _cli() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run raprober hand-eye calibration.")
    parser.add_argument("--port", default=DEFAULT_CONFIG.arm_port)
    parser.add_argument("--camera", default=DEFAULT_CONFIG.camera.index_or_path)
    parser.add_argument(
        "--mode",
        choices=["active", "manual"],
        default="active",
        help="active = IK moves the tip to known points (recommended); "
        "manual = you hand-guide the tip onto the cube.",
    )
    parser.add_argument("--samples", type=int, default=6, help="manual mode: number of points")
    parser.add_argument(
        "--cube-height", type=float, default=0.025, help="active mode: cube height (m)"
    )
    parser.add_argument(
        "--calib-clearance",
        type=float,
        default=None,
        help="active mode: hold tip this far above cube top (m), default from config "
        f"({DEFAULT_CONFIG.grasp.calib_clearance_m})",
    )
    parser.add_argument(
        "--calib-detect-retract",
        choices=["lift_home", "home", "observe"],
        default=None,
        help="how to clear the arm before cube detection (default: lift_home)",
    )
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    config.arm_port = args.port
    config.camera.index_or_path = args.camera
    if args.calib_detect_retract is not None:
        config.calib_grid.calib_detect_retract = args.calib_detect_retract
    if args.mode == "active":
        run_active_calibration(
            config,
            cube_height_m=args.cube_height,
            calib_clearance_m=args.calib_clearance,
        )
    else:
        run_calibration(config, num_samples=args.samples)


if __name__ == "__main__":
    _cli()
