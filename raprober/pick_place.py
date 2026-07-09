"""End-to-end scripted pick-and-place of a black cube.

Pipeline
--------
1. Move to the observe pose (or skip with ``--skip-observe`` if the arm already
   clears the fixed overhead camera).
2. Detect the green cube -> pixel centroid.
3. Map pixel -> base-frame table (x, y) via the calibrated homography.
4. IK to hover above the cube, descend, close the gripper.
5. Lift, transport to the fixed place location, open the gripper.
6. Return home.

Run:  ``python -m raprober.pick_place``  (requires prior motor calibration and
a saved hand-eye homography — see README).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .arm import Arm
from .calibration import HandEyeCalib, load_handeye, pixel_to_table
from .config import DEFAULT_CONFIG, RaproberConfig
from .kinematics_ik import ArmIK
from .perception import detect_cube, draw_detection

logger = logging.getLogger(__name__)


class PickPlace:
    """Orchestrates a single black-cube grasp-and-place cycle."""

    def __init__(self, config: RaproberConfig, skip_observe: bool = False):
        self.config = config
        self.skip_observe = skip_observe
        self.arm = Arm(config)
        self.ik = ArmIK(config)
        self.handeye = load_handeye(config.handeye_path)
        self.H = self.handeye.H

    def connect(self) -> None:
        self.arm.connect()

    def disconnect(self) -> None:
        self.arm.disconnect()

    # --- primitives ------------------------------------------------------ #
    def move_to_xyz(
        self,
        x: float,
        y: float,
        z: float,
        gripper: float | None = None,
        max_err_m: float = 0.02,
        align_down: bool = True,
        **move_kw,
    ) -> None:
        """Solve IK for the target tip position and move there.

        Raises ``RuntimeError`` if the IK solution cannot reach within
        ``max_err_m`` of the target. Use ``align_down=False`` for hover / lift /
        transport — forcing the gripper straight down at z≈0.10 m often conflicts
        with exact (x, y) and falsely looks "unreachable".

        ``move_kw`` forwards to ``Arm.move_to_joints`` (e.g. ``tol_deg``, ``settle_s``).
        """
        current = self.arm.get_joints()
        target = self.ik.solve(current, (x, y, z), align_down=align_down)
        err = self.ik.position_error_m(target, (x, y, z))
        if err > max_err_m:
            hint = ""
            if align_down and err > 0.025:
                probe = self.ik.solve(current, (x, y, z), align_down=False)
                probe_err = self.ik.position_error_m(probe, (x, y, z))
                if probe_err <= max_err_m:
                    hint = (
                        " Position is reachable without gripper-down constraint — "
                        "use align_down=False for hover/lift, align_down=True only for grasp."
                    )
            elif not self.handeye.table_in_bounds(x, y):
                hint = (
                    f" Table (x,y) is outside the calibrated region "
                    f"x∈[{self.handeye.table_x_min:.2f},{self.handeye.table_x_max:.2f}] "
                    f"y∈[{self.handeye.table_y_min:.2f},{self.handeye.table_y_max:.2f}] — "
                    "vision/homography extrapolation, not arm reach."
                )
            raise RuntimeError(
                f"IK target ({x:.3f}, {y:.3f}, {z:.3f}) unreachable: residual {err * 1000:.0f} mm "
                f"> {max_err_m * 1000:.0f} mm.{hint}"
            )
        if gripper is not None:
            target["gripper"] = float(gripper)
        roll_deg = self.config.grasp.grasp_wrist_roll_deg
        if align_down and roll_deg is not None:
            target["wrist_roll"] = float(roll_deg)
            # Spin wrist in air before the arm moves down — hover often leaves wrist_roll
            # near ±180° (align_down=False); streaming it together with a z-descent
            # leaves the gripper hitting the cube with the wrong face.
            wrist_kw = dict(move_kw)
            wrist_kw.setdefault("tol_deg", 2.0)
            wrist_kw.setdefault("timeout_s", 20.0)
            wrist_kw.setdefault("settle_s", 0.15)
            self.arm.ensure_wrist_roll(float(roll_deg), **wrist_kw)
        self.arm.move_to_joints(target, **move_kw)
        if align_down and roll_deg is not None:
            wrist_kw = dict(move_kw)
            wrist_kw.setdefault("tol_deg", 2.0)
            wrist_kw.setdefault("timeout_s", 20.0)
            self.arm.ensure_wrist_roll(float(roll_deg), **wrist_kw)
            self._verify_wrist_roll(float(roll_deg))
            self._log_grasp_orientation("grasp orient")

    def _log_grasp_orientation(self, label: str) -> None:
        """FK log: jaw opening direction in the table plane (helps debug side vs end-on)."""
        joints = self.arm.get_joints()
        r = self.ik.forward(joints)[:3, :3]
        jaw_x = r @ np.array([1.0, 0.0, 0.0])
        horiz = float(np.hypot(jaw_x[0], jaw_x[1]))
        if horiz < 1e-3:
            jaw_deg = float("nan")
        else:
            jaw_deg = float(np.degrees(np.arctan2(jaw_x[1], jaw_x[0])))
        logger.info(
            "%s: wrist_roll=%.1f° jaw_open_xy=%.1f° down_align=%.2f",
            label,
            joints["wrist_roll"],
            jaw_deg,
            self.ik.down_alignment(joints),
        )

    def _verify_wrist_roll(self, commanded_deg: float) -> None:
        """Log if the wrist_roll motor did not reach the IK/grasp target."""
        actual = self.arm.get_joints()["wrist_roll"]
        err = abs(actual - commanded_deg)
        tol = self.config.grasp.grasp_wrist_roll_tol_deg
        if err > tol:
            logger.warning(
                "wrist_roll tracking: commanded %.1f deg, actual %.1f deg (error %.1f). "
                "Gripper may still hit the cube with the wrong face — check motor calib "
                "or try grasp_wrist_roll_deg = 90 / -90.",
                commanded_deg,
                actual,
                err,
            )
        else:
            logger.info("wrist_roll at grasp: %.1f deg (target %.1f)", actual, commanded_deg)

    def plan_trajectory(self, cube_xy: tuple[float, float]) -> list[dict]:
        """Compute all waypoints + IK diagnostics for a given cube (x, y).

        Pure kinematics — no hardware, no motion. Returns one dict per waypoint
        with target xyz, IK residual (mm) and downward alignment. Use this
        (``--dry-run``) to validate heights/reachability before powering the arm.
        """
        g = self.config.grasp
        pick_z = g.pick_z
        place_z = g.place_z
        cx, cy = cube_xy
        px, py = self.config.place_xy
        waypoints = [
            ("hover_cube", cx, cy, g.hover_z),
            ("pick_approach", cx, cy, pick_z),
            ("lift", cx, cy, g.lift_z),
            ("transport", px, py, g.lift_z),
            ("place", px, py, place_z),
            ("retreat", px, py, g.hover_z),
        ]
        seed = {n: 0.0 for n in self.config.motor_names}
        plan = []
        for label, x, y, z in waypoints:
            sol = self.ik.solve(seed, (x, y, z))
            res_mm = 1000 * self.ik.position_error_m(sol, (x, y, z))
            plan.append(
                {
                    "label": label,
                    "target": (round(x, 4), round(y, 4), round(z, 4)),
                    "residual_mm": round(res_mm, 1),
                    "down": round(self.ik.down_alignment(sol), 2),
                    "reachable": res_mm <= 20.0,
                    "joints": {k: round(v, 1) for k, v in sol.items()},
                }
            )
            seed = sol  # chain the guess like the real run
        return plan

    def _save_detect_debug(self, img: np.ndarray, det, tag: str) -> None:
        out_dir = Path(self.config.handeye_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"pick_{tag}.png"
        cv2.imwrite(out, cv2.cvtColor(draw_detection(img, det), cv2.COLOR_RGB2BGR))

    def locate_cube(self) -> tuple[float, float] | None:
        """Return the cube's base-frame (x, y) from a camera frame, or None."""
        if self.skip_observe:
            logger.info("Skipping observe pose (--skip-observe); detecting from current arm pose.")
        else:
            self.arm.go_observe()
        img = self.arm.read_image()
        # At observe_pose the arm is out of the way — search the full frame.
        # ``cube.roi`` (y≥150) drops cubes that project to the upper image (v<150).
        det = detect_cube(img, replace(self.config.cube, roi=None))
        if det is None:
            debug = Path(self.config.handeye_path).parent / "pick_no_detect.png"
            cv2.imwrite(debug, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if self.skip_observe:
                logger.warning(
                    "No cube detected at current pose. Saved %s — retry without "
                    "--skip-observe or move the arm so it does not block the camera.",
                    debug,
                )
            else:
                logger.warning(
                    "No cube detected from the observe pose (full-frame search). "
                    "Saved %s — check cube is green, lit, and in view.",
                    debug,
                )
            return None
        self._save_detect_debug(img, det, "observe_detect")
        x, y = pixel_to_table(det.u, det.v, self.H)
        if not self.handeye.pixel_in_bounds(det.u, det.v):
            logger.warning(
                "Cube pixel (%.0f, %.0f) outside calibrated region "
                "(u∈[%.0f,%.0f], v∈[%.0f,%.0f]) — table coords may be wrong.",
                det.u,
                det.v,
                self.handeye.pixel_u_min,
                self.handeye.pixel_u_max,
                self.handeye.pixel_v_min,
                self.handeye.pixel_v_max,
            )
        if not self.handeye.table_in_bounds(x, y):
            logger.warning(
                "Mapped table (%.3f, %.3f) outside calibrated workspace "
                "(x∈[%.2f,%.2f], y∈[%.2f,%.2f]) — re-calibrate or move cube inward.",
                x,
                y,
                self.handeye.table_x_min,
                self.handeye.table_x_max,
                self.handeye.table_y_min,
                self.handeye.table_y_max,
            )
            return None
        logger.info("Cube pixel (%.1f, %.1f) -> table (%.4f, %.4f) m", det.u, det.v, x, y)
        return x, y

    def _apply_pick_offset(self, x: float, y: float) -> tuple[float, float]:
        ox, oy = self.config.pick_xy_offset
        if ox or oy:
            logger.info("Applying pick_xy_offset (%.4f, %.4f) m", ox, oy)
        return x + ox, y + oy

    def refine_xy_at_hover(
        self, x: float, y: float, max_delta_m: float = 0.04
    ) -> tuple[float, float]:
        """Second vision fix at hover height (closed-loop XY if cube still visible)."""
        if not self.config.refine_xy_at_hover:
            return x, y
        img = self.arm.read_image()
        det = detect_cube(img, self.config.cube)
        if det is None:
            logger.info(
                "Hover refine: cube not in frame (arm may occlude); keeping (%.4f, %.4f)",
                x,
                y,
            )
            return x, y
        nx, ny = pixel_to_table(det.u, det.v, self.H)
        if not self.handeye.table_in_bounds(nx, ny):
            logger.warning(
                "Hover refine: table (%.3f, %.3f) outside calib bounds; keeping (%.4f, %.4f)",
                nx,
                ny,
                x,
                y,
            )
            return x, y
        dx, dy = nx - x, ny - y
        dist = float(np.hypot(dx, dy))
        logger.info(
            "Hover refine: pixel (%.0f,%.0f) delta=(%.0f, %.0f) mm -> (%.4f, %.4f)",
            det.u,
            det.v,
            dx * 1000,
            dy * 1000,
            nx,
            ny,
        )
        if dist > max_delta_m:
            logger.warning(
                "Hover refine delta %.0f mm > %.0f mm limit; ignoring (bad detection?)",
                dist * 1000,
                max_delta_m * 1000,
            )
            return x, y
        return nx, ny

    def log_tip_xy_error(self, x: float, y: float, label: str) -> None:
        """Log FK tip (x,y) vs commanded — helps debug pick misses."""
        joints = self.arm.get_joints()
        ax, ay, _ = self.ik.tip_position(joints)
        ex, ey = (ax - x) * 1000, (ay - y) * 1000
        logger.info(
            "%s: commanded (%.4f, %.4f) FK tip (%.4f, %.4f) error (%.0f, %.0f) mm",
            label,
            x,
            y,
            ax,
            ay,
            ex,
            ey,
        )

    def descend_to(
        self,
        x: float,
        y: float,
        z_top: float,
        z_bottom: float,
        gripper: float,
        steps: int,
        **move_kw,
    ) -> None:
        """Vertical approach in ``steps`` increments, streamed without pauses between."""
        steps = max(1, steps)
        move_kw.setdefault("tol_deg", 5.0)
        move_kw.setdefault("settle_s", 0.0)
        move_kw.setdefault("timeout_s", 15.0)
        final_settle = move_kw.pop("final_settle_s", 0.15)
        for i in range(1, steps + 1):
            alpha = i / steps
            z = z_top + (z_bottom - z_top) * alpha
            logger.info("  descend step %d/%d -> z=%.3f m", i, steps, z)
            kw = dict(move_kw)
            if i == steps:
                kw["settle_s"] = final_settle
            kw.setdefault("align_down", True)
            self.move_to_xyz(x, y, z, gripper=gripper, **kw)

    def _fast_move_kw(self) -> dict:
        """Low-latency motion profile for lift / transport / home."""
        g = self.config.grasp
        return {
            "tol_deg": 6.0,
            "settle_s": g.transport_settle_s,
            "timeout_s": 20.0,
            "poll_dt": g.transport_poll_dt,
            "step_dt": g.motion_step_dt,
            "min_steps": g.motion_min_steps,
        }

    def _place_descend_kw(self) -> dict:
        g = self.config.grasp
        return {
            "tol_deg": 6.0,
            "settle_s": g.place_descend_settle_s,
            "timeout_s": 12.0,
            "poll_dt": g.transport_poll_dt,
            "step_dt": g.motion_step_dt,
            "min_steps": g.motion_min_steps,
        }

    # --- full cycle ------------------------------------------------------ #
    def run_once(self) -> bool:
        """Execute one pick-and-place cycle. Returns True on success."""
        g = self.config.grasp
        pick_z = g.pick_z
        place_z = g.place_z
        px, py = self.config.place_xy

        logger.info(
            "Place target: (x=%.3f, y=%.3f) — override with --place-x/--place-y or config.place_xy",
            px,
            py,
        )

        located = self.locate_cube()
        if located is None:
            return False
        cx, cy = self._apply_pick_offset(*located)
        logger.info(
            "Cube target (%.4f, %.4f) | pick z=%.3f | place z=%.3f",
            cx,
            cy,
            pick_z,
            place_z,
        )

        logger.info("Step 1/7: open gripper, hover above cube (z=%.3f)", g.hover_z)
        self.arm.open_gripper()
        self.move_to_xyz(cx, cy, g.hover_z, gripper=g.gripper_open, align_down=False)

        pre_refine = (cx, cy)
        cx, cy = self.refine_xy_at_hover(cx, cy)
        if (cx, cy) != pre_refine:
            logger.info("Step 1b: re-hover after refine -> (%.4f, %.4f)", cx, cy)
            self.move_to_xyz(cx, cy, g.hover_z, gripper=g.gripper_open, align_down=False)

        self.log_tip_xy_error(cx, cy, "After hover")

        if g.grasp_wrist_roll_deg is not None:
            logger.info(
                "Step 1c: pre-orient gripper down + wrist_roll=%.1f° at hover (before descend)",
                g.grasp_wrist_roll_deg,
            )
            self.move_to_xyz(cx, cy, g.hover_z, gripper=g.gripper_open, align_down=True)

        logger.info("Step 2/7: slow descend (%d steps) to pick z=%.3f (open gripper)", g.descend_steps, pick_z)
        self.descend_to(cx, cy, g.hover_z, pick_z, g.gripper_open, g.descend_steps)
        self.log_tip_xy_error(cx, cy, "At pick height")

        logger.info("Step 3/7: close gripper (target=%.0f)", g.gripper_closed)
        self.arm.close_gripper()

        logger.info("Step 4/7: lift to z=%.3f", g.lift_z)
        self.move_to_xyz(
            cx, cy, g.lift_z, gripper=g.gripper_closed, align_down=False, **self._fast_move_kw()
        )

        logger.info("Step 5/7: transport to place (%.4f, %.4f) at z=%.3f", px, py, g.lift_z)
        self.move_to_xyz(
            px, py, g.lift_z, gripper=g.gripper_closed, align_down=False, **self._fast_move_kw()
        )

        logger.info("Step 6/7: lower to place z=%.3f and release", place_z)
        self.move_to_xyz(
            px, py, place_z, gripper=g.gripper_closed, align_down=True, **self._place_descend_kw()
        )
        self.arm.open_gripper()

        logger.info("Step 7/7: lift, then return home")
        self.move_to_xyz(
            px, py, g.lift_z, gripper=g.gripper_open, align_down=False, **self._fast_move_kw()
        )
        self.arm.go_home(**self._fast_move_kw())
        return True


def _cli() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Black-cube pick-and-place.")
    parser.add_argument("--port", default=DEFAULT_CONFIG.arm_port)
    parser.add_argument("--camera", default=DEFAULT_CONFIG.camera.index_or_path)
    parser.add_argument("--place-x", type=float, default=None, help="override place x (m)")
    parser.add_argument("--place-y", type=float, default=None, help="override place y (m)")
    parser.add_argument(
        "--pick-offset-x",
        type=float,
        default=None,
        help="add to vision x before pick (m); + = forward",
    )
    parser.add_argument(
        "--pick-offset-y",
        type=float,
        default=None,
        help="add to vision y before pick (m); + = left",
    )
    parser.add_argument(
        "--no-refine-hover",
        action="store_true",
        help="disable second vision fix at hover height",
    )
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument(
        "--skip-observe",
        action="store_true",
        help="do not move to observe_pose before detection (use when home/rest "
        "already clears the fixed overhead camera)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="plan the trajectory and print IK diagnostics without moving the arm",
    )
    parser.add_argument("--cube-x", type=float, default=None, help="cube x (m) for --dry-run")
    parser.add_argument("--cube-y", type=float, default=None, help="cube y (m) for --dry-run")
    parser.add_argument(
        "--cube-height",
        type=float,
        default=None,
        help="cube height (m); sets place_z = table_z + height + margin",
    )
    args = parser.parse_args()

    config = DEFAULT_CONFIG
    config.arm_port = args.port
    config.camera.index_or_path = args.camera
    if args.cube_height is not None:
        config.grasp.cube_height_m = args.cube_height
    if args.place_x is not None and args.place_y is not None:
        config.place_xy = (args.place_x, args.place_y)
    ox = config.pick_xy_offset[0] if args.pick_offset_x is None else args.pick_offset_x
    oy = config.pick_xy_offset[1] if args.pick_offset_y is None else args.pick_offset_y
    config.pick_xy_offset = (ox, oy)
    if args.no_refine_hover:
        config.refine_xy_at_hover = False

    # Offline planning: no hardware needed if a cube (x, y) is supplied.
    if args.dry_run and args.cube_x is not None and args.cube_y is not None:
        from .kinematics_ik import ArmIK

        planner = PickPlace.__new__(PickPlace)
        planner.config = config
        planner.ik = ArmIK(config)
        print(f"\nDry-run plan for cube=({args.cube_x}, {args.cube_y}), place={config.place_xy}:")
        print(f"{'step':<12}{'target (x,y,z)':<28}{'res(mm)':>9}{'down':>7}  reach")
        for wp in planner.plan_trajectory((args.cube_x, args.cube_y)):
            print(
                f"{wp['label']:<12}{str(wp['target']):<28}{wp['residual_mm']:>9}"
                f"{wp['down']:>7}  {'OK' if wp['reachable'] else 'UNREACHABLE'}"
            )
        return

    task = PickPlace(config, skip_observe=args.skip_observe)
    task.connect()
    try:
        for c in range(args.cycles):
            logger.info("=== cycle %d/%d ===", c + 1, args.cycles)
            ok = task.run_once()
            if not ok:
                logger.warning("Cycle %d aborted (no cube).", c + 1)
                task.arm.go_home()
    finally:
        task.disconnect()


if __name__ == "__main__":
    _cli()
