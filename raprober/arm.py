"""High-level wrapper around the SO100 follower for scripted pick-and-place.

Adds convenience on top of ``SOFollower``:
* named key poses (home / observe) from config,
* time-streamed joint interpolation (continuous motion, not step-and-wait),
* simple gripper open/close,
* one-shot camera capture returning an RGB frame.
"""

from __future__ import annotations

import logging
import math
import time
from contextlib import contextmanager

import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

from .config import RaproberConfig

logger = logging.getLogger(__name__)

CAMERA_KEY = "wrist"  # observation key for this arm's camera

_CLAMP_WARNING = "Relative goal position magnitude had to be clamped to be safe"


@contextmanager
def _quiet_expected_clamp_warnings():
    """Suppress lerobot clamp warnings during closed-loop moves.

    ``max_relative_target`` intentionally limits each step to ~12 deg. Our
    ``move_to_joints`` / ``set_gripper`` loops re-send the goal until arrival, so
    every iteration triggers a clamp — expected, not an error. lerobot logs it
    via ``logging.warning`` on the root logger, which floods the terminal.
    """

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return _CLAMP_WARNING not in record.getMessage()

    flt = _Filter()
    root = logging.getLogger()
    root.addFilter(flt)
    try:
        yield
    finally:
        root.removeFilter(flt)


class Arm:
    """Scripted controller for a single SO100 arm + its camera."""

    def __init__(self, config: RaproberConfig, *, use_camera: bool = True):
        self.config = config
        self.motor_names = list(config.motor_names)
        self.use_camera = use_camera

        cam_cfg = {}
        if use_camera:
            cam_cfg = {
                CAMERA_KEY: OpenCVCameraConfig(
                    index_or_path=config.camera.index_or_path,
                    width=config.camera.width,
                    height=config.camera.height,
                    fps=config.camera.fps,
                )
            }
        robot_cfg = SO100FollowerConfig(
            port=config.arm_port,
            id=config.arm_id,
            cameras=cam_cfg,
            use_degrees=config.use_degrees,
            max_relative_target=config.max_relative_target,
        )
        self.robot = SO100Follower(robot_cfg)

    # --- lifecycle ------------------------------------------------------- #
    def connect(self, calibrate: bool = True) -> None:
        self.robot.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        self.robot.disconnect()

    def __enter__(self) -> "Arm":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        try:
            self.go_home()
        finally:
            self.disconnect()

    # --- state ----------------------------------------------------------- #
    def get_joints(self) -> dict[str, float]:
        """Return current joint positions as ``{motor: deg}`` (no camera read)."""
        obs = self.robot.bus.sync_read("Present_Position", num_retry=5)
        return {m: float(v) for m, v in obs.items()}

    def read_image(self) -> np.ndarray:
        """Return the latest camera frame as an HxWx3 RGB uint8 array."""
        if not self.use_camera:
            raise RuntimeError("No camera configured on this Arm (use_camera=False).")
        cam = self.robot.cameras[CAMERA_KEY]
        return cam.read_latest()

    # --- motion ---------------------------------------------------------- #
    def _stream_joint_targets(
        self,
        start: dict[str, float],
        targets: dict[str, float],
        *,
        step_dt: float,
        min_steps: int,
    ) -> int:
        """Send linearly interpolated joint waypoints on a fixed timer.

        Does not poll the bus between steps — the arm keeps moving while the
        next command is already queued. Returns the number of steps sent.
        """
        keys = list(targets.keys())
        max_rel = float(self.config.max_relative_target or 25.0)
        max_delta = max(abs(targets[k] - start[k]) for k in keys) if keys else 0.0
        n_steps = max(min_steps, 1, math.ceil(max_delta / max_rel)) if max_delta > 0 else 1

        with _quiet_expected_clamp_warnings():
            for i in range(1, n_steps + 1):
                alpha = i / n_steps
                waypoint = {f"{k}.pos": start[k] + alpha * (targets[k] - start[k]) for k in keys}
                self.robot.send_action(waypoint)
                if i < n_steps:
                    time.sleep(step_dt)
        return n_steps

    def move_to_joints(
        self,
        target: dict[str, float],
        tol_deg: float = 4.0,
        timeout_s: float = 12.0,
        poll_dt: float = 0.04,
        settle_s: float = 0.12,
        step_dt: float | None = None,
        min_steps: int | None = None,
    ) -> float:
        """Move to ``target`` with continuous time-streamed interpolation.

        lerobot's ``max_relative_target`` limits each command to a small delta.
        The old closed-loop "send goal → poll until arrived → repeat" made each
        clamp step fast but left a visible pause between steps while the bus was
        read and tolerance checked.

        Here we stream many small waypoints on a fixed ``step_dt`` timer without
        polling between them, then run a short final convergence loop only if
        needed. Only keys present in ``target`` are moved.
        """
        g = self.config.grasp
        if step_dt is None:
            step_dt = g.motion_step_dt
        if min_steps is None:
            min_steps = g.motion_min_steps

        keys = [k for k in self.motor_names if k in target]
        start = self.get_joints()
        targets = {k: float(target[k]) for k in keys}
        goal = {f"{k}.pos": targets[k] for k in keys}

        n = self._stream_joint_targets(start, targets, step_dt=step_dt, min_steps=min_steps)
        logger.debug("move_to_joints: streamed %d waypoints (step_dt=%.0f ms)", n, step_dt * 1000)

        final_deadline = time.time() + min(timeout_s, g.motion_final_poll_max_s)
        with _quiet_expected_clamp_warnings():
            while time.time() < final_deadline:
                present = self.get_joints()
                max_err = max(abs(present[k] - targets[k]) for k in keys)
                if max_err <= tol_deg:
                    break
                self.robot.send_action(goal)
                time.sleep(poll_dt)

        if settle_s > 0:
            time.sleep(settle_s)
        present = self.get_joints()
        errs = {k: abs(present[k] - targets[k]) for k in keys}
        max_err = max(errs.values()) if errs else 0.0
        worst = max(errs, key=errs.get) if errs else ""
        warn_at = max(tol_deg + 4.0, 8.0)
        if max_err > warn_at:
            logger.warning(
                "move_to_joints: max joint error %.1f deg on %s (> %.1f). "
                "The arm may not have reached the commanded pose.",
                max_err,
                worst,
                warn_at,
            )
        elif max_err > tol_deg:
            logger.debug(
                "move_to_joints: marginal residual %.1f deg on %s (within safe range).",
                max_err,
                worst,
            )
        return max_err

    def move_to_pose_list(self, pose: list[float], **kwargs) -> None:
        """Move using a list ordered like ``config.motor_names``."""
        target = {name: pose[i] for i, name in enumerate(self.motor_names)}
        self.move_to_joints(target, **kwargs)

    def ensure_wrist_roll(self, deg: float, **kwargs) -> None:
        """Hold other joints and drive only ``wrist_roll`` (post-move correction)."""
        kwargs.setdefault("tol_deg", 4.0)
        kwargs.setdefault("settle_s", 0.08)
        self.move_to_joints({"wrist_roll": float(deg)}, **kwargs)

    def go_home(self, *, via_observe: bool | None = None, open_gripper: bool = True, **kwargs) -> None:
        """Move to the folded rest pose.

        When ``via_observe`` is True (default from ``config.home_via_observe``),
        the arm visits ``observe_pose`` first so joint interpolation does not cut
        through the base on the way back from a forward/table reach.

        By default the gripper is opened at home (``home_pose``'s gripper value is
        ignored) so the arm does not look like it is "grasping" again after release.
        """
        if via_observe is None:
            via_observe = self.config.home_via_observe
        if via_observe:
            transit = dict(kwargs)
            transit.setdefault("settle_s", 0.1)
            logger.info("Moving to observe pose first (safe transit to home)")
            self.go_observe(**transit)
        logger.info("Moving to home pose")
        target = {name: self.config.home_pose[i] for i, name in enumerate(self.motor_names)}
        if open_gripper:
            target["gripper"] = self.config.grasp.gripper_open
        self.move_to_joints(target, **kwargs)

    def go_observe(self, **kwargs) -> None:
        logger.info("Moving to observe pose")
        self.move_to_pose_list(self.config.observe_pose, **kwargs)

    # --- torque (for hand-guided calibration) ---------------------------- #
    def disable_torque(self) -> None:
        """Release all motors so the arm can be moved by hand."""
        self.robot.bus.disable_torque()

    def enable_torque(self) -> None:
        """Re-engage motor torque; servos hold their current position."""
        self.robot.bus.enable_torque()

    # --- gripper --------------------------------------------------------- #
    def set_gripper(
        self,
        value: float,
        tol: float = 4.0,
        timeout_s: float = 4.0,
        poll_dt: float = 0.03,
        settle_s: float | None = None,
        step_dt: float | None = None,
    ) -> None:
        """Stream gripper to ``value``, then briefly verify arrival."""
        if settle_s is None:
            settle_s = self.config.grasp.gripper_settle_s
        g = self.config.grasp
        if step_dt is None:
            step_dt = g.motion_step_dt

        start_val = self.get_joints()["gripper"]
        target_val = float(value)
        start = {"gripper": start_val}
        targets = {"gripper": target_val}
        goal = {"gripper.pos": target_val}

        self._stream_joint_targets(start, targets, step_dt=step_dt, min_steps=4)

        present = start_val
        deadline = time.time() + min(timeout_s, g.motion_final_poll_max_s)
        with _quiet_expected_clamp_warnings():
            while time.time() < deadline:
                present = self.get_joints()["gripper"]
                if abs(present - target_val) <= tol:
                    break
                self.robot.send_action(goal)
                time.sleep(poll_dt)
            else:
                logger.warning(
                    "set_gripper: timed out at %.1f (target %.1f)", present, value
                )
        if settle_s > 0:
            time.sleep(settle_s)

    def open_gripper(self, **kwargs) -> None:
        self.set_gripper(self.config.grasp.gripper_open, **kwargs)

    def close_gripper(self, **kwargs) -> None:
        # Cube thickness often blocks full closure; allow ~8 units residual.
        kwargs.setdefault("tol", 8.0)
        self.set_gripper(self.config.grasp.gripper_closed, **kwargs)
