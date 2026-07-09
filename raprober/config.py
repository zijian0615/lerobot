"""Central configuration for the raprober black-cube pick-and-place tool.

Everything hardware- or task-specific lives here so the rest of the code stays
generic. Adjust the values below to match your physical setup, then run the
calibration script (``calibration.py``) once to generate the pixel->table
homography.

Coordinate conventions
----------------------
* Joint positions are in **degrees** (``use_degrees=True`` on the follower).
* End-effector / table positions are in **meters**, expressed in the robot
  **base frame** as returned by the SO100 forward kinematics (tip frame ``jaw``).
* The table is assumed to be a flat horizontal plane at a fixed height
  ``table_z`` in the base frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
RAPROBER_DIR = Path(__file__).resolve().parent
# IMPORTANT: use the "new_calib" URDF. Its joint zeros/signs/limits match the
# lerobot motor-calibration convention (verified: real home pose sits right at
# these joint limits). The old Simulation/SO100/so100.urdf uses a DIFFERENT joint
# convention (e.g. shoulder_lift limit [0, 200] deg) that does NOT match the
# motors, which corrupts FK/IK and makes the arm move to contorted/wrong poses.
URDF_PATH = RAPROBER_DIR / "assets" / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
CALIBRATION_DIR = RAPROBER_DIR / "calib_out"
HANDEYE_PATH = CALIBRATION_DIR / "handeye_homography.json"
# Standard SO100/SO101 motor ordering (matches FeetechMotorsBus in SOFollower).
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
# End-effector frame in the new_calib URDF (lerobot's standard EE frame). Its
# local +z axis points along the grasp approach (straight down in a top-down grasp).
TIP_FRAME_NAME = "gripper_frame_link"


@dataclass
class CameraSettings:
    """OpenCV camera parameters for the arm that performs the grasp."""

    index_or_path: str = "/dev/video4"
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class GreenCubeParams:
    """Thresholds for segmenting a GREEN cube on a light-colored table.

    Detection is done in HSV: a saturated green object occupies a hue band with
    high saturation and value, so we keep pixels inside
    ``[h_min, s_min, v_min] .. [h_max, s_max, v_max]``. OpenCV hue is 0-179;
    green sits around 40-80. Widen the hue band or lower ``s_min``/``v_min`` if
    the cube is missed; tighten them if background gets picked up. A dark (black)
    arm/gripper and red/blue tape are naturally rejected (wrong hue/low sat).
    """

    h_min: int = 35  # min hue for green
    h_max: int = 85  # max hue for green
    s_min: int = 70  # min saturation (reject washed-out / pale wood)
    v_min: int = 50  # min value (reject dark shadows)
    s_max: int = 255
    v_max: int = 255
    min_area_px: int = 300  # ignore blobs smaller than this (noise)
    max_area_px: int = 20_000  # ignore huge blobs
    morph_kernel: int = 5  # morphological open/close kernel size
    # Region of interest (x, y, w, h) in pixels. The green cube sits in the LOWER
    # part of the fixed overhead frame, and the (dark) arm sits at the top, so we
    # keep the lower band. Green segmentation already rejects the arm, so this is
    # mostly a safety crop. ``None`` = whole frame. Tune to your framing.
    roi: tuple[int, int, int, int] | None = (0, 150, 640, 330)


@dataclass
class CalibGridParams:
    """Active hand-eye calibration grid in base-frame meters (+x forward, +y left).

    Wider spacing spreads samples in pixel space (overhead camera compresses rows).
    All points must stay reachable by IK and visible from the calibration detect
    retract pose (see ``calib_detect_retract`` — usually ``lift_home``).
    """

    xs: list[float] = field(default_factory=lambda: [0.10, 0.17, 0.24, 0.30])
    ys: list[float] = field(default_factory=lambda: [-0.10, -0.03, 0.05, 0.12])
    min_pixel_sep: float = 30.0  # reject duplicate cube placement (px)
    min_detect_area_px: int = 600  # warn / retry if blob smaller than this
    # How to move the arm out of the overhead camera view before cube detection.
    # ``lift_home`` (default): lift to hover_z above the sample, then fold to
    # home — clears the table even when ``observe_pose`` still occludes a side.
    # ``observe``: legacy behaviour (observe_pose only). ``home``: fold to home
    # without lifting first (can sweep over the cube — prefer lift_home).
    calib_detect_retract: str = "lift_home"


@dataclass
class GraspParams:
    """Heights and gripper openings (all meters / gripper units)."""

    # tip-frame z at grasp height in the base frame (m). Measured via diagnose
    # CHECK 5 in the new_calib frame. Negative is expected: the base-frame origin
    # sits above the table, so the table plane has a negative z.
    table_z: float = -0.019
    grasp_z_offset: float = 0.005  # tip z when empty gripper closes on the cube
    cube_height_m: float = 0.025  # physical cube height (m)
    pick_z_margin: float = 0.010  # open gripper: stop this far above cube top (fingers hang lower)
    place_z_margin: float = 0.003  # closed gripper + cube: release clearance above cube top
    hover_z: float = 0.10  # safe hover height above the table before/after grasp
    lift_z: float = 0.15  # height to lift to after grasping before transporting
    # Active hand-eye calibration only: hold the tip this far ABOVE the cube top so
    # you can slide the cube under the gripper. Does not affect pick-and-place.
    calib_clearance_m: float = 0.04

    gripper_open: float = 100.0  # gripper.pos when fully open (RANGE_0_100)
    gripper_closed: float = 10.0  # gripper.pos when closed on the cube (tune if slip)
    gripper_settle_s: float = 0.6  # hold time after gripper reaches target
    # Vertical approach steps (pick = cautious; place = fewer / faster after grasp).
    descend_steps: int = 5
    place_descend_steps: int = 1  # single move down at place (z target is already safe)
    # Post-grasp motion tuning.
    transport_settle_s: float = 0.08
    place_descend_settle_s: float = 0.1
    transport_poll_dt: float = 0.02  # final convergence poll interval (after streaming)
    # Smooth joint streaming: send interpolated waypoints every step_dt without
    # reading the bus between steps (avoids "fast segment + long pause" jerkiness).
    motion_step_dt: float = 0.03
    motion_min_steps: int = 10
    motion_final_poll_max_s: float = 2.0

    # --- IK (task-based: position + downward axis alignment) ---
    # Gripper local axis (in the tip frame) that should point down when grasping.
    # Empirically the SO100 "jaw" frame's local z axis is the approach axis.
    approach_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    # Position task weight (high => reach the (x,y,z) target within ~mm).
    ik_position_weight: float = 1e3
    # Axis-alignment weight (points the gripper down; free rotation about it).
    ik_align_weight: float = 1.0
    # Lock wrist_roll during top-down grasp so the jaws straddle the cube from the
    # sides instead of a random spin about the approach axis (which often makes a
    # gripper end-face hit the cube first). None = leave wrist_roll free (old).
    grasp_wrist_roll_deg: float | None = -2.8
    ik_wrist_roll_weight: float = 500.0
    grasp_wrist_roll_tol_deg: float = 5.0  # warn if the motor misses this after a grasp move
    # placo is an iterative QP solver: iterate this many times per IK call.
    ik_iters: int = 60

    @property
    def grasp_z(self) -> float:
        """Tip z reference when closed on cube (from CHECK 5); not used as descend target."""
        return self.table_z + self.grasp_z_offset

    @property
    def pick_z(self) -> float:
        """Tip z to descend with OPEN gripper before closing (above cube top; fingers hang down)."""
        return self.table_z + self.cube_height_m + self.pick_z_margin

    @property
    def place_z(self) -> float:
        """Tip z for placing with cube in CLOSED gripper."""
        return self.table_z + self.cube_height_m + self.place_z_margin


@dataclass
class RaproberConfig:
    """Top-level config bundle passed around the tool."""

    # --- hardware ---
    arm_port: str = "/dev/ttyACM0"
    arm_id: str = "so100_1"  # calibration id used by lerobot-calibrate
    camera: CameraSettings = field(default_factory=CameraSettings)
    use_degrees: bool = True
    # Max joint step (deg) per command. THIS is why motion looks "segment by segment":
    # lerobot clamps every send_action to this delta; a 90 deg move needs ~90/25 ≈ 4 steps.
    # 12 = very safe but slow; 25 = reasonable for scripted pick-place on a clear table.
    max_relative_target: float | None = 25.0

    # --- key joint poses (degrees, order = MOTOR_NAMES) ---
    # "rest" pose used at connect/disconnect (arm folded, safe). Last value is
    # gripper — ``go_home(open_gripper=True)`` overrides it and opens the gripper.
    home_pose: list[float] = field(
        default_factory=lambda: [-3.0, -101.1, 97.2, 80.3, -8.1, 1.9]
    )
    # "observe" pose: with a FIXED overhead camera (eye-to-hand), this pose's only
    # job is to move the arm OUT of the camera's view of the table so it does not
    # occlude the cube during detection. The pixel->table homography is global
    # (camera is fixed), so this pose does NOT need to match the calibration pose.
    observe_pose: list[float] = field(
        default_factory=lambda: [32.4, 6.4, 12.7, 72.2, 2.4, 25.5]
    )

    # --- fixed place location (base-frame meters, new_calib frame: +x forward). ---
    place_xy: tuple[float, float] = (0.20, 0.10)

    # Constant shift added to vision (x, y) before pick (m). Tune if grasp consistently
    # misses in the same direction (positive x = forward, positive y = left).
    pick_xy_offset: tuple[float, float] = (0.0, 0.0)

    # After hovering, try one more camera read to correct XY (fixed overhead cam).
    refine_xy_at_hover: bool = True

    # Fold to home via observe_pose first — avoids the gripper sweeping into the base
    # when returning from a forward reach over the table.
    home_via_observe: bool = True

    # --- sub-configs ---
    cube: GreenCubeParams = field(default_factory=GreenCubeParams)
    grasp: GraspParams = field(default_factory=GraspParams)
    calib_grid: CalibGridParams = field(default_factory=CalibGridParams)

    # --- kinematics ---
    urdf_path: str = str(URDF_PATH)
    tip_frame: str = TIP_FRAME_NAME
    motor_names: list[str] = field(default_factory=lambda: list(MOTOR_NAMES))

    # --- calibration output ---
    handeye_path: str = str(HANDEYE_PATH)


DEFAULT_CONFIG = RaproberConfig()
