"""Forward / inverse kinematics for the SO100 arm (placo, task-based).

The SO100 has only 5 non-gripper DOF, so a full 6-DOF pose target is
over-constrained: the solver cannot satisfy both position and a full 3x3
orientation, and trades one off against the other (unreliable, see git history).

Instead we use exactly the constraints a top-down grasp needs:

* a **position task** on the tip frame (high weight -> reached within ~mm), and
* an **axis-alignment task** that points the gripper's local approach axis
  straight down (world -Z), leaving the free rotation about that axis
  unconstrained.

This reliably gives sub-mm position error *and* a downward gripper for all
reachable table points. We build the placo solver directly here (rather than via
``lerobot.model.RobotKinematics``) because that wrapper only exposes a full-pose
frame task.
"""

from __future__ import annotations

import numpy as np
import placo

from .config import RaproberConfig

WORLD_DOWN = np.array([0.0, 0.0, -1.0])


class ArmIK:
    """Task-based FK/IK helper bound to a specific SO100 URDF + motor order."""

    def __init__(self, config: RaproberConfig):
        self.config = config
        self.motor_names = list(config.motor_names)
        # placo solves over the non-gripper joints; the gripper is passthrough.
        self._ik_joint_names = [n for n in self.motor_names if n != "gripper"]
        self.tip = config.tip_frame
        self.approach_axis = np.asarray(config.grasp.approach_axis, dtype=float)
        self.position_weight = float(config.grasp.ik_position_weight)
        self.align_weight = float(config.grasp.ik_align_weight)
        self.grasp_wrist_roll_deg = config.grasp.grasp_wrist_roll_deg
        self.wrist_roll_weight = float(config.grasp.ik_wrist_roll_weight)
        self.n_iters = int(config.grasp.ik_iters)

        self.robot = placo.RobotWrapper(config.urdf_path)

    # --- helpers --------------------------------------------------------- #
    def _set_joints(self, joints: dict[str, float]) -> None:
        for name in self._ik_joint_names:
            self.robot.set_joint(name, np.deg2rad(float(joints[name])))
        self.robot.update_kinematics()

    def _read_joints(self) -> dict[str, float]:
        return {n: float(np.rad2deg(self.robot.get_joint(n))) for n in self._ik_joint_names}

    # --- forward kinematics --------------------------------------------- #
    def forward(self, joints: dict[str, float]) -> np.ndarray:
        """Return the 4x4 tip pose in the base frame for the given joint dict."""
        self._set_joints(joints)
        return self.robot.get_T_world_frame(self.tip)

    def tip_position(self, joints: dict[str, float]) -> np.ndarray:
        """Return the tip (x, y, z) in the base frame (meters)."""
        return self.forward(joints)[:3, 3]

    def down_alignment(self, joints: dict[str, float]) -> float:
        """Cosine between the gripper approach axis and world-down (1 = perfect)."""
        r = self.forward(joints)[:3, :3]
        return float((r @ self.approach_axis) @ WORLD_DOWN)

    def position_error_m(self, joints: dict[str, float], target_xyz) -> float:
        """Euclidean tip-position error (meters) for a solved joint dict."""
        return float(np.linalg.norm(np.asarray(target_xyz, dtype=float) - self.tip_position(joints)))

    # --- inverse kinematics --------------------------------------------- #
    def _solve_once(
        self,
        seed_joints: dict[str, float],
        target_xyz: np.ndarray,
        align_down: bool,
        n_iters: int,
        wrist_roll_deg: float | None = None,
    ) -> dict[str, float]:
        """Single placo IK run from one seed configuration."""
        self._set_joints(seed_joints)
        solver = placo.KinematicsSolver(self.robot)
        solver.mask_fbase(True)

        pos_task = solver.add_position_task(self.tip, target_xyz)
        pos_task.configure("pos", "soft", self.position_weight)

        if align_down:
            align_task = solver.add_axisalign_task(self.tip, self.approach_axis, WORLD_DOWN)
            align_task.configure("align", "soft", self.align_weight)

        if wrist_roll_deg is not None:
            roll_task = solver.add_joints_task()
            roll_task.set_joints({"wrist_roll": np.deg2rad(float(wrist_roll_deg))})
            # Hard constraint: a soft task is often dropped when hover leaves wrist_roll
            # near ±180° and the solver trades it for position during descend.
            roll_task.configure("wrist_roll", "hard", 1.0)

        for _ in range(max(1, n_iters)):
            solver.solve(True)
            self.robot.update_kinematics()

        result = dict(seed_joints)  # keep gripper (and any extras) untouched
        result.update(self._read_joints())
        return result

    def _seed_variants(self, current_joints: dict[str, float]) -> list[dict[str, float]]:
        """Current pose plus spread-out fallback seeds to escape local minima.

        placo IK is seed-sensitive (from some configurations it walks the wrong
        way on ``shoulder_pan`` and gets stuck). We try the current pose first,
        then a fan of nominal poses spanning the pan range so at least one basin
        reaches the target.
        """
        seeds = [dict(current_joints)]
        nominal = {
            "shoulder_lift": 20.0,
            "elbow_flex": 20.0,
            "wrist_flex": 40.0,
            "wrist_roll": 0.0,
            "gripper": current_joints.get("gripper", 0.0),
        }
        for pan in (-90.0, -45.0, 0.0, 45.0, 90.0):
            seeds.append({"shoulder_pan": pan, **nominal})
        return seeds

    def solve(
        self,
        current_joints: dict[str, float],
        target_xyz: np.ndarray | tuple[float, float, float],
        align_down: bool = True,
        n_iters: int | None = None,
        tol_m: float = 0.008,
    ) -> dict[str, float]:
        """Inverse kinematics with multi-seed restarts (robust to local minima).

        Tries the current pose first, then several fallback seeds, and returns the
        best solution (lowest position error, tie-broken by downward alignment).
        Stops early once a seed reaches ``tol_m`` with a downward gripper.

        Args:
            current_joints: current joint dict (preferred initial guess).
            target_xyz: desired tip position in base frame (meters).
            align_down: if True, also point the gripper approach axis downward.
            n_iters: solver iterations per seed. Defaults to config.
            tol_m: position tolerance for the early-exit "good enough" check.

        Returns:
            Joint dict for the non-gripper joints (gripper preserved from input).
        """
        target_xyz = np.asarray(target_xyz, dtype=float)
        n_iters = self.n_iters if n_iters is None else n_iters
        wrist_roll_deg: float | None = None
        if align_down and self.grasp_wrist_roll_deg is not None:
            wrist_roll_deg = float(self.grasp_wrist_roll_deg)

        best: dict[str, float] | None = None
        best_score: tuple[float, float] | None = None
        for seed in self._seed_variants(current_joints):
            cand = self._solve_once(
                seed, target_xyz, align_down, n_iters, wrist_roll_deg=wrist_roll_deg
            )
            res = self.position_error_m(cand, target_xyz)
            down = self.down_alignment(cand) if align_down else 1.0
            score = (res, -down)  # lower residual wins; then more downward
            if best_score is None or score < best_score:
                best, best_score = cand, score
            if res <= tol_m and down >= 0.9:
                break  # good enough, no need to try more seeds
        return best if best is not None else dict(current_joints)
