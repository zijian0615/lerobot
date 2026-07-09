"""raprober: scripted black-cube pick-and-place for an SO100 arm.

Modules:
    config          - central hardware/task configuration
    perception      - OpenCV black-cube detection
    kinematics_ik   - SO100 FK/IK wrapper (placo)
    arm             - high-level SO100 follower controller
    calibration     - hand-eye (pixel -> table) homography calibration
    pick_place      - end-to-end grasp-and-place orchestration
"""

from .config import DEFAULT_CONFIG, RaproberConfig

__all__ = ["DEFAULT_CONFIG", "RaproberConfig"]
