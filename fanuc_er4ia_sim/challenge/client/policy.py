"""Your policy. Edit this file.

act(obs) returns the arm's next target, or a list of targets to play in order. When they have been played, act is
called again with a new observation. act has no time limit: the arm holds still while it runs.
"""
from __future__ import annotations

import numpy as np


class Policy:
    CAMERAS = ["table", "side", "wrist"]  # the cameras you want, any subset
    IMAGE_SIZE = (640, 360)               # (width, height) of the images you get: the whole 16:9 camera frame, resized
                                          # (not cropped), as in the sim and the recorded datasets
    IMAGE_FORMAT = "png"                  # or "jpeg": about 7x smaller, for a slow connection
    ACTION = "joints"                     # what act returns: "joints" or "tcp" (see below)
    PREFETCH = False                      # True: if act returns many actions at once, plan the next ones while
                                          # these still play, so the arm does not stop between them
    DEPTH = []                            # ["table"] and/or ["wrist"]: also those cameras' depth, in obs["depth"]

    def __init__(self, checkpoint: str | None = None):
        pass  # load your model here

    def reset(self) -> None:
        """Called at the start of every rollout."""

    def act(self, obs: dict):
        """obs["images"]   {camera: (H, W, 3) uint8 RGB}
        obs["joints"]      [J1..J6 deg (the v0.5 twin's convention), gripper 0 open / 1 closed]
        obs["tcp_pose"]    the fingertip [x, y, z mm, w, p, r deg]
        obs["trial"]["instruction"]  the task in words (you may ignore it)
        obs["depth"]       only with DEPTH: {"table": (400, 640), "wrist": (480, 848)} uint16, millimetres along each
                           camera's optical axis, 0 = no depth. calibration.geometry.depth_to_base(load()["table_depth"],
                           depth) gives (N, 3) points in the robot base frame (metres); for the wrist, which moves with
                           the arm, pass joints=obs["joints"] too.

        Return the next target:
          ACTION = "joints"  [J1, J2, J3, J4, J5, J6, gripper]
          ACTION = "tcp"     [x, y, z, w, p, r, gripper]; fingers straight down is w, p, r = -180, 0, R,
                             and the table surface is at z ~ -335 mm. No IK needed.
        Example (tcp): 3 cm above the table at x 450, y 0, fingers down, gripper open
            return [450, 0, -305, -180, 0, 0, 0]
        Return where the arm should be, not a step from where it is.
        """
        return obs["joints"]  # placeholder: stay where you are
