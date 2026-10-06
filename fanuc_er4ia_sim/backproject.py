"""
Pixel <-> table-plane geometry for the perception pipeline.

The VLM returns 2D image coordinates; the planner needs 3D table coordinates.
Because we control the camera in simulation, this conversion is exact rather
than estimated: intersect the pixel's view ray with the known table plane.

USD convention notes, both of which are easy to get backwards:
  - ComputeLocalToWorldTransform is row-vector convention, so a point maps as
    p_world = [p, 1] @ M, the camera position is row 3, and the camera basis
    vectors are rows 0..2.
  - USD cameras look down their local -Z with +Y up.
"""
import json
import numpy as np


class CameraModel:
    def __init__(self, cam_json):
        if isinstance(cam_json, str):
            with open(cam_json) as f:
                cam_json = json.load(f)
        self.j = cam_json
        self.W, self.H = cam_json["resolution"]
        f = cam_json["focal_length_mm"]
        self.tan_h = (cam_json["horizontal_aperture_mm"] / 2.0) / f
        self.tan_v = (cam_json["vertical_aperture_mm"] / 2.0) / f
        M = np.array(cam_json["camera_to_world"], dtype=float)
        self.R = M[:3, :3]          # rows are the camera basis in world space
        self.eye = M[3, :3]
        self.table_z = cam_json["table_top_z"]

    # ---------------------------------------------------------------- forward
    def project(self, p_world):
        """World point -> (u, v) pixels. Returns None if behind the camera."""
        p = np.asarray(p_world, float) - self.eye
        cam = self.R @ p                      # world -> camera (R is orthonormal)
        if cam[2] >= -1e-9:                   # camera looks down -Z
            return None
        x = cam[0] / (-cam[2]) / self.tan_h   # NDC, -1..1
        y = cam[1] / (-cam[2]) / self.tan_v
        u = (x + 1.0) * 0.5 * self.W
        v = (1.0 - y) * 0.5 * self.H
        return np.array([u, v])

    # ---------------------------------------------------------------- inverse
    def ray(self, u, v):
        """Pixel -> (origin, unit direction) in world space.

        u, v are CONTINUOUS image coordinates, matching project(). Adding a
        half-pixel here (as you would for integer pixel indices) breaks the
        round trip by about a millimetre at this camera distance.
        """
        x = u / self.W * 2.0 - 1.0
        y = 1.0 - v / self.H * 2.0
        d_cam = np.array([x * self.tan_h, y * self.tan_v, -1.0])
        d_world = d_cam @ self.R              # inverse of R @ v for orthonormal R
        return self.eye, d_world / np.linalg.norm(d_world)

    def to_plane(self, u, v, z=None):
        """Pixel -> point on the horizontal plane at height z (default table top)."""
        z = self.table_z if z is None else z
        o, d = self.ray(u, v)
        if abs(d[2]) < 1e-9:
            return None
        t = (z - o[2]) / d[2]
        if t <= 0:
            return None
        return o + t * d

    def to_table_xy(self, u_norm, v_norm, height_above_table=0.0):
        """Normalised (0..1) image coords -> (x, y) in the ROBOT BASE frame.

        The robot base sits at the table top, so robot-frame z = 0 there.
        """
        p = self.to_plane(u_norm * self.W, v_norm * self.H,
                          z=self.table_z + height_above_table)
        return None if p is None else np.array([p[0], p[1]])
