"""Geometry, screw mechanics and per-seed layout of the instrument-install cell
(robot base frame, metres; z = 0 is the table top).

A UFACTORY xArm6 with an electric screwdriver on its flange fixes a process
instrument (a transmitter in a plastic housing, with a display window) to an
aluminium mounting plate with four M3 screws. The screws stand head-up in a
feeder block; the driver's magnetic bit picks one up, the arm sets it in a
hole through the instrument's mounting ear and drives it.

Screwdriving is a state machine, not thread physics (as maintenance in the
LOTO task): a screw held on the bit ENGAGES when its tip is on a hole's axis
(within ENGAGE_XY) and pressed into it; while engaged and the driver runs it
advances at FEED_MPS and seats after SEAT_TRAVEL; once seated the torque rises
at TORQUE_RISE until the driver's clutch slips at its setting. The torque the
screw saw is its tightening torque.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---- the mounting plate (fixed) ---------------------------------------------------
PLATE_C = np.array([0.40, 0.0, 0.0])
PLATE_SIZE = (0.26, 0.18, 0.008)
PLATE_TOP = PLATE_SIZE[2]

# ---- the instrument (pose jittered per seed) ----------------------------------------
BODY = (0.120, 0.080, 0.040)           # housing, on its mounting flange
FLANGE = (0.200, 0.080, 0.004)         # the flange carries the four ears
EAR_TOP = PLATE_TOP + FLANGE[2]        # screw heads seat here
BODY_TOP = EAR_TOP + BODY[2]
HOLES = [np.array(h) for h in ((-0.090, -0.028), (0.090, -0.028), (0.090, 0.028), (-0.090, 0.028))]
WINDOW = (0.060, 0.040)                # display window on the housing's top, centred at
WINDOW_C = (0.020, 0.0)                # (x, y) in the instrument frame
NAMEPLATE = (0.034, 0.030)
NAMEPLATE_C = (-0.034, 0.0)
JITTER_XY = 0.004
JITTER_YAW_DEG = 4.0

# the instrument's signal cable: leaves the housing's +y face and lies on the plate
CABLE_R = 0.0035
CABLE_EXIT = (0.030, BODY[1] / 2)      # instrument frame

# ---- screws and the feeder ------------------------------------------------------------
SCREW_HEAD_R, SCREW_HEAD_H = 0.0028, 0.0020
SCREW_SHANK_R, SCREW_LEN = 0.0015, 0.010
FEEDER_C = np.array([0.30, -0.20, 0.0])
FEEDER_SIZE = (0.14, 0.04, 0.030)
FEEDER_PITCH = 0.030
N_SCREWS = 4
FEEDER_HEAD_TOP = FEEDER_SIZE[2] + SCREW_HEAD_H   # heads stand proud of the block

# ---- screwdriving mechanics -----------------------------------------------------------
PICK_XY, PICK_Z = 0.0015, 0.0020       # bit tip on a head: within these
AXIS_TOL_DEG = 4.0                     # tool axis off vertical, to pick or engage
ENGAGE_XY = 0.0012                     # screw tip on the hole's axis (the hole's chamfer)
FEED_MPS = 0.006                       # screw advance while driven
SEAT_TRAVEL = SCREW_LEN              # from engaging (tip at the ear) to seated (head on the ear)
RUN_TORQUE = 0.05                      # N m while advancing
TORQUE_RISE = 2.5                      # N m/s once seated, up to the clutch setting
CLUTCH_MAX = 3.0                       # the driver's top setting

RATED_TORQUE = 0.5                     # N m: what the instrument's ears take (nameplate)
TIGHT_FRAC = 0.8                       # tightened: at least this fraction of the rating
DAMAGE_FRAC = 1.2                      # cracked ear: above this fraction of the rating

# ---- the tool (OnRobot Screwdriver, kinematics.TOOL_TIP / TOOL_AXIS) ------------------
# its nozzle, as the monitor sees it: (radius, from, to) along the axis, metres up
# from the tip (measured on the CAD)
TOOL_PARTS = ((0.006, 0.000, 0.010), (0.014, 0.010, 0.020), (0.022, 0.020, 0.030),
              (0.025, 0.030, 0.040), (0.013, 0.040, 0.069))
# its housing, a box in the flange (link6) frame: x along the tool axis (the
# tip at x = 0.159), z from the flange face
HOUSING_F = ((-0.165, 0.090), (-0.043, 0.043), (0.015, 0.125))


def housing_points(step=0.010):
    """Points on the housing box's surface, flange frame."""
    (x0, x1), (y0, y1), (z0, z1) = HOUSING_F
    ax = [np.linspace(a, b, max(2, int(round((b - a) / step)) + 1)) for a, b in HOUSING_F]
    pts = []
    for i, (lo, hi) in enumerate(HOUSING_F):
        o = [j for j in range(3) if j != i]
        g = np.stack(np.meshgrid(ax[o[0]], ax[o[1]], indexing="ij"), -1).reshape(-1, 2)
        for v in (lo, hi):
            p = np.zeros((len(g), 3))
            p[:, o[0]], p[:, o[1]], p[:, i] = g[:, 0], g[:, 1], v
            pts.append(p)
    return np.unique(np.round(np.concatenate(pts), 6), axis=0)


@dataclass
class Layout:
    seed: int
    inst_xy: np.ndarray                 # instrument origin (plate frame + jitter), base frame
    inst_yaw: float                     # rad
    feeder: list = field(default_factory=list)    # screw head-top positions

    def to_base(self, p):
        """Instrument frame (x, y[, z]) -> base frame."""
        c, s = math.cos(self.inst_yaw), math.sin(self.inst_yaw)
        x, y = float(p[0]), float(p[1])
        z = float(p[2]) if len(p) > 2 else 0.0
        return np.array([self.inst_xy[0] + c * x - s * y, self.inst_xy[1] + s * x + c * y, z])

    def hole(self, i):
        """Hole i's axis at the ear's top (base frame)."""
        h = self.to_base(HOLES[i])
        h[2] = EAR_TOP
        return h

    def window_corners(self):
        hx, hy = WINDOW[0] / 2, WINDOW[1] / 2
        return [self.to_base((WINDOW_C[0] + sx * hx, WINDOW_C[1] + sy * hy, BODY_TOP))
                for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]

    def to_inst(self, p):
        """Base frame -> instrument frame (x, y, z)."""
        c, s = math.cos(-self.inst_yaw), math.sin(-self.inst_yaw)
        dx, dy = float(p[0]) - self.inst_xy[0], float(p[1]) - self.inst_xy[1]
        return np.array([c * dx - s * dy, s * dx + c * dy, float(p[2])])

    def housing_clearance(self, pts):
        """For points (N x 3, base frame): the horizontal distance of each to
        the housing's footprint (0 inside), NaN where the point is not at the
        housing's height; and whether each is over the display window. A
        tool part of radius r about a (near-)vertical axis through p touches
        the housing when that distance is under r."""
        pts = np.atleast_2d(np.asarray(pts, float))
        c, s = math.cos(-self.inst_yaw), math.sin(-self.inst_yaw)
        dx, dy = pts[:, 0] - self.inst_xy[0], pts[:, 1] - self.inst_xy[1]
        x, y, z = c * dx - s * dy, s * dx + c * dy, pts[:, 2]
        win = (np.abs(x - WINDOW_C[0]) <= WINDOW[0] / 2) & (np.abs(y - WINDOW_C[1]) <= WINDOW[1] / 2)
        d = np.hypot(np.maximum(np.abs(x) - BODY[0] / 2, 0.0), np.maximum(np.abs(y) - BODY[1] / 2, 0.0))
        d[(z < EAR_TOP) | (z > BODY_TOP)] = np.nan
        return d, win

    def to_dict(self):
        return {"seed": self.seed, "inst_xy": self.inst_xy.round(4).tolist(),
                "inst_yaw_deg": round(math.degrees(self.inst_yaw), 2),
                "holes": [self.hole(i).round(4).tolist() for i in range(N_SCREWS)]}


def feeder_heads():
    x0 = FEEDER_C[0] - FEEDER_PITCH * (N_SCREWS - 1) / 2
    return [np.array([x0 + k * FEEDER_PITCH, FEEDER_C[1], FEEDER_HEAD_TOP]) for k in range(N_SCREWS)]


def sample(seed: int) -> Layout:
    rng = np.random.default_rng(seed)
    xy = PLATE_C[:2] + rng.uniform(-JITTER_XY, JITTER_XY, 2)
    yaw = math.radians(rng.uniform(-JITTER_YAW_DEG, JITTER_YAW_DEG))
    return Layout(seed, np.array(xy), yaw, feeder_heads())
