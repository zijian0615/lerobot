"""Geometry and per-seed episodes of the mail-sorting cell, version 2 (robot
base frame, metres; z = 0 is the table top).

A UFACTORY xArm6 with a two-cup vacuum gripper sorts eight sealed DL
envelopes, lying in a loose stack in the input tray (a grey tote), into two
outgoing letter trays -- the bins -- each with a label card in a holder on
its far end: AUTHORIZED (built in the left slot, +y) and STANDARD (right).

The envelopes are moved by a state machine, not contact physics (as the
screws of assembly/instrument): see mechanics.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---- the input tray (a tote) and the bins (letter trays) ------------------------------
TOTE_C = np.array([0.40, 0.0])
TOTE = (0.34, 0.26, 0.085)              # outer x, y, height
WALL = 0.005
BIN = (0.28, 0.17, 0.065)
SLOTS = {"L": np.array([0.30, 0.36]), "R": np.array([0.30, -0.36])}   # +y is the robot's left
BUILT = {"AUTHORIZED": "L", "STANDARD": "R"}                          # bin -> slot as built
BINS = tuple(BUILT)
CARD = (0.12, 0.032)                    # label card, in a holder on the bin's far (+x) end
CARD_TILT_DEG = 55.0

# ---- envelopes -------------------------------------------------------------------------
ENV = (0.220, 0.110, 0.003)             # DL
N_ENV, N_MARKED = 8, 4
MARK = "Encrypted Notice"
LABELS_OTHER = ("Public Notice", "Invoice", "Newsletter", "Account Statement",
                "Delivery Notice", "Price List")
PAPERS = {"white": (246, 244, 238), "manila": (226, 205, 160), "grey": (236, 238, 242)}
INKS = {"black": (25, 25, 30), "red": (170, 30, 30), "blue": (30, 50, 150)}
JITTER_XY = (0.030, 0.022)
JITTER_YAW_DEG = 14.0

# ---- the vacuum gripper (flange frame of the xArm6: tool along +Z) ---------------------
TOOL_LEN = 0.200                        # flange face -> cup lips
TOOL = (np.array([0.0, 0.0, TOOL_LEN]), np.array([0.0, 0.0, 1.0]))
CUP_SPAN = 0.090                        # the two cups, along the flange +Y (the "spread")
CUP_R = 0.017
SEAL_Z = (0.002, 0.003)                 # cups on a face: lips within these below/above it
AXIS_TOL_DEG = 6.0
BIN_OF = {True: "AUTHORIZED", False: "STANDARD"}       # marked -> its bin


@dataclass
class Envelope:
    label: str
    paper: str
    ink: str
    xy: np.ndarray
    yaw: float                          # rad, the long side's direction
    z: float                            # face (top) height

    @property
    def marked(self):
        return self.label == MARK

    def corners(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        hx, hy = ENV[0] / 2, ENV[1] / 2
        return [self.xy + [c * a - s * b, s * a + c * b] for a, b in
                ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]

    def covers(self, p, margin=0.0):
        """p (x, y) within the envelope's outline."""
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        dx, dy = p[0] - self.xy[0], p[1] - self.xy[1]
        return (abs(c * dx - s * dy) <= ENV[0] / 2 - margin
                and abs(s * dx + c * dy) <= ENV[1] / 2 - margin)

    def to_dict(self):
        return {"label": self.label, "paper": self.paper, "ink": self.ink,
                "xy": np.round(self.xy, 4).tolist(), "yaw_deg": round(math.degrees(self.yaw), 1),
                "z": round(self.z, 4)}


@dataclass
class Episode:
    seed: int
    envelopes: list = field(default_factory=list)       # bottom of the stack first

    def to_dict(self):
        return {"seed": self.seed, "envelopes": [e.to_dict() for e in self.envelopes]}


def texture_name(label, paper, ink):
    return f"mailv2_{label.replace(' ', '_').lower()}_{paper}_{ink}"


def sample(seed: int) -> Episode:
    rng = np.random.default_rng([seed, 2])
    labels = [MARK] * N_MARKED + list(rng.choice(LABELS_OTHER, N_ENV - N_MARKED, replace=False))
    rng.shuffle(labels)
    ep = Episode(seed)
    for k, lab in enumerate(labels):
        xy = TOTE_C + rng.uniform(-1, 1, 2) * JITTER_XY
        yaw = math.radians(rng.uniform(-JITTER_YAW_DEG, JITTER_YAW_DEG)) + (math.pi if rng.random() < 0.5 else 0.0)
        ep.envelopes.append(Envelope(str(lab), str(rng.choice(list(PAPERS))), str(rng.choice(list(INKS))),
                                     xy, yaw, WALL + ENV[2] * (k + 1)))
    return ep


def bin_inner(slot_xy):
    """(xmin, xmax, ymin, ymax) of a bin's inside, standing in a slot."""
    hx, hy = BIN[0] / 2 - WALL, BIN[1] / 2 - WALL
    return slot_xy[0] - hx, slot_xy[0] + hx, slot_xy[1] - hy, slot_xy[1] + hy


def in_bin(p, slot_xy):
    """p (x, y, z): inside a bin's walls and below its rim."""
    x0, x1, y0, y1 = bin_inner(slot_xy)
    return x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and p[2] <= BIN[2]


def in_tote(p):
    hx, hy = TOTE[0] / 2 - WALL, TOTE[1] / 2 - WALL
    return abs(p[0] - TOTE_C[0]) <= hx and abs(p[1] - TOTE_C[1]) <= hy
