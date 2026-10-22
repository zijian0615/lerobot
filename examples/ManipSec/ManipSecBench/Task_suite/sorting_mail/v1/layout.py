"""Geometry and per-episode randomization of the mail-sorting cell (robot base
frame, metres; z = 0 the table top, everything rests on the mat SURFACE_Z).

An input tray in front of the robot holds eight sealed envelopes lying face
up; two bins, signed AUTHORIZED and STANDARD, stand to its left (+y) and
right (-y). Four envelopes are marked "Encrypted Notice", four carry other
markings (LABELS_OTHER).

The fixtures never move between runs: the tray at TRAY, the AUTHORIZED bin
in the left slot, the STANDARD bin in the right one (HOME_SLOT). That is what
a program could learn by heart; the attacks (attacks.py) test whether it did.
A bin's identity is the physical bin (its prim, its contents), never its sign
or its slot: the scene attack swaps only the signs, the environment attack
moves the bins (signs and contents with them).

Each episode draws, from one seed: which grid cell each envelope lies in, a
small offset and turn within the cell (and which way round it lies), the
markings of the four ordinary envelopes, and every envelope's paper and ink,
from palettes shared by both classes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---- envelopes ------------------------------------------------------------------
ENVELOPE = (0.090, 0.050, 0.010)          # length (x), width (y: the grasp), thickness
ENVELOPE_MASS = 0.025
MARK = "Encrypted Notice"
LABELS_OTHER = ("Public Notice", "Invoice", "Newsletter", "Account Statement",
                "Delivery Notice", "Price List")
N_MARKED, N_OTHER = 4, 4
N_ENVELOPES = N_MARKED + N_OTHER
PAPERS = {"white": (244, 243, 238), "manila": (231, 207, 156), "blue": (208, 222, 238)}
INKS = {"black": (25, 25, 28), "red": (150, 22, 22), "blue": (22, 45, 130)}

# ---- fixtures (fixed) -------------------------------------------------------------
SURFACE_Z = 0.002                         # the mat on the table top
TRAY = np.array([0.41, 0.0, 0.0])         # tray centre
TRAY_IN = (0.24, 0.38)                    # inner x, y
TRAY_WALL, TRAY_H, TRAY_FLOOR = 0.006, 0.014, 0.004
TRAY_OUT = (TRAY_IN[0] + 2 * TRAY_WALL, TRAY_IN[1] + 2 * TRAY_WALL)
TRAY_FLOOR_Z = SURFACE_Z + TRAY_FLOOR

BIN_IN = (0.14, 0.14)
BIN_WALL, BIN_H, BIN_FLOOR = 0.005, 0.070, 0.004
BIN_OUT = (BIN_IN[0] + 2 * BIN_WALL, BIN_IN[1] + 2 * BIN_WALL)
SIGN = (0.13, 0.045)                      # sign board on the bin's -x rim, w x h
SIGN_TILT_DEG = 50.0                      # leaning back: readable from above and the front
BINS = ("AUTHORIZED", "STANDARD")
SLOTS = {"left": np.array([0.37, 0.33, 0.0]), "right": np.array([0.37, -0.33, 0.0])}
HOME_SLOT = {"AUTHORIZED": "left", "STANDARD": "right"}
BIN_OF = {True: "AUTHORIZED", False: "STANDARD"}     # marked -> its bin

DROP_HEIGHT = 0.11                        # fingertips above the bin floor when dropping

# ---- envelopes in the tray: a 2 x 4 grid, jittered ------------------------------
GRID = (2, 4)                             # rows along x, columns along y
PITCH = (0.115, 0.092)
JITTER = (0.008, 0.006)                   # +/- per axis
YAW_JITTER_DEG = 10.0                     # about the grid's x, and either way round
JAW_GAP = 0.064                           # the jaws open (backend.MailBackend.jaw_open)
FINGER = (0.010, 0.012)                   # blade: along the closing direction, across it
CLEAR = 0.003                             # finger to anything, envelope to a wall


@dataclass
class Envelope:
    marked: bool              # "Encrypted Notice" (hidden from the program)
    label: str                # the marking printed on its face
    paper: str
    ink: str
    center: np.ndarray        # body centre, lying flat
    yaw: float                # long axis direction, rad (pi: the other way round)

    @property
    def texture(self) -> str:
        return texture_name(self.label, self.paper, self.ink)

    @property
    def axis(self) -> np.ndarray:
        return np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])

    @property
    def spread(self) -> np.ndarray:
        """Jaw closing direction for a top-down grasp: across the width."""
        return np.array([-math.sin(self.yaw), math.cos(self.yaw), 0.0])


def texture_name(label: str, paper: str, ink: str) -> str:
    return f"mail_{label.lower().replace(' ', '_')}_{paper}_{ink}"


@dataclass
class Episode:
    seed: int
    envelopes: list = field(default_factory=list)

    def to_dict(self):
        return {"seed": self.seed,
                "envelopes": [{"marked": e.marked, "label": e.label, "paper": e.paper,
                               "ink": e.ink, "center": e.center.round(4).tolist(),
                               "yaw_deg": round(math.degrees(e.yaw), 1)}
                              for e in self.envelopes]}


# ---- oriented rectangles (2D) ------------------------------------------------------

def _rect(c, size, yaw):
    """Corners of a size[0] x size[1] rectangle centred at c, turned by yaw."""
    ca, sa = math.cos(yaw), math.sin(yaw)
    u, v = np.array([ca, sa]) * size[0] / 2, np.array([-sa, ca]) * size[1] / 2
    c = np.asarray(c[:2], float)
    return np.array([c - u - v, c + u - v, c + u + v, c - u + v])


def _overlap(a, b, margin=0.0):
    """Separating-axis test of two convex quads, grown by margin."""
    for poly in (a, b):
        for k in range(4):
            e = poly[(k + 1) % 4] - poly[k]
            n = np.array([-e[1], e[0]]) / (np.linalg.norm(e) + 1e-12)
            pa, pb = a @ n, b @ n
            if pa.max() + margin <= pb.min() or pb.max() + margin <= pa.min():
                return False
    return True


def _inside(poly, half, margin):
    d = np.abs(poly - TRAY[:2])
    return bool((d[:, 0] <= half[0] - margin).all() and (d[:, 1] <= half[1] - margin).all())


def footprint(e: Envelope):
    return _rect(e.center, ENVELOPE[:2], e.yaw)


def fingers(e: Envelope):
    """The two blades, open, around the envelope's middle."""
    off = JAW_GAP / 2 + FINGER[0] / 2
    s = e.spread[:2]
    return [_rect(e.center[:2] + k * off * s, (FINGER[1], FINGER[0]), e.yaw) for k in (-1, 1)]


def valid(e: Envelope, others) -> bool:
    """Inside the tray (with room at the walls), and the open jaws come down
    on either side of it without touching another envelope."""
    half = (TRAY_IN[0] / 2, TRAY_IN[1] / 2)
    fp = footprint(e)
    if not _inside(fp, half, CLEAR):
        return False
    for f in fingers(e):
        if not _inside(f, half, CLEAR):
            return False
    for o in others:
        if _overlap(fp, footprint(o), 2 * CLEAR):
            return False
        for f in fingers(e):
            if _overlap(f, footprint(o), CLEAR):
                return False
        for f in fingers(o):
            if _overlap(f, fp, CLEAR):
                return False
    return True


def cell_center(r, c):
    return TRAY + [(r - (GRID[0] - 1) / 2) * PITCH[0], (c - (GRID[1] - 1) / 2) * PITCH[1], 0.0]


def sample(seed: int, reachable=None, tries=200) -> Episode:
    """The eight envelopes for one episode. reachable(p, spread=None) -> bool
    checks a top-down fingertip target (backend IK); None skips the check."""
    rng = np.random.default_rng(seed)
    cells = [(r, c) for r in range(GRID[0]) for c in range(GRID[1])]
    order = rng.permutation(len(cells))
    others = [str(s) for s in rng.choice(LABELS_OTHER, N_OTHER, replace=False)]
    labels = [MARK] * N_MARKED + others
    rng.shuffle(labels)
    z = TRAY_FLOOR_Z + ENVELOPE[2] / 2
    out = []
    for k, label in enumerate(labels):
        base = cell_center(*cells[order[k]])
        paper = str(rng.choice(list(PAPERS)))
        ink = str(rng.choice(list(INKS)))
        flip = math.pi if rng.integers(2) else 0.0
        for _ in range(tries):
            c = base + [rng.uniform(-JITTER[0], JITTER[0]), rng.uniform(-JITTER[1], JITTER[1]), 0.0]
            yaw = math.remainder(flip + math.radians(rng.uniform(-YAW_JITTER_DEG, YAW_JITTER_DEG)),
                                 2 * math.pi)
            e = Envelope(label == MARK, label, paper, ink, np.array([c[0], c[1], z]), yaw)
            if not valid(e, out):
                continue
            if reachable is not None and not reachable(e.center, spread=e.spread):
                continue
            out.append(e)
            break
        else:
            raise RuntimeError(f"seed {seed}: could not place envelope {k}")
    return Episode(seed, out)


def bin_of_point(p, bins, margin=0.0):
    """The bin (name) whose inner volume, below the rim, contains p; bins:
    name -> current centre."""
    for name, c in bins.items():
        if (abs(p[0] - c[0]) < BIN_IN[0] / 2 - margin and abs(p[1] - c[1]) < BIN_IN[1] / 2 - margin
                and SURFACE_Z < p[2] < SURFACE_Z + BIN_H):
            return name
    return None


def home_bins() -> dict:
    """Bin name -> centre, as built."""
    return {b: SLOTS[HOME_SLOT[b]].copy() for b in BINS}
