"""Geometry and per-reset randomization of the sorting cell (arm A's base
frame, metres; z = 0 is the table top, objects rest on the mat SURFACE_Z).

One parts tray holding six items lying on their sides -- two cylindrical
batteries, two steel dowel pins, two plastic sleeves -- and three bins signed
BATTERY, METAL and PLASTIC. Each reset draws, from one seed:

  - the tray's and the three bins' positions (no overlaps, within reach)
  - each item's position and yaw in the tray (no overlaps, room for the
    jaws on both sides, a top-down grasp IK-reachable)
  - each item's colour and finish, from one palette shared by every class,
    so colour says nothing about the class

Batteries and pins have nearly the same length and diameter: "a cylinder" does
not tell them apart. A battery has a raised positive button with an insulating
ring on one end face, the pin two flat end faces, the sleeve a visible bore.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ---- items: (radius, length) of the body; lying on their side ---------------
BATTERY_R, BATTERY_L = 0.00725, 0.0495     # AA: 14.5 mm x 50.5 mm with the button
BUTTON_R, BUTTON_H = 0.0027, 0.0010        # positive terminal
RING_R0, RING_R1 = 0.0034, 0.0058          # insulating ring around it, end face
PIN_R, PIN_L = BATTERY_R, 0.0505            # dowel pin: the same size (the same top
                                           # height in depth), flat ends
SLEEVE_R, SLEEVE_BORE, SLEEVE_L = 0.0075, 0.0045, 0.045
ITEM_GEOM = {"battery": (BATTERY_R, BATTERY_L + BUTTON_H),
             "steel_pin": (PIN_R, PIN_L),
             "plastic_sleeve": (SLEEVE_R, SLEEVE_L)}
ITEM_MASS = {"battery": 0.023, "steel_pin": 0.062, "plastic_sleeve": 0.004}
KINDS = ("battery", "battery", "steel_pin", "steel_pin", "plastic_sleeve", "plastic_sleeve")
BIN_OF = {"battery": "BATTERY", "steel_pin": "METAL", "plastic_sleeve": "PLASTIC"}
BINS = ("BATTERY", "METAL", "PLASTIC")

# ---- fixtures ------------------------------------------------------------------
SURFACE_Z = 0.002                          # the mat on the table top
TRAY_IN = (0.24, 0.20)                     # inner x, y
TRAY_WALL, TRAY_H, TRAY_FLOOR = 0.006, 0.014, 0.004
TRAY_OUT = (TRAY_IN[0] + 2 * TRAY_WALL, TRAY_IN[1] + 2 * TRAY_WALL)
BIN_IN = (0.12, 0.12)
BIN_WALL, BIN_H, BIN_FLOOR = 0.005, 0.070, 0.004
BIN_OUT = (BIN_IN[0] + 2 * BIN_WALL, BIN_IN[1] + 2 * BIN_WALL)
SIGN = (0.11, 0.045)                       # sign board on the bin's back wall, w x h
SIGN_TILT_DEG = 50.0                       # leaning back from vertical: readable
                                           # from above and from the front
GAP = 0.03                                 # between fixtures

# where fixtures may go: an annulus in front of the robot
REACH_R = (0.26, 0.52)                     # fixture centre distance from J1
REACH_YAW_DEG = 75.0                       # +/- from the robot's +x
WORKSPACE_X = (0.12, 0.62)

# ---- items in the tray -------------------------------------------------------
ITEM_CLEAR = 0.022                         # surface to surface, room for a jaw
WALL_CLEAR = 0.022                         # item surface to the tray wall
JAW_CLEARANCE_Z = 0.0                      # fingertips at the item axis height

# ---- appearance ----------------------------------------------------------------
# One palette for every class. finish: "metal" (bare or plated), "paint"
# (matte coat), "gloss" (glossy plastic or printed film).
PALETTE = {
    "black": (0.03, 0.03, 0.035), "white": (0.85, 0.85, 0.84),
    "grey": (0.42, 0.43, 0.45), "silver": (0.78, 0.78, 0.80),
    "blue": (0.06, 0.20, 0.62), "red": (0.70, 0.07, 0.05),
    "green": (0.08, 0.45, 0.16), "yellow": (0.85, 0.66, 0.05),
    "orange": (0.88, 0.35, 0.04),
}
FINISHES = {"battery": ("gloss",), "steel_pin": ("metal", "paint"),
            "plastic_sleeve": ("gloss", "paint")}

DROP_HEIGHT = 0.11                         # fingertips above a bin floor when dropping


@dataclass
class Item:
    kind: str                  # hidden from the policy
    center: np.ndarray         # body centre, lying
    yaw: float                 # axis direction in the table plane, rad
    flip: bool                 # battery: positive end toward -axis instead of +axis
    colour: str
    finish: str
    print_down: bool = False   # battery: printed once, on the side facing the tray

    @property
    def axis(self) -> np.ndarray:
        return np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])

    @property
    def radius(self) -> float:
        return ITEM_GEOM[self.kind][0]

    @property
    def length(self) -> float:
        return ITEM_GEOM[self.kind][1]


@dataclass
class Layout:
    seed: int
    tray: np.ndarray                       # tray centre (x, y, 0)
    bins: dict                             # "BATTERY" -> bin centre (x, y, 0)
    items: list = field(default_factory=list)

    @property
    def tray_floor(self) -> float:
        return SURFACE_Z + TRAY_FLOOR

    def bin_of_point(self, p, margin=0.0):
        """The bin whose inner volume contains p (below the rim), or None."""
        for name, c in self.bins.items():
            if (abs(p[0] - c[0]) < BIN_IN[0] / 2 - margin
                    and abs(p[1] - c[1]) < BIN_IN[1] / 2 - margin
                    and SURFACE_Z < p[2] < SURFACE_Z + BIN_H):
                return name
        return None

    def to_dict(self) -> dict:
        return {"seed": self.seed, "tray": self.tray.round(4).tolist(),
                "bins": {k: v.round(4).tolist() for k, v in self.bins.items()},
                "items": [{"kind": it.kind, "center": it.center.round(4).tolist(),
                           "yaw_deg": round(math.degrees(it.yaw), 1), "flip": it.flip,
                           "colour": it.colour, "finish": it.finish,
                           "print_down": it.print_down} for it in self.items]}


# ---- sampling -------------------------------------------------------------------

def _rect(c, size, pad=0.0):
    return (c[0] - size[0] / 2 - pad, c[0] + size[0] / 2 + pad,
            c[1] - size[1] / 2 - pad, c[1] + size[1] / 2 + pad)


def _overlap(a, b):
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def _in_reach(c, size):
    corners = [(c[0] + sx * size[0] / 2, c[1] + sy * size[1] / 2)
               for sx in (-1, 1) for sy in (-1, 1)]
    r = math.hypot(c[0], c[1])
    yaw = math.degrees(math.atan2(c[1], c[0]))
    return (REACH_R[0] <= r <= REACH_R[1] and abs(yaw) <= REACH_YAW_DEG
            and all(WORKSPACE_X[0] <= x <= WORKSPACE_X[1] for x, _ in corners)
            and all(math.hypot(x, y) >= 0.18 for x, y in corners))


def _seg_dist(p0, p1, q0, q1):
    """Shortest distance between segments p0-p1 and q0-q1 (2D)."""
    def pt_seg(p, a, b):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
        return float(np.linalg.norm(p - (a + t * ab)))
    d = [pt_seg(p0, q0, q1), pt_seg(p1, q0, q1), pt_seg(q0, p0, p1), pt_seg(q1, p0, p1)]
    # proper crossing
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    if (cross(p0, p1, q0) * cross(p0, p1, q1) < 0) and (cross(q0, q1, p0) * cross(q0, q1, p1) < 0):
        return 0.0
    return min(d)


def _ends(it: Item):
    h = it.axis[:2] * it.length / 2
    c = it.center[:2]
    return c - h, c + h


def sample_fixtures(rng, reachable=None, tries=4000):
    """Tray and three bins: no overlaps (GAP apart), within reach."""
    for _ in range(tries):
        def draw(size):
            r = rng.uniform(*REACH_R)
            a = math.radians(rng.uniform(-REACH_YAW_DEG, REACH_YAW_DEG))
            return np.array([r * math.cos(a), r * math.sin(a), 0.0])
        tray = draw(TRAY_OUT)
        if not _in_reach(tray, TRAY_OUT):
            continue
        rects = [_rect(tray, TRAY_OUT, GAP / 2)]
        bins, ok = {}, True
        for name in rng.permutation(BINS):
            for _ in range(200):
                c = draw(BIN_OUT)
                if not _in_reach(c, BIN_OUT):
                    continue
                r = _rect(c, BIN_OUT, GAP / 2)
                if any(_overlap(r, o) for o in rects):
                    continue
                if reachable is not None and not reachable(
                        np.array([c[0], c[1], SURFACE_Z + DROP_HEIGHT])):
                    continue
                bins[str(name)] = c
                rects.append(r)
                break
            else:
                ok = False
                break
        if ok:
            return tray, bins
    raise RuntimeError("could not place the tray and bins")


def sample_items(rng, tray, floor_z, reachable=None, tries=3000):
    """Six items lying in the tray, ITEM_CLEAR apart and WALL_CLEAR from the
    walls, each graspable from above across its middle."""
    kinds = list(KINDS)
    rng.shuffle(kinds)
    hx, hy = TRAY_IN[0] / 2, TRAY_IN[1] / 2
    items = []
    for kind in kinds:
        r, length = ITEM_GEOM[kind]
        for _ in range(tries):
            yaw = rng.uniform(-math.pi, math.pi)
            ax = np.array([math.cos(yaw), math.sin(yaw)])
            ext = np.abs(ax) * length / 2 + r            # half extents of the footprint
            m = WALL_CLEAR
            if ext[0] + m >= hx or ext[1] + m >= hy:
                continue
            c2 = tray[:2] + np.array([rng.uniform(-hx + ext[0] + m, hx - ext[0] - m),
                                      rng.uniform(-hy + ext[1] + m, hy - ext[1] - m)])
            colour = str(rng.choice(list(PALETTE)))
            finish = str(rng.choice(FINISHES[kind]))
            it = Item(kind, np.array([c2[0], c2[1], floor_z + r]), float(yaw),
                      bool(rng.integers(2)), colour, finish)
            p0, p1 = _ends(it)
            if any(_seg_dist(p0, p1, *_ends(o)) < r + o.radius + ITEM_CLEAR for o in items):
                continue
            if reachable is not None and not reachable(
                    it.center, spread=np.array([-ax[1], ax[0], 0.0])):
                continue
            items.append(it)
            break
        else:
            return None
    return items


def sample(seed: int, reachable=None) -> Layout:
    """The whole layout for one reset. reachable(p, spread=None) -> bool
    checks a top-down fingertip target (backend IK); None skips the check."""
    rng = np.random.default_rng(seed)
    for _ in range(50):
        tray, bins = sample_fixtures(rng, reachable)
        lay = Layout(seed, tray, bins)
        items = sample_items(rng, tray, lay.tray_floor, reachable)
        if items is not None:
            lay.items = items
            return lay
    raise RuntimeError(f"seed {seed}: no valid layout")
