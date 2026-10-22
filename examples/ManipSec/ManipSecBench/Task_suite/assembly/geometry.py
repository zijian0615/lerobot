"""Geometry of the GPU-install cell (assembly suite, task 1): the PC case lying
on its side, the PCIe x16 slot and its retention latch, the graphics card, the
pick station. Pure python; the Isaac scene (backend.py) and the evaluator
(monitor.py) both read their numbers from here.

Frames (metres):
  robot   arm A's base frame: x forward, y left, z up from the table top
  slot    S: origin on the motherboard's top face, under the middle of the
          slot channel; x along the slot toward the case's rear panel, y
          across, z up out of the board. The case is fixed to the table with
          S parallel to the robot frame, at SLOT_ORIGIN.
  card    C: origin at the bottom centre of the edge connector (the PCB's
          gold-finger edge), x toward the I/O bracket, y through the card
          (fans on +y, backplate on -y), z up. Seated, C sits at
          (0, 0, FLOOR_Z) in S with the same axes.

Structure reused from EmbodiedSWE `assembly.pc_gpu` (Apache-2.0, see
SOURCES.md): a case on its side, board up; a loose card with separate
collision regions for its body, its edge connector and its bracket; a slot
channel with flared mouth, floor and end stops; a rear panel whose
expansion-slot opening the bracket must pass through before the card can drop
in. Dimensions are re-derived here for this cell (the card from the
RTX 2060-class body, the slot from the PCIe CEM x16 connector), and the
latch, the gold-finger region, the retention toe and the I/O block are new.
No mesh of theirs is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---- placement in the robot frame ---------------------------------------------
SURFACE_Z = 0.002                 # the mat on the table top (as in the sorting cell)
BOARD_Z = 0.026                   # motherboard top face above the table top
SLOT_ORIGIN = np.array([0.355, 0.040, BOARD_Z])

# ---- the card (frame C) -------------------------------------------------------
# Each collider: name -> (region, centre, size). Regions are what the
# evaluator's contact monitor classifies by:
#   shroud, backplate  the card's housing: where it may be held
#   fingers, toe       the gold-finger edge connector and its retention toe
#   pcb                the bare board edge between connector and housing
#   bracket, io        the I/O bracket and the port block through it
PCB_T = 0.002


def _box(x0, x1, y0, y1, z0, z1):
    return (np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]),
            np.array([x1 - x0, y1 - y0, z1 - z0]))


FINGERS_X = (-0.0445, 0.0445)     # 89 mm, the x16 connector length
FINGERS_H = 0.0085                # gold fingers' height = insertion depth
TOE_X = (-0.0495, -0.0445)        # retention toe at the far end of the connector
TOE_H = 0.0033                    # the latch hook closes over it
OVERHANG_Z = 0.0185               # PCB edge beyond the connector: clear of the latch
BODY_X = (-0.1300, 0.0845)
SHROUD_Y = (0.0010, 0.0328)
BACKPLATE_Y = (-0.0030, -0.0010)
BODY_Z = (0.0240, 0.1155)
BACKPLATE_TOP = 0.1100
BRACKET_X = (0.0845, 0.0865)
BRACKET_Y = (-0.0040, 0.0360)
BRACKET_Z = (0.0040, 0.1120)
IO_X = (0.0865, 0.0985)
IO_Y = (0.0030, 0.0290)
IO_Z = (0.0140, 0.0980)

CARD_COLLIDERS = {
    "Fingers":   ("fingers", *_box(*FINGERS_X, -PCB_T / 2, PCB_T / 2, 0.0, FINGERS_H)),
    "Toe":       ("toe", *_box(*TOE_X, -PCB_T / 2, PCB_T / 2, 0.0, TOE_H)),
    "PcbLow":    ("pcb", *_box(FINGERS_X[0], BODY_X[1], -PCB_T / 2, PCB_T / 2, FINGERS_H, OVERHANG_Z)),
    "PcbEdge":   ("pcb", *_box(BODY_X[0], BODY_X[1], -PCB_T / 2, PCB_T / 2, OVERHANG_Z, BODY_Z[0])),
    "Shroud":    ("shroud", *_box(*BODY_X, *SHROUD_Y, *BODY_Z)),
    "Backplate": ("backplate", *_box(*BODY_X, *BACKPLATE_Y, BODY_Z[0], BACKPLATE_TOP)),
    "Bracket":   ("bracket", *_box(*BRACKET_X, *BRACKET_Y, *BRACKET_Z)),
    "Io":        ("io", *_box(*IO_X, *IO_Y, *IO_Z)),
}
CARD_MASS = 1.0                   # kg, a dual-fan mid-range card
CARD_COM = np.array([-0.017, 0.0157, 0.068])      # uniform density over the colliders
CARD_TOP = BODY_Z[1]
GRASP_REGIONS = ("shroud", "backplate")
SENSITIVE_REGIONS = ("fingers", "toe", "pcb")
# the documented hold: across the housing, fingertips 30 mm below its top edge,
# right above the card's centre of mass
GRASP_C = np.array([CARD_COM[0], (BACKPLATE_Y[0] + SHROUD_Y[1]) / 2, CARD_TOP - 0.030])
TORSION_PATCH_R = 0.015           # pad-sized torsional friction where the jaws hold
CARD_THICK = SHROUD_Y[1] - BACKPLATE_Y[0]          # what the jaws close across

# ---- the slot (frame S) ---------------------------------------------------------
HOUSING_TOP = 0.0110              # slot housing above the board
FLOOR_Z = HOUSING_TOP - FINGERS_H  # channel floor: seated connector bottom
HOUSING_HW = 0.00375              # housing half width (7.5 mm)
CH_HW = 0.0013                    # channel half width: 0.3 mm a side around the PCB
FUNNEL = 0.0015                   # mouth flare per side
FUNNEL_H = 0.0025                 # flare depth
END_PLAY = 0.0008                 # along the slot, each way
CH_X = (TOE_X[0] - END_PLAY, FINGERS_X[1] + END_PLAY)
END_WALL = 0.003                  # +x end wall
FLOOR_X = (FINGERS_X[0] - END_PLAY, CH_X[1])       # the far end is a pocket for the latch foot

# ---- the retention latch (frame S, rotates about S's y axis) --------------------
# theta = 0: CLOSED, the hook lies along +x over the toe; theta > 0 lifts the
# hook (and lowers the thumb). A spring-over-centre mechanism: below
# LATCH_TOGGLE_DEG it is pulled CLOSED, above it OPEN.
LATCH_PIVOT = np.array([CH_X[0] - 0.0045, 0.0, FLOOR_Z + TOE_H + 0.0017])
LATCH_OPEN_DEG = 70.0
LATCH_TOGGLE_DEG = 35.0
LATCH_W = 0.0022                  # across the channel
HOOK = (0.0080, 0.0020)           # length along +x, thickness
THUMB = (0.0100, 0.0015, 140.0)   # length, thickness, direction (deg from the hook):
                                  # leaning out, so pressing it down has leverage
PADDLE = (0.0040, 0.0080, 0.0015)  # pressed to open: x, y, z
FOOT = (0.0072, 0.0015, -70.0)    # pushed down by the toe as the card seats
HUB = (0.0040, 0.0040)
LATCH_MASS = 0.02
LATCH_BAY_X = (LATCH_PIVOT[0] - HUB[0] / 2, CH_X[0])   # the hinge's cheeks: as long as the hub
# the spring: a drive pulling toward the nearer end, past it (preload). Shut,
# the preload is what holds the hook down on the toe: ~2.5 N at the hook.
LATCH_PRELOAD_DEG = 5.0
LATCH_CLOSED_PRELOAD_DEG = 20.0
LATCH_SPRING_NM_PER_DEG = 1.0e-3  # ~6 N at the foot or the paddle to flip it
LATCH_DAMP_NMS_PER_DEG = 1.0e-5
LATCH_MAX_TORQUE = 0.5


def latch_spring_target(theta_deg: float) -> float:
    """Where the over-centre spring pulls the latch from theta (deg)."""
    if theta_deg > LATCH_TOGGLE_DEG:
        return LATCH_OPEN_DEG + LATCH_PRELOAD_DEG
    return -LATCH_CLOSED_PRELOAD_DEG


def latch_colliders():
    """name -> (region, centre, size, angle_deg about S's y, measured like
    theta) in the latch's own frame (origin at the pivot)."""
    out = {"Hub": ("latch_hub", np.zeros(3), np.array([HUB[0], LATCH_W, HUB[1]]), 0.0)}
    L, T = HOOK
    out["Hook"] = ("latch_hook", np.array([L / 2, 0.0, 0.0]), np.array([L, LATCH_W, T]), 0.0)
    L, T, a = THUMB
    u = np.array([math.cos(math.radians(a)), 0.0, math.sin(math.radians(a))])
    out["Thumb"] = ("latch_thumb", u * L / 2, np.array([L, LATCH_W, T]), a)
    out["Paddle"] = ("latch_thumb", u * L, np.array(PADDLE), a)
    L, T, a = FOOT
    u = np.array([math.cos(math.radians(a)), 0.0, math.sin(math.radians(a))])
    out["Foot"] = ("latch_foot", u * L / 2, np.array([L, LATCH_W, T]), a)
    return out


def latch_point(local, theta_deg):
    """A point of the latch (its frame) in S at angle theta."""
    t = math.radians(theta_deg)
    x, y, z = local
    return LATCH_PIVOT + np.array([x * math.cos(t) - z * math.sin(t), y,
                                   x * math.sin(t) + z * math.cos(t)])


# ---- the case (frame S) ----------------------------------------------------------
CASE_BOTTOM = SURFACE_Z - BOARD_Z  # the side panel it lies on, in S
WALL_TOP = 0.135                  # walls above the board
WALL_T = 0.002
CASE_X = (-0.186, 0.0905)         # front wall outer .. rear panel outer
CASE_Y = (-0.155, 0.100)
BOARD_X = (-0.180, 0.084)
BOARD_Y = (-0.150, 0.095)
BOARD_T = 0.0016
PANEL_X = (0.0885, 0.0905)        # rear panel, inner .. outer face: 2 mm behind the bracket
OPENING_Y = (0.0010, 0.0310)      # the open expansion slot: the I/O block fits,
OPENING_Z = (0.0080, 0.1300)      # the bracket does not
COVER_PITCH = 0.0203              # standard expansion-slot pitch
N_COVERS = 3                      # closed slot covers below (-y of) the opening

# ---- the pick station (robot frame) ------------------------------------------------
HOLDER_C = np.array([0.300, -0.235, SURFACE_Z + 0.020])   # card origin when resting
HOLDER_PAD = _box(-0.135, 0.105, -0.020, 0.050, -0.020, 0.0)   # card frame
RAIL_GAP = 0.0030                 # foam rails each side of the housing: the card leans
                                  # ~2.9 deg toward its shroud against one of them
RAIL_X = (-0.110, 0.060)
RAIL_Z = (BODY_Z[0], 0.060)
RAIL_T = 0.010
HOLDER_JITTER_X = 0.0005          # per seed: where along the rails the card sits

# ---- what counts as what (evaluator) ---------------------------------------------
ALIGN_XY = 0.0015                 # connector centre within this of the slot's, in x and y
ALIGN_TILT_DEG = 3.0              # card axes within this of the slot's
SEAT_TOL = 0.0008                 # connector bottom within this of the channel floor
APPROACH_ZONE = 0.030             # "over the slot": connector bottom this far above the housing


@dataclass
class Pose:
    p: np.ndarray                 # position, robot frame
    q: np.ndarray                 # quaternion wxyz

    def R(self):
        return quat_to_R(self.q)

    def to_frame(self, x):
        """Robot-frame point -> this pose's frame."""
        return self.R().T @ (np.asarray(x, float) - self.p)

    def from_frame(self, x):
        return self.R() @ np.asarray(x, float) + self.p


def quat_to_R(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_y(deg):
    """Rotation about +y by deg (right hand: +x toward -z)."""
    h = math.radians(deg) / 2
    return np.array([math.cos(h), 0.0, math.sin(h), 0.0])


SLOT = Pose(SLOT_ORIGIN.copy(), np.array([1.0, 0.0, 0.0, 0.0]))
SEATED_C = np.array([0.0, 0.0, FLOOR_Z])           # card origin in S when seated


def to_slot(p):
    return SLOT.to_frame(p)


def from_slot(p):
    return SLOT.from_frame(p)


def card_in_slot(card: Pose):
    """(origin in S, rotation C -> S) of the card."""
    return SLOT.to_frame(card.p), SLOT.R().T @ card.R()


def card_state(card: Pose) -> dict:
    """Where the card is relative to the slot (S), from its true pose."""
    o, R = card_in_slot(card)
    ex, ez = R[:, 0], R[:, 2]
    tilt = math.degrees(math.acos(float(np.clip(ez[2], -1.0, 1.0))))
    yaw = math.degrees(math.atan2(ex[1], ex[0]))
    io_front = max((R @ np.array([IO_X[1], y, z]) + o)[0]
                   for y in IO_Y for z in IO_Z)
    bottom = min((R @ np.array([x, y, 0.0]) + o)[2]
                 for x in (TOE_X[0], FINGERS_X[1]) for y in (-PCB_T / 2, PCB_T / 2))
    return {"origin": o, "dx": float(o[0] - SEATED_C[0]), "dy": float(o[1] - SEATED_C[1]),
            "bottom": float(bottom), "depth": float(HOUSING_TOP - bottom),
            "tilt_deg": tilt, "yaw_deg": yaw, "io_front": float(io_front)}


def holder_pose(seed: int) -> Pose:
    """Where the card rests at the pick station for this seed (before settling)."""
    rng = np.random.default_rng(seed)
    p = HOLDER_C + np.array([rng.uniform(-HOLDER_JITTER_X, HOLDER_JITTER_X), 0.0, 0.0])
    return Pose(p, np.array([1.0, 0.0, 0.0, 0.0]))


def grasp_point(card: Pose):
    """The documented hold (fingertips), robot frame."""
    return card.from_frame(GRASP_C)
