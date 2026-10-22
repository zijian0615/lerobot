"""LOTO fixture geometry and interaction tolerances.

Pure Python: shared by the Isaac Sim backend (which draws it) and the
kinematic backend (which only needs the numbers). All positions are in arm A's
robot base frame, metres, z up from the table top -- the frame fk.py uses.

Two arms. Arm A (the original ER-4iA) holds the open padlock by its body,
tool horizontal (+x), and hooks it onto the hasp. Arm B, across the table,
presses the shackle down from above until it clicks shut.

Lockout hardware (looking along +y, lever at OFF):

     bow  ┌────────┐                     the padlock's legs run along x: the
          │        │ free leg            free leg (+x) goes down through a
          │      ══╪══════  lever ear    hole near the -x edge of the lever's
   fixed  │      ══╪══════  bracket      ear and of the fixed bracket below it
   leg    │        │                     (they line up only at OFF); the fixed
        ┌─┴────────┴─┐                   leg stays outside the edge
        │    body    │
        └────────────┘

Open, the shackle stands RAISE higher, leaving a gap between the free leg's
tip and the body: arm A slides the plates' -x edge into that gap (moving +x),
arm B then presses the bow down so the free leg passes the hole into the body.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---- arm B ------------------------------------------------------------------
ARM_B_BASE = (0.40, 0.62, 0.0)           # in arm A's frame, on the same table
ARM_B_YAW_DEG = -90.0                    # facing -y, toward the disconnect
ARM_B_PARK = (0.40, 0.52, 0.40)          # fingertips when idle: leaning back,
                                         # clear of arm A's path to the lock

# ---- disconnect enclosure ---------------------------------------------------
ENCLOSURE_SIZE = (0.10, 0.14, 0.16)      # x, y, z
LEVER_LEN = 0.08                         # pivot -> grab point
LEVER_PIVOT_DY = -0.01                   # pivot offset from enclosure centre, y
LEVER_ON_DEG = -45.0                     # tilt in the y-z plane, from vertical
LEVER_OFF_DEG = +45.0                    # (+ is toward the hasp side, +y)
LEVER_LATCH_DEG = 40.0                   # past this the detent holds it OFF

# ---- lockout hasp: lever ear over a fixed bracket, two holes ----------------
HASP_PLATE_T = 0.004                     # each plate
HASP_HALF_X = 0.035                      # plates span disconnect.x +/- this
HASP_HOLE_FROM_EDGE = 0.009              # hole centres in from the -x edge (3 mm web)
HASP_HOLES_DY = (0.0075, 0.035)          # hole A, hole B, from the +y face: a
                                         # lock in A hangs clear of the enclosure
HASP_EDGE_DY = 0.045                     # plates' outer edge beyond the +y face
HASP_BRACKET_DY = -0.012                 # bracket starts this far inside the face
HOLE_HALF = 0.006                        # half-width of the (square) holes: 12 mm

# ---- padlock (body frame: origin at the body centre, legs along x) ----------
LOCK_BODY = (0.030, 0.015, 0.042)        # x, y, z (tall enough for the 36 mm travel)
SHACKLE_S = 0.018                        # leg spacing: free leg +x, fixed leg -x
SHACKLE_R = 0.0025
BOW_T = 0.005
BOW_CLOSED = 0.050                       # bow top above the body centre, shut
RAISE = 0.036                            # extra shackle height when open
FREE_LEG_LEN = 0.041                     # bow top -> free-leg tip
PEG_TOL = HOLE_HALF - SHACKLE_R          # 2 mm lateral clearance in a hole
BODY_GRASP = (-0.006, 0.0, 0.0)          # fingertips, tool along +x
LOCK_STAND_H = 0.13                      # the open lock waits on a stand

# ---- terminal block (test point and service terminals) ----------------------
TERMINAL_SIZE = (0.06, 0.08, 0.05)

# ---- other props ---------------------------------------------------------------
TAG_SIZE = (0.040, 0.004, 0.068)         # plate in the x-z plane
TAG_DX = -0.030                          # plate hangs beside the lock body
TAG_HANG_DZ = 0.008                      # hook point below the hasp, fixed leg
TRAY_SIZE = (0.12, 0.12, 0.016)
KNOB_SIZE = (0.014, 0.014, 0.022)        # fits the 18 mm jaws

# width the jaws stall at when gripping each item (pneumatic: commanded shut,
# stopped by the part)
GRIP_WIDTH = {"lock": LOCK_BODY[1], "lever": KNOB_SIZE[0], "tag": TAG_SIZE[1]}

# ---- maintenance: a screw driven into the disconnect's cover -----------------
SCREW_LEN = 0.020                        # shank, tip to underside of the head
SCREW_HEAD = (0.010, 0.004)              # head diameter, height
SCREW_TOL = 0.002                        # tip within this of the hole axis
COVER_SIZE = (0.050, 0.040, 0.003)       # cover plate on the enclosure top

# ---- clamp ammeter: NVIDIA SimReady Clamp_Multimeter_B01, already clamped
# around the supply riser (a cable coming up through a table grommet into the
# terminal block). It reads only once its power button has been pressed.
METER_USD = ("/Isaac/SimReady/Industrial/Tools/Testing/Clamp_Multimeter_B01/"
             "sm_testingTool_multimeter_clamp_b01_01.usd")
METER_RING = (-0.090, 0.003)             # jaw hole centre, asset frame (mesh)
METER_BUTTON = (0.0185, -0.013, 0.024)   # power button (button_01) top centre
METER_Z = 0.05                           # asset bottom above the table
RISER_H = 0.10                           # supply riser height
CABLE_R = 0.006
METER_PRESS_R = 0.006                    # fingertip to the button's centre (xy)
METER_PRESS_DZ = 0.002                   # ... and down to within this of its top
GRIP_WIDTH.update(screw=SCREW_HEAD[0])

# ---- interaction tolerances --------------------------------------------------
GRASP_R = 0.025       # tool tip to a grasp point, to pick it up
PLACE_R = 0.030       # released this close to its tag point, the tag hangs there
PRESS_R = 0.010       # arm B's fingertips this close (xy) to the bow to press it
PROBE_R = 0.015       # tool tip to the test point, for the probe to read
PROBE_DWELL_S = 0.20  # held in contact this long, the reading counts as a verify
GRIP_CLOSED = 0.5     # gripper command above this is "closed"
LOCK_GRIP_COS = 0.9   # jaws must close within ~25 deg of the lock's thin (y) axis
LOCK_TOOL_HORIZ = 0.3  # and the tool must be near horizontal (|approach.z| <)


@dataclass
class Layout:
    disconnect: np.ndarray
    test_point: np.ndarray
    lock_start: np.ndarray       # body centre of the open lock on its stand
    tag_start: np.ndarray        # tag hole, tag standing on the tray
    screw_start: np.ndarray      # screw head top, screw standing on the tray
    screw_hole: np.ndarray       # hole in the cover, at its top surface
    riser: np.ndarray            # supply riser foot (the meter is clamped on it)

    @classmethod
    def from_cfg(cls, cfg):
        a = lambda v: np.asarray(v, float)
        return cls(a(cfg.disconnect_pose), a(cfg.test_point_pose),
                   a(cfg.lock_start_pose), a(cfg.tag_start_pose),
                   a(cfg.screw_start_pose), a(cfg.screw_hole_pose),
                   a(cfg.meter_riser_pose))

    # ---- lever: pivots about an x-parallel axis on the enclosure top -------
    @property
    def lever_pivot(self) -> np.ndarray:
        return self.disconnect + [0.0, LEVER_PIVOT_DY, ENCLOSURE_SIZE[2]]

    def lever_tip(self, deg: float) -> np.ndarray:
        t = math.radians(deg)
        return self.lever_pivot + LEVER_LEN * np.array([0.0, math.sin(t), math.cos(t)])

    def lever_angle_toward(self, p) -> float:
        """Lever angle that points the grab point at p (clamped to its travel)."""
        d = np.asarray(p, float) - self.lever_pivot
        deg = math.degrees(math.atan2(d[1], d[2]))
        return float(np.clip(deg, LEVER_ON_DEG, LEVER_OFF_DEG))

    # ---- hasp ---------------------------------------------------------------
    @property
    def face_y(self) -> float:
        return float(self.disconnect[1] + ENCLOSURE_SIZE[1] / 2)

    @property
    def plate_top(self) -> float:
        """Top of the lever ear (centred on the knob height at OFF)."""
        return float(self.lever_tip(LEVER_OFF_DEG)[2] + HASP_PLATE_T / 2)

    @property
    def plate_bottom(self) -> float:
        return self.plate_top - 2 * HASP_PLATE_T

    @property
    def plate_x(self) -> tuple:
        return self.disconnect[0] - HASP_HALF_X, self.disconnect[0] + HASP_HALF_X

    @property
    def plate_y(self) -> tuple:
        return self.face_y + HASP_BRACKET_DY, self.face_y + HASP_EDGE_DY

    @property
    def holes(self) -> list:
        """Hole centres at the top of the plate stack, A then B."""
        x = self.plate_x[0] + HASP_HOLE_FROM_EDGE
        return [np.array([x, self.face_y + dy, self.plate_top]) for dy in HASP_HOLES_DY]

    def in_plates(self, p, r=0.0) -> bool:
        """Is point p (grown by r) inside the plate stack's volume?"""
        (x0, x1), (y0, y1) = self.plate_x, self.plate_y
        return (x0 - r <= p[0] <= x1 + r and y0 - r <= p[1] <= y1 + r
                and self.plate_bottom - r <= p[2] <= self.plate_top + r)

    # ---- padlock --------------------------------------------------------------
    @staticmethod
    def bow_top(center, travel) -> np.ndarray:
        """travel = how far the shackle is pressed in from open (0..RAISE)."""
        return np.asarray(center, float) + [0.0, 0.0, BOW_CLOSED + RAISE - travel]

    @staticmethod
    def free_leg(center, travel) -> tuple:
        """(x, y, z_tip, z_top) of the free leg."""
        top = Layout.bow_top(center, travel)
        return top[0] + SHACKLE_S / 2, top[1], top[2] - FREE_LEG_LEN, top[2] - BOW_T

    def hook_center(self, i: int) -> np.ndarray:
        """Body centre of an open lock hooked on hole i, ready to press: the
        free leg over the hole, its tip just above the plates, the body top
        just below them."""
        h = self.holes[i]
        # the plates sit centred in the gap: equal clearance above and below
        gap = BOW_CLOSED + RAISE - FREE_LEG_LEN - LOCK_BODY[2] / 2
        margin = (gap - 2 * HASP_PLATE_T) / 2
        return np.array([h[0] - SHACKLE_S / 2, h[1],
                         self.plate_bottom - margin - LOCK_BODY[2] / 2])

    def locked_center(self, i: int) -> np.ndarray:
        """Body centre of a shut lock hanging in hole i (bow on the plate)."""
        h = self.holes[i]
        return np.array([h[0] - SHACKLE_S / 2, h[1],
                         self.plate_top + BOW_T - BOW_CLOSED])

    def tag_site(self, i: int) -> np.ndarray:
        """Where a tag's hole hooks onto the fixed leg of the lock in hole i."""
        c = self.locked_center(i)
        return np.array([c[0] - SHACKLE_S / 2, c[1], self.plate_bottom - TAG_HANG_DZ])

    @staticmethod
    def lock_grasp(center) -> np.ndarray:
        return np.asarray(center, float) + BODY_GRASP

    # ---- maintenance screw ---------------------------------------------------
    @staticmethod
    def screw_tip(head_top) -> np.ndarray:
        return np.asarray(head_top, float) - [0.0, 0.0, SCREW_HEAD[1] + SCREW_LEN]

    def screw_head_at(self, progress: float) -> np.ndarray:
        """Head top of a screw engaged in the hole, driven in by `progress`."""
        return self.screw_hole + [0.0, 0.0, SCREW_HEAD[1] + SCREW_LEN * (1 - progress)]

    # ---- clamp ammeter ---------------------------------------------------------
    @property
    def meter_origin(self) -> np.ndarray:
        """Where the meter asset's origin sits (jaw hole around the riser)."""
        return np.array([self.riser[0] - METER_RING[0], self.riser[1] - METER_RING[1],
                         METER_Z])

    @property
    def meter_button(self) -> np.ndarray:
        """Top centre of the meter's power button."""
        return self.meter_origin + METER_BUTTON

    # ---- props at rest ---------------------------------------------------------
    @property
    def tray_center(self) -> np.ndarray:
        return np.array([self.tag_start[0], self.tag_start[1], TRAY_SIZE[2] / 2])

    @property
    def stand_top(self) -> np.ndarray:
        return np.array([self.lock_start[0], self.lock_start[1], LOCK_STAND_H])

    def tag_rest_z(self) -> float:
        return float(self.tag_start[2] - TRAY_SIZE[2])

    def rest_z(self, prop: str) -> float:
        """Grasp-point height of a prop lying on the table (dropped)."""
        start = {"tag": self.tag_start, "screw": self.screw_start}[prop]
        return float(start[2] - TRAY_SIZE[2])
