# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""One parameter set for the 8 x SO-100 loop cell: part sizes, unit layout, chute routes.

``build.py`` turns this into printable CAD, a BOM and the 3D page's layout, so the page and
the parts never disagree. Lengths in mm for parts, metres for the table layout (z up, x right,
y towards the far side of the table = top of the sketch).
"""

from __future__ import annotations

import math

# ---------------- printed base ----------------
TILE = 160.0  # base tile edge (mm); every arm stands on one whole tile, units are 2x1 .. 3x2 tiles
TILE_T = 10.0  # tile thickness
PAD_T = 5.0  # TPU foot pads under the tile corners; taller than a joiner plate + its flush screws
JOINER_T = 4.0  # seam plate under two tiles, countersunk M4
GRID = 20.0  # M4 heat-set insert grid on every tile (first hole 10 mm from an edge)
PT = (TILE_T + PAD_T) / 1000  # tile top above the table (m)

# ---------------- carrier: printed puck + bought clear tube + printed collar/cap ----------------
TUBE_OD, TUBE_L = 25.0, 100.0  # bought clear PET/PC tube, open both ends, glued into the puck
PUCK_D = 45.0
SOLE_T = 1.0  # glued-on sole that keeps the rolling balls in
BASE_H = 8.0  # puck base above the sole
NECK_D, NECK_H = 30.0, 8.0
SOCKET_D, SOCKET_DEPTH = 25.4, 9.0
BB_D, BB_LIP_D, BB_RING = 8.0, 7.2, 17.5  # 3 steel balls Ø8 per puck -> it rolls, not slides
BB_PROTRUDE = BB_D / 2 - SOLE_T - math.sqrt((BB_D / 2) ** 2 - (BB_LIP_D / 2) ** 2)  # ~1.26 mm
PUCK_TOP = SOLE_T + BASE_H + NECK_H  # 17 mm above the sole bottom
# heights above the chute floor (the balls touch the floor)
SOLE_Z = BB_PROTRUDE
TUBE_BOTTOM_Z = SOLE_Z + PUCK_TOP - SOCKET_DEPTH
TUBE_TOP_Z = TUBE_BOTTOM_Z + TUBE_L
GRIP_Z = 70.0  # where the gripper takes the tube

COLLAR_ID, COLLAR_OD, COLLAR_H = 25.3, 29.0, 12.0
THREAD_RADIAL, THREAD_LEAD, THREAD_SPAN, THREAD_TOP = 1.0, 16.0, 60.0, 9.5  # 4-start lug finish
CAP_OD, CAP_H, CAP_TOP_T, CAP_CLEAR = 36.0, 16.0, 2.5, 0.3
LUG_W, LUG_H = 3.0, 1.5

BALL_D = 16.0  # bought, uniform diameter (red / green / blue)
N_BOTTLES, N_CAPS, N_BALLS = 9, 6, 33

# ---------------- chutes ----------------
FLOOR_T, WALL_T = 3.0, 3.0
END_Z = 0.045  # chute floor at every inlet stop (m)
FAMILIES = {
    # inner width W, wall height H above the floor, corner radius R, slope, start/end module lengths
    "bottle": dict(W=PUCK_D + 4, H=20.0, R=80.0, slope=3.0, flare=80.0, gate=80.0, item=PUCK_D, beam_z=10.0),
    "cap": dict(W=CAP_OD + 5, H=12.0, R=80.0, slope=9.0, flare=80.0, gate=80.0, item=CAP_OD, beam_z=8.0),
    "ball": dict(W=BALL_D + 3, H=11.0, R=48.0, slope=3.0, flare=64.0, gate=64.0, item=BALL_D, beam_z=8.0),
}
STRAIGHTS = (160.0, 80.0)

# Gate: an SG90 lies on its side in a plate under the floor, shaft across the chute (along y). The
# lever swings in a vertical plane: level = pin up through the floor slot, turned ``release_deg`` =
# pin tip under the floor. The pin rises off-centre, where the head item (against the end wall) and
# the next one touching it leave a wedge-shaped gap, so it always fits between them and holds the
# rest back while an arm lifts the head item. Frame: gate module, x along the flow from its upstream
# end, y across, z = 0 at the floor bottom.
SG90 = dict(body=(22.8, 12.2), ear_w=32.4, hole_pitch=27.8, shaft_off=5.5, below_ears=15.9, shaft_tip=11.0)
LEVER_R, LEVER_T, LEVER_H = 18.0, 3.0, 6.0  # pivot -> pin, bar thickness (y), bar height (z)
SHAFT_Z = -12.0
PIN_D = 2.8
SERVO_PLATE_DY = (-15.5, -11.5)  # plate faces relative to the lever plane; ears rest on the +y face


def gate_geom(fam: str) -> dict:
    """Where everything in a gate module sits (mm, gate frame)."""
    f = FAMILIES[fam]
    ln, item = f["gate"], f["item"]
    head_x = ln - 4 - 1 - item / 2
    pin_x = ln - 4 - 1 - item  # where the head item touches the next one
    r = item / 2
    pin_y = math.ceil(math.sqrt(r * r - (r - (PIN_D + 1) / 2) ** 2)) + 1.0  # wedge gap >= pin + 1 mm
    hold = min(10.0, f["H"] / 2)
    pin_len = FLOOR_T + hold - (SHAFT_Z + LEVER_H / 2)
    for d in (1, -1):  # lever runs from the shaft towards +x or -x
        for off in (SG90["shaft_off"], -SG90["shaft_off"]):
            shaft_x = pin_x - d * LEVER_R
            cx = shaft_x + off
            if cx - 18 >= 0 and cx + 18 <= ln - 4:
                break
        else:
            continue
        break
    else:
        raise ValueError(f"{fam}: no room for the SG90 under the gate")
    # release angle: smallest turn that drops the pin tip 1 mm under the floor top
    rel = 0.0
    while True:
        a = math.radians(rel)
        tip_z = SHAFT_Z - LEVER_R * math.sin(a) + (LEVER_H / 2 + pin_len) * math.cos(a)
        if tip_z < FLOOR_T - 1.0 or rel > 60:
            break
        rel += 1.0
    a = math.radians(rel)
    # where the pin crosses the floor top while turning -> slot from pin_x outward
    reach = 0.0
    for k in range(int(rel) + 1):
        b = math.radians(k)
        bx, bz = LEVER_R * math.cos(b) + (LEVER_H / 2) * math.sin(b), -LEVER_R * math.sin(b) + (LEVER_H / 2) * math.cos(b)
        t = (FLOOR_T - SHAFT_Z - bz) / math.cos(b)
        if 0 <= t <= pin_len:
            reach = max(reach, bx + t * math.sin(b) - LEVER_R)
    slot = sorted((pin_x - d * 1.6, pin_x + d * (reach + 1.6)))
    return dict(head_x=head_x, beam_z=FLOOR_T + f["beam_z"], pin_x=pin_x, pin_y=pin_y, dir=d, shaft_x=shaft_x, shaft_z=SHAFT_Z, servo_cx=cx,
                pin_len=pin_len, hold=hold, release_deg=rel, slot=slot, plate_y=(pin_y + SERVO_PLATE_DY[0], pin_y + SERVO_PLATE_DY[1]))  # standard straight lengths; the rest is one trim piece per leg
SADDLE_L, SADDLE_T, SADDLE_STEP = 30.0, 4.0, 0.5  # joint saddle; downstream side 0.5 mm lower
FOOT_H = 20.0  # post foot incl. boss; spacers stack on top
SPACERS = (20.0, 10.0, 5.0, 2.0)

# Arms are SO-100 from TheRobotStudio's URDF: see so100.py / so100_model.json.
# Every arm owns one whole tile (``plate``: its centre): the base and its clamp dogs stay inside
# that 160 x 160 mm square, and no chute, nest, camera or cone post may enter it.
ARM_PLATE = TILE

# ---------------- layout (m) ----------------
# Table 1.45 x 1.19 m. Tiles sit on a 160 mm grid (tile centres at multiples of 0.16 m). The bottle
# loop runs round the table edge (top y=0.48, right x=0.64, bottom y=-0.48, left x=-0.66) with the
# nests on it; every arm stands on its own tile just inside the loop, so the middle stays free for
# the ball lanes, the cap chute and the funnel -> hopper -> sorter path.
UNITS = [
    dict(id="U1", name="灌装", sub="取球放入瓶口", rect=[-0.24, 0.24, -0.08, 0.56], nest=[-0.16, 0.48], arms=["A1"], inP="P51", outP="P12"),
    dict(id="U2", name="封盖拧紧", sub="扶瓶 + 放盖拧紧", rect=[0.40, 0.08, 0.72, 0.40], nest=[0.64, 0.24], arms=["A2", "A3"], inP="P12", outP="P23"),
    dict(id="U3", name="质检追溯", sub="相机读序列和码", rect=[0.40, -0.24, 0.72, -0.08], nest=[0.64, -0.16], arms=["A4"], inP="P23", outP="P34"),
    dict(id="U4", name="拆盖", sub="扶瓶 + 拧开", rect=[-0.08, -0.56, 0.24, -0.24], nest=[0.08, -0.48], arms=["A5", "A6"], inP="P34", outP="P45"),
    dict(id="U5", name="倒球", sub="倾倒 + 查空", rect=[-0.56, -0.56, -0.24, -0.24], nest=[-0.32, -0.48], arms=["A7"], inP="P45", outP="P51"),
    dict(id="U6", name="分拣", sub="相机按色分球", rect=[-0.40, -0.08, -0.24, 0.08], nest=None, arms=["A8"]),
]
# ``plate``: centre of the arm's own tile. ``xy`` (pan axis) and ``yaw`` (base yaw, degrees, squared to the
# tile so the clamp dogs sit along its edges) come from ``python -m loop_cell.build --place``.
ARMS = [
    dict(id="A1", unit="U1", plate=[-0.16, 0.32], xy=[-0.16, 0.305], yaw=270, role="取球放入瓶口;也负责把瓶子从入口搬进工位、再送到出口。"),
    dict(id="A2", unit="U2", plate=[0.48, 0.32], xy=[0.47, 0.29], yaw=0, role="扶瓶;负责瓶子进出工位。"),
    dict(id="A3", unit="U2", plate=[0.48, 0.16], xy=[0.465, 0.16], yaw=0, role="从盖滑道取盖,放到瓶口,转约 90–120° 锁紧。"),
    dict(id="A4", unit="U3", plate=[0.48, -0.16], xy=[0.465, -0.17], yaw=0, role="把瓶子举到相机前转一圈,读出球的颜色序列和托座码。"),
    dict(id="A5", unit="U4", plate=[0.16, -0.32], xy=[0.13, -0.32], yaw=270, role="扶瓶;负责瓶子进出工位。"),
    dict(id="A6", unit="U4", plate=[0.0, -0.32], xy=[0.0, -0.32], yaw=90, role="反向转开瓶盖,放进盖滑道送回 U2。"),
    dict(id="A7", unit="U5", plate=[-0.32, -0.32], xy=[-0.305, -0.32], yaw=0, role="把瓶子倒扣在漏斗上倒出小球,确认倒空后放回。"),
    dict(id="A8", unit="U6", plate=[-0.32, 0.0], xy=[-0.35, -0.015], yaw=270, role="从料斗出口取球,相机认颜色,放进红、绿、蓝球道。"),
]


def _lane(offset: float) -> list[list[float]]:
    """45° ball lane from U6 up to U1; ``offset`` shifts it sideways (m)."""
    ex, ey, length = -0.305, 0.335, 0.25
    d = (math.sqrt(0.5), math.sqrt(0.5))
    n = (-d[1] * offset, d[0] * offset)
    end = (ex + n[0], ey + n[1])
    start = (end[0] - d[0] * length, end[1] - d[1] * length)
    return [[round(start[0], 4), round(start[1], 4)], [round(end[0], 4), round(end[1], 4)]]


# Route = polyline from the start of the first module to the stop point at the inlet (m).
ROUTES = {
    "P12": dict(family="bottle", pts=[[-0.115, 0.48], [0.64, 0.48], [0.64, 0.31]]),
    "P23": dict(family="bottle", pts=[[0.64, 0.195], [0.64, -0.09]]),
    "P34": dict(family="bottle", pts=[[0.64, -0.205], [0.64, -0.48], [0.15, -0.48]]),
    "P45": dict(family="bottle", pts=[[0.035, -0.48], [-0.25, -0.48]]),
    "P51": dict(family="bottle", pts=[[-0.365, -0.48], [-0.66, -0.48], [-0.66, 0.48], [-0.23, 0.48]]),
    "CAP": dict(family="cap", pts=[[0.03, -0.19], [0.31, 0.12]]),
    "red": dict(family="ball", pts=_lane(0.026)),
    "green": dict(family="ball", pts=_lane(0.0)),
    "blue": dict(family="ball", pts=_lane(-0.026)),
    "M1": dict(family="ball", pts=[[-0.50, -0.29], [-0.50, -0.215]], transit=True, end_z=0.118),  # just above the hopper rim
    "M2": dict(family="ball", pts=[[-0.50, -0.12], [-0.50, -0.05]], no_flare=True),
}
FUNNEL = dict(xy=[-0.50, -0.29], top_r=45.0, out_r=11.0, h=35.0)  # mm sizes
SPOUT_GAP = BALL_D + 2.0  # funnel / hopper spout bottom above the chute floor: a ball rolls out under it
POUR_Z = dict(above=0.26, at=0.235)  # A7's gripper height over the funnel, then while pouring (m)
HOPPER = dict(xy=[-0.50, -0.17], top_r=50.0, out_r=10.5, h=40.0)
INSPECT = [0.59, -0.16, 0.21]  # gripper tip while the bottle is in front of the U3 camera
CAMERAS = {"U3": dict(at=[0.70, -0.16, 0.19], look=[0.59, -0.16, 0.19]), "U6": dict(at=[-0.43, -0.05, 0.17], look=[-0.50, -0.05, 0.05])}
TABLE = dict(x=[-0.725, 0.725], y=[-0.595, 0.595])
