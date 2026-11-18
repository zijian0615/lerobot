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
TILE = 200.0  # base tile edge (mm); units are 2x1 or 2x2 tiles
TILE_T = 10.0  # tile thickness
PAD_T = 2.0  # TPU foot pads under the tiles
GRID = 25.0  # M4 heat-set insert grid on every tile (first hole 12.5 mm from an edge)
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
    "bottle": dict(W=PUCK_D + 4, H=20.0, R=100.0, slope=3.0, flare=100.0, gate=100.0, item=PUCK_D, beam_z=10.0),
    "cap": dict(W=CAP_OD + 5, H=12.0, R=100.0, slope=9.0, flare=100.0, gate=100.0, item=CAP_OD, beam_z=8.0),
    "ball": dict(W=BALL_D + 3, H=11.0, R=60.0, slope=3.0, flare=80.0, gate=80.0, item=BALL_D, beam_z=8.0),
}
STRAIGHTS = (200.0, 100.0)  # standard straight lengths; the rest is one trim piece per leg
SADDLE_L, SADDLE_T, SADDLE_STEP = 30.0, 4.0, 0.5  # joint saddle; downstream side 0.5 mm lower
FOOT_H = 20.0  # post foot incl. boss; spacers stack on top
SPACERS = (20.0, 10.0, 5.0, 2.0)

# Arms are SO-100 from TheRobotStudio's URDF: see so100.py / so100_model.json. ``xy`` below is the
# pan axis; the base yaw is chosen by routes.arm_yaws so the arm faces the middle of its work.

# ---------------- layout (m) ----------------
UNITS = [
    dict(id="U1", name="灌装", sub="取球放入瓶口", rect=[-0.2, 0.2, 0.2, 0.6], nest=[0.0375, 0.5375], arms=["A1"], inP="P51", outP="P12"),
    dict(id="U2", name="封盖拧紧", sub="扶瓶 + 放盖拧紧", rect=[0.4, 0.0, 0.8, 0.4], nest=[0.6, 0.2125], arms=["A2", "A3"], inP="P12", outP="P23"),
    dict(id="U3", name="质检追溯", sub="相机读序列和码", rect=[0.4, -0.4, 0.8, -0.2], nest=[0.7, -0.3], arms=["A4"], inP="P23", outP="P34"),
    dict(id="U4", name="拆盖", sub="扶瓶 + 拧开", rect=[-0.2, -0.8, 0.2, -0.4], nest=[0.0, -0.6], arms=["A5", "A6"], inP="P34", outP="P45"),
    dict(id="U5", name="倒球", sub="倾倒 + 查空", rect=[-0.6, -0.6, -0.2, -0.2], nest=[-0.5125, -0.275], arms=["A7"], inP="P45", outP="P51"),
    dict(id="U6", name="分拣", sub="相机按色分球", rect=[-0.6, 0.0, -0.2, 0.2], nest=None, arms=["A8"]),
]
ARMS = [
    dict(id="A1", unit="U1", xy=[0.0375, 0.3875], role="取球放入瓶口;也负责把瓶子从入口搬进工位、再送到出口。"),
    dict(id="A2", unit="U2", xy=[0.725, 0.2125], role="扶瓶;负责瓶子进出工位。"),
    dict(id="A3", unit="U2", xy=[0.4875, 0.3], role="从盖滑道取盖,放到瓶口,转约 90–120° 锁紧。"),
    dict(id="A4", unit="U3", xy=[0.4875, -0.275], role="把瓶子举到相机前转一圈,读出球的颜色序列和托座码。"),
    dict(id="A5", unit="U4", xy=[0.0125, -0.7125], role="扶瓶;负责瓶子进出工位。"),
    dict(id="A6", unit="U4", xy=[-0.0625, -0.4875], role="反向转开瓶盖,放进盖滑道送回 U2。"),
    dict(id="A7", unit="U5", xy=[-0.4125, -0.4625], role="把瓶子倒扣在漏斗上倒出小球,确认倒空后放回。"),
    dict(id="A8", unit="U6", xy=[-0.4625, 0.125], role="从料斗出口取球,相机认颜色,放进红、绿、蓝球道。"),
]


def _lane(offset: float) -> list[list[float]]:
    """45° ball lane from U6 to U1; ``offset`` shifts it sideways (m)."""
    ex, ey, length = -0.12, 0.46, 0.269
    d = (math.sqrt(0.5), math.sqrt(0.5))
    n = (-d[1] * offset, d[0] * offset)
    end = (ex + n[0], ey + n[1])
    start = (end[0] - d[0] * length, end[1] - d[1] * length)
    return [[round(start[0], 4), round(start[1], 4)], [round(end[0], 4), round(end[1], 4)]]


# Route = polyline from the start of the first module to the stop point at the inlet (m).
ROUTES = {
    "P12": dict(family="bottle", pts=[[0.1625, 0.5375], [0.6, 0.5375], [0.6, 0.3375]]),
    "P23": dict(family="bottle", pts=[[0.6, 0.0875], [0.6, -0.2375]]),
    "P34": dict(family="bottle", pts=[[0.6, -0.35], [0.6, -0.6], [0.1375, -0.6]]),
    "P45": dict(family="bottle", pts=[[-0.1125, -0.6], [-0.3125, -0.6], [-0.3125, -0.3625]]),
    "P51": dict(family="bottle", pts=[[-0.55, -0.3625], [-0.775, -0.3625], [-0.775, 0.5625], [-0.1125, 0.5625]]),
    "CAP": dict(family="cap", pts=[[0.1125, -0.45], [0.4625, 0.1375]]),
    "red": dict(family="ball", pts=_lane(0.026)),
    "green": dict(family="ball", pts=_lane(0.0)),
    "blue": dict(family="ball", pts=_lane(-0.026)),
    "M1": dict(family="ball", pts=[[-0.4, -0.235], [-0.4, -0.145]], transit=True, end_z=0.095),
    "M2": dict(family="ball", pts=[[-0.4, -0.05], [-0.4, 0.0375]], no_flare=True),
}
FUNNEL = dict(xy=[-0.4, -0.235], top_r=45.0, out_r=11.0, h=35.0)  # mm sizes
HOPPER = dict(xy=[-0.4, -0.1], top_r=50.0, out_r=10.5, h=40.0)
INSPECT = [0.69, -0.3, 0.21]  # gripper tip while the bottle is in front of the U3 camera
CAMERAS = {"U3": dict(at=[0.775, -0.3, 0.19], look=[0.69, -0.3, 0.19]), "U6": dict(at=[-0.3, 0.0375, 0.17], look=[-0.4, 0.0375, 0.05])}
TABLE = dict(x=[-0.85, 0.9], y=[-0.9, 0.7])
