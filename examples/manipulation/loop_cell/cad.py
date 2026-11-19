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

"""Printable parts of the loop cell, parametric in CadQuery (mm, z up, print orientation)."""

from __future__ import annotations

import math

import cadquery as cq

from . import layout as L

M4_CLEAR, M4_INSERT, M4_NUT_AF = 4.4, 5.6, 7.0


def _cyl(r: float, h: float, z0: float = 0.0, x: float = 0.0, y: float = 0.0) -> cq.Workplane:
    return cq.Workplane("XY").workplane(offset=z0).center(x, y).circle(r).extrude(h)


def _hex_trap(af: float, depth: float, z0: float = 0.0) -> cq.Workplane:
    return cq.Workplane("XY").workplane(offset=z0).polygon(6, af / math.cos(math.pi / 6)).extrude(depth)


# ---------------- base ----------------
def tile() -> cq.Workplane:
    t, skin = L.TILE_T, 4.0
    body = cq.Workplane("XY").box(L.TILE, L.TILE, t, centered=(True, True, False)).edges("|Z").fillet(3)
    depth = t - skin
    n, g = round(L.TILE / L.GRID), L.GRID
    pockets = cq.Workplane("XY").rarray(2 * g, 2 * g, n // 2, n // 2).rect(2 * g - 3, 2 * g - 3).extrude(depth)
    inner = cq.Workplane("XY").box(L.TILE - 8, L.TILE - 8, depth, centered=(True, True, False))
    body = body.cut(pockets.intersect(inner))
    pts = [(-L.TILE / 2 + g / 2 + g * i, -L.TILE / 2 + g / 2 + g * j) for i in range(n) for j in range(n)]
    body = body.union(cq.Workplane("XY").pushPoints(pts).circle(5).extrude(depth))
    return body.cut(cq.Workplane("XY").pushPoints(pts).circle(M4_INSERT / 2).extrude(t))


def joiner() -> cq.Workplane:
    """Plate under the middle of a seam: 4 countersunk M4 into the inserts at ±GRID/2 of the seam.

    Thinner than the foot pads, so the tiles still stand on their pads (z=0 faces the table)."""
    g, t = L.GRID, L.JOINER_T
    p = cq.Workplane("XY").box(2 * g + 10, 2 * g + 10, t, centered=(True, True, False)).edges("|Z").fillet(3)
    pts = [(x, y) for x in (-g / 2, g / 2) for y in (-g / 2, g / 2)]
    p = p.cut(cq.Workplane("XY").pushPoints(pts).circle(M4_CLEAR / 2).extrude(t))
    sink = cq.Solid.makeCone(8.4 / 2, M4_CLEAR / 2, 2.0)
    for x, y in pts:
        p = p.cut(cq.Workplane("XY").add(sink.translate(cq.Vector(x, y, 0))))
    return p


def foot_pad() -> cq.Workplane:
    """TPU pad under each tile corner, centred 15 mm in from both edges (print in TPU)."""
    return cq.Workplane("XY").box(20, 20, L.PAD_T, centered=(True, True, False)).edges("|Z").fillet(2)


def clamp_dog(flange_h: float = 6.0) -> cq.Workplane:
    """Holds the SO-100 base flange down onto its tile; slot for an M4 into an insert.

    Short enough (24 mm behind the lip) to sit between the base and the tile edge on the two
    roomy sides of every arm tile. ``flange_h``: measure your SO-100 base flange and regenerate.
    """
    h = flange_h + 4
    body = cq.Workplane("XY").box(24, 20, h, centered=(False, True, False))
    lip = cq.Workplane("XY").box(8, 20, 4, centered=(False, True, False)).translate((24, 0, h - 4))
    body = body.union(lip)
    slot = cq.Workplane("XY").center(12, 0).slot2D(L.GRID - 2, M4_CLEAR).extrude(h)
    return body.cut(slot).edges("|Z").fillet(1)


# ---------------- carrier ----------------
def puck_body() -> cq.Workplane:
    """Sits on a glued sole; 3 Ø8 steel balls in blind bores make it roll."""
    s, b = L.SOLE_T, L.BASE_H
    body = _cyl(L.PUCK_D / 2, b).edges("<Z").chamfer(0.6)
    body = body.union(_cyl(L.NECK_D / 2, L.NECK_H, b)).edges(">Z").chamfer(0.8)
    sock_bottom = b + L.NECK_H - L.SOCKET_DEPTH
    body = body.cut(_cyl(L.SOCKET_D / 2, L.SOCKET_DEPTH + 1, sock_bottom))
    body = body.cut(cq.Workplane("XY").workplane(offset=b + L.NECK_H - 1.2).circle(L.SOCKET_D / 2 + 1.2).workplane(offset=1.2).circle(L.SOCKET_D / 2).loft())
    bore_depth = L.BB_D / 2 + math.sqrt((L.BB_D / 2) ** 2 - (L.BB_LIP_D / 2) ** 2) + L.BB_D / 2 - s + 0.2
    for k in range(3):
        a = math.radians(90 + 120 * k)
        body = body.cut(_cyl(L.BB_D / 2 + 0.15, bore_depth, 0, L.BB_RING * math.cos(a), L.BB_RING * math.sin(a)))
    # flat for an ArUco / ID sticker on the neck
    return body.cut(cq.Workplane("XY").box(10, 14, L.NECK_H, centered=(False, True, False)).translate((L.NECK_D / 2 - 1.0, 0, b)))


def puck_sole() -> cq.Workplane:
    sole = _cyl(L.PUCK_D / 2, L.SOLE_T)
    for k in range(3):
        a = math.radians(90 + 120 * k)
        sole = sole.cut(_cyl(L.BB_LIP_D / 2, L.SOLE_T, 0, L.BB_RING * math.cos(a), L.BB_RING * math.sin(a)))
    return sole


def _thread_segment(r0: float, z_low: float, angle_deg: float) -> cq.Workplane:
    span = L.THREAD_SPAN
    height = L.THREAD_LEAD * span / 360.0
    helix = cq.Wire.makeHelix(L.THREAD_LEAD, height, r0)
    prof = cq.Workplane("XZ").polyline([(r0 - 0.3, -1.0), (r0 + L.THREAD_RADIAL, -0.5), (r0 + L.THREAD_RADIAL, 0.5), (r0 - 0.3, 1.0)]).close()
    seg = prof.sweep(cq.Workplane().add(helix), isFrenet=True)
    return seg.rotate((0, 0, 0), (0, 0, 1), angle_deg).translate((0, 0, z_low))


def collar() -> cq.Workplane:
    """Glued round the tube top. 4-start lug finish: any cap angle locks within ~120°."""
    r0 = L.COLLAR_OD / 2
    body = _cyl(r0, L.COLLAR_H).cut(_cyl(L.COLLAR_ID / 2, L.COLLAR_H))
    height = L.THREAD_LEAD * L.THREAD_SPAN / 360.0
    z_low = L.THREAD_TOP - height
    for k in range(4):
        body = body.union(_thread_segment(r0, z_low, 90 * k))
    # click detent where the lug parks, 30° in from each segment's high end
    lug_seat_z = z_low - 1.0 - 0.2 - L.LUG_H / 2
    for k in range(4):
        a = math.radians(90 * k + L.THREAD_SPAN - 30)
        bump = cq.Workplane("XY").sphere(0.7).translate((r0 * math.cos(a), r0 * math.sin(a), lug_seat_z))
        body = body.union(bump)
    return body


def cap() -> cq.Workplane:
    """Printed upside down (top on the bed). Lugs ride under the collar thread."""
    bore_r = L.COLLAR_OD / 2 + L.THREAD_RADIAL + L.CAP_CLEAR
    inner_depth = L.CAP_H - L.CAP_TOP_T
    body = _cyl(L.CAP_OD / 2, L.CAP_H).edges("<Z").chamfer(0.8)
    body = body.cut(_cyl(bore_r, inner_depth))
    # seated: cap inner top on the collar top -> lug top sits 0.2 under the thread's low end
    height = L.THREAD_LEAD * L.THREAD_SPAN / 360.0
    lug_top_collar = (L.THREAD_TOP - height) - 1.0 - 0.2
    lug_top = lug_top_collar + (inner_depth - L.COLLAR_H)
    lug_r_in = L.COLLAR_OD / 2 + 0.3
    for k in range(4):
        lug = cq.Workplane("XY").box(bore_r - lug_r_in + 0.4, L.LUG_W, L.LUG_H, centered=(False, True, False))
        lug = lug.translate((lug_r_in, 0, lug_top - L.LUG_H)).rotate((0, 0, 0), (0, 0, 1), 90 * k + 45)
        body = body.union(lug)
    rib = cq.Workplane("XY").box(L.CAP_OD - 10, 4, 4, centered=(True, True, False)).translate((0, 0, L.CAP_H)).edges("|Z").fillet(1.5)
    return body.union(rib)


def fit_coupon() -> cq.Workplane:
    """Print first: tube socket (3 sizes), ball lip (3), heat-set insert holes (3)."""
    p = cq.Workplane("XY").box(120, 40, 3, centered=(False, False, False))
    for i, d in enumerate((25.2, 25.4, 25.6)):
        x = 18 + 30 * i
        p = p.union(_cyl(d / 2 + 2, 9, 3, x, 20)).cut(_cyl(d / 2, 12, 0, x, 20))
    for i, d in enumerate((7.0, 7.2, 7.4)):
        p = p.cut(_cyl(d / 2, 3, 0, 108, 8 + 12 * i))
    for i, d in enumerate((5.5, 5.6, 5.7)):
        p = p.union(_cyl(5, 8, 3, 96, 8 + 12 * i)).cut(_cyl(d / 2, 11, 0, 96, 8 + 12 * i))
    return p


# ---------------- chute modules (local frame: x along the flow, y across, floor bottom at z=0) ----------------
def _fam(name: str) -> dict:
    return L.FAMILIES[name]


def _locator_holes(part: cq.Workplane, pts: list[tuple[float, float]]) -> cq.Workplane:
    return part.cut(cq.Workplane("XY").pushPoints(pts).circle(1.6).extrude(2.5))


def straight(fam: str, length: float) -> cq.Workplane:
    f = _fam(fam)
    w, h, t = f["W"], f["H"], L.WALL_T
    floor = cq.Workplane("XY").box(length, w + 2 * t, L.FLOOR_T, centered=(False, True, False))
    walls = [cq.Workplane("XY").box(length, t, L.FLOOR_T + h, centered=(False, False, False)).translate((0, s * (w / 2) + (0 if s > 0 else -t), 0)) for s in (1, -1)]
    part = floor.union(walls[0]).union(walls[1])
    q = w / 4
    xs = (8, length - 8) if length >= 24 else (length / 2,)  # short trims: one pair in the middle
    return _locator_holes(part, [(x, s * q) for x in xs for s in (1, -1)])


def curve(fam: str) -> cq.Workplane:
    """90° left turn: starts at (0,-R) heading +x, ends at (R,0) heading +y. Reverse flow = right turn."""
    f = _fam(fam)
    w, h, t, r = f["W"], f["H"], L.WALL_T, f["R"]
    quad = cq.Workplane("XY").box(r + w, r + w, 100, centered=(False, False, False)).translate((0, -(r + w), -10))

    def ring(r_in, r_out, height):
        return _cyl(r_out, height).cut(_cyl(r_in, height)).intersect(quad)

    part = ring(r - w / 2 - t, r + w / 2 + t, L.FLOOR_T)
    part = part.union(ring(r - w / 2 - t, r - w / 2, L.FLOOR_T + h))
    part = part.union(ring(r + w / 2, r + w / 2 + t, L.FLOOR_T + h + 6))  # taller outer wall
    da = 8.0 / r
    pts = []
    for a in (-math.pi / 2 + da, -da):
        for rr in (r - w / 4, r + w / 4):
            pts.append((rr * math.cos(a), rr * math.sin(a)))
    return _locator_holes(part, pts)


def flare(fam: str) -> cq.Workplane:
    """Drop zone: 20 mm wider mouth narrowing to the chute width."""
    f = _fam(fam)
    w, h, t, ln = f["W"], f["H"], L.WALL_T, f["flare"]
    w0, xn = w + 20, 0.6 * ln
    floor = cq.Workplane("XY").polyline([(0, -w0 / 2 - t), (xn, -w / 2 - t), (ln, -w / 2 - t), (ln, w / 2 + t), (xn, w / 2 + t), (0, w0 / 2 + t)]).close().extrude(L.FLOOR_T)
    part = floor
    for s in (1, -1):
        wall = cq.Workplane("XY").polyline([(0, s * w0 / 2), (xn, s * w / 2), (ln, s * w / 2), (ln, s * (w / 2 + t)), (xn, s * (w / 2 + t)), (0, s * (w0 / 2 + t))]).close().extrude(L.FLOOR_T + h)
        part = part.union(wall)
    q = w / 4
    return _locator_holes(part, [(8, q), (8, -q), (ln - 8, q), (ln - 8, -q)])


def gate(fam: str) -> cq.Workplane:
    """Inlet stop: end wall, IR beam holes at the head item, an off-centre slot for the hold-back pin
    and a plate under the floor for an SG90 lying on its side (pin lever: ``gate_lever``)."""
    f = _fam(fam)
    w, h, t, ln, item = f["W"], f["H"], L.WALL_T, f["gate"], f["item"]
    part = straight(fam, ln)
    part = part.union(cq.Workplane("XY").box(4, w + 2 * t, L.FLOOR_T + h + 8, centered=(False, True, False)).translate((ln - 4, 0, 0)))
    head_x = ln - 4 - 1 - item / 2
    beam = cq.Workplane("XZ").center(head_x, L.FLOOR_T + f["beam_z"]).circle(2.6).extrude(w + 2 * t + 2, both=True)
    part = part.cut(beam)
    g = L.gate_geom(fam)
    s0, s1 = g["slot"]
    part = part.cut(cq.Workplane("XY").center((s0 + s1) / 2, g["pin_y"]).rect(s1 - s0, L.PIN_D + 0.4).extrude(L.FLOOR_T))
    # SG90 on its side: a plate hanging under the floor with the body window and two ear screws
    y0, y1 = g["plate_y"]
    cx, z0 = g["servo_cx"], g["shaft_z"]
    bw, bh = L.SG90["body"]
    plate = cq.Workplane("XY").box(36, y1 - y0, L.FLOOR_T - (z0 - 14), centered=(True, False, False)).translate((cx, y0, z0 - 14))
    plate = plate.cut(cq.Workplane("XZ").center(cx, z0).rect(bw + 0.4, bh + 0.6).extrude(-(y1 - y0)).translate((0, y0, 0)))
    plate = plate.cut(cq.Workplane("XZ").pushPoints([(cx - L.SG90["hole_pitch"] / 2, z0), (cx + L.SG90["hole_pitch"] / 2, z0)]).circle(0.9).extrude(-(y1 - y0)).translate((0, y0, 0)))
    return part.union(plate)


def gate_lever(fam: str) -> cq.Workplane:
    """Lever for the SG90 horn: pivot at x=3 (horn holes at x=3 and 9, along y), hold-back pin at
    x=3+LEVER_R pointing up. Mount it so the pin sits under the gate's off-centre slot."""
    g = L.gate_geom(fam)
    arm = cq.Workplane("XY").box(24, L.LEVER_T, L.LEVER_H, centered=(False, True, True)).edges("|Y").fillet(2.5)
    arm = arm.cut(cq.Workplane("XZ").pushPoints([(3, 0), (9, 0)]).circle(1.0).extrude(L.LEVER_T, both=True))
    return arm.union(_cyl(L.PIN_D / 2, g["pin_len"], L.LEVER_H / 2, 3 + L.LEVER_R, 0))


def saddle(fam: str, flat: bool = False) -> cq.Workplane:
    """Joint between two modules; sits on an M4 rod. Downstream half 0.5 mm lower."""
    f = _fam(fam)
    outer = f["W"] + 2 * L.WALL_T
    width = outer + 0.4 + 6
    base = cq.Workplane("XY").box(L.SADDLE_L / 2, width, L.SADDLE_T, centered=(False, True, False)).translate((-L.SADDLE_L / 2, 0, 0))
    base = base.union(cq.Workplane("XY").box(L.SADDLE_L / 2, width, L.SADDLE_T - L.SADDLE_STEP, centered=(False, True, False)))
    q = f["W"] / 4
    for x, top in ((-8, L.SADDLE_T), (8, L.SADDLE_T - L.SADDLE_STEP)):
        base = base.union(cq.Workplane("XY").pushPoints([(x, q), (x, -q)]).circle(1.45).extrude(2.2).translate((0, 0, top)))
    if not flat:
        for s in (1, -1):
            base = base.union(cq.Workplane("XY").box(L.SADDLE_L, 3, L.SADDLE_T + 8, centered=(True, False, False)).translate((0, s * (outer / 2 + 0.2) + (0 if s > 0 else -3), 0)))
    base = base.cut(_cyl(M4_CLEAR / 2, L.SADDLE_T))
    return base.cut(_hex_trap(M4_NUT_AF, 3.2))


def post_foot() -> cq.Workplane:
    disk = _cyl(30, 5).union(_cyl(8, L.FOOT_H - 5, 5))
    disk = disk.cut(_cyl(M4_CLEAR / 2, L.FOOT_H)).cut(_hex_trap(M4_NUT_AF, 3.5))
    g = L.GRID
    return disk.cut(cq.Workplane("XY").pushPoints([(-g / 2, g), (g / 2, g), (-g / 2, -g), (g / 2, -g)]).circle(M4_CLEAR / 2).extrude(5))


def spacer(h: float) -> cq.Workplane:
    return _cyl(8, h).cut(_cyl(M4_CLEAR / 2, h))


def nest() -> cq.Workplane:
    """Puck seat with a 45° lead-in; bolts to the grid at (±1.5, ±1.5) grid steps (±30 mm)."""
    t, floor, straight_h = 9.0, 2.0, 3.0
    body = cq.Workplane("XY").box(75, 75, t, centered=(True, True, False)).edges("|Z").fillet(4)
    pocket_r = L.PUCK_D / 2 + 0.4
    body = body.cut(_cyl(pocket_r, t, floor))
    cone = cq.Workplane("XY").workplane(offset=floor + straight_h).circle(pocket_r).workplane(offset=t - floor - straight_h).circle(30).loft()
    body = body.cut(cone)
    b = 1.5 * L.GRID
    pts = [(x, y) for x in (-b, b) for y in (-b, b)]
    body = body.cut(cq.Workplane("XY").pushPoints(pts).circle(M4_CLEAR / 2).extrude(t))
    return body.cut(cq.Workplane("XY").workplane(offset=t - 3).pushPoints(pts).circle(4.2).extrude(3))


def _cone_hopper(top_r: float, out_r: float, h: float, wall: float = 2.0, tab: bool = True) -> cq.Workplane:
    spout = 6.0
    prof = [(out_r, 0), (out_r + wall, 0), (out_r + wall, spout), (top_r + wall, spout + h), (top_r, spout + h), (out_r, spout)]
    body = cq.Workplane("XZ").polyline(prof).close().revolve(360, (0, 0, 0), (0, 1, 0))
    if tab:
        z = spout + h * 0.6
        r_at = out_r + wall + (top_r - out_r) * 0.6
        t = cq.Workplane("XY").box(top_r - r_at + 28, 16, 4, centered=(False, True, False)).translate((r_at - 4, 0, z))
        t = t.cut(_cyl(M4_CLEAR / 2, 4, z, top_r + 14, 0))
        body = body.union(t)
    return body


def hopper() -> cq.Workplane:
    """Replaces the cup: walls ~42°, outlet 1.3 x ball so balls leave one at a time."""
    return _cone_hopper(L.HOPPER["top_r"], L.HOPPER["out_r"], L.HOPPER["h"])


def funnel() -> cq.Workplane:
    return _cone_hopper(L.FUNNEL["top_r"], L.FUNNEL["out_r"], L.FUNNEL["h"])


def camera_post(h: float) -> cq.Workplane:
    """Upright for a small USB camera; bolts to the grid. Camera plate: 2 x M2 at 20 mm."""
    base = cq.Workplane("XY").box(50, 30, 5, centered=(True, True, False)).cut(cq.Workplane("XY").pushPoints([(-L.GRID, 0), (L.GRID, 0)]).circle(M4_CLEAR / 2).extrude(5))
    col = cq.Workplane("XY").box(12, 12, h, centered=(True, True, False)).translate((0, 0, 5))
    head = cq.Workplane("XY").box(30, 4, 30, centered=(True, True, False)).translate((0, 8, h - 25))
    head = head.cut(cq.Workplane("XZ").pushPoints([(-10, h - 10), (10, h - 10)]).circle(1.1).extrude(-20, both=True))
    return base.union(col).union(head)
