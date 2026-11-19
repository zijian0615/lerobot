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

"""Chute routes -> exact centreline (lines + arcs), module list, joint posts, arm reach check.

Standard library only, so the layout can be checked without a CAD kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import layout as L


@dataclass
class Prim:
    kind: str  # "line" | "arc"
    length: float
    p0: tuple[float, float]
    p1: tuple[float, float]
    center: tuple[float, float] | None = None
    radius: float = 0.0
    a0: float = 0.0  # start angle (rad) around center
    sweep: float = 0.0  # signed (+ = counter-clockwise = left turn)

    def at(self, u: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """Point and unit direction at distance ``u`` from this primitive's start."""
        if self.kind == "line":
            dx, dy = self.p1[0] - self.p0[0], self.p1[1] - self.p0[1]
            n = math.hypot(dx, dy) or 1.0
            t = u / n
            return (self.p0[0] + dx * t, self.p0[1] + dy * t), (dx / n, dy / n)
        a = self.a0 + self.sweep * (u / self.length)
        s = 1.0 if self.sweep > 0 else -1.0
        p = (self.center[0] + self.radius * math.cos(a), self.center[1] + self.radius * math.sin(a))
        return p, (-math.sin(a) * s, math.cos(a) * s)


@dataclass
class Module:
    kind: str  # flare | straight | curve | gate
    length_mm: float
    s0: float
    s1: float
    turn: str = ""  # curves: "left" / "right" in the flow direction


@dataclass
class Route:
    name: str
    family: str
    prims: list[Prim]
    length: float  # to the stop point (m)
    end_ext: float  # stop point -> outer face of the end wall (m); 0 for transit
    z_end: float
    slope_deg: float
    transit: bool
    no_flare: bool
    modules: list[Module] = field(default_factory=list)
    joints: list[dict] = field(default_factory=list)

    def at(self, s: float):
        if s <= 0:
            p, d = self.prims[0].at(0.0)
            return (p[0] + d[0] * s, p[1] + d[1] * s), d
        acc = 0.0
        for pr in self.prims:
            if s <= acc + pr.length:
                return pr.at(s - acc)
            acc += pr.length
        p, d = self.prims[-1].at(self.prims[-1].length)
        e = s - acc
        return (p[0] + d[0] * e, p[1] + d[1] * e), d

    def z(self, s: float) -> float:
        return self.z_end + (self.length - s) * math.tan(math.radians(self.slope_deg))

    @property
    def fam(self) -> dict:
        return L.FAMILIES[self.family]


def _unit(v):
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n)


def build_route(name: str, spec: dict) -> Route:
    fam = L.FAMILIES[spec["family"]]
    R = fam["R"] / 1000
    pts = [tuple(p) for p in spec["pts"]]
    prims: list[Prim] = []
    cur = pts[0]
    for i in range(1, len(pts) - 1):
        a, c, b = pts[i - 1], pts[i], pts[i + 1]
        d1 = _unit((c[0] - a[0], c[1] - a[1]))
        d2 = _unit((b[0] - c[0], b[1] - c[1]))
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        theta = math.acos(max(-1.0, min(1.0, d1[0] * d2[0] + d1[1] * d2[1])))
        t = R * math.tan(theta / 2)
        t1 = (c[0] - d1[0] * t, c[1] - d1[1] * t)
        t2 = (c[0] + d2[0] * t, c[1] + d2[1] * t)
        prims.append(Prim("line", math.dist(cur, t1), cur, t1))
        n = (-d1[1], d1[0]) if cross > 0 else (d1[1], -d1[0])
        center = (t1[0] + n[0] * R, t1[1] + n[1] * R)
        a0 = math.atan2(t1[1] - center[1], t1[0] - center[0])
        sweep = theta if cross > 0 else -theta
        prims.append(Prim("arc", R * theta, t1, t2, center, R, a0, sweep))
        cur = t2
    prims.append(Prim("line", math.dist(cur, pts[-1]), cur, pts[-1]))
    prims = [p for p in prims if p.length > 1e-9]
    length = sum(p.length for p in prims)
    transit = bool(spec.get("transit"))
    end_ext = 0.0 if transit else (fam["item"] / 2 + 1 + 4) / 1000
    r = Route(name, spec["family"], prims, length, end_ext, float(spec.get("end_z", L.END_Z)), fam["slope"], transit, bool(spec.get("no_flare")))
    _modules(r)
    _joints(r)
    return r


def _fill(avail_mm: float) -> list[float]:
    if avail_mm < -0.5:
        raise ValueError(f"leg too short by {-avail_mm:.1f} mm")
    out: list[float] = []
    long, short = L.STRAIGHTS
    n = int(avail_mm // long)
    out += [long] * n
    rem = avail_mm - long * n
    if rem > 0.5:
        if abs(rem - short) < 0.5:
            out.append(short)
        elif rem < 15.0 and out:
            out[-1] = round(out[-1] + rem, 1)
        else:
            out.append(round(rem, 1))
    return out


def _modules(r: Route) -> None:
    fam = r.fam
    s = 0.0
    lines = [i for i, p in enumerate(r.prims) if p.kind == "line"]
    first_line, last_line = lines[0], lines[-1]
    for i, p in enumerate(r.prims):
        if p.kind == "arc":
            if abs(abs(p.sweep) - math.pi / 2) > 1e-3:
                raise ValueError(f"{r.name}: only 90° corners have a printed module")
            r.modules.append(Module("curve", p.length * 1000, s, s + p.length, "left" if p.sweep > 0 else "right"))
            s += p.length
            continue
        start, end = s, s + p.length
        if i == first_line and not (r.transit or r.no_flare):
            fl = fam["flare"] / 1000
            r.modules.append(Module("flare", fam["flare"], start, start + fl))
            start += fl
        gate_mod = None
        if i == last_line and not r.transit:
            se = end + r.end_ext
            gate_mod = Module("gate", fam["gate"], se - fam["gate"] / 1000, se)
            end = gate_mod.s0
        for ln in _fill((end - start) * 1000):
            r.modules.append(Module("straight", ln, start, start + ln / 1000))
            start += ln / 1000
        if gate_mod:
            r.modules.append(gate_mod)
        s += p.length


def _ground(x: float, y: float) -> float:
    for u in L.UNITS:
        x0, y0, x1, y1 = u["rect"]
        if x0 <= x <= x1 and y0 <= y <= y1:
            return L.PT
    return 0.0


def _joints(r: Route) -> None:
    cuts = [r.modules[0].s0] + [m.s1 for m in r.modules]
    for s in cuts:
        (x, y), d = r.at(s)
        zf = r.z(s)
        g = _ground(x, y)
        post = (zf - g) * 1000 - L.FLOOR_T - L.SADDLE_T
        fill = post - L.FOOT_H
        stack, rest = [], fill
        for sp in L.SPACERS:
            while rest >= sp - 1e-6:
                stack.append(sp)
                rest -= sp
        r.joints.append(dict(s=s, xy=(x, y), dir=d, floor_z=zf, ground=g, post_mm=post, spacers=stack, adjust_mm=rest, ok=fill >= 0))


def build_all() -> dict[str, Route]:
    return {k: build_route(k, v) for k, v in L.ROUTES.items()}


# ---------------- arm reach (SO-100 model, same maths as the 3D page) ----------------
def arm_targets(routes: dict[str, Route]) -> dict[str, list[tuple[str, tuple[float, float, float]]]]:
    """Every point each arm's gripper has to reach (pinch point, world m), with a label."""
    g = L.GRIP_Z / 1000
    top = L.TUBE_TOP_Z / 1000
    nest_z = L.PT + 0.002
    units = {u["id"]: u for u in L.UNITS}
    out: dict[str, list] = {a["id"]: [] for a in L.ARMS}

    def stop(rt):
        (x, y), _ = rt.at(rt.length)
        return x, y, rt.z(rt.length)

    def drop(rt):
        s = drop_s(rt)
        (x, y), _ = rt.at(s)
        return x, y, rt.z(s)

    for u in L.UNITS:
        if not u.get("inP"):
            continue
        holder = u["arms"][0]
        ix, iy, iz = stop(routes[u["inP"]])
        ox, oy, oz = drop(routes[u["outP"]])
        nx, ny = u["nest"]
        out[holder] += [("入口取瓶", (ix, iy, iz + g)), ("入口上方", (ix, iy, iz + g + 0.06)),
                        ("工位", (nx, ny, nest_z + g)), ("工位上方", (nx, ny, nest_z + g + 0.06)),
                        ("出口放瓶", (ox, oy, oz + g)), ("出口上方", (ox, oy, oz + g + 0.06))]
    r = L.BALL_D / 2000
    for c in ("red", "green", "blue"):
        x, y, z = stop(routes[c])
        out["A1"] += [(f"{c} 球道末端取球", (x, y, z + r)), (f"{c} 球道上方", (x, y, z + r + 0.05))]
        x, y, z = drop(routes[c])
        out["A8"] += [(f"{c} 球道入口放球", (x, y, z + r + 0.004)), (f"{c} 球道入口上方", (x, y, z + r + 0.034))]
    nx, ny = units["U1"]["nest"]
    out["A1"].append(("瓶口上方投球", (nx, ny, nest_z + top + 0.035)))
    cap_c = (L.TUBE_TOP_Z - (L.CAP_H - L.CAP_TOP_T) + L.CAP_H / 2) / 1000
    x, y, z = stop(routes["CAP"])
    out["A3"] += [("盖滑道末端取盖", (x, y, z + L.CAP_H / 2000)), ("盖滑道上方", (x, y, z + L.CAP_H / 2000 + 0.05))]
    for uid, arm in (("U2", "A3"), ("U4", "A6")):
        nx, ny = units[uid]["nest"]
        out[arm] += [("瓶口取/放盖", (nx, ny, nest_z + cap_c)), ("瓶口上方", (nx, ny, nest_z + cap_c + 0.05))]
    for uid, arm in (("U2", "A2"), ("U4", "A5")):
        nx, ny = units[uid]["nest"]
        out[arm].append(("侧面扶瓶", (nx, ny, nest_z + 0.05)))
    x, y, z = drop(routes["CAP"])
    out["A6"] += [("盖滑道入口放盖", (x, y, z + L.CAP_H / 2000 + 0.004)), ("盖滑道入口上方", (x, y, z + L.CAP_H / 2000 + 0.05))]
    out["A4"].append(("相机前", tuple(L.INSPECT)))
    fx, fy = L.FUNNEL["xy"]
    out["A7"] += [("漏斗上方", (fx - 0.004, fy + 0.03, L.POUR_Z["above"])), ("倒扣位置", (fx - 0.004, fy + 0.035, L.POUR_Z["at"]))]
    x, y, z = stop(routes["M2"])
    out["A8"] += [("料斗出口取球", (x, y, z + r)), ("料斗出口上方", (x, y, z + r + 0.05))]
    return out


def arm_yaws(targets=None) -> dict[str, float]:
    """Base yaw per arm (rad). The base is squared to its tile (``yaw`` in layout, degrees)."""
    return {a["id"]: math.radians(a["yaw"]) for a in L.ARMS}


def reach_report(routes: dict[str, Route]) -> list[dict]:
    from . import arm_model as A

    targets = arm_targets(routes)
    yaws = arm_yaws()
    rows = []
    for a in L.ARMS:
        b = a["xy"]
        for label, p in targets[a["id"]]:
            rows.append(dict(arm=a["id"], what=label, horiz=math.hypot(p[0] - b[0], p[1] - b[1]), z=p[2], tilt=A.tilt_needed(b, L.PT, yaws[a["id"]], p)))
    return rows


# ---------------- arm tiles ----------------
def obstacles(routes: dict[str, Route]) -> list[tuple[float, float, float, str]]:
    """(x, y, radius, what) samples of everything that must stay off an arm's tile."""
    out = []
    for k, r in routes.items():
        half = (r.fam["W"] / 2 + L.WALL_T) / 1000
        n = int((r.length + r.end_ext) / 0.005) + 1
        out += [(*r.at(i * 0.005)[0], half, f"滑道 {k}") for i in range(n)]
        out += [(j["xy"][0], j["xy"][1], POST_R, f"{k} 支柱") for j in r.joints]
    for u in L.UNITS:
        if u["nest"]:
            h = NEST / 2
            out += [(u["nest"][0] + dx, u["nest"][1] + dy, 0.0, f"{u['id']} 工位") for dx in (-h, 0, h) for dy in (-h, 0, h)]
    for c, cam in L.CAMERAS.items():
        out.append((cam["at"][0], cam["at"][1], 0.025, f"{c} 相机"))
    for nm, spec in (("漏斗", L.FUNNEL), ("料斗", L.HOPPER)):
        out.append((spec["xy"][0] + spec["top_r"] / 1000 + 0.014, spec["xy"][1], POST_R, f"{nm}支柱"))
    return out


POST_R = 0.030  # post foot radius
NEST = 0.075  # nest block edge


def base_corners(xy, yaw) -> list[tuple[float, float]]:
    from . import arm_model as A

    x0, y0, x1, y1 = A.P["base_box"]
    c, s = math.cos(yaw), math.sin(yaw)
    return [(xy[0] + c * lx - s * ly, xy[1] + s * lx + c * ly) for lx, ly in ((x0, y0), (x0, y1), (x1, y0), (x1, y1))]


def plate_hits(pc, obs) -> set[str]:
    h = L.ARM_PLATE / 2000
    return {w for x, y, m, w in obs if abs(x - pc[0]) < h + m - 1e-6 and abs(y - pc[1]) < h + m - 1e-6}


def base_clearance(routes: dict[str, Route]) -> list[str]:
    """Each arm's tile: inside its unit, base on it, nothing else on it, no two arms on one tile."""
    obs = obstacles(routes)
    h = L.ARM_PLATE / 2000
    problems = []
    for a in L.ARMS:
        pc = a["plate"]
        u = next(u for u in L.UNITS if u["id"] == a["unit"])
        if not (u["rect"][0] - 1e-6 <= pc[0] - h and pc[0] + h <= u["rect"][2] + 1e-6 and u["rect"][1] - 1e-6 <= pc[1] - h and pc[1] + h <= u["rect"][3] + 1e-6):
            problems.append(f"{a['id']} 的拼接块不在 {u['id']} 底板里")
        if not all(abs(x - pc[0]) <= h - PLATE_MARGIN and abs(y - pc[1]) <= h - PLATE_MARGIN for x, y in base_corners(a["xy"], math.radians(a["yaw"]))):
            problems.append(f"{a['id']} 底座伸出自己的拼接块")
        for w in sorted(plate_hits(pc, obs)):
            problems.append(f"{a['id']} 的拼接块上有{w}")
        for b in L.ARMS:
            if b["id"] > a["id"] and abs(b["plate"][0] - pc[0]) < 2 * h - 1e-6 and abs(b["plate"][1] - pc[1]) < 2 * h - 1e-6:
                problems.append(f"{a['id']} 和 {b['id']} 的拼接块重叠")
    return problems


PLATE_MARGIN = 0.008  # base edge to tile edge, room for the clamp dogs' lips


def place_arms(routes: dict[str, Route]) -> dict[str, dict]:
    """Best pan axis + squared yaw for each arm on its tile: smallest worst tool tilt, then mean."""
    from . import arm_model as A

    targets = arm_targets(routes)
    h = L.ARM_PLATE / 2000
    out = {}
    steps = [k * 0.005 for k in range(-6, 7)]
    for a in L.ARMS:
        pc, best = a["plate"], None
        for dx in steps:
            for dy in steps:
                xy = (round(pc[0] + dx, 4), round(pc[1] + dy, 4))
                for q in range(4):
                    if not all(abs(x - pc[0]) <= h - PLATE_MARGIN and abs(y - pc[1]) <= h - PLATE_MARGIN for x, y in base_corners(xy, q * math.pi / 2)):
                        continue
                    tilts = [A.tilt_needed(xy, L.PT, q * math.pi / 2, p) for _, p in targets[a["id"]]]
                    if any(t is None for t in tilts):
                        continue
                    key = (max(tilts), sum(tilts) / len(tilts))
                    if best is None or key < best["key"]:
                        best = dict(key=key, xy=list(xy), yaw=90 * q)
        out[a["id"]] = best
    return out


def drop_s(rt: Route) -> float:
    """Where an arm puts an item on the route: inside the wide part of the flare."""
    return 0.0 if (rt.transit or rt.no_flare) else rt.fam["flare"] / 1000 * 0.35
