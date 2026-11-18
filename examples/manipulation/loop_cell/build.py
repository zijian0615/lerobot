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

"""Build the loop cell from ``layout.py``: STEP/STL parts, BOM.md, layout.json, the 3D page.

Needs CadQuery (any recent version) for the parts; ``--no-cad`` skips them::

    cd examples/manipulation
    python -m loop_cell.build            # everything
    python -m loop_cell.build --no-cad   # BOM + page only
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from . import layout as L
from .routes import build_all, drop_s, reach_report

HERE = Path(__file__).resolve().parent
PLA = 1.24  # g/cm³


def _fmt(x: float) -> str:
    return f"{x:g}"


def part_list(routes) -> list[dict]:
    """Every printed part with its quantity; ``make`` builds it in CadQuery."""
    n_tiles = sum(round((u["rect"][2] - u["rect"][0]) / 0.2) * round((u["rect"][3] - u["rect"][1]) / 0.2) for u in L.UNITS)
    joiners = sum(2 if (u["rect"][3] - u["rect"][1]) < 0.3 else 5 for u in L.UNITS)
    parts = [
        dict(name="tile_200", make=("tile",), qty=n_tiles, mat="PLA", note="底座拼接块,25 mm 网格装 M4 热熔螺母"),
        dict(name="tile_joiner", make=("joiner",), qty=joiners, mat="PLA", note="拼接块底面跨缝连接板"),
        dict(name="tile_foot_pad", make=("foot_pad",), qty=4 * n_tiles, mat="TPU", note="拼接块四角脚垫"),
        dict(name="arm_clamp_dog", make=("clamp_dog",), qty=4 * len(L.ARMS), mat="PETG", note="压住 SO-101 底座;先量底座法兰厚度再改参数"),
        dict(name="puck_body", make=("puck_body",), qty=L.N_BOTTLES + 1, mat="PETG", note="托座;3 颗 Ø8 钢珠装进盲孔"),
        dict(name="puck_sole", make=("puck_sole",), qty=L.N_BOTTLES + 1, mat="PETG", note="粘在托座底面,挡住钢珠"),
        dict(name="tube_collar", make=("collar",), qty=L.N_BOTTLES + 1, mat="PETG", note="粘在管口,四头卡口螺纹 + 卡位凸点"),
        dict(name="cap", make=("cap",), qty=L.N_CAPS + 1, mat="PETG", note="顶面朝下打印"),
        dict(name="nest", make=("nest",), qty=sum(1 for u in L.UNITS if u["nest"]), mat="PLA", note="工位托座,带 45° 导口"),
        dict(name="hopper", make=("hopper",), qty=1, mat="PLA", note="暂存料斗,单列出球"),
        dict(name="funnel", make=("funnel",), qty=1, mat="PLA", note="U5 倒球漏斗"),
        dict(name="fit_coupon", make=("fit_coupon",), qty=1, mat="PETG", note="先打这个:试管孔、钢珠挡圈、热熔孔各 3 档"),
    ]
    for cam, spec in L.CAMERAS.items():
        h = round((spec["at"][2] - L.PT) * 1000 + 10)
        parts.append(dict(name=f"camera_post_{cam}_{h}", make=("camera_post", h), qty=1, mat="PLA", note=f"{cam} 相机立柱"))
    by_fam: dict[str, Counter] = {}
    joints = Counter()
    flats = Counter()
    for r in routes.values():
        c = by_fam.setdefault(r.family, Counter())
        for m in r.modules:
            key = (m.kind, round(m.length_mm, 1) if m.kind == "straight" else 0)
            c[key] += 1
        joints[r.family] += len(r.joints)
        if r.modules and r.modules[0].kind == "flare":
            flats[r.family] += 1
    for fam, c in by_fam.items():
        for (kind, ln), q in sorted(c.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
            if kind == "straight":
                parts.append(dict(name=f"chute_{fam}_S{_fmt(ln)}", make=("straight", fam, ln), qty=q, mat="PETG", note="直段" + ("" if ln in L.STRAIGHTS else "(按布局裁出的长度)")))
            elif kind == "curve":
                parts.append(dict(name=f"chute_{fam}_C90_R{_fmt(L.FAMILIES[fam]['R'])}", make=("curve", fam), qty=q, mat="PETG", note="90° 弯段,反向用就是另一侧转弯"))
            elif kind == "flare":
                parts.append(dict(name=f"chute_{fam}_flare", make=("flare", fam), qty=q, mat="PETG", note="入口喇叭段(臂放东西的地方)"))
            elif kind == "gate":
                parts.append(dict(name=f"chute_{fam}_gate", make=("gate", fam), qty=q, mat="PETG", note="末端闸门:挡墙 + 对射光电孔 + SG90 托架"))
                parts.append(dict(name=f"gate_lever_{fam}", make=("gate_lever", fam), qty=q, mat="PETG", note="闸门挡销摇臂,按舵盘修长度"))
        parts.append(dict(name=f"saddle_{fam}", make=("saddle", fam), qty=joints[fam] - flats[fam], mat="PLA", note="接头托架,装在 M4 螺杆上"))
        if flats[fam]:
            parts.append(dict(name=f"saddle_{fam}_flat", make=("saddle", fam, True), qty=flats[fam], mat="PLA", note="喇叭口起点用的平托架"))
    total_joints = sum(joints.values()) + 2  # + hopper and funnel posts
    sp = Counter()
    for r in routes.values():
        for j in r.joints:
            sp.update(j["spacers"])
    for s, q in extra_posts(routes)["spacers"].items():
        sp[s] += q
    parts.append(dict(name="post_foot", make=("post_foot",), qty=total_joints, mat="PLA", note="支柱底座,M4 螺杆从中间穿过"))
    for s in L.SPACERS:
        if sp[s]:
            parts.append(dict(name=f"post_spacer_{_fmt(s)}", make=("spacer", s), qty=sp[s], mat="PLA", note=f"{_fmt(s)} mm 垫块"))
    return parts


def extra_posts(routes) -> dict:
    """Posts under the hopper and funnel tabs."""
    out = {"rows": [], "spacers": Counter()}
    for name, spec, rt, ground in (("hopper", L.HOPPER, routes["M2"], 0.0), ("funnel", L.FUNNEL, routes["M1"], L.PT)):
        spout_bottom = rt.z(0) * 1000 + 2
        tab_z = spout_bottom + 6 + spec["h"] * 0.6
        post = tab_z - ground * 1000
        fill, stack = post - L.FOOT_H, []
        for s in L.SPACERS:
            while fill >= s - 1e-6:
                stack.append(s)
                fill -= s
        out["spacers"].update(stack)
        out["rows"].append((name, post, stack, fill))
    return out


def make_part(cad, spec):
    fn, *args = spec
    return getattr(cad, fn)(*args)


def export_parts(parts, out: Path) -> dict[str, float]:
    import cadquery as cq

    from . import cad

    (out / "step").mkdir(parents=True, exist_ok=True)
    (out / "stl").mkdir(parents=True, exist_ok=True)
    vols = {}
    for p in parts:
        wp = make_part(cad, p["make"])
        shape = wp.val()
        if not shape.isValid():
            raise RuntimeError(f"{p['name']}: invalid solid")
        cq.exporters.export(wp, str(out / "step" / f"{p['name']}.step"))
        cq.exporters.export(wp, str(out / "stl" / f"{p['name']}.stl"), tolerance=0.05, angularTolerance=0.15)
        vols[p["name"]] = shape.Volume()
        print(f"  {p['name']:32s} x{p['qty']:<3d} {shape.Volume() / 1000:7.1f} cm³")
    return vols


def page_layout(routes) -> dict:
    def dense(r):
        n = max(2, int(r.length / 0.004))
        return [[round(c, 5) for c in r.at(r.length * i / n)[0]] for i in range(n + 1)]

    rts = {}
    for k, r in routes.items():
        f = r.fam
        rts[k] = dict(
            family=r.family, dense=dense(r), length=r.length, z_end=r.z_end, slope=r.slope_deg, end_ext=r.end_ext,
            w=f["W"] / 1000, h=f["H"] / 1000, flare=0 if (r.transit or r.no_flare) else f["flare"] / 1000, transit=r.transit,
            drop_s=drop_s(r), modules=[dict(kind=m.kind, s0=m.s0, s1=m.s1) for m in r.modules],
            joints=[dict(s=j["s"], x=j["xy"][0], y=j["xy"][1], z=j["floor_z"], g=j["ground"]) for j in r.joints],
        )
    mm = 1 / 1000
    return dict(
        PT=L.PT, END_Z=L.END_Z, tile=L.TILE * mm, table=L.TABLE, units=L.UNITS, arms=L.ARMS, routes=rts, arm=L.ARM,
        funnel=dict(xy=L.FUNNEL["xy"], top_r=L.FUNNEL["top_r"] * mm, out_r=L.FUNNEL["out_r"] * mm, h=L.FUNNEL["h"] * mm, z0=routes["M1"].z(0) + 0.002),
        hopper=dict(xy=L.HOPPER["xy"], top_r=L.HOPPER["top_r"] * mm, out_r=L.HOPPER["out_r"] * mm, h=L.HOPPER["h"] * mm, z0=routes["M2"].z(0) + 0.002),
        inspect=L.INSPECT, cameras=L.CAMERAS,
        carrier=dict(puck_d=L.PUCK_D * mm, puck_top=(L.SOLE_Z + L.PUCK_TOP) * mm, base_h=(L.SOLE_Z + L.SOLE_T + L.BASE_H) * mm, neck_d=L.NECK_D * mm,
                     tube_d=L.TUBE_OD * mm, tube_bottom=L.TUBE_BOTTOM_Z * mm, tube_top=L.TUBE_TOP_Z * mm, collar_d=L.COLLAR_OD * mm,
                     collar_h=L.COLLAR_H * mm, grip_z=L.GRIP_Z * mm),
        cap=dict(d=L.CAP_OD * mm, h=L.CAP_H * mm, seat=(L.TUBE_TOP_Z - (L.CAP_H - L.CAP_TOP_T)) * mm),
        ball_d=L.BALL_D * mm, counts=dict(bottles=L.N_BOTTLES, caps=L.N_CAPS, balls=L.N_BALLS),
    )


def write_bom(parts, vols, routes, reach, out: Path) -> None:
    lines = ["# 八臂循环产线:打印零件与物料清单", "", "由 `python -m loop_cell.build` 从 `layout.py` 生成,不要手改。", ""]
    lines += ["## 关键尺寸", "", "| 项 | 值 |", "|---|---|"]
    lines += [
        f"| 底座拼接块 | {L.TILE:g} × {L.TILE:g} × {L.TILE_T:g} mm,M4 热熔螺母网格 {L.GRID:g} mm,台面以上 {L.PT * 1000:g} mm(含 {L.PAD_T:g} mm TPU 脚垫) |",
        f"| 托座 | Ø{L.PUCK_D:g} mm,高 {L.PUCK_TOP:g} mm(含 {L.SOLE_T:g} mm 底片),3 颗 Ø{L.BB_D:g} 钢珠凸出 {L.BB_PROTRUDE:.2f} mm |",
        f"| 透明管 | Ø{L.TUBE_OD:g} × {L.TUBE_L:g} mm(两端开口,下端粘进托座 {L.SOCKET_DEPTH:g} mm);管口离滑道底面 {L.TUBE_TOP_Z:.1f} mm |",
        f"| 瓶口套环 / 瓶盖 | 套环 Ø{L.COLLAR_OD:g} × {L.COLLAR_H:g} mm,四头卡口(每段 {L.THREAD_SPAN:g}°,导程 {L.THREAD_LEAD:g} mm);瓶盖 Ø{L.CAP_OD:g} × {L.CAP_H:g} mm |",
        f"| 小球 | Ø{L.BALL_D:g} mm 成品塑料球,红绿蓝共 {L.N_BALLS} 个 |",
    ]
    for fam, f in L.FAMILIES.items():
        lines.append(f"| {fam} 滑道 | 内宽 {f['W']:g} mm,墙高 {f['H']:g} mm,弯道半径 {f['R']:g} mm,坡度 {f['slope']:g}°,喇叭段 {f['flare']:g} / 闸门段 {f['gate']:g} mm |")
    lines += ["", "## 打印零件", "", "| 零件 | 数量 | 材料 | 单件体积 cm³ | 合计实体重量 g | 说明 |", "|---|---:|---|---:|---:|---|"]
    total_g = 0.0
    for p in parts:
        v = vols.get(p["name"])
        g = (v / 1000 * PLA * p["qty"]) if v else 0
        total_g += g
        lines.append(f"| `{p['name']}` | {p['qty']} | {p['mat']} | {v / 1000:.1f} | {g:.0f} | {p['note']} |" if v else f"| `{p['name']}` | {p['qty']} | {p['mat']} | – | – | {p['note']} |")
    if vols:
        lines += ["", f"实体重量合计约 **{total_g / 1000:.1f} kg**(按 100% 实心、密度 1.24 g/cm³);实际用料按填充率大约是它的 50–80%。"]
    lines += ["", "## 外购件", "", "| 物料 | 数量 | 说明 |", "|---|---:|---|"]
    n_gates = sum(1 for r in routes.values() if not r.transit)
    n_posts = sum(len(r.joints) for r in routes.values()) + 2
    lines += [
        f"| 透明管 Ø{L.TUBE_OD:g} × {L.TUBE_L:g} mm | {L.N_BOTTLES + 1} | PET 或 PC,两端开口 |",
        f"| Ø{L.BB_D:g} mm 钢珠 | {3 * (L.N_BOTTLES + 1)} | 托座滚动用 |",
        f"| Ø{L.BALL_D:g} mm 彩色塑料球 | {L.N_BALLS + 6} | 红绿蓝各 {(L.N_BALLS + 6) // 3},同一批次 |",
        f"| SG90 舵机 | {n_gates} | 每个闸门一个 |",
        f"| 对射红外光电(5 mm) | {n_gates} 对 | 装在闸门的 Ø5.2 孔 |",
        f"| M4 热熔螺母 | 约 {4 * len(L.ARMS) + 4 * 5 + 4 * 2 + 4 * part_qty(parts, 'tile_joiner')} | 只装在用到的孔位(臂压块、工位、相机、连接板) |",
        f"| M4 螺丝 / 螺母 | 约 {4 * part_qty(parts, 'tile_joiner') + 4 * len(L.ARMS) + 30} / {2 * n_posts} | 螺母:每根支柱上下各一 |",
        f"| M4 螺杆 | {n_posts} 根 | 每个支柱一根,长度见下方支柱表 |",
        "| PTFE 胶带 | 1 卷 | 贴在盖滑道底面和弯道外墙 |",
        "| 瞬干胶 / 环氧 | – | 托座底片、管子和套环 |",
    ]
    lines += ["", "## 每条滑道的拼装顺序", ""]
    for k, r in routes.items():
        seq = " → ".join(f"{m.kind}{'' if m.kind in ('flare', 'gate') else ' ' + format(m.length_mm, '.0f')}{' (' + m.turn + ')' if m.turn else ''}" for m in r.modules)
        lines.append(f"- **{k}**({r.family},{r.length * 1000:.0f} mm,起点底面高 {r.z(0) * 1000:.0f} mm → 末端 {r.z_end * 1000:.0f} mm):{seq}")
    lines += ["", "## 支柱高度", "", "从地面(台面或拼接块顶面)到接头托架底面。螺杆长度 = 支柱高度 + 20 mm。", "", "| 滑道 | 接头 | 地面 | 支柱 mm | 垫块 | 余量 mm(螺母调) |", "|---|---:|---|---:|---|---:|"]
    for k, r in routes.items():
        for i, j in enumerate(r.joints):
            g = "拼接块" if j["ground"] > 0 else "台面"
            lines.append(f"| {k} | {i} | {g} | {j['post_mm']:.0f} | {' + '.join(_fmt(s) for s in j['spacers']) or '–'} | {j['adjust_mm']:.1f} |")
    for name, post, stack, rest in extra_posts(routes)["rows"]:
        lines.append(f"| {name} | – | {'拼接块' if name == 'funnel' else '台面'} | {post:.0f} | {' + '.join(_fmt(s) for s in stack) or '–'} | {rest:.1f} |")
    lines += ["", "## 机械臂可达性", "", "末端朝下够不到时,允许工具倾斜;倾斜越大越吃力(超过 60° 需要调布局)。", "", "| 臂 | 作业点 | 水平距离 cm | 高度 cm | 需要倾斜 |", "|---|---|---:|---:|---:|"]
    for row in reach:
        t = "够不到" if row["tilt"] is None else f"{row['tilt']:.0f}°"
        lines.append(f"| {row['arm']} | {row['what']} | {row['horiz'] * 100:.1f} | {row['z'] * 100:.1f} | {t} |")
    lines += ["", "## 打印建议", "",
              "- 先打 `fit_coupon`,用实际的管子、钢珠、热熔螺母试配合,再改 `layout.py` 里的 `SOCKET_D`、`BB_LIP_D`、`cad.M4_INSERT`。",
              "- 滑道零件按导出的方向打印(底面贴平台),层纹顺着流动方向;底面贴 PTFE 胶带的只有盖滑道。",
              "- 瓶盖顶面朝下打印;托座、套环 PETG,0.2 mm 层高。",
              "- 闸门的舵机托架和摇臂尺寸按 SG90 标称值画的,装之前用实物比一下。",
              "- `arm_clamp_dog` 的唇口高度要按你的 SO-101 底座法兰厚度改(`cad.clamp_dog(flange_h=...)`)。"]
    (out / "BOM.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def part_qty(parts, name):
    return sum(p["qty"] for p in parts if p["name"] == name)


def inject_page(data: dict) -> None:
    tpl = (HERE / "page_template.html").read_text(encoding="utf-8")
    marker = "/*@LAYOUT@*/null"
    if marker not in tpl:
        raise RuntimeError("page_template.html has no layout marker")
    (HERE / "loop_cell.html").write_text(tpl.replace(marker, json.dumps(data, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-cad", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "cad")
    a = ap.parse_args(argv)
    routes = build_all()
    reach = reach_report(routes)
    bad = [r for r in reach if r["tilt"] is None or r["tilt"] > 60]
    for r in bad:
        print(f"WARNING reach: {r['arm']} {r['what']} tilt={r['tilt']}")
    for k, r in routes.items():
        for j in r.joints:
            if not j["ok"]:
                print(f"WARNING post too short on {k} at s={j['s']:.3f}: {j['post_mm']:.0f} mm")
    parts = part_list(routes)
    a.out.mkdir(parents=True, exist_ok=True)
    vols = {} if a.no_cad else export_parts(parts, a.out)
    write_bom(parts, vols, routes, reach, a.out)
    data = page_layout(routes)
    (a.out / "layout.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    inject_page(data)
    print(f"parts: {len(parts)} kinds, {sum(p['qty'] for p in parts)} pieces -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
