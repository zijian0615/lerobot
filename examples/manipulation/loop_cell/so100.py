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

"""SO-100 arm model from TheRobotStudio's URDF (SO-ARM100, Apache-2.0).

Writes ``so100_mesh.json`` (the 13 meshes as base64 float32 vertices + indices) and ``so100_model.json`` (joint chain,
which mesh belongs to which link, the gripper pinch point, and a planar IK model)::

    python -m loop_cell.so100 --src <dir with so100.urdf and assets/>   # local copy
    python -m loop_cell.so100                                          # download from GitHub

Needs numpy + trimesh.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO100"
TIP_BACK = 0.02  # pinch point: 2 cm back from the end of the fixed jaw
TIP_SIDE = -0.003  # and 3 mm towards the moving jaw (centre of a ~25 mm part)


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def _tf(xyz, rpy):
    m = np.eye(4)
    m[:3, :3] = _rpy(*rpy)
    m[:3, 3] = xyz
    return m


def _download(dst: Path) -> None:
    (dst / "assets").mkdir(parents=True, exist_ok=True)
    urdf = urllib.request.urlopen(f"{RAW}/so100.urdf", timeout=30).read()
    (dst / "so100.urdf").write_bytes(urdf)
    for mesh in sorted({m.get("filename") for m in ET.fromstring(urdf).iter("mesh")}):
        (dst / mesh).write_bytes(urllib.request.urlopen(f"{RAW}/{mesh}", timeout=60).read())


def build(src: Path, out_dir: Path) -> dict:
    import trimesh

    root = ET.parse(src / "so100.urdf").getroot()
    joints, visuals = [], {}
    for link in root.findall("link"):
        seen = []
        for v in link.findall("visual"):
            f = v.find("geometry/mesh").get("filename")
            if f in seen:
                continue
            seen.append(f)
            mat = v.find("material")
            kind = "motor" if (mat is not None and "sts" in (mat.get("name") or "")) or "Motor" in f else "printed"
            visuals.setdefault(link.get("name"), []).append(dict(mesh=Path(f).stem, kind=kind))
    world = {"base": np.eye(4)}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in o.get("xyz").split()]
        rpy = [float(v) for v in o.get("rpy").split()]
        lim = j.find("limit")
        parent, child = j.find("parent").get("link"), j.find("child").get("link")
        world[child] = world[parent] @ _tf(xyz, rpy)
        joints.append(dict(name=j.get("name"), parent=parent, child=child, xyz=xyz, rpy=rpy,
                           axis=[float(v) for v in j.find("axis").get("xyz").split()],
                           lower=float(lim.get("lower")), upper=float(lim.get("upper"))))
    by = {j["name"]: j for j in joints}

    def axis_w(name):
        j = by[name]
        return world[j["child"]][:3, :3] @ np.array(j["axis"])

    def pos(name):
        return world[by[name]["child"]][:3, 3]

    pan_axis = axis_w("shoulder_pan")
    assert np.allclose(pan_axis, [0, 0, 1], atol=1e-3), "expected a vertical pan axis"
    pitch = [axis_w(n) for n in ("shoulder_lift", "elbow_flex", "wrist_flex")]
    for a in pitch:
        assert np.allclose(a, pitch[0], atol=1e-3), "pitch axes must be parallel"
    n = pitch[0]
    pan_xy = pos("shoulder_pan")[:2]
    e_r = np.cross(np.array([0, 0, 1.0]), n)  # forward direction at pan = 0 (sign fixed below)

    # gripper pinch point from the fixed jaw mesh
    gw = world["gripper"]
    roll_axis = axis_w("wrist_roll")
    jaw = trimesh.load(src / "assets" / "Fixed_Jaw.stl")
    jv = (gw @ np.c_[jaw.vertices, np.ones(len(jaw.vertices))].T).T[:, :3]
    roll_p = gw[:3, 3]
    tool = -roll_axis if ((jv - roll_p) @ -roll_axis).max() > ((jv - roll_p) @ roll_axis).max() else roll_axis
    tool = tool / np.linalg.norm(tool)
    side = np.cross(n, tool)
    side /= np.linalg.norm(side)
    reach = ((jv - roll_p) @ tool).max()
    tip_w = roll_p + tool * (reach - TIP_BACK) + side * TIP_SIDE
    if (tip_w[:2] - pan_xy) @ e_r[:2] < 0:
        e_r = -e_r
    tip_local = np.linalg.inv(gw) @ np.append(tip_w, 1.0)
    tool_local = gw[:3, :3].T @ tool

    def plane(p):
        d = p[:2] - pan_xy
        return [float(d @ e_r[:2]), float(p[2])]

    p1, p2, p3, pt = plane(pos("shoulder_lift")), plane(pos("elbow_flex")), plane(pos("wrist_flex")), plane(tip_w)
    lateral = float((tip_w[:2] - pan_xy) @ n[:2])

    def ang(a, b):
        return math.atan2(b[1] - a[1], b[0] - a[0])

    # rotating +q about the pitch axis turns plane angles by sign * q
    sign = 1.0 if np.cross(e_r, [0, 0, 1.0]) @ n > 0 else -1.0
    model = dict(
        source="TheRobotStudio/SO-ARM100 Simulation/SO100 (Apache-2.0)",
        joints=joints, visuals=visuals,
        tip=dict(link="gripper", xyz=[float(v) for v in tip_local[:3]], tool=[float(v) for v in tool_local]),
        base_bottom=float(min(((world[l] @ np.c_[m.vertices, np.ones(len(m.vertices))].T).T[:, 2]).min()
                              for l, vs in visuals.items() for m in [trimesh.load(src / "assets" / f"{vs[0]['mesh']}.stl")] if l == "base")),
        plane=dict(
            pan_xy=[float(v) for v in pan_xy], forward_yaw=math.atan2(e_r[1], e_r[0]), sign=sign, lateral=lateral,
            p1=p1, L1=math.dist(p1, p2), L2=math.dist(p2, p3), LT=math.dist(p3, pt),
            a1_0=ang(p1, p2), a2_0=ang(p2, p3), a3_0=ang(p3, pt), phi0=math.atan2(tool @ np.array([0, 0, 1.0]), tool[:2] @ e_r[:2]),
            limits={k: [by[k]["lower"], by[k]["upper"]] for k in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")},
        ),
    )
    base_pts = []
    for vs in visuals["base"]:
        m = trimesh.load(src / "assets" / f"{vs['mesh']}.stl")
        base_pts.append(m.vertices[:, :2])
    bp = np.vstack(base_pts) - pan_xy
    model["plane"]["base_box"] = [float(bp[:, 0].min()), float(bp[:, 1].min()), float(bp[:, 0].max()), float(bp[:, 1].max())]

    meshes = {}
    for f in sorted((src / "assets").glob("*.stl")):
        m = trimesh.load(f)
        m.merge_vertices()
        idx = np.asarray(m.faces, dtype=np.uint32).ravel()
        idx_t = "u16" if idx.max() < 65536 else "u32"
        meshes[f.stem] = dict(
            pos=base64.b64encode(np.asarray(m.vertices, dtype="<f4").tobytes()).decode(),
            idx=base64.b64encode(idx.astype("<u2" if idx_t == "u16" else "<u4").tobytes()).decode(),
            idx_type=idx_t,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "so100_mesh.json").write_text(json.dumps(dict(source=model["source"], meshes=meshes)), encoding="utf-8")
    (out_dir / "so100_model.json").write_text(json.dumps(model, indent=1), encoding="utf-8")
    return model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, help="folder with so100.urdf and assets/ (default: download)")
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args(argv)
    if a.src is None:
        tmp = Path(tempfile.mkdtemp(prefix="so100_"))
        _download(tmp)
        a.src = tmp
    m = build(a.src, a.out)
    p = m["plane"]
    print(f"SO-100: L1={p['L1']:.4f} L2={p['L2']:.4f} wrist->pinch={p['LT']:.4f} m, lift axis at r={p['p1'][0]:.4f} z={p['p1'][1]:.4f}")
    print(f"wrote {a.out / 'so100_mesh.json'} ({(a.out / 'so100_mesh.json').stat().st_size / 1e6:.2f} MB) and so100_model.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
