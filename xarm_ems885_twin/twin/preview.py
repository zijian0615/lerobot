"""Orthographic preview of the twin (top + side), numpy/OpenCV only, to check the stage geometry without Isaac Sim.

    uv run python xarm_ems885_twin/twin/preview.py --out preview.png                 # both arms at home
    uv run python xarm_ems885_twin/twin/preview.py --read-robot xarm --out now.png    # Robot1 at its live joints (read-only)

Draws the aligned scan texture (top view) and every link mesh posed by the same Chain FK that omni_twin.py uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_scan import Grid, render_scan_topdown  # noqa: E402
from build_usd import ASSETS, CONFIG, OUT_JSON, load_stl  # noqa: E402
from glb import load_glb  # noqa: E402
from xarm_model import Chain, quat_to_mat  # noqa: E402

RES = 0.002
ARM_BGR = {"xarm": (235, 235, 230), "xarm2": (200, 215, 245)}


def posed_triangles(chain, joint_values, cache):
    """[(arm, (N,3,3) world triangles)] for every mesh geom."""
    frames = chain.forward(joint_values)
    out = []
    for b in chain.bodies:
        R, p = frames[b["name"]]
        arm = b["arm"]
        for g in b["geoms"]:
            if g["type"] == "mesh":
                if g["file"] not in cache:
                    cache[g["file"]] = load_stl(ASSETS / g["file"])
                tris = cache[g["file"]]
            else:  # box -> 12 triangles
                c, s = np.asarray(g["pos"]), np.asarray(g["size"]) / 2
                v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s + c
                idx = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
                       (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
                tris = v[np.array(idx)]
            gR = quat_to_mat(g.get("quat", [1, 0, 0, 0]))
            local = tris @ gR.T + np.asarray(g.get("pos", [0, 0, 0])) if g["type"] == "mesh" else tris
            out.append((arm, local @ R.T + p))
    return out


def draw(img, tris_2d, depth, colour, shade):
    order = np.argsort(depth)
    for i in order:
        c = tuple(int(v * shade[i]) for v in colour)
        cv2.fillConvexPoly(img, np.round(tris_2d[i]).astype(np.int32), c, lineType=cv2.LINE_AA)


def render(chain, joint_values, scan=None):
    cache = {}
    parts = posed_triangles(chain, joint_values, cache)
    # top view: same Grid as align_scan (rows = -x, cols = +y), flipped left-right at the end
    grid = Grid(-0.45, 1.0, -1.45, 0.45, RES)
    top = np.full((grid.h, grid.w, 3), 40, np.uint8)
    if scan is not None:
        img, valid = render_scan_topdown(scan[0], scan[1], grid, z_band=(-1.0, 0.2))
        top[valid] = img[valid]
    P = grid.to_px()
    # side view: looking along +y (x to the right, z up)
    sx0, sx1, sz0, sz1 = -0.45, 1.0, -0.1, 1.0
    side = np.full((int((sz1 - sz0) / RES), int((sx1 - sx0) / RES), 3), 40, np.uint8)
    light = np.array([0.3, -0.4, 0.87])
    for arm, t in parts:
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        shade = 0.35 + 0.65 * np.abs(n @ light)
        colour = ARM_BGR.get(arm, (200, 200, 200))
        uv = np.einsum("ij,nkj->nki", P[:2, :2], t[:, :, :2]) + P[:2, 2]
        draw(top, uv, t[:, :, 2].mean(1), colour, shade)
        su = np.stack([(t[:, :, 0] - sx0) / RES, (sz1 - t[:, :, 2]) / RES], axis=-1)
        draw(side, su, t[:, :, 1].mean(1) * -1, colour, shade)
    top = np.ascontiguousarray(top[:, ::-1])       # Grid is the view from below; flip to the overhead camera's view
    # table line in the side view
    cv2.line(side, (0, int(sz1 / RES)), (side.shape[1], int(sz1 / RES)), (60, 140, 200), 1)
    for img, label in ((top, "top, like the overhead camera (up=+x, right=-y)"), (side, "side (right=+x, up=+z)")):
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    h = max(top.shape[0], side.shape[0])
    pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))  # noqa: E731
    return np.hstack([pad(top), pad(side)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chain", type=Path, default=OUT_JSON)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--read-robot", action="append", default=[], metavar="ARM",
                    help="pose this arm from its live joints (read-only SDK connection)")
    ap.add_argument("--no-scan", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    chain = Chain.load(a.chain)
    state = {arm: (info.get("home_joints") or [0.0] * 6, 0.0) for arm, info in chain.arms.items()}
    for arm in a.read_robot:
        from sources import read_xarm_once
        q, drive = read_xarm_once(chain.arms[arm]["ip"])
        state[arm] = (q, drive)
        print(f"{arm}: joints(deg) {np.round(np.degrees(q), 1).tolist()} gripper drive {drive:.2f} rad")
    scan = None
    if not a.no_scan:
        cfg = json.loads(a.config.read_text())
        base = a.config.resolve().parent
        T = np.asarray(json.loads((base / cfg["scan"]["alignment"]).read_text())["T_world_scan"], float)
        scan = (load_glb(base / cfg["scan"]["glb"]), T)
    img = render(chain, chain.joint_values(state), scan)
    cv2.imwrite(str(a.out), img)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
