"""Rebuild the mat texture from real camera frames instead of the thresholded site-scan print.

Inputs (data/real_mat/): frames exported from a real rollout (table + wrist, with joints) and the sim
depth rendered at the same joints (render_pose.py), plus the camera solution from calib_solve.py.

1. Table camera: every frame is projected onto the mat plane on the texture grid; pixels the sim says
   are covered by the arm, the bin or the screw (dilated for cables and shadows) are dropped, and the
   per-pixel median over frames removes what is left of the arm.
2. Wrist camera: frames looking down from close range (sharper) are projected the same way, colour-
   matched to the table result, and laid over it where at least --min-wrist frames saw a pixel.
3. Clutter: dark blobs that are not near any line of the old texture (markers, spare screws, the zip
   tie, the tripod) are inpainted.

Writes assets/mat_print_real.png on the same grid as assets/mat_print.png (row 0 = front edge, +X;
column 0 = -Y edge), so scene_config.MAT_TEXTURE can point at it.

--scan reads a scan_mat.py directory instead (table parks + wrist grid over a cleared surface, sim depth
from render_pose.py --pose <dir>/shots.json --out <dir>). The surface is not assumed to carry the old
print: clutter is any dark blob, the tool's red laser spot is removed, the yellow tape is kept, and
unseen areas are filled with the surface colour.

    uv run python fanuc_er4ia_sim/make_real_mat.py
    uv run python fanuc_er4ia_sim/make_real_mat.py --scan --dir fanuc_er4ia_sim/data/mat_scan_v2 \
        --out fanuc_er4ia_sim/assets/surface_real.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bg_composite as B  # noqa: E402
import fk  # noqa: E402
import scene_config as C  # noqa: E402

HERE = Path(__file__).resolve().parent
PX_PER_M = 2000  # same grid as make_mat_texture.py


def grid_points():
    (x0, x1), (y0, y1) = C.TABLE_X, C.TABLE_Y
    rows, cols = int(round((x1 - x0) * PX_PER_M)), int(round((y1 - y0) * PX_PER_M))
    x = x1 - (np.arange(rows) + 0.5) / PX_PER_M
    y = y0 + (np.arange(cols) + 0.5) / PX_PER_M
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.stack([xx, yy, np.full_like(xx, C.SURFACE_DZ)], -1)  # base frame, on the mat


def project(points, T_base_cam, fx, k1, size):
    """Pixel maps (map_x, map_y) and depth for base-frame points through an OpenCV camera."""
    T = np.linalg.inv(T_base_cam)
    pc = points @ T[:3, :3].T + T[:3, 3]
    z = pc[..., 2]
    x, y = pc[..., 0] / z, pc[..., 1] / z
    r2 = x * x + y * y
    d = 1 + k1 * r2
    w, h = size
    return (fx * x * d + w / 2).astype(np.float32), (fx * y * d + h / 2).astype(np.float32), z


def remap(img, mx, my):
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def nanmedian_stack(stack):
    with np.errstate(all="ignore"):
        return np.nanmedian(np.stack(stack), axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=HERE / "data" / "real_mat")
    parser.add_argument("--solution", type=Path, default=HERE / "data" / "cam_calib" / "solution.json")
    parser.add_argument("--out", type=Path, default=HERE / "assets" / "mat_print_real.png")
    parser.add_argument("--occluder-margin-px", type=int, default=30)
    parser.add_argument("--min-wrist", type=int, default=3)
    parser.add_argument("--paper", type=float, default=225.0,
                        help="scale so the mat's paper reads this grey: the texture is albedo; each camera's own "
                             "response is fitted separately (fit_cam_tone.py)")
    parser.add_argument("--wrist-max-range-m", type=float, default=0.35)
    parser.add_argument("--wrist-max-tilt-deg", type=float, default=35.0,
                        help="only views whose optical axis is this close to straight down")
    parser.add_argument("--scan", action="store_true", help="--dir is a scan_mat.py directory (see above)")
    parser.add_argument("--clean-box", type=float, nargs=4, action="append", default=[],
                        metavar=("X0", "X1", "Y0", "Y1"),
                        help="base-frame box (m) to inpaint, e.g. a tripod foot standing on the surface; repeatable")
    parser.add_argument("--no-wrist", action="store_true",
                        help="table camera only (a plain surface gains little from the wrist and its patches show)")
    args = parser.parse_args()

    sol = json.loads(args.solution.read_text())
    shots = json.loads((args.dir / "shots.json").read_text())
    if args.scan:  # scan_mat.py: one image per shot, "kind" says which camera
        shots = [{**sh, "images": {sh["kind"]: sh["image"]}} for sh in shots]
    pts = grid_points()
    H, W = pts.shape[:2]

    # ---- table camera
    T_bt = np.array(sol["table"]["T_base_cam"])
    tmx, tmy, _ = project(pts, T_bt, sol["table"]["fx"], sol["table"]["k1"], sol["table"]["size"])
    table_stack = []
    for shot in shots:
        if "table" not in shot["images"]:
            continue
        img = cv2.imread(str(args.dir / shot["images"]["table"])).astype(np.float32)
        depth = np.load(args.dir / f"sim_{shot['tag']}" / "sim_table_depth.npy")
        _, above = B.classify("table", depth)
        occluded = cv2.dilate(above.astype(np.uint8), np.ones((2 * args.occluder_margin_px + 1,) * 2, np.uint8))
        ok = remap(255 - occluded * 255, tmx, tmy) > 128
        ortho = remap(img, tmx, tmy)
        ortho[~ok] = np.nan
        table_stack.append(ortho)
    if args.scan:
        # Few parks, each hiding a different side behind the arm, its cables and its shadow, none of
        # which the sim depth covers fully: per pixel keep the park whose view is closest to plain surface
        # (least saturated, brightest).
        stack = np.stack(table_stack)
        hsv = np.stack([cv2.cvtColor(np.nan_to_num(f).astype(np.uint8), cv2.COLOR_BGR2HSV) for f in table_stack])
        score = hsv[..., 1].astype(np.float32) + 0.5 * (255 - hsv[..., 2].astype(np.float32))
        score[np.isnan(stack[..., 0])] = np.inf
        pick = np.argmin(score, axis=0)
        table = np.take_along_axis(stack, pick[None, ..., None].repeat(3, -1), 0)[0]
    else:
        table = nanmedian_stack(table_stack)
    seen_table = ~np.isnan(table[..., 0])
    print(f"[mat] table camera covers {seen_table.mean() * 100:.0f}% of the mat")

    # ---- wrist camera (sharper, close range)
    T_fw = np.array(sol["wrist"]["T_flange_cam"])
    ww, wh = sol["wrist"]["size"]
    wrist_stack = []
    for shot in shots:
        if "wrist" not in shot["images"] or args.no_wrist:
            continue
        R, p = fk.fk(np.radians(shot["sim_joints_deg"]))
        T_bf = np.eye(4)
        T_bf[:3, :3], T_bf[:3, 3] = R, p
        T_bw = T_bf @ T_fw
        tilt = np.degrees(np.arccos(np.clip(-T_bw[2, 2], -1, 1)))
        if tilt > args.wrist_max_tilt_deg:
            continue
        mx, my, z = project(pts, T_bw, sol["wrist"]["fx"], sol["wrist"]["k1"], sol["wrist"]["size"])
        img = cv2.imread(str(args.dir / shot["images"]["wrist"])).astype(np.float32)
        ok = (z > 0.05) & (z < args.wrist_max_range_m) & (mx > 2) & (my > 2) & (mx < ww - 3) & (my < wh - 3)
        if ok.mean() < 0.001:
            continue
        # the bin, the screw and anything else standing on the mat: from the sim depth at this pose
        depth = np.load(args.dir / f"sim_{shot['tag']}" / "sim_wrist_depth.npy")
        fxw = sol["wrist"]["fx"]
        u, v = np.meshgrid(np.arange(ww) + 0.5, np.arange(wh) + 0.5)
        zz = np.where(np.isfinite(depth), depth, 1e3)
        cam_pts = np.stack([(u - ww / 2) / fxw * zz, (v - wh / 2) / fxw * zz, zz], -1)
        height = (cam_pts @ T_bw[:3, :3].T + T_bw[:3, 3])[..., 2]
        standing = (height > C.SURFACE_DZ + 0.004) & (height < 1.0)
        standing = cv2.dilate(standing.astype(np.uint8), np.ones((41, 41), np.uint8))
        ok &= remap(255 - standing * 255, mx, my) > 128
        ortho = remap(img, mx, my)
        ortho[~ok] = np.nan
        wrist_stack.append(ortho)
    result = np.nan_to_num(table)
    if wrist_stack:
        count = np.sum([~np.isnan(s[..., 0]) for s in wrist_stack], axis=0)
        wrist = nanmedian_stack(wrist_stack)
        use = (count >= args.min_wrist) & ~np.isnan(wrist[..., 0])
        both = use & seen_table
        # per-channel gain/offset so the wrist camera's colours match the table camera's
        for c in range(3):
            a, b = np.polyfit(wrist[both, c], table[both, c], 1)
            wrist[..., c] = wrist[..., c] * a + b
        alpha = cv2.GaussianBlur(use.astype(np.float32), (0, 0), 6)[..., None] * use[..., None]
        result = np.where(use[..., None], np.nan_to_num(wrist) * alpha + result * (1 - alpha), result)
        print(f"[mat] wrist camera ({len(wrist_stack)} frames) refines {use.mean() * 100:.1f}% of the mat")
        seen = seen_table | use
    else:
        seen = seen_table
    result = np.clip(result, 0, 255).astype(np.uint8)

    # ---- clutter that is not part of the print: a dark blob mostly off the old texture's lines
    old = cv2.imread(str(HERE / C.MAT_TEXTURE))
    old = cv2.resize(old, (W, H), interpolation=cv2.INTER_AREA)
    old_dark = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY) < 110
    if args.scan:  # a new surface: nothing dark belongs to it except what sits on the coloured tape
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        tape = (hsv[..., 1] > 80) & (hsv[..., 0] > 15) & (hsv[..., 0] < 40)
        tape = cv2.morphologyEx(tape.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        old_dark = cv2.dilate(tape, np.ones((31, 31), np.uint8)) > 0  # the tape and its lettering
    printed = cv2.dilate(old_dark.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local = cv2.medianBlur(gray.astype(np.uint8), 61).astype(np.float32)
    # relative darkness finds thin things; absolute darkness finds big ones (tripod, marker squares)
    dark = ((gray < 0.6 * local) | (gray < 0.45 * np.median(gray[seen]))) & seen
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8))
    on_print = np.bincount(lab.ravel(), weights=printed.ravel(), minlength=n) / np.maximum(st[:, cv2.CC_STAT_AREA], 1)
    clutter = np.isin(lab, [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 40 and on_print[k] < 0.5])
    # thick dark things (screws: 6 mm across) survive an opening that removes the 1-2 mm printed lines;
    # anything smaller than the printed black pad or the tape marks is not part of the print
    thick = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(thick)
    clutter |= np.isin(lab2, [k for k in range(1, n2) if 60 <= st2[k, cv2.CC_STAT_AREA] < 3000])
    if args.scan:  # the tool's laser spot: small, saturated red, away from the tape's lettering
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        red = ((hsv[..., 0] < 10) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 60) & ~printed & seen
        n4, lab4, st4, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8))
        clutter |= np.isin(lab4, [k for k in range(1, n4) if st4[k, cv2.CC_STAT_AREA] < 400])
    clutter = cv2.dilate(clutter.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    # the calibration markers (and their stickers) sit where calib_solve put them
    for name, T in ({} if args.scan else sol.get("table_markers_base", {})).items():
        x, y = T[0][3], T[1][3]
        r0 = int((C.TABLE_X[1] - x - 0.040) * PX_PER_M)
        c0 = int((y - C.TABLE_Y[0] - 0.030) * PX_PER_M)
        clutter[max(r0, 0):r0 + int(0.080 * PX_PER_M), max(c0, 0):c0 + int(0.060 * PX_PER_M)] = True
    for x0, x1, y0, y1 in args.clean_box:
        r0, r1 = int((C.TABLE_X[1] - x1) * PX_PER_M), int((C.TABLE_X[1] - x0) * PX_PER_M)
        c0, c1 = int((y0 - C.TABLE_Y[0]) * PX_PER_M), int((y1 - C.TABLE_Y[0]) * PX_PER_M)
        clutter[max(r0, 0):r1, max(c0, 0):c1] = True
    if not args.scan:  # scan: clutter is filled with the unseen areas below, in one pass
        result = cv2.inpaint(result, clutter.astype(np.uint8), 9, cv2.INPAINT_TELEA)

    # ---- never seen (under the arm's base, the front strip): the old print recoloured to the real mat
    paper = np.median(result[seen & ~printed & ~clutter], axis=0)
    g = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    ink = np.median(result[seen & ~clutter & (g <= np.percentile(g[seen & ~clutter], 1))], axis=0)
    old_hsv = cv2.cvtColor(old, cv2.COLOR_BGR2HSV)
    fallback = np.where(old_dark[..., None], ink, paper).astype(np.float32)
    coloured = old_hsv[..., 1] > 80  # the yellow CAUTION band and the red lettering keep their own colour
    fallback[coloured] = old[coloured]
    if args.scan:  # no old print to fall back on: plain surface; the tape keeps its scanned colour
        fallback[:] = paper
        coloured = printed & seen
    # small unseen holes (under the screw and the bin) are inpainted from the real print around them so
    # its thin lines carry through; only large areas (front strip, under the arm's base) use the old print
    n3, lab3, st3, _ = cv2.connectedComponentsWithStats((~seen).astype(np.uint8))
    small = np.isin(lab3, [k for k in range(1, n3) if st3[k, cv2.CC_STAT_AREA] < 0.01 * PX_PER_M**2])
    if args.scan:
        # A plain surface: grow every hole (unseen or clutter) in from its edges, at 1/4 size then upsampled.
        # The hole is widened first so the dark rim it leaves (unseen pixels are 0) is not grown inwards.
        q = 4
        hole = cv2.dilate((~seen | clutter).astype(np.uint8), np.ones((31, 31), np.uint8))
        hole[printed & seen & ~clutter] = 0  # the tape and its lettering stay as scanned
        hole_lo = cv2.resize(hole.astype(np.float32), (W // q, H // q), interpolation=cv2.INTER_AREA) > 0
        hole_lo = cv2.dilate(hole_lo.astype(np.uint8), np.ones((3, 3), np.uint8))
        lo = cv2.inpaint(cv2.resize(result, (W // q, H // q), interpolation=cv2.INTER_AREA), hole_lo, 5,
                         cv2.INPAINT_TELEA)
        fill = cv2.resize(cv2.GaussianBlur(lo, (0, 0), 3), (W, H), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 4)[..., None]
        result = (result.astype(np.float32) * (1 - alpha) + fill.astype(np.float32) * alpha).astype(np.uint8)
        small = np.zeros_like(small)
        seen = np.ones_like(seen)
    result = cv2.inpaint(result, small.astype(np.uint8), 9, cv2.INPAINT_TELEA)
    large = ~seen & ~small
    alpha = cv2.GaussianBlur(large.astype(np.float32), (0, 0), 8)[..., None]
    result = result.astype(np.float32) * (1 - alpha) + fallback * alpha
    # under the bin (+5 mm) nothing real is ever visible and its walls smear into every view: plain paper
    bx, by = C.BIN_XY
    half = C.BIN_OUTER[0] / 2 + 0.005
    r0, r1 = int((C.TABLE_X[1] - bx - half) * PX_PER_M), int((C.TABLE_X[1] - bx + half) * PX_PER_M)
    c0, c1 = int((by - half - C.TABLE_Y[0]) * PX_PER_M), int((by + half - C.TABLE_Y[0]) * PX_PER_M)
    if not args.scan:
        result[r0:r1, c0:c1] = paper
    result[~coloured] *= args.paper / float(np.median(paper))
    result = np.clip(result, 0, 255).astype(np.uint8)
    print(f"[mat] inpainted {clutter.mean() * 100:.2f}% clutter; {(~seen).mean() * 100:.1f}% never seen, "
          f"filled with the old print in paper {np.round(paper).tolist()} / ink {np.round(ink).tolist()}")
    cv2.imwrite(str(args.out), result)
    cv2.imwrite(str(args.dir / "clutter_mask.png"), clutter * 255)
    print(f"[mat] -> {args.out} ({W}x{H})")


if __name__ == "__main__":
    main()
