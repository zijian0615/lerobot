"""Locate the yellow bin and the screw on the real table from the calibrated fixed cameras.

Read-only: works on saved stills (snap_cams.py), the robot is not touched.

  bin    the BIN_OUTER x BIN_WALL_H box is projected into the table and side views and its pose
         (x, y, yaw) fitted so the silhouettes cover the yellow pixels (IoU, both views).
  screw  the dark blob near the bin in the table view, back-projected onto the table at the screw's
         lying height; yaw from its long axis, pointing head (wider end) -> tip.

All poses in the robot base frame (scene_config SCREWS / BIN_XY). Writes <dir>/objects.json and an
overlay <dir>/objects_overlay.jpg.

    uv run python fanuc_er4ia_sim/locate_objects.py --dir fanuc_er4ia_sim/data/objects_v2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_config as C  # noqa: E402

HERE = Path(__file__).resolve().parent
SCREW_LYING_Z = C.SURFACE_DZ + C.SCREW_SHANK_D / 2  # axis height of a screw lying on the surface


def load_cam(path: Path, name: str) -> dict:
    s = json.loads(path.read_text())[name]
    return {"T": np.array(s["T_base_cam"]), "fx": s["fx"], "k1": s["k1"], "size": tuple(s["size"])}


def project(cam: dict, pts: np.ndarray) -> np.ndarray:
    T = np.linalg.inv(cam["T"])
    pc = pts @ T[:3, :3].T + T[:3, 3]
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    d = 1 + cam["k1"] * (x * x + y * y)
    w, h = cam["size"]
    return np.stack([cam["fx"] * x * d + w / 2, cam["fx"] * y * d + h / 2], -1)


def back_project(cam: dict, uv: np.ndarray, z: float) -> np.ndarray:
    """Pixel -> base-frame point on the plane z (undistorting k1 by fixed-point iteration)."""
    w, h = cam["size"]
    xd, yd = (uv[0] - w / 2) / cam["fx"], (uv[1] - h / 2) / cam["fx"]
    x, y = xd, yd
    for _ in range(10):
        d = 1 + cam["k1"] * (x * x + y * y)
        x, y = xd / d, yd / d
    ray = cam["T"][:3, :3] @ np.array([x, y, 1.0])
    o = cam["T"][:3, 3]
    return o + ray * (z - o[2]) / ray[2]


def box_corners(x: float, y: float, yaw_deg: float, side: float = C.BIN_OUTER[0],
                height: float = C.BIN_WALL_H) -> np.ndarray:
    hx = hy = side / 2
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    xy = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]) @ np.array([[c, s], [-s, c]]) + [x, y]
    z0, z1 = C.SURFACE_DZ, C.SURFACE_DZ + height
    return np.concatenate([np.c_[xy, np.full(4, z0)], np.c_[xy, np.full(4, z1)]])


def silhouette(cam: dict, pose) -> np.ndarray:
    uv = project(cam, box_corners(*pose))
    mask = np.zeros(cam["size"][::-1], np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(uv.astype(np.int32)), 1)
    return mask.astype(bool)


def yellow_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= 8) & (hsv[..., 0] <= 35) & (hsv[..., 1] > 90) & (hsv[..., 2] > 80)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return m.astype(bool)


def largest_blob_near(mask: np.ndarray, uv0: np.ndarray, radius: float) -> np.ndarray:
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    best = min(range(1, n), key=lambda k: np.hypot(*(cen[k] - uv0)), default=None)
    if best is None or np.hypot(*(cen[best] - uv0)) > radius:
        raise SystemExit(f"no blob within {radius} px of {np.round(uv0)}")
    return lab == best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True, help="dir with table.png and side.png")
    parser.add_argument("--table-solution", type=Path, default=HERE / "data" / "cam_calib" / "solution.json")
    parser.add_argument("--side-solution", type=Path, default=HERE / "data" / "cam_calib_v2" / "solution.json")
    parser.add_argument("--bin-height", type=float, default=None, help="measured bin height (m): hold it in the fit")
    args = parser.parse_args()

    cams = {"table": load_cam(args.table_solution, "table"), "side": load_cam(args.side_solution, "side")}
    imgs = {name: cv2.imread(str(args.dir / f"{name}.png")) for name in cams}

    # ---- bin: in the table view, the compact yellow blob of about the bin's projected size (the tape
    # strips and the arm are long or huge); its back-projection seeds the side view; then fit the box
    cam = cams["table"]
    expect = (C.BIN_OUTER[0] * cam["fx"] / np.linalg.norm(cam["T"][:3, 3])) ** 2  # rough top-down area, px
    n, lab, st, cen = cv2.connectedComponentsWithStats(yellow_mask(imgs["table"]).astype(np.uint8))
    compact = [k for k in range(1, n) if 0.4 * expect < st[k, cv2.CC_STAT_AREA] < 4 * expect
               and 0.6 < st[k, cv2.CC_STAT_WIDTH] / st[k, cv2.CC_STAT_HEIGHT] < 1.6]
    if len(compact) != 1:
        raise SystemExit(f"[bin] expected one bin-sized yellow blob in the table view, found {len(compact)}")
    targets = {"table": lab == compact[0]}
    seed_xy = back_project(cam, cen[compact[0]], C.SURFACE_DZ + C.BIN_WALL_H / 2)[:2]
    uv0 = project(cams["side"], box_corners(*seed_xy, 0.0)).mean(0)
    targets["side"] = largest_blob_near(yellow_mask(imgs["side"]), uv0, radius=250)

    def cost(p):
        iou = []
        for name, cam in cams.items():
            sil = silhouette(cam, p)
            iou.append((sil & targets[name]).sum() / max((sil | targets[name]).sum(), 1))
        return -float(np.mean(iou))

    # pose first at the configured size, then pose + outer side + height together
    starts = [(seed_xy[0], seed_xy[1], yaw) for yaw in (-30, -15, 0, 15, 30)]
    best = min((minimize(cost, s, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-4}) for s in starts),
               key=lambda r: r.fun)
    iou_cfg = -best.fun
    if args.bin_height is None:
        full = minimize(cost, [*best.x, C.BIN_OUTER[0], C.BIN_WALL_H], method="Nelder-Mead",
                        options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 2000})
        bx, by, byaw, bside, bh = full.x
    else:
        full = minimize(lambda v: cost([*v, args.bin_height]), [*best.x, C.BIN_OUTER[0]], method="Nelder-Mead",
                        options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 2000})
        (bx, by, byaw, bside), bh = full.x, args.bin_height
    byaw = (byaw + 45) % 90 - 45  # a square bin: yaw modulo 90
    print(f"[bin] xy ({bx:.4f}, {by:.4f}) m, yaw {byaw:+.1f} deg, outer {bside * 1000:.1f} mm, height {bh * 1000:.1f} mm, "
          f"IoU {-full.fun:.3f} (at the configured {C.BIN_OUTER[0] * 1000:.0f}x{C.BIN_WALL_H * 1000:.0f} mm: {iou_cfg:.3f}); "
          f"scene_config BIN_XY {C.BIN_XY}, moved {np.hypot(bx - C.BIN_XY[0], by - C.BIN_XY[1]) * 1000:.0f} mm")

    # ---- screw: the dark elongated blob on the table near the bin, in the table view
    cam = cams["table"]
    gray = cv2.cvtColor(imgs["table"], cv2.COLOR_BGR2GRAY)
    local = cv2.medianBlur(gray, 51)
    dark = ((gray.astype(np.int16) < local.astype(np.int16) - 50) & ~cv2.dilate(
        targets["table"].astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark)
    bin_uv = project(cam, box_corners(bx, by, byaw, bside, bh)).mean(0)
    px_len = C.SCREW_LENGTH * cam["fx"] / np.linalg.norm(cam["T"][:3, 3] - [bx, by, 0])
    # screw-sized: its bounding box spans about one screw length (a cable is longer, a speck shorter)
    cands = [k for k in range(1, n) if 0.04 * px_len**2 < st[k, cv2.CC_STAT_AREA] < 2 * px_len**2
             and 0.5 * px_len < np.hypot(st[k, cv2.CC_STAT_WIDTH], st[k, cv2.CC_STAT_HEIGHT]) < 1.6 * px_len
             and np.hypot(*(cen[k] - bin_uv)) < 600]
    if not cands:
        raise SystemExit("[screw] no screw-sized dark blob near the bin")
    print(f"[screw] {len(cands)} screw-sized blob(s) within 600 px of the bin; taking the nearest")
    k = min(cands, key=lambda k: np.hypot(*(cen[k] - bin_uv)))
    v, u = np.nonzero(lab == k)
    # long axis in the table plane: back-project every blob pixel, PCA there (no perspective skew)
    pts = np.array([back_project(cam, np.array([uu, vv]), SCREW_LYING_Z)[:2] for uu, vv in zip(u, v, strict=True)])
    centre = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    axis = vt[0]
    t = (pts - centre) @ axis
    normal = (pts - centre) @ np.array([-axis[1], axis[0]])
    lo, hi = np.percentile(t, 3), np.percentile(t, 97)
    width = lambda sel: np.ptp(normal[sel]) if sel.sum() > 3 else 0.0  # noqa: E731
    head_at_lo = width(t < lo + 0.3 * (hi - lo)) > width(t > hi - 0.3 * (hi - lo))
    head_to_tip = axis if head_at_lo else -axis
    head = centre + head_to_tip * (lo if head_at_lo else -hi)
    syaw = float(np.degrees(np.arctan2(head_to_tip[1], head_to_tip[0])))
    print(f"[screw] centre ({centre[0]:.4f}, {centre[1]:.4f}) m, head ({head[0]:.4f}, {head[1]:.4f}), "
          f"head->tip yaw {syaw:+.1f} deg, seen length {(hi - lo) * 1000:.0f} mm "
          f"(scene_config SCREWS[0] {C.SCREWS[0]})")

    out = {"bin": {"xy": [float(bx), float(by)], "yaw_deg": float(byaw), "outer_m": float(bside),
                   "height_m": float(bh), "iou": float(-full.fun)},
           "screw": {"centre_xy": centre.tolist(), "head_xy": head.tolist(), "yaw_deg": syaw,
                     "seen_length_m": float(hi - lo)}}
    (args.dir / "objects.json").write_text(json.dumps(out, indent=1))

    # ---- overlay: fitted bin box (green), yellow target (magenta outline), screw axis (red)
    tiles = []
    for name, cam in cams.items():
        img = imgs[name].copy()
        cnt, _ = cv2.findContours(targets[name].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnt, -1, (255, 0, 255), 2)
        uv = project(cam, box_corners(bx, by, byaw, bside, bh)).astype(int)
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
            cv2.line(img, tuple(uv[a]), tuple(uv[b]), (0, 255, 0), 2)
        tip = head + head_to_tip * C.SCREW_LENGTH
        seg = project(cam, np.array([[*head, SCREW_LYING_Z], [*tip, SCREW_LYING_Z]])).astype(int)
        cv2.line(img, tuple(seg[0]), tuple(seg[1]), (0, 0, 255), 2)
        cv2.circle(img, tuple(seg[0]), 6, (0, 0, 255), 2)
        c = uv.mean(0).astype(int)
        crop = img[max(c[1] - 220, 0):c[1] + 220, max(c[0] - 400, 0):c[0] + 400]
        tiles.append(cv2.resize(crop, (800, 440)))
    cv2.imwrite(str(args.dir / "objects_overlay.jpg"), np.hstack(tiles))
    print(f"[locate] -> {args.dir / 'objects.json'}, {args.dir / 'objects_overlay.jpg'}")


if __name__ == "__main__":
    main()
