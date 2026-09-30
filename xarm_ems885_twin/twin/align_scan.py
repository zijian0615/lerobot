"""Place the Scaniverse table scan in the twin's world frame (Robot1 base frame) using the overhead camera.

    uv run python xarm_ems885_twin/twin/align_scan.py --photo overhead.png          # empty-table frame from /dev/video0
    uv run python xarm_ems885_twin/twin/align_scan.py --capture /dev/video0         # grab the frame itself (read-only)

1. Overhead pixels -> world XY: homography fitted to the tabletop calibration samples (uv of the object vs. the TCP
   position the operator jogged onto it, `true_base_mm` = Robot1 base frame).
2. Scan -> level: glTF +Y up becomes +Z up, the dominant plane (the table) is rotated flat and moved to z = 0.
3. Both are rendered top-down at the same resolution; the purple tic-tac-toe tape and the beige mat are segmented in both and
   a 2-D rigid transform (rotation + translation, the scan keeps its metric scale) is found by a coarse rotation search
   and ECC refinement. The mat is not centred on the grid, which breaks the grid's 4-fold symmetry.

Writes <out>/scan_alignment.json (T_world_scan: 4x4, glTF scan coordinates -> world metres) and two check images:
alignment_topdown.png (camera plate with the scan's tape and mat outlines) and alignment_camera.png (scan texture projected
back into the photo). Redo it whenever the scan, the table or the camera changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glb import load_glb  # noqa: E402

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / "twin_config.json"
RES_M = 0.002                       # top-down render resolution
PLANE_TOL_M = 0.006                 # RANSAC inlier band for the table plane


def load_config(path=CONFIG):
    cfg = json.loads(Path(path).read_text())
    base = Path(path).resolve().parent
    cfg["_dir"] = base
    return cfg


def resolve(cfg, rel):
    return (cfg["_dir"] / rel).resolve()


# --------------------------------------------------------------------------------------------------------------- camera
def homography_from_samples(samples_path):
    """H (3x3): overhead pixel (u, v) -> world (x, y) metres, fitted to every calibration sample. Also returns stats."""
    samples = json.loads(Path(samples_path).read_text())["samples"]
    uv = np.array([s["uv"] for s in samples], dtype=float)
    xy = np.array([s["true_base_mm"] for s in samples], dtype=float) / 1000.0
    if len(uv) < 4:
        raise SystemExit(f"{samples_path}: need >= 4 samples for a homography, got {len(uv)}")
    H, _ = cv2.findHomography(uv, xy, 0)
    res = np.linalg.norm(cv2.perspectiveTransform(uv[None], H)[0] - xy, axis=1)
    loo = []
    for i in range(len(uv)):
        keep = np.arange(len(uv)) != i
        Hi, _ = cv2.findHomography(uv[keep], xy[keep], 0)
        loo.append(np.linalg.norm(cv2.perspectiveTransform(uv[i:i + 1][None], Hi)[0, 0] - xy[i]))
    stats = {"n": len(uv), "rms_mm": float(np.sqrt(np.mean(res ** 2)) * 1000),
             "max_mm": float(res.max() * 1000), "loo_rms_mm": float(np.sqrt(np.mean(np.square(loo))) * 1000)}
    return H, stats


class Grid:
    """Raster of the world XY plane: row 0 = x1 (rows run along -x), columns run along +y.

    Note this is the table seen from BELOW (a mirror image of the overhead camera, whose image right is -y). It is only used
    internally: the camera plate and the scan are rasterised with the same mapping, so the transform found between them
    is a proper rotation (checked: det > 0). preview.py flips it left-right for display.
    """

    def __init__(self, x0, x1, y0, y1, res=RES_M):
        self.x0, self.x1, self.y0, self.y1, self.res = x0, x1, y0, y1, res
        self.w = int(round((y1 - y0) / res))      # columns run along +y
        self.h = int(round((x1 - x0) / res))      # rows run along -x (row 0 = x1, the far edge, like the camera)

    def to_px(self):
        """3x3 affine: world (x, y, 1) -> pixel (col, row, 1)."""
        return np.array([[0.0, 1.0 / self.res, -self.y0 / self.res],
                         [-1.0 / self.res, 0.0, self.x1 / self.res],
                         [0.0, 0.0, 1.0]])

    def world_of(self, cols, rows):
        return self.x1 - (np.asarray(rows) + 0.5) * self.res, self.y0 + (np.asarray(cols) + 0.5) * self.res


def camera_topdown(photo_bgr, H_uv_world, grid):
    """Warp the overhead photo onto the world grid. Pixels outside the photo are black (mask returned)."""
    M = grid.to_px() @ H_uv_world                  # photo pixel -> grid pixel
    img = cv2.warpPerspective(photo_bgr, M, (grid.w, grid.h), flags=cv2.INTER_LINEAR)
    valid = cv2.warpPerspective(np.full(photo_bgr.shape[:2], 255, np.uint8), M, (grid.w, grid.h), flags=cv2.INTER_NEAREST)
    return img, valid > 0


# ----------------------------------------------------------------------------------------------------------------- scan
def gltf_to_zup():
    """glTF (+Y up, +Z towards the viewer) -> right-handed +Z up: (x, y, z) -> (x, -z, y)."""
    return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


def fit_table_plane(pts, iters=400, tol=PLANE_TOL_M, seed=0):
    """RANSAC plane among near-horizontal candidates (normal within 30 deg of +z). Returns (normal, d, inlier mask)."""
    rng = np.random.default_rng(seed)
    best = (None, None, np.zeros(len(pts), bool))
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        if n[2] < math.cos(math.radians(30)):
            continue
        inl = np.abs(pts @ n - n @ a) < tol
        if inl.sum() > best[2].sum():
            best = (n, float(n @ a), inl)
    n, _, inl = best
    if n is None:
        raise RuntimeError("no table plane found in the scan")
    # least-squares refit on the inliers
    P = pts[inl]
    c = P.mean(0)
    n = np.linalg.svd(P - c)[2][2]
    n = n if n[2] > 0 else -n
    return n, float(n @ c), np.abs(pts @ n - n @ c) < tol


def rot_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def level_transform(positions):
    """4x4: glTF scan coords -> levelled frame (table plane = z 0, +z up). Also returns plane stats."""
    R0 = gltf_to_zup()
    p = positions @ R0.T
    n, d, inl = fit_table_plane(p)
    R1 = rot_between(n, np.array([0.0, 0.0, 1.0]))
    T = np.eye(4)
    T[:3, :3] = R1 @ R0
    T[2, 3] = -d                                    # after R1, the plane is z = d
    tilt = math.degrees(math.acos(min(1.0, n[2])))
    return T, {"table_inliers": int(inl.sum()), "vertices": len(p), "tilt_deg": tilt}


def apply(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def render_scan_topdown(mesh, T, grid, z_band=(-0.03, 0.03), samples_per_px=2.0):
    """Texture-coloured top view of the mesh parts inside z_band (the table surface), world grid given by T."""
    tex = mesh.texture()[:, :, ::-1]                                  # BGR like the photo
    th, tw = tex.shape[:2]
    P = apply(T, mesh.positions)
    F = mesh.faces
    zc = P[F, 2].mean(1)
    F = F[(zc > z_band[0]) & (zc < z_band[1])]
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    ab, ac = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    area = 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    n_s = np.maximum(1, np.ceil(area / grid.res ** 2 * samples_per_px)).astype(int)
    idx = np.repeat(np.arange(len(F)), n_s)
    rng = np.random.default_rng(0)
    r1, r2 = rng.random(len(idx)), rng.random(len(idx))
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    w0 = 1 - r1 - r2
    pts = w0[:, None] * a[idx] + r1[:, None] * b[idx] + r2[:, None] * c[idx]
    uva, uvb, uvc = mesh.uv[F[idx, 0]], mesh.uv[F[idx, 1]], mesh.uv[F[idx, 2]]
    uv = w0[:, None] * uva + r1[:, None] * uvb + r2[:, None] * uvc
    col = tex[np.clip((uv[:, 1] * th).astype(int), 0, th - 1), np.clip((uv[:, 0] * tw).astype(int), 0, tw - 1)]
    px = (grid.to_px() @ np.c_[pts[:, :2], np.ones(len(pts))].T).T
    cc, rr = px[:, 0].astype(int), px[:, 1].astype(int)
    ok = (cc >= 0) & (cc < grid.w) & (rr >= 0) & (rr < grid.h)
    img = np.zeros((grid.h, grid.w, 3), np.uint8)
    img[rr[ok], cc[ok]] = col[ok]
    filled = np.zeros((grid.h, grid.w), np.uint8)
    filled[rr[ok], cc[ok]] = 255
    # close pinholes between samples
    img = cv2.dilate(img, np.ones((3, 3), np.uint8)) * (filled[..., None] == 0) + img * (filled[..., None] > 0)
    filled = cv2.dilate(filled, np.ones((3, 3), np.uint8))
    return img.astype(np.uint8), filled > 0


# ------------------------------------------------------------------------------------------------------------- matching
def segment(img_bgr, valid):
    """(purple tape mask, beige mat mask) as float32 0/1."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    purple = (h >= 118) & (h <= 160) & (s >= 60) & (v >= 50) & valid
    # the mat is bright and warm (hue ~20). The scan's texture is much paler (saturation ~30 vs ~100 in the camera), so
    # separate it from the blue-grey table (hue ~100) by hue, not saturation.
    mat = (v >= 130) & (s >= 15) & (s <= 170) & (h >= 5) & (h <= 40) & ~purple & valid
    k = np.ones((5, 5), np.uint8)
    purple = cv2.morphologyEx(purple.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mat = cv2.morphologyEx(mat.astype(np.uint8), cv2.MORPH_CLOSE, k)
    mat = np.maximum(mat, purple)                     # the tape lies on the mat
    return purple.astype(np.float32), mat.astype(np.float32)


def feature_image(purple, mat):
    return cv2.GaussianBlur(purple * 1.0 + mat * 0.35, (0, 0), 3)


def _score(a, b, valid):
    a, b = a[valid], b[valid]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def register(src_feat, dst_feat, dst_valid, src_purple, dst_purple, step_deg=2.0):
    """2-D rigid warp (2x3, src pixel -> dst pixel) maximising the correlation of the feature images."""
    h, w = dst_feat.shape
    sc = np.argwhere(src_purple > 0).mean(0)[::-1]       # (col, row) centroids of the tape
    dc = np.argwhere(dst_purple > 0).mean(0)[::-1]
    best = (-2.0, None)
    for ang in np.arange(0.0, 360.0, step_deg):
        M = cv2.getRotationMatrix2D(tuple(sc), ang, 1.0)
        M[:, 2] += dc - sc
        warped = cv2.warpAffine(src_feat, M, (w, h))
        # small translation refinement by phase correlation
        (dx, dy), _ = cv2.phaseCorrelate(warped.astype(np.float64), dst_feat.astype(np.float64))
        if abs(dx) < 60 and abs(dy) < 60:
            M[:, 2] += (dx, dy)
            warped = cv2.warpAffine(src_feat, M, (w, h))
        s = _score(warped, dst_feat, dst_valid)
        if s > best[0]:
            best = (s, M.copy())
    coarse_score, M = best
    warp = M.astype(np.float32)
    try:
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6)
        _, warp = cv2.findTransformECC(dst_feat, src_feat, cv2.invertAffineTransform(warp).astype(np.float32),
                                       cv2.MOTION_EUCLIDEAN, crit, dst_valid.astype(np.uint8), 5)
        warp = cv2.invertAffineTransform(warp)
    except cv2.error as e:  # ECC can fail to converge; keep the coarse result
        print(f"ECC refinement failed ({e}); using the coarse match")
    final = _score(cv2.warpAffine(src_feat, warp, (w, h)), dst_feat, dst_valid)
    return warp, coarse_score, final


def chamfer_mm(src_mask, dst_mask, valid, res=RES_M):
    """Symmetric mean distance [mm] between the tape outlines of the aligned scan and the camera."""
    def edges(m):
        m = ((m > 0.5) & valid).astype(np.uint8)
        return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0

    es, ed = edges(src_mask), edges(dst_mask)
    if not es.any() or not ed.any():
        return float("nan")
    d_to_dst = cv2.distanceTransform((~ed).astype(np.uint8), cv2.DIST_L2, 5)
    d_to_src = cv2.distanceTransform((~es).astype(np.uint8), cv2.DIST_L2, 5)
    return float(0.5 * (d_to_dst[es].mean() + d_to_src[ed].mean()) * res * 1000)


def iou(a, b, valid):
    a, b = (a > 0.5) & valid, (b > 0.5) & valid
    return float((a & b).sum() / max((a | b).sum(), 1))


# ------------------------------------------------------------------------------------------------------------------ run
def capture(device, width=640, height=480):
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    frame = None
    for _ in range(20):                              # let exposure settle
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        raise SystemExit(f"could not read {device}")
    return frame


def align(cfg, photo_bgr, out_dir):
    H, hstats = homography_from_samples(resolve(cfg, cfg["overhead"]["calib_samples"]))
    print(f"overhead homography: n={hstats['n']} rms {hstats['rms_mm']:.1f} mm, max {hstats['max_mm']:.1f} mm, "
          f"leave-one-out {hstats['loo_rms_mm']:.1f} mm")
    ph, pw = photo_bgr.shape[:2]
    corners = cv2.perspectiveTransform(np.array([[[0, 0], [pw, 0], [pw, ph], [0, ph]]], float), H)[0]
    pad = 0.05

    mesh = load_glb(resolve(cfg, cfg["scan"]["glb"]))
    T_level, lstats = level_transform(mesh.positions)
    print(f"scan: {lstats['vertices']} vertices, table plane {lstats['table_inliers']} inliers, tilt {lstats['tilt_deg']:.1f} deg")
    lp = apply(T_level, mesh.positions)
    ext = lp[np.abs(lp[:, 2]) < 0.03]
    # the scan rendered in its own levelled frame, same resolution as the camera plate
    sgrid = Grid(ext[:, 0].min() - pad, ext[:, 0].max() + pad, ext[:, 1].min() - pad, ext[:, 1].max() + pad)
    scan_img, scan_valid = render_scan_topdown(mesh, T_level, sgrid)
    scan_purple, scan_mat = segment(scan_img, scan_valid)

    # camera plate: the photo's footprint, grown so the whole scan fits in any rotation (no cropping)
    cx, cy = corners[:, 0].mean(), corners[:, 1].mean()
    half = max(np.abs(corners[:, 0] - cx).max(), np.abs(corners[:, 1] - cy).max(),
               0.5 * math.hypot(sgrid.x1 - sgrid.x0, sgrid.y1 - sgrid.y0)) + pad
    grid = Grid(cx - half, cx + half, cy - half, cy + half)
    cam_img, cam_valid = camera_topdown(photo_bgr, H, grid)
    cam_purple, cam_mat = segment(cam_img, cam_valid)
    if scan_purple.sum() < 200 or cam_purple.sum() < 200:
        raise SystemExit(f"tape not found (scan {int(scan_purple.sum())} px, camera {int(cam_purple.sum())} px); "
                         "check the photo shows the purple grid")

    # paste the whole scan raster into the middle of the camera canvas
    canvas = (grid.h, grid.w)
    sh, sw = scan_purple.shape
    oy, ox = (canvas[0] - sh) // 2, (canvas[1] - sw) // 2
    assert oy >= 0 and ox >= 0, "canvas smaller than the scan raster"
    tgt = (slice(oy, oy + sh), slice(ox, ox + sw))
    src_feat, src_p, src_m = (np.zeros(canvas, np.float32) for _ in range(3))
    src_feat[tgt], src_p[tgt], src_m[tgt] = feature_image(scan_purple, scan_mat), scan_purple, scan_mat
    dst_feat = feature_image(cam_purple, cam_mat)
    warp, coarse, final = register(src_feat, dst_feat, cam_valid, src_p, cam_purple)

    # compose: scan glTF -> levelled -> scan raster px -> canvas px (offset) -> camera grid px (warp) -> world
    S = sgrid.to_px().copy()
    S[0, 2] += ox
    S[1, 2] += oy
    W = np.vstack([warp, [0, 0, 1]])
    A2 = np.linalg.inv(grid.to_px()) @ W @ S       # levelled scan xy -> world xy (2-D rigid, may contain a reflection check)
    R2, t2 = A2[:2, :2], A2[:2, 2]
    det = np.linalg.det(R2)
    if det < 0:
        raise RuntimeError("alignment came out mirrored; the pixel-frame bookkeeping is wrong")
    T_plane = np.eye(4)
    T_plane[:2, :2], T_plane[:2, 3] = R2, t2
    T_world_scan = T_plane @ T_level
    yaw = math.degrees(math.atan2(R2[1, 0], R2[0, 0]))

    warped_p = cv2.warpAffine(src_p, warp, (grid.w, grid.h))
    warped_m = cv2.warpAffine(src_m, warp, (grid.w, grid.h))
    metrics = {"coarse_corr": coarse, "final_corr": final,
               "tape_iou": iou(warped_p, cam_purple, cam_valid), "mat_iou": iou(warped_m, cam_mat, cam_valid),
               "tape_chamfer_mm": chamfer_mm(warped_p, cam_purple, cam_valid),
               "yaw_deg": yaw, "scale_check": float(math.sqrt(abs(det)))}
    print("alignment: " + ", ".join(f"{k} {v:.3f}" for k, v in metrics.items()))

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "_doc": "T_world_scan maps the GLB's glTF coordinates (metres, +Y up) into the twin world (Robot1 base frame, "
                "+Z up, table top z = 0). Written by xarm_ems885_twin/twin/align_scan.py.",
        "glb": str(cfg["scan"]["glb"]),
        "T_world_scan": T_world_scan.tolist(),
        "level": lstats,
        "homography": hstats,
        "metrics": metrics,
    }
    (out_dir / "scan_alignment.json").write_text(json.dumps(result, indent=2) + "\n")

    # check image 1: camera plate + scan outlines
    vis = cam_img.copy()
    for mask, colour in ((warped_m, (0, 255, 255)), (warped_p, (255, 0, 255))):
        cnts, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, colour, 1)
    cv2.imwrite(str(out_dir / "alignment_topdown.png"), vis)
    # check image 2: scan texture projected into the photo through the homography
    world_img, world_valid = render_scan_topdown(mesh, T_world_scan, grid)
    Hinv = np.linalg.inv(grid.to_px() @ H)
    back = cv2.warpPerspective(world_img, Hinv, (pw, ph))
    bvalid = cv2.warpPerspective(world_valid.astype(np.uint8) * 255, Hinv, (pw, ph)) > 0
    blend = photo_bgr.copy()
    blend[bvalid] = (0.5 * photo_bgr[bvalid] + 0.5 * back[bvalid]).astype(np.uint8)
    cv2.imwrite(str(out_dir / "alignment_camera.png"), np.hstack([photo_bgr, blend]))
    print(f"wrote {out_dir / 'scan_alignment.json'} and check images in {out_dir}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--photo", type=Path, help="overhead frame (same camera pose as the calibration)")
    src.add_argument("--capture", metavar="DEVICE", help="grab a frame from this camera, e.g. /dev/video0")
    ap.add_argument("--out", type=Path, default=None, help="output dir (default: next to the GLB)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    photo = capture(a.capture) if a.capture else cv2.imread(str(a.photo))
    if photo is None:
        ap.error(f"could not read {a.photo}")
    out = a.out or resolve(cfg, cfg["scan"]["alignment"]).parent
    if a.capture:
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "alignment_photo.png"), photo)
    align(cfg, photo, out)


if __name__ == "__main__":
    main()
