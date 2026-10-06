"""
Build the printed-mat texture from the Scaniverse site scan.

    /isaacsim/python.sh make_mat_texture.py --scan ~/Downloads/CSI-site-digital-model.zip

Writes assets/mat_print.png: the real mat artwork (reach arcs, screw target,
BATTERY KIT, grids, "#" tape mark, black pad, table seams) plus the yellow
CAUTION edge bands, rectified into the ROBOT frame at 2 px/mm. The scene maps
it onto the tabletop, so every printed feature sits where it does in the real
cell -- the earlier hand-drawn arcs and rectangle were guesses, and wrong: the
arcs are a stock FANUC LR Mate mat ("200iD", "200iD/4S") and are NOT centred on
the ER-4iA.

Committed as an asset because recipients do not have the scan. Re-run only if
the scan or the registration below changes.

How the scan is registered to the robot (all measured on the orthophoto this
script renders, 1 px = 1 mm):
  - J1 axis: centre of the base-plate bolt pattern, pixel (842, 780), +/- ~10 mm
  - yaw: the table edges and base-plate edges agree on a 1.37 deg rotation of
    the scan relative to its pixel grid (slope 0.024)
  - robot +X is image -u (the work area), +Y is image +v
Cross-checked: a homography from the real-cell video onto this orthophoto fits
8 printed-feature correspondences to ~2 mm RMS.
"""
import argparse, io, os, sys, zipfile

ap = argparse.ArgumentParser()
ap.add_argument("--scan", default=os.path.expanduser("~/Downloads/CSI-site-digital-model.zip"),
                help="the site-scan zip, or the 204107 .glb inside it")
ap.add_argument("--out", default=None)
ap.add_argument("--ortho", default=None, help="also save the raw orthophoto here")
args = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(args.out or os.path.join(HERE, "assets", "mat_print.png"))

import numpy as np
import cv2
import trimesh

sys.path.insert(0, HERE)
import scene_config as C

GLB_NAME = "Scaniverse 2026-09-22 204107.glb"      # the full-table scan
J1_PX = np.array([842.0, 780.0])
SLOPE = 0.024                                       # tan(1.37 deg)
RES = 0.001                                         # orthophoto m/px
TEX_PX_PER_M = 2000                                 # output texture resolution


def load_scan(path):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(GLB_NAME))
            return trimesh.load(io.BytesIO(z.read(name)), file_type="glb")
    return trimesh.load(path)


def orthophoto(scene):
    """Top-down orthographic render of the scan's texture, 1 px = 1 mm.

    Point-splat rather than rasterise: sample ~3 points per mm^2 on every face,
    colour them from the texture, keep the top-most per pixel.
    """
    m = scene.dump(concatenate=True)
    V, F = np.asarray(m.vertices), np.asarray(m.faces)
    uv = np.asarray(m.visual.uv)
    tex = np.asarray(m.visual.material.baseColorTexture.convert("RGB"))
    th, tw = tex.shape[:2]
    # tabletop normal = area-weighted mean of the upward faces (glTF is Y-up)
    n, a = m.face_normals, m.area_faces
    upf = (n @ np.array([0, 1.0, 0])) > 0.9
    z = (n[upf] * a[upf, None]).sum(0); z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    P = V @ np.stack([x, y, z]).T
    rng = np.random.default_rng(0)
    nper = np.clip((a / 1e-6 * 3).astype(int), 1, 4000)
    idx = np.repeat(np.arange(len(F)), nper)
    r1 = np.sqrt(rng.random(len(idx))); r2 = rng.random(len(idx))
    b = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], 1)
    pts = (P[F][idx] * b[:, :, None]).sum(1)
    uvs = (uv[F][idx] * b[:, :, None]).sum(1)
    col = tex[np.clip(((1 - uvs[:, 1]) * (th - 1)).astype(int), 0, th - 1),
              np.clip((uvs[:, 0] * (tw - 1)).astype(int), 0, tw - 1)]
    mn = P[:, :2].min(0)
    gx = ((pts[:, 0] - mn[0]) / RES).astype(int)
    gy = ((pts[:, 1] - mn[1]) / RES).astype(int)
    img = np.zeros((gy.max() + 1, gx.max() + 1, 3), np.uint8)
    order = np.argsort(pts[:, 2])
    img[gy[order], gx[order]] = col[order]
    img = cv2.cvtColor(img[::-1], cv2.COLOR_RGB2BGR)     # +v = robot +Y
    # fill the splatting speckle, but not the large unscanned regions
    hole = (img.sum(2) == 0).astype(np.uint8)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(hole)
    small = (np.isin(lab, np.nonzero(st[:, 4] < 30)[0]) & (lab > 0)).astype(np.uint8)
    img = cv2.inpaint(img, small, 3, cv2.INPAINT_TELEA)
    valid = cv2.erode((img.sum(2) > 0).astype(np.uint8), np.ones((9, 9), np.uint8))
    return img, valid.astype(bool)


def robot_to_px(X, Y):
    """Robot-frame metres -> orthophoto pixels (inverse of the registration)."""
    M = np.array([[-1.0, SLOPE], [SLOPE, 1.0]]) / 1000.0
    Mi = np.linalg.inv(M)
    du = Mi[0, 0] * X + Mi[0, 1] * Y
    dv = Mi[1, 0] * X + Mi[1, 1] * Y
    return du + J1_PX[0], dv + J1_PX[1]


def main():
    img, valid = orthophoto(load_scan(os.path.expanduser(args.scan)))
    if args.ortho:
        cv2.imwrite(os.path.abspath(args.ortho), img)
    g = img.astype(np.float32).mean(2)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat, hue = hsv[..., 1].astype(np.float32), hsv[..., 0]
    grey = sat < 90
    # Printed lines by hysteresis: a single threshold either breaks the arcs
    # into dashes or lets in the scan's glare mottling. Strong line pixels seed
    # components of the weak mask; components shorter than 40 mm are dropped.
    d = cv2.GaussianBlur(g, (0, 0), 10) - g
    strong = ((d > 28) & grey).astype(np.uint8)
    weak = cv2.morphologyEx(((d > 15) & grey).astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((3, 3), np.uint8))
    nl, lab, st, _ = cv2.connectedComponentsWithStats(weak)
    keep = np.zeros(nl, bool)
    keep[np.unique(lab[strong > 0])] = True
    keep &= np.maximum(st[:, 2], st[:, 3]) >= 40
    keep[0] = False
    dark_line = keep[lab]
    solid = g < 95
    red = ((hue <= 10) | (hue >= 165)) & (sat > 90) & (hsv[..., 2] > 60)

    # sample the classification on the robot-frame texture grid
    (x0, x1), (y0, y1) = C.TABLE_X, C.TABLE_Y
    H = int(round((x1 - x0) * TEX_PX_PER_M)); W = int(round((y1 - y0) * TEX_PX_PER_M))
    Xg = x1 - (np.arange(H) + 0.5) / TEX_PX_PER_M          # row 0 = front edge
    Yg = y0 + (np.arange(W) + 0.5) / TEX_PX_PER_M          # col 0 = -Y edge
    X, Y = np.meshgrid(Xg, Yg, indexing="ij")
    mu, mv = robot_to_px(X, Y)
    mu, mv = mu.astype(np.float32), mv.astype(np.float32)

    def warp(mask):
        return cv2.remap(mask.astype(np.float32), mu, mv, cv2.INTER_LINEAR,
                         borderValue=0.0)

    # Printed lines are ~1.5 mm; the detection comes out ~3 mm and ragged.
    # Blur the resampled line mask and cut it high to thin and smooth it;
    # solid areas (tape, pad, thick grid) keep a plain threshold.
    line_p = cv2.GaussianBlur(warp(dark_line & grey & valid), (0, 0), 1.2)
    ink = (line_p > 0.62) | (warp(solid & grey & valid) > 0.5)
    txt = warp(red & valid) > 0.5
    # under the robot base and its cable bundle the scan shows the robot, not
    # the mat -- the simulated base covers this patch anyway
    ink &= ~((X < 0.10) & (Y > -0.16) & (Y < 0.27))

    # yellow CAUTION bands: on the front and both side edges, not along the
    # back edge against the enclosure (site scan + cell video agree)
    bw = C.EDGE_BAND_W
    band = (X > x1 - bw) | (Y < y0 + bw) | (Y > y1 - bw)
    out = np.empty((H, W, 3), np.uint8)
    out[:] = (236, 237, 238)                           # BGR, laminated white
    out[ink & ~band] = (38, 36, 36)
    out[band] = (40, 196, 226)
    out[band & txt] = (40, 40, 170)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, out)
    print(f"[mat] {W}x{H} px ({(y1-y0)*1000:.0f} x {(x1-x0)*1000:.0f} mm) -> {OUT}")
    print(f"[mat] ink covers {ink.mean()*100:.1f}% of the table")


main()
