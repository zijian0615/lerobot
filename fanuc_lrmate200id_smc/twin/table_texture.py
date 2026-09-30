"""Table texture for the twin from the overhead camera (numpy + Pillow; no pxr, no OpenCV).

    .venv-twin/bin/python twin/table_texture.py --frames '../examples/*/runs/*/perceive_*/capture_rgb.png' --since 20260921_022200

1. pixel -> table xy: a homography fitted to the calibration samples (uv, true_table_xy), for the plane z = 0.01 the samples were
   taken on. On the fanuc samples it is ~5 mm RMSE (6.9 mm leave-one-out) vs ~14 mm for the stored K + affine model.
2. many frames -> per-pixel median: the arm and the objects move between captures, the printed table does not, so the median is a
   "background plate" without the arm painted on. One frame works too, but the arm/objects then appear painted on the table.
3. the robot zone (base + shoulder, tall, smeared by the plane assumption) is cut out: the twin draws its own robot there.
4. the plate is rectified into the table frame and written as a PNG (+ .json with its bounds) for `omni_twin.py --table-texture`.
   Texture rows run along +x (top row = x_min), columns along +y (left column = y_min): the real camera's orientation.
Limits: the homography is only accurate inside the calibration points (extrapolation degrades), tall things are smeared by the
plane assumption, objects that stayed put across all frames stay in the plate, and pixels outside the image get the table colour.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "..", "..", "examples", "tabletop_perception", "calib", "table_xy_calib_samples_fanuc.json")
OUT = os.path.join(HERE, "..", "table_texture.png")
TABLE_BOUNDS = (-0.45, 0.45, -0.35, 0.35)      # (x_min, x_max, y_min, y_max) = calib table_polygon_xy
# Where the robot's base and shoulder stand (table frame). They are tall, so the plane homography smears them across the
# table; the twin draws its own 3D robot there, so this box is cut out of the texture. (x_min, x_max, y_min, y_max)
ROBOT_ZONE = (-0.45, 0.10, -0.16, 0.24)      # the +y side is wider: the hoses swing out to about y = 0.22


def _normalise(p):
    m = p.mean(0)
    s = np.sqrt(2) / np.mean(np.linalg.norm(p - m, axis=1))
    return np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1.0]])


def fit_homography(uv, xy):
    """3x3 H with xy ~ H @ [u, v, 1] (normalised DLT). Needs >= 4 points, not all collinear."""
    uv, xy = np.asarray(uv, float), np.asarray(xy, float)
    if len(uv) < 4:
        raise ValueError("a homography needs at least 4 point pairs")
    Tu, Tx = _normalise(uv), _normalise(xy)
    a = (Tu @ np.c_[uv, np.ones(len(uv))].T).T
    b = (Tx @ np.c_[xy, np.ones(len(xy))].T).T
    rows = []
    for (u, v, _), (x, y, _) in zip(a, b):
        rows += [[u, v, 1, 0, 0, 0, -x * u, -x * v, -x], [0, 0, 0, u, v, 1, -y * u, -y * v, -y]]
    H = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    H = np.linalg.inv(Tx) @ H @ Tu
    return H / H[2, 2]


def apply_homography(H, pts):
    p = np.asarray(H) @ np.c_[np.asarray(pts, float), np.ones(len(pts))].T
    return (p[:2] / p[2]).T


def load_samples(path):
    with open(path) as f:
        d = json.load(f)
    s = d["samples"]
    return np.array([x["uv"] for x in s]), np.array([x["true_table_xy"] for x in s]), tuple(s[0]["image_hw"])


def leave_one_out_mm(uv, xy):
    return np.array([np.linalg.norm(apply_homography(fit_homography(np.delete(uv, i, 0), np.delete(xy, i, 0)), uv[i:i + 1])[0] - xy[i])
                     for i in range(len(uv))]) * 1000


def median_frames(images):
    """Per-pixel median of equally sized uint8 images."""
    if len(images) == 1:
        return images[0]
    return np.median(np.stack(images), axis=0).astype(np.uint8)


def rectify(img, H, bounds=TABLE_BOUNDS, px_per_m=1000, fill=None, masks=()):
    """Resample `img` (H x W x 3) onto the table plane. Returns (texture rows x cols x 3 uint8, valid mask). Pixels that fall
    outside the image or inside a table-frame mask box (x_min, x_max, y_min, y_max) get `fill` (default: the median colour
    of the valid pixels, i.e. the table's own colour)."""
    x0, x1, y0, y1 = bounds
    rows, cols = int(round((x1 - x0) * px_per_m)), int(round((y1 - y0) * px_per_m))
    X, Y = np.meshgrid(x0 + (np.arange(rows) + 0.5) / px_per_m, y0 + (np.arange(cols) + 0.5) / px_per_m, indexing="ij")
    uv = apply_homography(np.linalg.inv(H), np.c_[X.ravel(), Y.ravel()])
    u, v = uv[:, 0], uv[:, 1]
    h, w = img.shape[:2]
    ok = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    for mx0, mx1, my0, my1 in masks:
        ok &= ~((X.ravel() >= mx0) & (X.ravel() <= mx1) & (Y.ravel() >= my0) & (Y.ravel() <= my1))
    uc, vc = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
    iu, iv = np.minimum(uc.astype(int), w - 2), np.minimum(vc.astype(int), h - 2)
    fu, fv = (uc - iu)[:, None], (vc - iv)[:, None]
    im = img.astype(np.float32)
    out = (im[iv, iu] * (1 - fu) * (1 - fv) + im[iv, iu + 1] * fu * (1 - fv)
           + im[iv + 1, iu] * (1 - fu) * fv + im[iv + 1, iu + 1] * fu * fv)
    out[~ok] = np.median(out[ok], axis=0) if fill is None and ok.any() else (fill or (128, 128, 128))
    return np.clip(out + 0.5, 0, 255).astype(np.uint8).reshape(rows, cols, 3), ok.reshape(rows, cols)


def select_frames(patterns, since=None, last=30):
    """Frame paths matching the globs, optionally only runs at/after the `YYYYMMDD_HHMMSS` stamp `since`, newest `last`."""
    def stamp(p):
        m = re.findall(r"\d{8}_\d{6}", p)
        return m[-1] if m else ""
    files = sorted({f for p in patterns for f in glob.glob(p, recursive=True)}, key=lambda f: (stamp(f), f))
    if since:
        files = [f for f in files if stamp(f) >= since]
    return files[-last:] if last else files


def write_texture(png_path, tex, bounds, px_per_m, meta=None):
    from PIL import Image
    Image.fromarray(tex).save(png_path)
    with open(os.path.splitext(png_path)[0] + ".json", "w") as f:
        json.dump({"bounds": list(bounds), "px_per_m": px_per_m, **(meta or {})}, f, indent=1)


def load_texture_meta(png_path):
    with open(os.path.splitext(png_path)[0] + ".json") as f:
        m = json.load(f)
    return tuple(m["bounds"]), m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", action="append", required=True, metavar="GLOB", help="overhead camera images (repeatable)")
    ap.add_argument("--since", help="only frames from runs at/after this YYYYMMDD_HHMMSS (use the time of the last camera move / recalibration)")
    ap.add_argument("--last", type=int, default=30, help="use at most the newest N frames (0 = all)")
    ap.add_argument("--samples", default=SAMPLES, help="calibration samples JSON (uv -> true_table_xy)")
    ap.add_argument("--bounds", type=float, nargs=4, default=TABLE_BOUNDS, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    ap.add_argument("--px-per-m", type=int, default=1000)
    ap.add_argument("--mask", type=float, nargs=4, action="append", metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help=f"table-frame box to cut out (repeatable; default: the robot zone {ROBOT_ZONE})")
    ap.add_argument("--no-robot-mask", action="store_true", help="paint the robot base/shoulder into the texture too")
    ap.add_argument("--fill", type=int, nargs=3, help="colour outside the image / masks (default: the table's median colour)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    from PIL import Image

    uv, xy, image_hw = load_samples(a.samples)
    H = fit_homography(uv, xy)
    fit = np.linalg.norm(apply_homography(H, uv) - xy, axis=1) * 1000
    loo = leave_one_out_mm(uv, xy)
    print(f"homography from {len(uv)} samples: fit RMSE {np.sqrt((fit ** 2).mean()):.1f} mm, leave-one-out RMSE "
          f"{np.sqrt((loo ** 2).mean()):.1f} mm (max {loo.max():.1f})")

    files = select_frames(a.frames, a.since, a.last)
    imgs = []
    for f in files:
        im = np.asarray(Image.open(f).convert("RGB"))
        if im.shape[:2] != image_hw:
            print(f"skip {f}: {im.shape[:2]} != calibration image {image_hw}")
            continue
        imgs.append(im)
    if not imgs:
        ap.error("no usable frames")
    if len(imgs) == 1:
        print("WARNING: one frame only; the arm and objects in it will be painted onto the table (use several frames)")
    print(f"median of {len(imgs)} frames ({files[0]} ... {files[-1]})")
    masks = a.mask if a.mask else [] if a.no_robot_mask else [ROBOT_ZONE]
    tex, ok = rectify(median_frames(imgs), H, a.bounds, a.px_per_m, tuple(a.fill) if a.fill else None, masks)
    write_texture(a.out, tex, a.bounds, a.px_per_m,
                  {"frames": len(imgs), "homography": H.tolist(), "loo_rmse_mm": float(np.sqrt((loo ** 2).mean())),
                   "covered": float(ok.mean())})
    print(f"wrote {a.out} {tex.shape[1]}x{tex.shape[0]} px, {100 * ok.mean():.0f}% of the table inside the camera image")


if __name__ == "__main__":
    main()
