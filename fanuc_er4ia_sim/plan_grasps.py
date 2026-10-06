"""
Stage 3: turn Cosmos3-Nano's 2D detections into a 3D pick-and-place plan.

    .venv/bin/python plan_grasps.py --run run/

Steps:
  1. back-project each detection onto the table plane (exact -- we own the camera)
  2. optionally refine the centre with a colour/intensity blob inside the
     detection's own box ("VLM proposes, classical vision refines")
  3. score everything against ground truth so the accuracy is visible, not assumed
  4. emit an ordered pick-and-place plan into the bin

Writes run/plan.json and run/detections_overlay.png.
"""
import argparse, json, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backproject import CameraModel
import scene_config as C

# BGR reference colours for the cube materials, for the refinement step
_CUBE_REF = {"red": (45, 35, 165), "yellow": (40, 160, 200), "blue": (160, 45, 30)}


# Expected blob size in pixels for each class, used to reject a refinement that
# has latched onto the wrong thing. Derived from the object's real size and the
# camera scale (~1.3 mm/px at the table for the world camera).
# max_shift caps how far refinement may move the centre. Refinement is meant to
# be a local snap onto the object's own pixels; without a cap it becomes a search
# and will happily jump onto a larger neighbour that shares the colour -- which
# is exactly how the black plastic parts stole the screw detections.
_EXPECT = {
    "cube":  dict(area=(60, 1200),  aspect=(1.0, 2.2), max_shift=30.0),
    # M6 x 30 countersunk screw: ~30 px long, ~12 px across the head at ~1 mm/px
    "screw": dict(area=(25, 700),   aspect=(1.4, 9.0), max_shift=24.0),
}


def _blob_axis(mask_xy):
    """Given a screw blob's pixel coordinates, return its centroid and long axis.

    The grasp is across the MIDDLE of the shank. The M6 screws are countersunk:
    their head is a 90 deg cone, and parallel jaws closing on a cone push the
    part out along its axis rather than gripping it. The shank is a plain 6 mm
    cylinder, and the silhouette centroid of a 30 mm countersunk screw lies on it.
    """
    pts = mask_xy.astype(np.float32)
    c = pts.mean(0)
    u, sdv, vt = np.linalg.svd(pts - c, full_matrices=False)
    return c, vt[0]


def detect_coord_convention(det, W, H):
    """Work out which coordinate space the model used, from the whole reply.

    Cosmos3-Nano is not consistent between runs on the same prompt: one run
    returned [0.43, 0.71], the next [439, 714]. The latter is the Qwen-VL
    0..1000 grid its understanding tower inherits, NOT absolute pixels -- reading
    it as pixels puts every object out by ~100 mm.

    Decided once over every number in the reply rather than per detection, since
    a single object's coordinates are often ambiguous on their own.
    """
    vals = []
    for o in det.get("objects", []):
        vals += [float(x) for x in o.get("center", [])]
        vals += [float(x) for x in (o.get("bbox") or [])]
    b = det.get("bin") or {}
    vals += [float(x) for x in b.get("center", [])]
    vals += [float(x) for x in (b.get("bbox") or [])]
    if not vals:
        return "unit", 1.0
    hi = max(vals)
    if hi <= 1.5:
        return "unit", 1.0            # 0..1
    if hi <= 1000.0:
        return "grid1000", 1000.0     # Qwen-VL 0..1000
    return "pixel", None              # absolute pixels


def make_as_pixels(convention, divisor, W, H):
    def as_pixels(vals):
        v = [float(x) for x in vals]
        if convention == "pixel":
            return v
        scale = [W, H] * (len(v) // 2 + 1)
        return [v[i] / divisor * scale[i] for i in range(len(v))]
    return as_pixels


def refine_centre(img, cam, det, as_pixels, pad=1.6):
    """Snap a detection centre to the object's own pixels.

    The VLM localises to roughly a cube-width; good enough to say what is where,
    not good enough to close a 20 mm gripper on it. Searching a window around the
    reported box for the matching blob recovers the rest.
    Returns (u, v, method) in pixels. method is "refined", "raw" (no blob found)
    or "rejected" (a blob was found but its size/shape does not match the class,
    which usually means the detection itself is wrong).
    """
    H, W = img.shape[:2]
    cu, cv_ = as_pixels(det["center"])
    bb = det.get("bbox")
    if bb:
        bb = as_pixels(bb)
        bw = max(abs(bb[2] - bb[0]), 12.0)
        bh = max(abs(bb[3] - bb[1]), 12.0)
    else:
        bw = bh = 26.0
    half_w, half_h = bw * pad / 2, bh * pad / 2
    x0, x1 = int(max(0, cu - half_w)), int(min(W, cu + half_w))
    y0, y1 = int(max(0, cv_ - half_h)), int(min(H, cv_ + half_h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return cu, cv_, "raw", None
    patch = img[y0:y1, x0:x1]

    cat = det.get("category", "")
    colour = (det.get("colour") or "").lower()
    if cat == "cube" and colour in _CUBE_REF:
        ref = np.array(_CUBE_REF[colour], dtype=np.float32)
        dist = np.linalg.norm(patch.astype(np.float32) - ref, axis=2)
        mask = (dist < 90).astype(np.uint8)
    elif cat == "screw":
        grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        mask = (grey < 90).astype(np.uint8)
    else:
        return cu, cv_, "raw", None

    # Screws: a 5 px round opening strips the mat's printed lines (~2 px wide
    # at ~1 mm/px), which are as dark as black oxide and otherwise merge with
    # any screw lying across them. The 6 mm shank survives it.
    kern = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) if cat == "screw"
            else np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return cu, cv_, "raw", None

    exp = _EXPECT.get(cat)
    best, best_score, saw_blob, best_i = None, -1e9, False, None
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        # elongation from the principal axes, NOT the bounding box: a screw
        # lying at 45 deg has a square bounding box and was being rejected
        ys_i, xs_i = np.nonzero(lab == i)
        if len(xs_i) >= 3:
            ev = np.linalg.eigvalsh(np.cov(np.stack([xs_i, ys_i]).astype(np.float64)))
            aspect = float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-3)))
        else:
            aspect = 1.0
        saw_blob = True
        if exp:
            if not (exp["area"][0] <= area <= exp["area"][1]):
                continue
            if not (exp["aspect"][0] <= aspect <= exp["aspect"][1]):
                continue
        px, py = cent[i]
        d = np.hypot(x0 + px - cu, y0 + py - cv_)
        # prefer close over merely large: a distant big blob is usually a
        # different object that happens to share the colour
        score = -d + 0.05 * area
        if score > best_score:
            best_score, best, best_i = score, (x0 + px, y0 + py), i
    if best is None:
        # blobs existed but none matched the class -> the detection is suspect
        return cu, cv_, ("rejected" if saw_blob else "raw"), None
    shift = np.hypot(best[0] - cu, best[1] - cv_)
    if exp and shift > exp["max_shift"]:
        return cu, cv_, "rejected", None
    if cat == "screw" and best_i is not None:
        ys, xs = np.nonzero(lab == best_i)
        if len(xs) >= 12:
            hc, axis = _blob_axis(np.stack([xs, ys], 1))
            return (x0 + float(hc[0]), y0 + float(hc[1]), "refined",
                    (float(axis[0]), float(axis[1])))
    return best[0], best[1], "refined", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run")
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--categories", default="all",
                    help="comma-separated categories to pick, or 'all'")
    a = ap.parse_args()
    run = a.run

    cam = CameraModel(os.path.join(run, "camera.json"))
    det = json.load(open(os.path.join(run, "detections.json")))
    gt = json.load(open(os.path.join(run, "ground_truth.json")))
    img = cv2.imread(os.path.join(run, "world_camera.png"))

    want = None if a.categories == "all" else set(a.categories.split(","))
    conv, div = detect_coord_convention(det, cam.W, cam.H)
    as_pixels = make_as_pixels(conv, div, cam.W, cam.H)
    print(f"coordinate convention reported by the model: {conv}"
          + (f" (dividing by {div:.0f})" if div else ""))

    # ---------------------------------------------------------------- locate
    objects = []
    for d in det.get("objects", []):
        cat = d.get("category", "other")
        if not d.get("graspable", False):
            continue
        # h is the object's CENTRE height above the work surface, used only to
        # back-project its image position onto the right plane. The arm's grasp
        # height is a fingertip clearance and is applied by the executor.
        if cat == "cube":
            h, jaw = C.CUBE_EDGE / 2, C.CUBE_EDGE / 2
        else:
            # a countersunk screw lying down rests on its head rim and shank,
            # so its axis sits between the two radii at mid-length
            h = (C.SCREW_HEAD_D + C.SCREW_SHANK_D) / 4
            jaw = C.SCREW_SHANK_D / 2
        u_raw, v_raw = as_pixels(d["center"])
        if a.no_refine:
            u, v, how, axis_px = u_raw, v_raw, "raw", None
        else:
            u, v, how, axis_px = refine_centre(img, cam, d, as_pixels)
        xy = cam.to_table_xy(u / cam.W, v / cam.H, height_above_table=h)
        xy_raw = cam.to_table_xy(u_raw / cam.W, v_raw / cam.H,
                                 height_above_table=h)
        spread = None
        if cat == "screw" and axis_px is not None and xy is not None:
            # back-project a short step along the blob's long axis to get the
            # screw's direction on the table, then close the jaws across it
            step = 8.0
            p2 = cam.to_table_xy((u + step * axis_px[0]) / cam.W,
                                 (v + step * axis_px[1]) / cam.H,
                                 height_above_table=h)
            if p2 is not None:
                d_world = np.array(p2) - np.array(xy)
                if np.linalg.norm(d_world) > 1e-6:
                    d_world /= np.linalg.norm(d_world)
                    spread = [float(-d_world[1]), float(d_world[0]), 0.0]

        objects.append(dict(valid=(how == "refined"), spread_dir=spread,
                            label=d.get("label", cat), category=cat,
                            colour=d.get("colour"), pixel=[float(u), float(v)],
                            pixel_raw=[float(u_raw), float(v_raw)],
                            xy=[float(xy[0]), float(xy[1])],
                            xy_raw=[float(xy_raw[0]), float(xy_raw[1])],
                            grasp_z=h, jaw_half=jaw, method=how))

    # ---------------------------------------------------------------- score
    print(f"{'detection':16s} {'method':8s} {'raw err':>9s} {'refined err':>12s}  nearest GT")
    rows = []
    for o in objects:
        cands = [g for g in gt["objects"] if g["category"] == o["category"]]
        if not cands:
            continue
        d_ref = [np.hypot(g["xyz"][0] - o["xy"][0], g["xyz"][1] - o["xy"][1]) for g in cands]
        j = int(np.argmin(d_ref))
        e_ref = d_ref[j] * 1000
        e_raw = np.hypot(cands[j]["xyz"][0] - o["xy_raw"][0],
                         cands[j]["xyz"][1] - o["xy_raw"][1]) * 1000
        o["gt_match"], o["err_mm"], o["err_raw_mm"] = cands[j]["name"], e_ref, e_raw
        rows.append((e_raw, e_ref))
        print(f"{o['label']:16s} {o['method']:8s} {e_raw:8.1f}mm {e_ref:11.1f}mm  {cands[j]['name']}")
    if rows:
        r = np.array(rows)
        print(f"\nmean localisation error: raw {r[:,0].mean():.1f} mm -> "
              f"refined {r[:,1].mean():.1f} mm   (cube is {C.CUBE_EDGE*1000:.0f} mm)")

    # detection recall
    for cat in ("cube", "screw"):
        n_gt = sum(1 for g in gt["objects"] if g["category"] == cat)
        n_d = sum(1 for o in objects if o["category"] == cat)
        print(f"recall {cat:6s}: {n_d}/{n_gt} detected")

    # ---------------------------------------------------------------- plan
    # Never try to pick up the bin. Cosmos3-Nano has reported the bin as a
    # "cube", which is a fair description of a cube-shaped block. The blob check
    # happened to filter it, but relying on that is luck; anything sitting in
    # the bin's footprint is excluded explicitly.
    bx, by = C.BIN_XY
    guard = max(C.BIN_OUTER) / 2 + 0.015
    for o in objects:
        if abs(o["xy"][0] - bx) < guard and abs(o["xy"][1] - by) < guard:
            o["valid"] = False
            o["method"] = "in-bin"

    rejected = [o for o in objects if not o["valid"]]
    if rejected:
        print(f"\nrejected {len(rejected)} detection(s) whose blob did not match "
              f"the expected size/shape for their class:")
        for o in rejected:
            print(f"   {o['label']:16s} ({o['method']}) at pixel "
                  f"({o['pixel_raw'][0]:.0f}, {o['pixel_raw'][1]:.0f})")

    picks = [o for o in objects if o["valid"]
             and (want is None or o["category"] in want)]
    # nearest-first from the robot base keeps the arm from crossing over objects
    picks.sort(key=lambda o: np.hypot(*o["xy"]))
    bin_xy = list(C.BIN_XY)
    plan = {"bin_xy": bin_xy, "bin_release_z": C.BIN_RELEASE_Z,
            "picks": [dict(label=o["label"], category=o["category"],
                           xy=o["xy"], grasp_z=o["grasp_z"],
                           jaw_half=o["jaw_half"],
                           spread_dir=o.get("spread_dir"),
                           gt_match=o.get("gt_match"), err_mm=o.get("err_mm"))
                      for o in picks]}
    with open(os.path.join(run, "plan.json"), "w") as f:
        json.dump(plan, f, indent=2)
    print(f"\nplan: {len(picks)} picks -> bin at {bin_xy}")
    for p in plan["picks"]:
        print(f"   {p['label']:16s} xy=({p['xy'][0]:.3f}, {p['xy'][1]:+.3f})")

    # ---------------------------------------------------------------- overlay
    ov = img.copy()
    for o in objects:
        pr = tuple(int(v) for v in o["pixel_raw"])
        pf = tuple(int(v) for v in o["pixel"])
        cv2.circle(ov, pr, 7, (0, 0, 255), 2)
        cv2.circle(ov, pf, 5, (0, 255, 0), -1)
        cv2.line(ov, pr, pf, (0, 200, 255), 1)
        cv2.putText(ov, o["label"], (pf[0] + 8, pf[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    for g in gt["objects"]:
        px = cam.project(np.array([g["xyz"][0], g["xyz"][1], g["xyz"][2] + cam.table_z]))
        if px is not None:
            cv2.drawMarker(ov, (int(px[0]), int(px[1])), (255, 255, 0),
                           cv2.MARKER_CROSS, 12, 1)
    cv2.putText(ov, "red=VLM raw   green=refined   cyan=ground truth",
                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(run, "detections_overlay.png"), ov)
    print(f"overlay: {run}/detections_overlay.png")


if __name__ == "__main__":
    main()
