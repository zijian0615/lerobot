import cv2, numpy as np, json

img = cv2.imread('calib_plate.png')
H, W = img.shape[:2]
edges = cv2.imread('edges.png', 0)
ys, xs = np.nonzero(edges)
pts = np.stack([xs, ys], 1).astype(float)

# Features that are STRAIGHT in the real world, given as approximate endpoints.
# Curved features (the reach-envelope arcs, the ellipse, the rounded table edge)
# are deliberately excluded -- they would poison a plumb-line fit.
CANDIDATES = {
    "diag_right_long":   ((659,   2), (747, 638)),   # long printed line, right periphery
    "sq_top":            ((357, 378), (562, 373)),   # centre square, top edge
    "sq_bottom":         ((358, 533), (566, 529)),   # centre square, bottom edge
    "sq_left":           ((356, 379), (359, 532)),   # centre square, left edge
    "sq_right":          ((563, 374), (567, 530)),   # centre square, right edge
    "tray_upper_left":   ((112, 302), (247, 189)),   # dark tray, top face edge
    "tray_upper_right":  ((248, 189), (386, 299)),
    "tray_lower_left":   ((113, 305), (243, 466)),
    "bkit_left":         ((604, 424), (658, 231)),   # battery-kit drawing outline
    "bkit_right":        ((742, 398), (782, 225)),
    "left_vertical":     ((186, 462), (180, 640)),
    "tl_grid_top":       ((146,  92), (240,  97)),   # small printed grid, top left
    "tr_grid_top":       ((679,  72), (762,  84)),   # small printed grid, top right
}

CORRIDOR = 4.0   # px; wide enough to catch the bow, tight enough to reject neighbours

out = {}
vis = img.copy()
for name, (p0, p1) in CANDIDATES.items():
    a = np.array(p0, float); b = np.array(p1, float)
    d = b - a; L = np.linalg.norm(d); u = d / L
    v = pts - a
    t = v @ u                      # along-segment coordinate
    n = np.abs(v[:, 0]*u[1] - v[:, 1]*u[0])   # perpendicular distance
    m = (t >= 0) & (t <= L) & (n <= CORRIDOR)
    sel = pts[m]
    # thin: keep one point per unit of arc length (median across the corridor)
    tt = t[m]
    bins = np.round(tt).astype(int)
    keep = []
    for bv in np.unique(bins):
        q = sel[bins == bv]
        keep.append(q.mean(0))
    keep = np.array(keep)
    out[name] = keep.tolist()
    col = tuple(int(c) for c in np.random.randint(0, 255, 3))
    for p in keep:
        cv2.circle(vis, (int(p[0]), int(p[1])), 1, col, -1)
    # straightness of the RAW points, as a sanity check
    c = keep.mean(0); uu, ss, vv = np.linalg.svd(keep - c)
    resid = np.sqrt((ss[1]**2) / len(keep))
    print(f"{name:18s} n={len(keep):4d} len={L:6.1f}px  raw straightness rms={resid:5.2f}px")

json.dump(out, open('plumb_lines.json', 'w'))
cv2.imwrite('plumb_vis.png', vis)
print('\ntotal lines:', len(out))
