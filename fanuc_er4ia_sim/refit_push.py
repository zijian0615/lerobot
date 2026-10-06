import cv2, numpy as np, json
from scipy.optimize import least_squares

img = cv2.imread('calib_plate_push.png')
H, W = img.shape[:2]
g = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 7, 60, 60)
edges = cv2.Canny(g, 30, 90, L2gradient=True)
ys, xs = np.nonzero(edges)
pts = np.stack([xs, ys], 1).astype(float)
print(f"image {W}x{H}, edge px {len(pts)}")

cands = json.load(open('candidates_push.json'))
# the long diagonal is a CABLE (verified on the grasp clip) -- never a plumb line
VETO = {"diag_right_long"}
CORRIDOR = 6.0

lines = {}
for k, (p0, p1) in cands.items():
    if k in VETO: continue
    a, b = np.array(p0), np.array(p1)
    d = b - a; L = np.linalg.norm(d); u = d / L
    v = pts - a
    t = v @ u
    n = np.abs(v[:, 0]*u[1] - v[:, 1]*u[0])
    m = (t >= 0) & (t <= L) & (n <= CORRIDOR)
    sel, tt = pts[m], t[m]
    if m.sum() < 30: 
        print(f"  {k}: too few points ({m.sum()}), skipped"); continue
    bins = np.round(tt).astype(int)
    keep = np.array([sel[bins == bv].mean(0) for bv in np.unique(bins)])
    lines[k] = keep

S = np.hypot(W, H) / 2.0

def rectify(p, prm):
    k1, cx, cy = prm
    q = (p - [cx, cy]) / S
    r2 = (q**2).sum(1, keepdims=True)
    return q * (1 + k1*r2) * S + [cx, cy]

def lr(q):
    c = q.mean(0); _, _, vt = np.linalg.svd(q - c); return (q - c) @ vt[1]

def cost(prm, sub): return np.concatenate([lr(rectify(lines[k], prm)) for k in sub])
def rms(prm, sub):
    r = cost(prm, sub); return float(np.sqrt((r**2).mean()))

sub = list(lines)
print(f"\nlines used: {len(sub)}")
for k in sub:
    print(f"   {k:18s} n={len(lines[k]):4d} raw rms={np.sqrt((lr(lines[k])**2).mean()):5.2f}px")

x0 = np.array([0.0, W/2, H/2])
print(f"\nrms no correction      : {rms(x0, sub):.3f} px")

# k1 only, principal point pinned at image centre
f1 = least_squares(lambda p: cost([p[0], W/2, H/2], sub), [0.0],
                   loss='soft_l1', f_scale=0.8)
p1 = np.array([f1.x[0], W/2, H/2])
print(f"rms k1 only (pp fixed) : {rms(p1, sub):.3f} px   k1={f1.x[0]:+.4f}")

# k1 + free principal point
f2 = least_squares(cost, x0, args=(sub,), loss='soft_l1', f_scale=0.8,
                   bounds=([-1, W*0.2, H*0.2], [1, W*0.8, H*0.8]))
print(f"rms k1 + free pp       : {rms(f2.x, sub):.3f} px   "
      f"k1={f2.x[0]:+.4f} pp=({f2.x[1]:.0f},{f2.x[2]:.0f})")

# bootstrap over lines on the k1-only model
rng = np.random.default_rng(0); boot = []
for _ in range(400):
    s2 = list(rng.choice(sub, len(sub), replace=True))
    boot.append(least_squares(lambda p: cost([p[0], W/2, H/2], s2), [0.0],
                              loss='soft_l1', f_scale=0.8).x[0])
boot = np.array(boot); lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"\nbootstrap k1: mean={boot.mean():+.4f} sd={boot.std():.4f} "
      f"95% CI=[{lo:+.4f}, {hi:+.4f}]  contains zero: {lo <= 0 <= hi}")

# leave-one-line-out generalisation
imp = []
for k in sub:
    tr = [x for x in sub if x != k]
    kk = least_squares(lambda p: cost([p[0], W/2, H/2], tr), [0.0],
                       loss='soft_l1', f_scale=0.8).x[0]
    r0 = np.sqrt((lr(lines[k])**2).mean())
    r1 = np.sqrt((lr(rectify(lines[k], [kk, W/2, H/2]))**2).mean())
    imp.append(r0 - r1)
imp = np.array(imp)
print(f"mean held-out improvement: {imp.mean():+.3f} px ({(imp>0).sum()}/{len(imp)} improved)")
