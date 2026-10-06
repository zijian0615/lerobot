import numpy as np, json
lines = {k: np.array(v) for k, v in json.load(open('plumb_lines.json')).items()}
W, H = 858, 644
cx, cy = W/2, H/2

def fit_line(p):
    c = p.mean(0); _, _, vt = np.linalg.svd(p - c)
    d = vt[0]; n = vt[1]
    return np.array([n[0], n[1], -n @ c])       # homogeneous line

def inter(l1, l2):
    p = np.cross(l1, l2); return p[:2] / p[2]

L = {k: fit_line(lines[k]) for k in ("sq_top", "sq_bottom", "sq_left", "sq_right")}
# corners, ordered TL, TR, BR, BL
corners = np.array([inter(L["sq_top"],    L["sq_left"]),
                    inter(L["sq_top"],    L["sq_right"]),
                    inter(L["sq_bottom"], L["sq_right"]),
                    inter(L["sq_bottom"], L["sq_left"])])
print("centre-square corners (px):")
for n, c in zip("TL TR BR BL".split(), corners): print(f"   {n} {c[0]:7.2f} {c[1]:7.2f}")
side = [np.linalg.norm(corners[i] - corners[(i+1) % 4]) for i in range(4)]
print(f"side lengths px: {['%.1f' % s for s in side]}  (ratio max/min "
      f"{max(side)/min(side):.3f})")

# homography from the unit square to the image quad
def homography(dst):
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    A = []
    for (X, Y), (u, v) in zip(src, dst):
        A.append([X, Y, 1, 0, 0, 0, -u*X, -u*Y, -u])
        A.append([0, 0, 0, X, Y, 1, -v*X, -v*Y, -v])
    _, _, vt = np.linalg.svd(np.array(A))
    return vt[-1].reshape(3, 3)

Hm = homography(corners)
a1, a2 = Hm[0, 0], Hm[0, 1]
b1, b2 = Hm[1, 0], Hm[1, 1]
c1, c2 = Hm[2, 0], Hm[2, 1]

# orthogonality of the two in-plane axes gives f^2 directly
num = -((a1 - cx*c1)*(a2 - cx*c2) + (b1 - cy*c1)*(b2 - cy*c2))
f2 = num / (c1*c2)
print(f"\northogonality constraint -> f^2 = {f2:.1f}")
if f2 > 0:
    f = np.sqrt(f2)
    hfov = 2*np.degrees(np.arctan(W/(2*f)))
    vfov = 2*np.degrees(np.arctan(H/(2*f)))
    print(f"f = {f:.1f} px   HFOV = {hfov:.1f} deg   VFOV = {vfov:.1f} deg")
else:
    print("f^2 NEGATIVE -> square is too close to fronto-parallel; f unidentifiable")
    f = None

# how well-conditioned is this? perturb the corners by +-0.5 px and refit
rng = np.random.default_rng(1); est = []
for _ in range(500):
    d = corners + rng.normal(0, 0.5, corners.shape)
    Hp = homography(d)
    A1, A2, B1, B2, C1, C2 = Hp[0,0], Hp[0,1], Hp[1,0], Hp[1,1], Hp[2,0], Hp[2,1]
    n = -((A1-cx*C1)*(A2-cx*C2) + (B1-cy*C1)*(B2-cy*C2))
    v = n/(C1*C2)
    if v > 0: est.append(np.sqrt(v))
est = np.array(est)
print(f"\nsensitivity to +-0.5px corner noise: {len(est)}/500 gave a real f")
if len(est):
    print(f"   f: median={np.median(est):.0f}px  5-95%=[{np.percentile(est,5):.0f}, "
          f"{np.percentile(est,95):.0f}]px")
    fov = 2*np.degrees(np.arctan(W/(2*est)))
    print(f"   HFOV: median={np.median(fov):.1f}deg  5-95%=[{np.percentile(fov,5):.1f}, "
          f"{np.percentile(fov,95):.1f}]deg")
