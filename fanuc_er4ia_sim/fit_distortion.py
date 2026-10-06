import numpy as np, json
from scipy.optimize import least_squares

lines = {k: np.array(v) for k, v in json.load(open('plumb_lines.json')).items()}
W, H = 858, 644
S = np.hypot(W, H) / 2.0

# Vetting. 'diag_right_long' is a CABLE lying across the table (confirmed by
# zooming: it crosses the printed arc and runs off the table edge), so it is not
# straight in the world. The rounded table edge and the printed reach-envelope
# arcs are genuinely curved and were never included.
VETTED = ["sq_top", "sq_bottom", "sq_left", "sq_right",
          "tray_upper_left", "tray_upper_right", "tray_lower_left",
          "bkit_left", "bkit_right", "left_vertical",
          "tl_grid_top", "tr_grid_top"]

def rectify(p, k1, cx=W/2, cy=H/2):
    q = (p - [cx, cy]) / S
    r2 = (q**2).sum(1, keepdims=True)
    return q * (1 + k1*r2) * S + [cx, cy]

def line_resid(q):
    c = q.mean(0); _, _, vt = np.linalg.svd(q - c)
    return (q - c) @ vt[1]

def cost(k1, subset):
    return np.concatenate([line_resid(rectify(lines[k], k1[0])) for k in subset])

def rms(k1, subset):
    r = cost([k1], subset); return float(np.sqrt((r**2).mean()))

# ---- k1-only fit, principal point fixed at image centre (data is too weak to
# ---- float it: see the bootstrap spread below)
fit = least_squares(cost, [0.0], args=(VETTED,), loss='soft_l1', f_scale=0.8)
k1 = fit.x[0]
print(f"k1 = {k1:+.4f}")
print(f"rms  no correction : {rms(0.0, VETTED):.3f} px")
print(f"rms  fitted k1     : {rms(k1,  VETTED):.3f} px")

# ---- bootstrap over LINES (not points): how stable is k1?
rng = np.random.default_rng(0)
boot = []
for _ in range(400):
    samp = list(rng.choice(VETTED, len(VETTED), replace=True))
    try:
        boot.append(least_squares(cost, [0.0], args=(samp,),
                                  loss='soft_l1', f_scale=0.8).x[0])
    except Exception:
        pass
boot = np.array(boot)
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"\nbootstrap k1: mean={boot.mean():+.4f}  sd={boot.std():.4f}"
      f"  95% CI=[{lo:+.4f}, {hi:+.4f}]")
print(f"CI contains zero: {lo <= 0 <= hi}")

# ---- leave-one-line-out: does k1 generalise to a line it never saw?
imp = []
for k in VETTED:
    tr = [x for x in VETTED if x != k]
    kk = least_squares(cost, [0.0], args=(tr,), loss='soft_l1', f_scale=0.8).x[0]
    r0 = np.sqrt((line_resid(rectify(lines[k], 0.0))**2).mean())
    r1 = np.sqrt((line_resid(rectify(lines[k], kk))**2).mean())
    imp.append(r0 - r1)
    print(f"  holdout {k:18s} k1={kk:+.4f}  rms {r0:5.2f} -> {r1:5.2f} px  ({r0-r1:+.2f})")
imp = np.array(imp)
print(f"\nmean held-out improvement: {imp.mean():+.3f} px "
      f"({(imp > 0).sum()}/{len(imp)} lines improved)")

# ---- peak displacement implied at the image corner
print(f"\nimplied corner displacement: {abs(k1)*S:.1f} px at r=S ({S:.0f} px)")
json.dump({"k1": k1, "S": S, "ci95": [lo, hi],
           "rms_raw": rms(0.0, VETTED), "rms_fit": rms(k1, VETTED),
           "held_out_gain_px": float(imp.mean())}, open('k1.json', 'w'), indent=2)
