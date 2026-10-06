import cv2, numpy as np, json
a = cv2.imread('calib_plate.png', 0)          # 858x644  (grasp clip)
b = cv2.imread('calib_plate_push.png', 0)     # 1270x838 (push clip)
sift = cv2.SIFT_create(4000)
ka, da = sift.detectAndCompute(a, None)
kb, db = sift.detectAndCompute(b, None)
bf = cv2.BFMatcher()
good = [m for m, n in bf.knnMatch(da, db, k=2) if m.distance < 0.7 * n.distance]
print(f"SIFT matches: {len(good)}")
src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
print(f"inliers: {int(mask.sum())}/{len(good)}")
print("homography:\n", np.round(Hm, 4))

# what kind of transform is it? decompose the scale
s = np.sqrt(abs(np.linalg.det(Hm[:2, :2])))
print(f"\napprox scale grasp->push: {s:.3f}")
print(f"resolution ratio: {1270/858:.3f} (w), {838/644:.3f} (h)")

cands = json.load(open('candidates.json'))
out = {}
for k, (p0, p1) in cands.items():
    q = cv2.perspectiveTransform(np.float32([[p0], [p1]]).reshape(-1, 1, 2), Hm)
    out[k] = [q[0, 0].tolist(), q[1, 0].tolist()]
json.dump(out, open('candidates_push.json', 'w'), indent=1)
print(f"\ntransferred {len(out)} candidates")
for k, v in list(out.items())[:4]:
    print(f"   {k:18s} {np.round(v[0],1)} -> {np.round(v[1],1)}")
