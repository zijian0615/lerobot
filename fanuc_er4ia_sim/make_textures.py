"""
Generate the procedural texture set used by the metal materials.

    ~/fanuc_er4ia_sim/.venv/bin/python make_textures.py

Writes tileable maps into textures/. Machined aluminium is not a flat grey: it
carries fine directional tool marks that break up the specular highlight, and
that break-up is most of what makes it read as metal. A constant-roughness
metallic surface under a featureless dome renders as dead grey no matter what
the metallic value is, which is why the environment map below matters too.
"""
import os
import numpy as np
import cv2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textures")
os.makedirs(OUT, exist_ok=True)
N = 1024
rng = np.random.default_rng(7)


def _streaks(n, along=220, across=1.1, seed=0):
    """Tileable anisotropic noise: heavily blurred along X, barely across Y.

    OpenCV rejects BORDER_WRAP for separable filters, so tile 3x3 and crop the
    centre instead -- the blur then sees real neighbours on every edge and the
    result tiles cleanly.
    """
    r = np.random.default_rng(seed)
    f = r.normal(0, 1, (n, n)).astype(np.float32)
    f = np.tile(f, (3, 3))
    f = cv2.GaussianBlur(f, (0, 0), sigmaX=along, sigmaY=across)
    f = f[n:2 * n, n:2 * n]
    f -= f.min(); f /= max(f.max(), 1e-6)
    return f


def brushed_aluminium():
    coarse = _streaks(N, along=380, across=0.8, seed=1)
    fine   = _streaks(N, along=120, across=0.45, seed=2)
    h = 0.62 * coarse + 0.38 * fine
    h = (h - h.mean()) / (h.std() + 1e-6)

    # albedo: bright aluminium with faint tonal drift
    alb = np.clip(0.82 + 0.020 * h, 0.0, 1.0)
    alb_img = np.dstack([alb * 0.985, alb * 0.995, alb])      # a touch cool
    cv2.imwrite(f"{OUT}/alu_albedo.png",
                (np.clip(alb_img, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])

    # roughness: the tool marks are the variation that shapes the highlight
    rough = np.clip(0.26 + 0.085 * h, 0.05, 0.85)
    cv2.imwrite(f"{OUT}/alu_roughness.png",
                (rough * 255).astype(np.uint8))

    # normal map from the same height field
    hh = (h - h.min()) / (h.max() - h.min() + 1e-6)
    gx = cv2.Sobel(hh, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(hh, cv2.CV_32F, 0, 1, ksize=3)
    s = 2.5
    nz = np.ones_like(gx)
    nrm = np.dstack([-gx * s, -gy * s, nz])
    nrm /= (np.linalg.norm(nrm, axis=2, keepdims=True) + 1e-6)
    nrm = ((nrm * 0.5 + 0.5) * 255).astype(np.uint8)
    cv2.imwrite(f"{OUT}/alu_normal.png", nrm[:, :, ::-1])
    print("wrote alu_albedo / alu_roughness / alu_normal")


def anodised():
    """Finer, more uniform finish for the anodised gripper body."""
    h = _streaks(N, along=150, across=0.9, seed=5)
    h = (h - h.mean()) / (h.std() + 1e-6)
    alb = np.clip(0.66 + 0.030 * h, 0, 1)
    cv2.imwrite(f"{OUT}/anod_albedo.png",
                (np.dstack([alb * 1.0, alb * 0.995, alb * 0.985]) * 255)
                .astype(np.uint8)[:, :, ::-1])
    rough = np.clip(0.46 + 0.09 * h, 0.1, 0.9)
    cv2.imwrite(f"{OUT}/anod_roughness.png", (rough * 255).astype(np.uint8))
    print("wrote anod_albedo / anod_roughness")


def studio_environment(w=2048, h=1024):
    """Equirectangular studio env so the metal has something to reflect."""
    yy = np.linspace(0, 1, h)[:, None]
    xx = np.linspace(0, 1, w)[None, :]
    # vertical gradient: bright ceiling -> mid wall -> darker floor
    # keep a decent floor value: the dome is the scene's ambient as well as the
    # metal's reflection source, and a dark lower hemisphere blacks out the room
    base = np.clip(1.30 - 0.85 * yy, 0.42, 1.30) * np.ones_like(xx)
    # two soft overhead panels, the classic studio look metals key off
    for cx, cw, amp in ((0.28, 0.10, 2.4), (0.72, 0.13, 1.6)):
        panel = np.exp(-((xx - cx) ** 2) / (2 * cw ** 2)) * \
                np.exp(-((yy - 0.16) ** 2) / (2 * 0.10 ** 2))
        base = base + amp * panel
    # faint horizon band
    base += 0.25 * np.exp(-((yy - 0.52) ** 2) / (2 * 0.02 ** 2))
    base = cv2.GaussianBlur(base.astype(np.float32), (0, 0), 6)
    img = np.dstack([base * 0.98, base * 0.99, base * 1.02])   # slightly cool
    img = np.clip(img / max(img.max(), 1e-6), 0, 1)
    img = np.clip(img * 1.9, 0, 1)           # overall lift
    cv2.imwrite(f"{OUT}/studio_env.png", (img * 255).astype(np.uint8)[:, :, ::-1])
    print("wrote studio_env")


if __name__ == "__main__":
    brushed_aluminium()
    anodised()
    studio_environment()
    print("textures in", OUT)
