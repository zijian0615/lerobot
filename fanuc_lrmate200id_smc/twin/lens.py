"""Real-lens view from a pinhole render (numpy + OpenCV; no pxr). Isaac Sim cameras have no lens distortion.

The video0 webcam has barrel distortion (overhead_camera.json "K" / "dist", OpenCV model). DistortedView renders a wider
pinhole image at the same focal length (so every pixel of the distorted frame has a source) and remaps it into the
distorted frame, so simulated overhead images line up with raw video0 frames of the same size.
"""

import numpy as np


def undistort_radial(distorted, dist, iterations=30):
    """Normalized undistorted coordinates for normalized distorted ones (..., 2); OpenCV k1, k2, k3, no tangential."""
    dist = np.zeros(5) if dist is None else np.concatenate([np.asarray(dist, dtype=np.float64).ravel(), np.zeros(5)])[:5]
    k1, k2, p1, p2, k3 = dist
    if p1 or p2:
        raise NotImplementedError("tangential distortion")
    rd = np.linalg.norm(distorted, axis=-1)
    # A fitted polynomial can turn over (k2 < 0): past its peak radius there is no inverse. Clamp to the peak there;
    # those pixels (the frame's far corners) are outside the calibrated region anyway.
    grid = np.linspace(0.0, 3.0, 30001)
    g2 = grid * grid
    slope = 1 + 3 * k1 * g2 + 5 * k2 * g2 * g2 + 7 * k3 * g2 * g2 * g2
    r_peak = grid[np.argmax(slope <= 0)] if (slope <= 0).any() else np.inf
    rd_peak = r_peak * (1 + k1 * r_peak**2 + k2 * r_peak**4 + k3 * r_peak**6) if np.isfinite(r_peak) else np.inf
    target = np.minimum(rd, rd_peak)
    r = target.copy()
    for _ in range(iterations):  # Newton on r (1 + k1 r^2 + k2 r^4 + k3 r^6) = rd, kept below the peak
        r2 = r * r
        f = r * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) - target
        df = 1 + 3 * k1 * r2 + 5 * k2 * r2 * r2 + 7 * k3 * r2 * r2 * r2
        r = np.clip(r - f / np.maximum(df, 1e-6), 0.0, r_peak)
    scale = np.where(rd > 0, r / np.maximum(rd, 1e-12), 1.0)
    return distorted * scale[..., None]


class DistortedView:
    def __init__(self, k, dist, image_wh, out_wh, margin=1.02):
        """`k`, `dist`: OpenCV intrinsics for frames of `image_wh`; `out_wh`: size of the distorted output (same aspect)."""
        k = np.asarray(k, dtype=np.float64)
        scale = out_wh[0] / image_wh[0]
        k_out = k.copy()
        k_out[0, 0] *= scale
        k_out[1, 1] *= scale
        k_out[0, 2] = (k[0, 2] + 0.5) * scale - 0.5   # pixel-index convention: centres at integers
        k_out[1, 2] = (k[1, 2] + 0.5) * scale - 0.5
        w, h = out_wh
        uu, vv = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        distorted = np.stack([(uu - k_out[0, 2]) / k_out[0, 0], (vv - k_out[1, 2]) / k_out[1, 1]], -1)
        norm = undistort_radial(distorted, dist)
        self.focal_px = float(k_out[0, 0])
        half_w = np.abs(norm[..., 0]).max() * margin * self.focal_px
        half_h = np.abs(norm[..., 1]).max() * margin * self.focal_px
        self.render_wh = (2 * int(np.ceil(half_w)), 2 * int(np.ceil(half_h)))
        # optical axis at the render centre (continuous W/2 = pixel index W/2 - 0.5)
        self.map_x = (self.focal_px * norm[..., 0] + self.render_wh[0] / 2 - 0.5).astype(np.float32)
        self.map_y = (self.focal_px * norm[..., 1] + self.render_wh[1] / 2 - 0.5).astype(np.float32)
        self.out_wh = tuple(out_wh)

    def usd_lens(self, focal_mm=20.0):
        """Pinhole USD lens (centred) for the wide render: same focal length in pixels as the distorted frame."""
        return {"focal_mm": focal_mm, "horizontal_aperture": focal_mm * self.render_wh[0] / self.focal_px,
                "vertical_aperture": focal_mm * self.render_wh[1] / self.focal_px}

    def apply(self, render):
        """Distorted out_wh image from a render_wh pinhole render."""
        import cv2

        if render.shape[1::-1] != self.render_wh:
            raise ValueError(f"render is {render.shape[1::-1]}, expected {self.render_wh}")
        return cv2.remap(render, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
