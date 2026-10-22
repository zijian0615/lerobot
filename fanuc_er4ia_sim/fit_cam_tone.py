"""Fit each camera's response: real ~= gain * render + bias per RGB channel, on mat pixels only.

The mat texture is an albedo; how bright it looks differs per camera (exposure, white balance and the
glossy mat seen at a grazing angle from the side). One affine map per camera, applied to the render
before the background is composited, absorbs that.

    uv run python fanuc_er4ia_sim/fit_cam_tone.py --real data/vis_check --sim data/vis_check/newmat
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bg_composite as B  # noqa: E402

HERE = Path(__file__).resolve().parent
TONE_FILE = HERE / "data" / "cam_tone.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", type=Path, required=True, help="dir with real_<cam>.png")
    parser.add_argument("--sim", type=Path, required=True, help="dir with sim_<cam>.png and sim_<cam>_depth.npy")
    parser.add_argument("--out", type=Path, default=TONE_FILE)
    parser.add_argument("--gain-only", action="store_true",
                        help="per-channel gain = median ratio, no bias: for a near-uniform surface the affine fit "
                             "has nothing to pin the slope and collapses to a constant")
    args = parser.parse_args()
    tone = {}
    for cam in ("table", "side", "wrist"):
        real = cv2.cvtColor(cv2.imread(str(args.real / f"real_{cam}.png")), cv2.COLOR_BGR2RGB).astype(np.float32)
        sim = cv2.cvtColor(cv2.imread(str(args.sim / f"sim_{cam}.png")), cv2.COLOR_BGR2RGB).astype(np.float32)
        if cam in B.FIXED:
            background, above = B.classify(cam, np.load(args.sim / f"sim_{cam}_depth.npy"))
            mat = ~background & ~above
        else:
            mat = np.ones(real.shape[:2], bool)
        mat &= cv2.erode(mat.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0  # stay off edges
        idx = np.flatnonzero(mat)[:: max(1, int(mat.sum() // 200_000))]
        gains, biases = [], []
        for c in range(3):
            s, r = sim[..., c].ravel()[idx], real[..., c].ravel()[idx]
            if args.gain_only:
                gains.append(float(np.median(r) / max(np.median(s), 1.0)))
                biases.append(0.0)
                continue
            fit = least_squares(lambda v: v[0] * s + v[1] - r, [1.0, 0.0], loss="huber", f_scale=10.0)
            gains.append(float(fit.x[0]))
            biases.append(float(fit.x[1]))
        tone[cam] = {"gain": gains, "bias": biases}
        before = np.abs(sim - real)[mat].mean()
        after = np.abs(np.clip(sim * gains + np.array(biases), 0, 255) - real)[mat].mean()
        print(f"[tone] {cam:<5} gain {np.round(gains, 3).tolist()} bias {np.round(biases, 1).tolist()}  "
              f"mat |diff| {before:.1f} -> {after:.1f}")
    args.out.write_text(json.dumps(tone, indent=1))
    print(f"[tone] -> {args.out}")


if __name__ == "__main__":
    main()
