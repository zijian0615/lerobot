"""Put the real background behind the sim render for the fixed cameras (table, side).

A pixel is background when its depth puts it outside the tabletop volume (floor, sky, the table's
own sides). Those pixels come from a real photo of the empty cell taken from the same camera; the
tabletop, the arm and the task objects stay rendered.

    uv run python fanuc_er4ia_sim/bg_composite.py build   # data/bg_plates/*.png -> plate_<cam>.png
    uv run python fanuc_er4ia_sim/bg_composite.py build fanuc_er4ia_sim/data/bg_plates_v2

A shot needs only the cameras it has ("images"); <dir>/clean_boxes.json ({"side": [[x0, y0, x1, y1, thr]]},
thr optional: colour distance from the wall that counts as leftover, 0 = refill the whole box)
overrides CLEAN_BOXES, which belong to the 2026-10-07 side camera and are only used for data/bg_plates.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_config as C  # noqa: E402

HERE = Path(__file__).resolve().parent
PLATE_DIR = HERE / "data" / "bg_plates"
FIXED = {"table": C.TABLE_CAM, "side": C.SIDE_CAM}
# Boxes (x0, y0, x1, y1, px) in a plate where the real arm's dress pack stayed in view in most source
# frames, so the median kept it. Wall-unlike pixels inside are inpainted. Static real things (the
# pole, the hanging cable, the ceiling light) lie outside the boxes and stay.
CLEAN_BOXES = {"side": [(930, 0, 1180, 360)]}
ROBOT_MARGIN_PX = 100  # real cables and hoses hang well past the sim arm; a median cannot drop them where the arm loitered


def _rays(cfg):
    """Per-pixel unit offsets so that world point = eye + depth * rays (depth along the optical axis)."""
    w, h = cfg["resolution"]
    fx = cfg["focal_length"] / cfg["horiz_aperture"] * w
    eye = np.array(cfg["eye"], float)
    fwd = np.array(cfg["look_at"], float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, cfg["up"])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    return eye, fwd + ((u - w / 2) / fx)[..., None] * right - ((v - h / 2) / fx)[..., None] * up


_RAYS = {name: _rays(cfg) for name, cfg in FIXED.items()}


def classify(name, depth):
    """(background, above_table) masks from a distance-to-image-plane depth map."""
    eye, rays = _RAYS[name]
    z = np.where(np.isfinite(depth), depth, 1e3)
    p = eye + z[..., None] * rays
    x, y, h = p[..., 0], p[..., 1], p[..., 2] - C.TABLE_TOP_Z
    on_cell = ((x > C.TABLE_X[0] - 0.01) & (x < C.TABLE_X[1] + 0.01)
               & (y > C.TABLE_Y[0] - 0.01) & (y < C.TABLE_Y[1] + 0.01) & (h > C.SURFACE_DZ - C.MAT_T - 0.005)
               & (h < 1.2))
    return ~on_cell, on_cell & (h > C.SURFACE_DZ + 0.004)


def load_plates(plate_dir=PLATE_DIR):
    return {name: cv2.cvtColor(cv2.imread(str(plate_dir / f"plate_{name}.png")), cv2.COLOR_BGR2RGB)
            for name in FIXED if (plate_dir / f"plate_{name}.png").is_file()}


def load_tone(path=HERE / "data" / "cam_tone.json"):
    """Per-camera {gain, bias} from fit_cam_tone.py; {} if not fitted."""
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def apply_tone(rgb, tone):
    """Map a render to the camera's real response: gain * rgb + bias per channel."""
    if not tone:
        return rgb
    out = rgb.astype(np.float32) * np.array(tone["gain"], np.float32) + np.array(tone["bias"], np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def mat_mask_moving(depth, T_world_cam, fx):
    """Mat pixels for a moving camera: depth -> world height (OpenCV camera axes, world = sim world)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    z = np.where(np.isfinite(depth), depth, 1e3)
    pc = np.stack([(u - w / 2) / fx * z, (v - h / 2) / fx * z, z], -1)
    height = (pc @ np.asarray(T_world_cam)[:3, :3].T + np.asarray(T_world_cam)[:3, 3])[..., 2] - C.TABLE_TOP_Z
    return (height > C.SURFACE_DZ - 0.005) & (height < C.SURFACE_DZ + 0.004)


def render_to_real(name, rgb, depth, plate=None, tone=None, gain=1.0, T_world_cam=None, fx=None):
    """Everything the recorder applies to a render: the camera's tone on the mat, then the real background.

    The tone map is fitted on mat pixels, so it is only applied there; the arm and the task objects keep
    their rendered colour. The wrist camera moves, so its mat pixels come from its pose (T_world_cam,
    OpenCV axes) and fx; it has no fixed background.
    """
    if tone:
        if name in FIXED:
            background, above = classify(name, depth)
            mat = (~background & ~above).astype(np.float32)
        else:
            mat = mat_mask_moving(depth, T_world_cam, fx).astype(np.float32)
        a = cv2.GaussianBlur(mat, (0, 0), 2)[..., None]
        rgb = (apply_tone(rgb, tone).astype(np.float32) * a + rgb.astype(np.float32) * (1 - a)).astype(np.uint8)
    return rgb if plate is None or name not in FIXED else composite(name, rgb, depth, plate, gain=gain)


def composite(name, rgb, depth, plate, gain=1.0, feather_px=2):
    """rgb, plate: HxWx3 uint8 RGB. gain scales the plate (per-episode brightness jitter)."""
    background, _ = classify(name, depth)
    alpha = cv2.GaussianBlur(background.astype(np.float32), (0, 0), feather_px)[..., None]
    out = rgb.astype(np.float32) * (1 - alpha) + np.clip(plate.astype(np.float32) * gain, 0, 255) * alpha
    return out.astype(np.uint8)


def build(plate_dir=PLATE_DIR):
    """Per-pixel median over the shots in which that pixel is background and well clear of the arm."""
    plate_dir = Path(plate_dir)
    shots = json.loads((plate_dir / "shots.json").read_text())
    boxes_file = plate_dir / "clean_boxes.json"
    clean_boxes = (json.loads(boxes_file.read_text()) if boxes_file.is_file()
                   else CLEAN_BOXES if plate_dir.resolve() == PLATE_DIR.resolve() else {})
    for name in FIXED:
        stack, needed = [], None
        used = [shot for shot in shots if name in shot["images"]]
        if not used:
            print(f"[plates] no {name} shots in {plate_dir}; plate_{name}.png not built")
            continue
        for shot in used:
            real = cv2.imread(str(plate_dir / shot["images"][name])).astype(np.float32)
            depth = np.load(plate_dir / f"sim_{shot['tag']}" / f"sim_{name}_depth.npy")
            background, above = classify(name, depth)
            arm = cv2.dilate(above.astype(np.uint8), np.ones((2 * ROBOT_MARGIN_PX + 1,) * 2, np.uint8)) > 0
            real[~(background & ~arm)] = np.nan
            stack.append(real)
            needed = background if needed is None else needed | background
        with np.errstate(all="ignore"):
            plate = np.nanmedian(np.stack(stack), axis=0)
        holes = np.isnan(plate[..., 0])
        plate = cv2.inpaint(np.nan_to_num(plate).astype(np.uint8), holes.astype(np.uint8), 7, cv2.INPAINT_TELEA)
        for x0, y0, x1, y1, *thr in clean_boxes.get(name, []):
            thr = thr[0] if thr else 18  # colour distance from the wall that counts as leftover; 0 = the whole box
            box = plate[y0:y1, x0:x1]
            ring = np.concatenate([plate[y0:y1, max(x0 - 40, 0):x0].reshape(-1, 3), plate[y0:y1, x1:x1 + 40].reshape(-1, 3)])
            wall = np.median(ring, axis=0)
            odd = (np.linalg.norm(box.astype(np.float32) - wall, axis=2) > thr) | (
                cv2.cvtColor(box, cv2.COLOR_BGR2HSV)[..., 1] > 45)
            mask = np.zeros(plate.shape[:2], np.uint8)
            mask[y0:y1, x0:x1] = cv2.dilate(odd.astype(np.uint8), np.ones((9, 9), np.uint8))
            # a smooth wall: grow it in from the mask's edge at 1/4 size (a full-size inpaint of a large
            # area leaves the leftover's blotches), then blend back
            h, w = mask.shape
            q = 4
            lo_mask = cv2.dilate((cv2.resize(mask.astype(np.float32), (w // q, h // q), interpolation=cv2.INTER_AREA)
                                  > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
            lo = cv2.inpaint(cv2.resize(plate, (w // q, h // q), interpolation=cv2.INTER_AREA), lo_mask, 5,
                             cv2.INPAINT_TELEA)
            fill = cv2.resize(cv2.GaussianBlur(lo, (0, 0), 3), (w, h), interpolation=cv2.INTER_LINEAR)
            a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 4)[..., None]
            plate = (plate.astype(np.float32) * (1 - a) + fill.astype(np.float32) * a).astype(np.uint8)
            print(f"[plates] plate_{name}.png: cleaned {mask.sum()} px of leftover arm in {(x0, y0, x1, y1)}")
        cv2.imwrite(str(plate_dir / f"plate_{name}.png"), plate)
        print(f"[plates] plate_{name}.png from {len(used)} shots; {(holes & needed).sum() / needed.sum() * 100:.1f}% "
              f"of the background seen in none of them (inpainted)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"] and len(sys.argv) <= 3:
        build(Path(sys.argv[2]) if len(sys.argv) == 3 else PLATE_DIR)
    else:
        raise SystemExit(__doc__)
