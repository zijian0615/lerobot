"""Placement check before each rollout: where the objects should go, and whether they are there.

A placement spec (written by the operator from what the team asked for, the team writes nothing):

    seed: 1                         # same seed, same 10 placements, for every team that uses this spec
    objects:
      red:  {colour: red,  xy_m: [0.40, -0.06], tol_mm: 20, yaw_deg: [-20, 20], size_mm: 16}
      blue: {colour: blue, xy_m: [0.40,  0.06], tol_mm: 20, yaw_deg: [-20, 20], size_mm: 16}

For each rollout a target inside each object's allowed area is drawn (repeatable from the seed and the rollout
index) and outlined on the live table image. The objects are then found by colour, fitted with a box of their
size (x, y, yaw from the calibrated table camera) and checked against the allowed area (|dx|, |dy| <= tol_mm from
xy_m, yaw in yaw_deg; a square repeats every 90 deg).

    operator_console.py run <run_id> --placement placement.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

CALIB = Path(__file__).resolve().parents[2] / "examples" / "csi_dataset" / "calib" / "object_calib"
sys.path.insert(0, str(CALIB))
import object_calib as oc  # noqa: E402  (colour masks, box fit, projection, table camera calibration)

MARGIN_MM = 5  # targets stay this far inside the allowed area, so a placement near the target passes


def load(path: Path) -> dict:
    spec = yaml.safe_load(Path(path).read_text())
    for name, o in spec["objects"].items():
        o.setdefault("colour", name)
        o.setdefault("tol_mm", 20)
        o.setdefault("yaw_deg", [-45, 45])
        o.setdefault("size_mm", 10)
        if o["colour"] not in oc.HSV:
            raise SystemExit(f"placement: no colour model for {o['colour']!r}; known: {', '.join(oc.HSV)}")
    return spec


def targets(spec: dict, rollout: int) -> dict:
    """Where each object goes for this rollout (the same for the same seed and rollout, on every run)."""
    rng = np.random.default_rng([int(spec.get("seed", 0)), rollout])
    out = {}
    for name, o in spec["objects"].items():
        r = (o["tol_mm"] - MARGIN_MM) / 1000
        lo, hi = o["yaw_deg"]
        out[name] = {"xy": [o["xy_m"][0] + rng.uniform(-r, r), o["xy_m"][1] + rng.uniform(-r, r)],
                     "yaw": float(rng.uniform(lo + 3, hi - 3) if hi - lo > 6 else (lo + hi) / 2)}
    return out


def _obj(name: str, o: dict, xy, yaw) -> dict:
    s = o["size_mm"] / 1000
    return {"name": name, "kind": "cube", "colour": o["colour"], "xy": list(xy), "yaw": yaw, "side": s, "height": s}


def check(spec: dict, cam: dict, img: np.ndarray, tgt: dict) -> tuple[dict, bool]:
    """Measure every object and check it against its allowed area. ({name: result}, all ok)."""
    res, all_ok = {}, True
    for name, o in spec["objects"].items():
        lo, hi = o["yaw_deg"]
        f = oc.fit(cam, img, _obj(name, o, o["xy_m"], (lo + hi) / 2), o["colour"], (o["tol_mm"] + 15) / 1000)
        if f is None:
            res[name], all_ok = {"found": False, "ok": False, "why": "not found: is it in the area, and its colour?"}, False
            continue
        dx, dy = ((np.array(f["xy"]) - o["xy_m"]) * 1000).tolist()
        yaw = (f["yaw_deg"] - (lo + hi) / 2 + 45) % 90 - 45 + (lo + hi) / 2  # the square's angle nearest the range
        why = []
        if abs(dx) > o["tol_mm"] or abs(dy) > o["tol_mm"]:
            why.append(f"{max(abs(dx), abs(dy)) - o['tol_mm']:.0f} mm outside the area")
        if not lo <= yaw <= hi:
            why.append(f"yaw {yaw:+.0f}° outside [{lo:+.0f}, {hi:+.0f}]°")
        t = tgt[name]
        res[name] = {"found": True, "xy": f["xy"], "yaw_deg": yaw, "dx_mm": dx, "dy_mm": dy,
                     "from_target_mm": float(np.hypot(*((np.array(f["xy"]) - t["xy"]) * 1000))),
                     "ok": not why, "why": "; ".join(why)}
        all_ok &= not why
    return res, all_ok


def lines(spec: dict, tgt: dict, res: dict | None = None) -> list[str]:
    out = []
    for name, o in spec["objects"].items():
        t = tgt[name]
        s = f"  {name:6s} target x {t['xy'][0] * 1000:5.0f}  y {t['xy'][1] * 1000:+5.0f} mm  yaw {t['yaw']:+4.0f}°"
        r = (res or {}).get(name)
        if r is not None:
            if not r["found"]:
                s += f"   ✗ {r['why']}"
            else:
                s += (f"   now x {r['xy'][0] * 1000:5.0f}  y {r['xy'][1] * 1000:+5.0f}  yaw {r['yaw_deg']:+4.0f}°  "
                      + ("✓" if r["ok"] else f"✗ {r['why']}"))
        out.append(s)
    return out


def overlay(spec: dict, cam: dict, img: np.ndarray, tgt: dict, res: dict | None = None) -> np.ndarray:
    """The allowed areas (thin), this rollout's targets (outlined) and what was measured (crosses)."""
    out = img.copy()
    bgr = {"red": (40, 40, 230), "blue": (230, 120, 20), "yellow": (0, 200, 255)}
    for name, o in spec["objects"].items():
        col = bgr.get(o["colour"], (255, 255, 255))
        t = o["tol_mm"] / 1000
        area = np.array([[o["xy_m"][0] + a * t, o["xy_m"][1] + b * t, oc.SURFACE_Z] for a, b in
                         ((1, 1), (-1, 1), (-1, -1), (1, -1))])
        cv2.polylines(out, [oc.project(cam, area).astype(np.int32)], True, col, 1, cv2.LINE_AA)
        uv = oc.project(cam, oc.box(*tgt[name]["xy"], tgt[name]["yaw"], o["size_mm"] / 1000, o["size_mm"] / 1000))
        cv2.polylines(out, [uv[:4].astype(np.int32)], True, col, 2, cv2.LINE_AA)
        r = (res or {}).get(name)
        if r and r["found"]:
            c = oc.project(cam, np.array([[*r["xy"], oc.SURFACE_Z]]))[0].astype(int)
            cv2.drawMarker(out, tuple(c), (0, 255, 0) if r["ok"] else (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    return out
