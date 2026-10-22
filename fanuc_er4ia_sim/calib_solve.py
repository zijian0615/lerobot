"""Solve the three real camera mounts from calib_capture.py samples and compare them with scene_config.

Frames: everything is in the robot base frame (sim world minus TABLE_TOP_Z, z = 0 on the base
mounting face); the flange frame is the sim J6_link frame (wrist centre, tool along +X).

Markers (all DICT_4X4_50 id 0, told apart by camera and by where they are):
  A, C, D  lie on the table   - seen by wrist and table; a detection is the nearest of TABLE_MARKERS
  B        is on the gripper  - seen by side

Unknowns: wrist  fx, k1, flange->camera (6)    table markers  base pose (6 each)
          side   fx, k1, base->camera (6)      marker B       flange pose (6)
          table  fx, k1, base->camera (6)

    uv run python fanuc_er4ia_sim/calib_solve.py
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
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fk  # noqa: E402
import scene_config as C  # noqa: E402

from lerobot.robots.fanuc.er4ia_fk import sim_from_controller_deg  # noqa: E402

WHICH = {"wrist": "table", "table": "table", "side": "B"}
# Rough base XY (m) of the table markers, only used to tell their detections apart.
TABLE_MARKERS = {"A": (0.504, -0.095), "C": (0.236, -0.477), "D": (0.256, 0.398)}


def marker_points(edge_m):
    h = edge_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)  # ArUco corner order


def to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
    T[:3, 3] = tvec
    return T


def from_T(T):
    return np.concatenate([Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])


def inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def flange_in_base(sample):
    R, p = fk.fk(np.radians(sim_from_controller_deg(sample["joints_deg"])))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def sim_camera(cfg, parent_is_flange):
    """scene_config camera -> (T_parent_cam in OpenCV axes, fx)."""
    eye = np.array(cfg["eye"], float)
    if not parent_is_flange:
        eye = eye - [0, 0, C.TABLE_TOP_Z]
    look = np.array(cfg["look_at"], float) - ([0, 0, 0] if parent_is_flange else [0, 0, C.TABLE_TOP_Z])
    f = look - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, np.array(cfg["up"], float))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([r, -u, f])  # cam x right, y down, z forward
    T[:3, 3] = eye
    fx = cfg["focal_length"] / cfg["horiz_aperture"] * cfg["resolution"][0]
    return T, fx


def K_of(fx, size):
    return np.array([[fx, 0, size[0] / 2], [0, fx, size[1] / 2], [0, 0, 1]])


def solve_side_only(args, side_obs, obj, size, sim_side, fx_s0):
    """No table markers seen (e.g. a new surface): solve only the side camera from gripper marker B.

    Unknowns: side fx, k1, base->camera (6), marker B flange pose (6). The wrist and table mounts are
    left as they are in scene_config.
    """
    def pnp(fx, corners):
        ok, rv, tv = cv2.solvePnP(obj, corners, K_of(fx, size), None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        return to_T(rv.ravel(), tv.ravel())

    # Seed: marker B on the gripper from the previous solution (--marker-b), so every corner is a known
    # base-frame point and one PnP over all samples gives the camera. The fingertip-down poses barely
    # rotate the flange, so a plain hand-eye seed is degenerate here.
    T_fB0 = np.array(json.loads(args.marker_b.read_text())["marker_B_flange"])
    pts = np.concatenate([(o[1] @ T_fB0 @ np.c_[obj, np.ones(4)].T).T[:, :3] for o in side_obs])
    img = np.concatenate([o[2] for o in side_obs])
    ok, rv, tv = cv2.solvePnP(pts, img, K_of(fx_s0, size), None)
    T_bs_seed = inv(to_T(rv.ravel(), tv.ravel()))
    x0 = np.concatenate([[fx_s0, 0.0], from_T(T_bs_seed), from_T(T_fB0)])
    if args.side_k1 is not None:
        x0[1] = args.side_k1
    free = [i for i in range(len(x0)) if (args.free_marker_b or i < 8) and not (i == 1 and args.side_k1 is not None)]

    def unpack(x):
        return x[0], x[1], to_T(x[2:5], x[5:8]), to_T(x[8:11], x[11:14])

    def project(x, T_bf):
        fx, k, T_bs, T_fB = unpack(x)
        T_cm = inv(T_bs) @ T_bf @ T_fB
        uv, _ = cv2.projectPoints(obj, Rotation.from_matrix(T_cm[:3, :3]).as_rotvec(), T_cm[:3, 3],
                                  K_of(fx, size), np.array([k, 0, 0, 0]))
        return uv.reshape(4, 2)

    def residuals(x):
        return np.concatenate([(project(x, T_bf) - c).ravel() for _, T_bf, c, _ in side_obs])

    def residuals_free(v):
        x = x0.copy()
        x[free] = v
        return residuals(x)

    def per_sample(x):
        return [float(np.sqrt(np.mean(np.linalg.norm(project(x, T_bf) - c, axis=1) ** 2))) for _, T_bf, c, _ in side_obs]

    def report(x, title):
        per = per_sample(x)
        allp = np.concatenate([np.linalg.norm(project(x, T_bf) - c, axis=1) for _, T_bf, c, _ in side_obs])
        print(f"\n{title}\n  side  RMS {np.sqrt(np.mean(allp ** 2)):6.2f} px   worst sample "
              f"#{side_obs[int(np.argmax(per))][0]} {max(per):.2f} px")

    print(f"[side-only] {len(side_obs)} observations of marker B; wrist/table kept from scene_config")
    report(x0, "initial guess (hand-eye)")
    x = x0.copy()
    x[free] = least_squares(residuals_free, x0[free], loss="soft_l1", f_scale=2.0, x_scale="jac").x
    report(x, "after refinement" + (" (marker B free)" if args.free_marker_b else " (marker B fixed)"))
    print("  per sample px: " + " ".join(f"#{o[0]}:{e:.1f}" for o, e in zip(side_obs, per_sample(x), strict=True)))
    fx, k, T_bs, T_fB = unpack(x)
    d = (T_bs[:3, 3] - sim_side[:3, 3]) * 1000
    ang = np.degrees(Rotation.from_matrix(sim_side[:3, :3].T @ T_bs[:3, :3]).magnitude())
    print(f"\n  side  position {np.round(T_bs[:3, 3] * 1000, 1)} mm (base frame) vs sim {np.round(sim_side[:3, 3] * 1000, 1)}"
          f"\n        moved {np.linalg.norm(d):6.1f} mm {np.round(d, 1)}, rotated {ang:5.2f} deg, fx {fx:7.1f} vs sim {fx_s0:7.1f}, k1 {k:+.3f}")
    print(f"  marker B (gripper) at {np.round(T_fB[:3, 3] * 1000, 1)} mm in J6 frame")
    allp = np.concatenate([np.linalg.norm(project(x, T_bf) - c, axis=1) for _, T_bf, c, _ in side_obs])
    out = {"side": {"T_base_cam": T_bs.tolist(), "fx": fx, "k1": k, "size": list(size)},
           "marker_B_flange": T_fB.tolist(), "rms_px": {"side": float(np.sqrt(np.mean(allp ** 2)))}}
    (args.dir / "solution.json").write_text(json.dumps(out, indent=1))
    print(f"\n-> {args.dir / 'solution.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=Path(__file__).with_name("data") / "cam_calib")
    parser.add_argument("--marker-b", type=Path, default=Path(__file__).with_name("data") / "cam_calib" / "solution.json",
                        help="side-only mode: solution.json whose marker_B_flange seeds (or fixes) marker B")
    parser.add_argument("--free-marker-b", action="store_true", help="side-only mode: also refine marker B")
    parser.add_argument("--side-k1", type=float, default=None,
                        help="side-only mode: hold k1 at this value (0 = pinhole, as rendered)")
    args = parser.parse_args()
    samples = json.loads((args.dir / "samples.json").read_text())
    edge = samples[0]["marker_mm"] / 1000.0
    obj = marker_points(edge)

    sim_wrist, fx_w0 = sim_camera(C.WRIST_CAM, True)
    sim_side, fx_s0 = sim_camera(C.SIDE_CAM, False)
    sim_table, fx_t0 = sim_camera(C.TABLE_CAM, False)
    sizes = {cam: tuple(samples[0]["images"][cam]["size"]) for cam in WHICH}

    def pnp(cam, fx, corners):
        ok, rv, tv = cv2.solvePnP(obj, corners, K_of(fx, sizes[cam]), None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        return to_T(rv.ravel(), tv.ravel())

    def nearest_marker(xy):
        name = min(TABLE_MARKERS, key=lambda k: np.hypot(*(np.subtract(TABLE_MARKERS[k], xy))))
        dist = np.hypot(*(np.subtract(TABLE_MARKERS[name], xy)))
        return name if dist < 0.06 else None

    # (sample index, T_base_flange, corners 4x2, marker name)
    obs = {cam: [] for cam in WHICH}
    table_seen = {}
    for i, s in enumerate(samples):
        T_bf = flange_in_base(s)
        for cam, info in s["images"].items():
            for m in info["markers"]:
                corners = np.array(m["corners"], float)
                if cam == "side":
                    obs[cam].append((i, T_bf, corners, "B"))
                    continue
                T_cm = pnp(cam, fx_w0 if cam == "wrist" else fx_t0, corners)
                T_bm = (T_bf @ sim_wrist @ T_cm) if cam == "wrist" else (sim_table @ T_cm)
                name = nearest_marker(T_bm[:2, 3])
                if name is None:
                    print(f"[warn] {cam} sample #{i}: marker at {np.round(T_bm[:2, 3], 3)} matches no table marker")
                    continue
                if cam == "table":  # static camera, static markers: keep one view of each
                    table_seen.setdefault(name, []).append(corners)
                    continue
                obs[cam].append((i, T_bf, corners, name))
    for name, views in table_seen.items():
        obs["table"].append((-1, np.eye(4), np.median(views, axis=0), name))
    names = sorted({o[3] for cam in ("wrist", "table") for o in obs[cam]})
    for cam in obs:
        print(f"{cam}: {len(obs[cam])} observations " + str(sorted({o[3] for o in obs[cam]})))
    seen_by_wrist = {o[3] for o in obs["wrist"]}
    for name in names:
        if name not in seen_by_wrist:
            print(f"[warn] marker {name} is only seen by the table camera; its pose rides on that camera alone")

    if not obs["wrist"]:
        solve_side_only(args, obs["side"], obj, sizes["side"], sim_side, fx_s0)
        return

    # ---- initial guesses
    # Both hand-eye problems have the form A_i X B_i = Y. Seed X from scene_config and Y from the
    # first sample, then least squares on the pose consistency.
    def solve_axby(As, Bs, X0):
        Y0 = As[0] @ X0 @ Bs[0]

        def res(v):
            X, Y = to_T(v[:3], v[3:6]), to_T(v[6:9], v[9:12])
            out = []
            for A, B in zip(As, Bs, strict=True):
                E = inv(Y) @ A @ X @ B
                out.extend(Rotation.from_matrix(E[:3, :3]).as_rotvec() * 0.3)  # 1 rad ~ 0.3 m
                out.extend(E[:3, 3])
            return np.asarray(out)

        v = least_squares(res, np.concatenate([from_T(X0), from_T(Y0)]), loss="soft_l1", f_scale=0.01).x
        return to_T(v[:3], v[3:6]), to_T(v[6:9], v[9:12])

    # table markers from the current wrist mount; marker A also seeds the hand-eye
    T_bm0 = {}
    for name in names:
        views = [T_bf @ sim_wrist @ pnp("wrist", fx_w0, c) for _, T_bf, c, n in obs["wrist"] if n == name]
        if not views:
            views = [sim_table @ pnp("table", fx_t0, c) for _, _, c, n in obs["table"] if n == name]
        T = np.mean(views, axis=0)
        T[:3, :3] = Rotation.from_matrix(T[:3, :3]).as_matrix()
        T_bm0[name] = T
    A_obs = [o for o in obs["wrist"] if o[3] == "A"]
    T_fw0, T_bm0["A"] = solve_axby([o[1] for o in A_obs], [pnp("wrist", fx_w0, o[2]) for o in A_obs], sim_wrist)
    # side (eye-to-hand): A = T_bf, X = T_fB, B = inv(T_cm), Y = T_bs
    T_cms = [pnp("side", fx_s0, o[2]) for o in obs["side"]]
    T_fB_seed = inv(obs["side"][0][1]) @ sim_side @ T_cms[0]
    T_fB0, T_bs0 = solve_axby([o[1] for o in obs["side"]], [inv(T) for T in T_cms], T_fB_seed)
    T_bt0 = sim_table

    # ---- joint refinement
    x0 = np.concatenate([[fx_w0, 0.0], from_T(T_fw0), [fx_s0, 0.0], from_T(T_bs0), [fx_t0, 0.0], from_T(T_bt0),
                         from_T(T_fB0)] + [from_T(T_bm0[n]) for n in names])

    def unpack(x):
        p = dict(fx_w=x[0], k_w=x[1], T_fw=to_T(x[2:5], x[5:8]), fx_s=x[8], k_s=x[9],
                 T_bs=to_T(x[10:13], x[13:16]), fx_t=x[16], k_t=x[17], T_bt=to_T(x[18:21], x[21:24]),
                 T_fB=to_T(x[24:27], x[27:30]))
        p["T_bm"] = {n: to_T(x[30 + 6 * k:33 + 6 * k], x[33 + 6 * k:36 + 6 * k]) for k, n in enumerate(names)}
        return p

    def project(cam, p, T_bf, name):
        if cam == "wrist":
            T_cm, fx, k = inv(p["T_fw"]) @ inv(T_bf) @ p["T_bm"][name], p["fx_w"], p["k_w"]
        elif cam == "table":
            T_cm, fx, k = inv(p["T_bt"]) @ p["T_bm"][name], p["fx_t"], p["k_t"]
        else:
            T_cm, fx, k = inv(p["T_bs"]) @ T_bf @ p["T_fB"], p["fx_s"], p["k_s"]
        rv = Rotation.from_matrix(T_cm[:3, :3]).as_rotvec()
        uv, _ = cv2.projectPoints(obj, rv, T_cm[:3, 3], K_of(fx, sizes[cam]), np.array([k, 0, 0, 0]))
        return uv.reshape(4, 2)

    def residuals(x):
        p = unpack(x)
        return np.concatenate([(project(cam, p, T_bf, n) - c).ravel() for cam in obs for _, T_bf, c, n in obs[cam]])

    def rms(p, cam):
        e = np.concatenate([np.linalg.norm(project(cam, p, T_bf, n) - c, axis=1) for _, T_bf, c, n in obs[cam]])
        return float(np.sqrt(np.mean(e**2)))

    def report(x, title):
        p = unpack(x)
        print(f"\n{title}")
        for cam in obs:
            per = [float(np.sqrt(np.mean(np.linalg.norm(project(cam, p, T_bf, n) - c, axis=1) ** 2)))
                   for _, T_bf, c, n in obs[cam]]
            worst = int(np.argmax(per))
            print(f"  {cam:<5} RMS {rms(p, cam):6.2f} px   worst sample #{obs[cam][worst][0]} "
                  f"(marker {obs[cam][worst][3]}) {per[worst]:.2f} px")

    report(x0, "initial guess (hand-eye)")
    result = least_squares(residuals, x0, loss="soft_l1", f_scale=2.0, x_scale="jac")
    x = result.x
    report(x, "after joint refinement")
    p = unpack(x)

    # ---- compare with scene_config
    def compare(name, T_new, T_old, fx_new, fx_old, frame):
        d = (T_new[:3, 3] - T_old[:3, 3]) * 1000
        ang = np.degrees(Rotation.from_matrix(T_old[:3, :3].T @ T_new[:3, :3]).magnitude())
        axis = np.degrees(np.arccos(np.clip(T_new[:3, 2] @ T_old[:3, 2], -1, 1)))
        print(f"  {name:<5} position {np.round(T_new[:3, 3] * 1000, 1)} mm ({frame})  vs sim {np.round(T_old[:3, 3] * 1000, 1)}"
              f"\n        moved {np.linalg.norm(d):6.1f} mm {np.round(d, 1)}, rotated {ang:5.2f} deg "
              f"(optical axis {axis:5.2f} deg), fx {fx_new:7.1f} vs sim {fx_old:7.1f}")

    print("\nsolved vs scene_config")
    compare("wrist", p["T_fw"], sim_wrist, p["fx_w"], fx_w0, "J6 frame")
    compare("side", p["T_bs"], sim_side, p["fx_s"], fx_s0, "base frame")
    compare("table", p["T_bt"], sim_table, p["fx_t"], fx_t0, "base frame")
    print(f"  k1: wrist {p['k_w']:+.3f}, side {p['k_s']:+.3f}, table {p['k_t']:+.3f}")
    for n, T in p["T_bm"].items():
        print(f"  marker {n} (table) at {np.round(T[:3, 3] * 1000, 1)} mm in base, "
              f"normal tilt {np.degrees(np.arccos(abs(T[2, 2]))):.1f} deg from vertical")
    print(f"  marker B (gripper) at {np.round(p['T_fB'][:3, 3] * 1000, 1)} mm in J6 frame")

    out = {
        "wrist": {"T_flange_cam": p["T_fw"].tolist(), "fx": p["fx_w"], "k1": p["k_w"], "size": sizes["wrist"]},
        "side": {"T_base_cam": p["T_bs"].tolist(), "fx": p["fx_s"], "k1": p["k_s"], "size": sizes["side"]},
        "table": {"T_base_cam": p["T_bt"].tolist(), "fx": p["fx_t"], "k1": p["k_t"], "size": sizes["table"]},
        "table_markers_base": {n: T.tolist() for n, T in p["T_bm"].items()},
        "marker_B_flange": p["T_fB"].tolist(),
        "rms_px": {cam: rms(p, cam) for cam in obs},
    }
    (args.dir / "solution.json").write_text(json.dumps(out, indent=1))
    print(f"\n-> {args.dir / 'solution.json'}")


if __name__ == "__main__":
    main()
