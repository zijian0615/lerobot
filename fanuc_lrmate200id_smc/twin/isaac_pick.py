"""Pick-and-place trajectory on the USD robot, with no MuJoCo.

The kinematic chain is the one baked into the Isaac stage. One Isaac Sim
process can solve this and then render the overhead and wrist cameras.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from joint_map import FINGER_OPEN_M
from usd_chain import Chain

# Same ready pose as the MuJoCo keyframe: tool +Z points world -Z.
READY_Q = np.array([0.0, 0.4693, 0.1612, 0.0, -1.2627, 0.0])
FPS = 30
SPEED_M_S = 0.16
APPROACH_Z = 0.10
GRASP_Z = 0.025
PLACE_Z = 0.07
CARRY_DROP_M = 0.015
TCP_TOL_M = 0.008
HOLD_FRAMES = 5
TABLE = (-0.45, 0.45, -0.35, 0.35)
_TOOL_DOWN = np.array([0.0, 0.0, -1.0])


class IkError(RuntimeError):
    pass


def load_desk(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _values(chain: Chain, q: np.ndarray, gripper_m: float = FINGER_OPEN_M) -> dict:
    return chain.joint_values(q, gripper_m)


def _tcp(chain: Chain, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation, position = chain.site_frame("tcp", _values(chain, q))
    return rotation, position


def ik_to(chain: Chain, q: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Damped least squares onto a TCP position. Tool +Z stays world -Z."""
    lo, hi = chain.arm_limits()
    q = np.array(q, dtype=float)
    position = target
    for _ in range(80):
        rotation, position = _tcp(chain, q)
        tool_z = rotation[:, 2]
        err = np.concatenate([target - position, np.cross(tool_z, _TOOL_DOWN)])
        if np.linalg.norm(err[:3]) < 0.001 and float(tool_z @ _TOOL_DOWN) > 0.998:
            return q
        jac = np.zeros((6, 6))
        eps = 1e-4
        for joint in range(6):
            nudged = q.copy()
            nudged[joint] += eps
            rotation_n, position_n = _tcp(chain, nudged)
            delta = rotation_n @ rotation.T
            omega = np.array(
                [delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0], delta[1, 0] - delta[0, 1]]
            ) / (2 * eps)
            jac[:3, joint] = (position_n - position) / eps
            jac[3:, joint] = omega
        step = jac.T @ np.linalg.solve(jac @ jac.T + (0.05**2) * np.eye(6), err)
        q = np.clip(q + 0.6 * step, lo, hi)
    if np.linalg.norm(target - position) > TCP_TOL_M:
        raise IkError(f"TCP missed {target.tolist()} by {np.linalg.norm(target - position):.4f} m")
    return q


def _box(spec: dict, center: list[float]) -> dict:
    return {
        "name": spec["name"],
        "center": center,
        "size": list(spec["size"]),
        "yaw": float(spec["yaw"]),
        "color": list(spec["color"]),
    }


def _objects(desk: dict, pick: dict, tcp: np.ndarray, carried: bool, placed: bool) -> list[dict]:
    container = desk["container"]
    pick_xy = np.asarray(pick["xy"], dtype=float)
    height = float(pick["size"][2])
    if placed:
        pick_center = [
            float(container["xy"][0]),
            float(container["xy"][1]),
            float(container["size"][2]) + height / 2,
        ]
    elif carried:
        pick_center = [float(tcp[0]), float(tcp[1]), float(tcp[2] - CARRY_DROP_M)]
    else:
        pick_center = [float(pick_xy[0]), float(pick_xy[1]), height / 2]
    objects = [
        _box(container, [float(container["xy"][0]), float(container["xy"][1]), float(container["size"][2]) / 2])
    ]
    for cube in desk["cubes"]:
        center = pick_center if cube["name"] == pick["name"] else [
            float(cube["xy"][0]),
            float(cube["xy"][1]),
            float(cube["size"][2]) / 2,
        ]
        objects.append(_box(cube, center))
    return objects


def _frame(chain: Chain, desk: dict, q: np.ndarray, gripper: float, pick: dict, carried: bool, placed: bool) -> dict:
    from joint_map import model_to_fanuc

    _, tcp = _tcp(chain, q)
    return {
        "joints_deg": [float(v) for v in model_to_fanuc(q, "coupled")],
        "gripper": float(gripper),
        "objects": _objects(desk, pick, tcp, carried, placed),
    }


def _move(chain, desk, frames, q, target, gripper, pick, carried, placed):
    _, start = _tcp(chain, q)
    steps = max(1, int(np.ceil(np.linalg.norm(target - start) / SPEED_M_S * FPS)))
    for i in range(1, steps + 1):
        q = ik_to(chain, q, start + (target - start) * (i / steps))
        frames.append(_frame(chain, desk, q, gripper, pick, carried, placed))
    return q


def _hold(chain, desk, frames, q, gripper, n, pick, carried, placed):
    for _ in range(n):
        frames.append(_frame(chain, desk, q, gripper, pick, carried, placed))
    return q


def _grip(chain, desk, frames, q, start, end, pick, placed: bool):
    for i in range(1, HOLD_FRAMES + 1):
        gripper = start + (end - start) * (i / HOLD_FRAMES)
        carried = gripper >= 0.5 and not placed
        frames.append(_frame(chain, desk, q, gripper, pick, carried, placed))
    return q


def solve_episode(chain: Chain, desk: dict, rng: np.random.Generator) -> dict:
    """One scripted grasp of a desk cube into the container. Joints are pendant degrees."""
    q = READY_Q.copy()
    pick = desk["cubes"][int(rng.integers(0, len(desk["cubes"])))]
    pick_xy = np.asarray(pick["xy"], dtype=float)
    box_xy = np.asarray(desk["container"]["xy"], dtype=float)
    frames: list[dict] = []
    _, home = _tcp(chain, q)
    above_pick = np.array([pick_xy[0], pick_xy[1], APPROACH_Z])
    at_pick = np.array([pick_xy[0], pick_xy[1], GRASP_Z])
    above_box = np.array([box_xy[0], box_xy[1], APPROACH_Z])
    at_box = np.array([box_xy[0], box_xy[1], PLACE_Z])

    q = _hold(chain, desk, frames, q, 0.0, HOLD_FRAMES, pick, False, False)
    q = _move(chain, desk, frames, q, above_pick, 0.0, pick, False, False)
    q = _move(chain, desk, frames, q, at_pick, 0.0, pick, False, False)
    q = _hold(chain, desk, frames, q, 0.0, HOLD_FRAMES, pick, False, False)
    q = _grip(chain, desk, frames, q, 0.0, 1.0, pick, False)
    q = _move(chain, desk, frames, q, above_pick, 1.0, pick, True, False)
    q = _move(chain, desk, frames, q, above_box, 1.0, pick, True, False)
    q = _move(chain, desk, frames, q, at_box, 1.0, pick, True, False)
    q = _hold(chain, desk, frames, q, 1.0, HOLD_FRAMES, pick, True, False)
    q = _grip(chain, desk, frames, q, 1.0, 0.0, pick, True)
    q = _move(chain, desk, frames, q, above_box, 0.0, pick, False, True)
    q = _move(chain, desk, frames, q, home, 0.0, pick, False, True)
    _hold(chain, desk, frames, q, 0.0, HOLD_FRAMES, pick, False, True)
    return {
        "task": desk["task"],
        "fps": FPS,
        "table": list(TABLE),
        "pick": pick["name"],
        "pick_xy": [float(v) for v in pick_xy],
        "container_xy": [float(v) for v in box_xy],
        "solver": "isaac_usd_chain",
        "frames": frames,
    }
