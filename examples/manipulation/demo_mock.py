# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Offline mock for solver + executor + orchestrator.

```bash
cd examples && python -m manipulation.demo_mock
```
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from shapely.geometry import Point, box

from manipulation.executor import ArmExecutor, ExecutorConfig
from manipulation.orchestrator import run_manipulation
from manipulation.solver import SolverConfig, solve
from manipulation.visualize import plot_solver_step

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _scene():
    table = box(-0.45, -0.35, 0.45, 0.35)
    ws = Point(-0.45, -0.35).buffer(0.5).intersection(table)
    objs = []
    specs = [
        ("grey_stuffed_animal", (-0.20, -0.10), 0.04),
        ("plastic_container", (-0.10, -0.05), 0.03),
        ("vr_headset", (0.15, 0.05), 0.05),
    ]
    footprints = []
    for name, (x, y), r in specs:
        fp = Point(x, y).buffer(r, cap_style=3)
        footprints.append(fp)
        objs.append(
            {
                "name": name,
                "xy": (x, y),
                "yaw": 0.0,
                "footprint": fp,
                "grasp_pose": (x, y, 0.05, 0.0),
            }
        )
    occupied = footprints[0]
    for fp in footprints[1:]:
        occupied = occupied.union(fp)
    free = table.difference(occupied.buffer(0.02))

    symbolic = {
        "objects": [
            {
                "name": "grey_stuffed_animal",
                "blocked_by": None,
                "in_workspace": ["xarm"],
            },
            {
                "name": "plastic_container",
                "blocked_by": "grey_stuffed_animal",
                "in_workspace": ["xarm"],
            },
            {"name": "vr_headset", "blocked_by": None, "in_workspace": ["xarm"]},
        ]
    }
    geometric = {
        "objects": objs,
        "free_space": free,
        "table_polygon": table,
        "workspaces": {"xarm": ws},
    }
    return symbolic, geometric


def _plan(_sym, _instruction, _arms):
    return [
        {
            "step": 1,
            "arm": "xarm",
            "primitive": "Grasp",
            "args": {"object": "grey_stuffed_animal"},
            "depends_on": [],
        },
        {
            "step": 2,
            "arm": "xarm",
            "primitive": "Place",
            "args": {"object": "grey_stuffed_animal", "destination": "free_space"},
            "depends_on": [1],
        },
        {
            "step": 3,
            "arm": "xarm",
            "primitive": "Grasp",
            "args": {"object": "plastic_container"},
            "depends_on": [2],
        },
        {
            "step": 4,
            "arm": "xarm",
            "primitive": "Place",
            "args": {"object": "plastic_container", "destination": "free_space"},
            "depends_on": [3],
        },
    ]


def main() -> None:
    symbolic, geometric = _scene()
    cfg = SolverConfig(grasp_height=0.05, margin=0.03)

    debug: dict = {}
    bound = solve(
        _plan(symbolic, "", ["xarm"]),
        geometric,
        cfg,
        lookahead=True,
        debug_step=2,
        debug_out=debug,
    )
    print("bound plan:")
    print(json.dumps(bound, indent=2))

    out = Path(__file__).resolve().parent / "demo_solver_step2.png"
    plot_solver_step(debug, show=False, save_path=out, title="Place free_space with lookahead")
    print(f"wrote {out}")

    # Mock arm state for executor verification.
    world_xy = {o["name"]: list(o["xy"]) for o in geometric["objects"]}
    held: dict[str, str | None] = {"xarm": None}
    pose = {"xarm": [0.0, 0.0, 0.2, 0.0]}
    width = {"xarm": 0.08}

    def move_to_pose(p):
        pose["xarm"] = list(p)

    def gripper(cmd: str):
        if cmd == "open":
            width["xarm"] = 0.08
            # If releasing, drop held object at current xy.
            obj = held["xarm"]
            if obj is not None:
                world_xy[obj] = [pose["xarm"][0], pose["xarm"][1]]
                held["xarm"] = None
        elif cmd == "close":
            width["xarm"] = 0.035
            # Pick nearest object (mock).
            x, y = pose["xarm"][0], pose["xarm"][1]
            best, best_d = None, 1e9
            for name, xy in world_xy.items():
                d = (xy[0] - x) ** 2 + (xy[1] - y) ** 2
                if d < best_d:
                    best, best_d = name, d
            held["xarm"] = best

    def perceive_geo():
        objs = []
        for name, xy in world_xy.items():
            objs.append({"name": name, "xy": tuple(xy), "yaw": 0.0, "footprint": Point(xy).buffer(0.03)})
        return {
            "objects": objs,
            "free_space": geometric["free_space"],
            "table_polygon": geometric["table_polygon"],
            "workspaces": geometric["workspaces"],
        }

    def perceive_pair():
        # Symbolic stays stable in this mock.
        return symbolic, perceive_geo()

    ex = ArmExecutor(
        "xarm",
        move_to_pose=move_to_pose,
        gripper=gripper,
        read_gripper_width=lambda: width["xarm"],
        perceive=perceive_geo,
        get_current_pose=lambda: tuple(pose["xarm"]),  # type: ignore[return-value]
        config=ExecutorConfig(place_xy_tol=0.03),
    )

    result = run_manipulation(
        "clear the plastic container",
        perceive=perceive_pair,
        plan_fn=_plan,
        executors={"xarm": ex},
        solver_config=cfg,
        arms=["xarm"],
        lookahead=True,
    )
    print("loop:", result.status, result.reason)
    print(json.dumps(result.results, indent=2))


if __name__ == "__main__":
    main()
