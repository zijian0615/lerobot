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
Offline demo of the tabletop Perception module with a mocked VLM.

Requires: numpy, shapely, matplotlib

```bash
cd examples && python -m tabletop_perception.demo_mock
```
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from shapely.geometry import box

from tabletop_perception.perception import Perception
from tabletop_perception.visualize import visualize_table_plane
from tabletop_perception.vlm import RawDetection


def _mock_vlm(_image: np.ndarray, _instruction: str, _prompt: str) -> list[RawDetection]:
    # Normalized 0-1000 coords on a synthetic 480x640 image.
    return [
        {
            "name": "red_cup",
            "box_2d": [350, 400, 550, 600],
            "grasp_point": [450, 500],
            "blocked_by": None,
        },
        {
            "name": "blue_block",
            "box_2d": [300, 200, 480, 360],
            "grasp_point": [100, 100],  # outside box -> centre fallback
            "blocked_by": "ghost_object",  # not in list -> null
        },
        {
            "name": "plate",
            "box_2d": [500, 550, 720, 850],
            "grasp_point": [610, 700],
            "blocked_by": "red_cup",
        },
    ]


def _synthetic_camera(
    image_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Downward-tilted camera above the table origin."""
    h, w = image_hw
    fx = fy = 600.0
    cx, cy = w / 2.0, h / 2.0
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)

    # Camera looks mostly -Z_table with a mild tilt; origin above table.
    # Table frame: z up. Camera at (0, 0, 0.8), rotated to look down.
    # R maps camera coords -> table coords.
    # Camera optical axis = +Z_cam should point toward -Z_table.
    r = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=float,
    )
    t = np.array([0.0, 0.0, 0.8], dtype=float)
    t_cam_table = np.eye(4, dtype=float)
    t_cam_table[:3, :3] = r
    t_cam_table[:3, 3] = t
    return k, t_cam_table


def main() -> None:
    h, w = 480, 640
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[:] = (220, 220, 220)

    k, t_cam_table = _synthetic_camera((h, w))
    table_polygon = box(-0.45, -0.35, 0.45, 0.35)
    arm_workspaces = {
        "left": box(-0.45, -0.05, 0.05, 0.35),
        "right": box(-0.05, -0.35, 0.45, 0.05),
    }

    perception = Perception(vlm_caller=_mock_vlm, footprint_buffer_m=0.02)
    symbolic_view, geometric_view = perception(
        image=image,
        K=k,
        T_cam_table=t_cam_table,
        grasp_height=0.05,
        instruction="pick up the red cup",
        arm_workspaces=arm_workspaces,
        table_polygon=table_polygon,
    )

    print("symbolic_view:")
    print(json.dumps(symbolic_view, indent=2))
    print("\ngeometric objects:")
    for obj in geometric_view["objects"]:
        print(
            f"  {obj['name']}: xy={obj['xy']}, yaw={obj['yaw']:.3f}, "
            f"grasp_pose={obj['grasp_pose']}"
        )

    out = Path(__file__).resolve().parent / "demo_table_plane.png"
    visualize_table_plane(
        geometric_view,
        show=False,
        save_path=out,
        title="Mock tabletop perception",
    )
    print(f"\nsaved visualisation → {out}")


if __name__ == "__main__":
    main()
