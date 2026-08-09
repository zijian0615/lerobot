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

"""Offline mock demo for the symbolic Planning module.

```bash
cd examples && python -m planner.demo_mock
```
"""

from __future__ import annotations

import json

from planner.planning import PlanValidationError, Planner, validate_plan

SCENE = {
    "objects": [
        {
            "name": "plastic_container",
            "blocked_by": "grey_stuffed_animal",
            "in_workspace": ["xarm"],
        },
        {
            "name": "grey_stuffed_animal",
            "blocked_by": None,
            "in_workspace": ["xarm"],
        },
        {
            "name": "vr_headset",
            "blocked_by": None,
            "in_workspace": ["xarm"],
        },
    ]
}

ARMS = ["xarm"]


def _mock_vlm(_scene, _instruction, _arms, feedback):
    # First attempt deliberately violates rule (f): Place without Grasp.
    if feedback is None:
        return [
            {
                "step": 1,
                "arm": "xarm",
                "primitive": "Place",
                "args": {"object": "vr_headset", "destination": "free_space"},
                "depends_on": [],
            }
        ]
    # Corrected plan: clear blocker, then grasp container.
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
    bad = [
        {
            "step": 1,
            "arm": "xarm",
            "primitive": "Grasp",
            "args": {"object": "vr_headset", "offset_m": 0.1},
            "depends_on": [],
        }
    ]
    try:
        validate_plan(bad, SCENE, ARMS)
        raise AssertionError("expected PlanValidationError for numeric args")
    except PlanValidationError as exc:
        assert exc.rule == "d", exc

    planner = Planner(vlm_caller=_mock_vlm, max_attempts=2)
    plan = planner(SCENE, "clear the plastic container", ARMS)
    print(json.dumps({"plan": plan}, indent=2))


if __name__ == "__main__":
    main()
