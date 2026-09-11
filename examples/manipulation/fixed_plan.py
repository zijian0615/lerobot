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

"""Rule-based plans that skip the planner VLM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _name(obj: Mapping[str, Any]) -> str:
    return str(obj.get("name", ""))


def _in_workspace(obj: Mapping[str, Any], arm: str) -> bool:
    ws = obj.get("in_workspace") or []
    return arm in [str(a) for a in ws]


def plan_screws_to_container_xarm(
    symbolic_view: Mapping[str, Any],
    instruction: str = "",
    arms: Sequence[str] | None = None,
    *,
    arm: str = "xarm",
) -> list[dict[str, Any]]:
    """
    Fixed plan: every table screw → one Grasp+Place on ``arm`` (default xarm / arm1).

    Skips objects already named like ``*_in_container*``. Destination is the first
    non-screw object whose name contains ``container``.
    """
    del instruction, arms  # interface-compatible with Planner.__call__
    objects = list(symbolic_view.get("objects") or [])
    if not objects:
        return []

    containers = [
        o
        for o in objects
        if "container" in _name(o).lower() and "screw" not in _name(o).lower()
    ]
    if not containers:
        raise ValueError(
            "Fixed plan: no container object found in symbolic_view "
            f"(names={[ _name(o) for o in objects ]})"
        )
    destination = _name(containers[0])

    screws = []
    for o in objects:
        n = _name(o).lower()
        if "screw" not in n:
            continue
        if "in_container" in n or n.endswith("_in_box"):
            continue
        if not _in_workspace(o, arm):
            continue
        screws.append(_name(o))

    plan: list[dict[str, Any]] = []
    step = 1
    for screw in screws:
        grasp_step = step
        plan.append(
            {
                "step": grasp_step,
                "arm": arm,
                "primitive": "Grasp",
                "args": {"object": screw},
                "depends_on": [],
            }
        )
        step += 1
        plan.append(
            {
                "step": step,
                "arm": arm,
                "primitive": "Place",
                "args": {"object": screw, "destination": destination},
                "depends_on": [grasp_step],
            }
        )
        step += 1
    return plan
