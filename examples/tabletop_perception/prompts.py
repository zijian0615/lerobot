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

"""VLM prompts for tabletop perception.

Coordinates are Gemini-style normalized integers in [0, 1000].
"""

from __future__ import annotations

DEFAULT_DETECTION_PROMPT = """
You are a robot perception module. Analyze this top-down image of a
tabletop scene.

Task instruction: "{instruction}"

Return three things for the objects relevant to this instruction.

TASK 1 - DETECTION
- Exclude the robot arm, gripper, and the table surface itself.
- Limit to 10 objects.
- Each object needs a unique lowercase snake_case name. If two objects
  look identical, disambiguate by color or position (e.g. red_cup,
  blue_cup) rather than numbering.
- box_2d format: [ymin, xmin, ymax, xmax], integers normalized to 0-1000.

TASK 2 - OCCLUSION
For each object, report which single other detected object lies on top
of it or blocks top-down gripper access to it. Use null when the object
is freely accessible from above. Report only the direct blocker, not a
chain.

TASK 3 - GRASP POINT
For each object, give one 2D point where a top-down parallel gripper
should close. Format [y, x], integers normalized to 0-1000.
- Choose a point on the object, not on its shadow or on a neighbouring
  object.
- Avoid any part of the object that is covered by another object.
- Prefer a point near the object's center of mass for stability.

Return a single JSON object, no markdown fencing, no explanation:

{{
  "objects": [
    {{
      "name": "box",
      "box_2d": [400, 300, 560, 520],
      "blocked_by": "bottle",
      "grasp_point": [530, 320]
    }}
  ]
}}
""".strip()
