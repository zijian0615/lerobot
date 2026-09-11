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

Return the fields below for the objects relevant to this instruction.

TASK 1 - DETECTION
- Exclude the robot arm, gripper, and the table surface itself.
- Limit to 10 objects.
- Each object needs a unique lowercase snake_case name. If two objects
  look identical, disambiguate by color or position (e.g. red_cup,
  blue_cup) rather than numbering.
- box_2d format: [ymin, xmin, ymax, xmax], integers normalized to 0-1000.
  This axis-aligned box is only a fallback; do not use it for orientation.

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

TASK 4 - OUTLINE (silhouette / top-face polygon)
Give 4-8 vertices that trace the visible object outline, in order
(clockwise or counterclockwise). Format [[y, x], ...], integers
normalized to 0-1000.
- For boxes / blocks / containers: the four corners of the top face,
  rotated with the object. Do NOT copy the axis-aligned box_2d corners.
- For elongated objects: a tight quad around the visible body.
- For round objects: 6-8 points on the rim.
- Vertices must lie on the object, not on shadows or the table.

TASK 5 - FACE / LONG AXIS (grasp yaw)
Two points on the object along the direction the gripper should align
to. Format [[y1, x1], [y2, x2]], integers normalized to 0-1000.
- Elongated objects (screws, pens, tools): the longest visible axis,
  near the tips.
- Boxes / rectangles: two endpoints of one longer top-face edge.
- Square objects: two endpoints of any top-face edge.
- Only truly round objects (balls, cups from above) may set this null.

Return a single JSON object, no markdown fencing, no explanation:

{{
  "objects": [
    {{
      "name": "box",
      "box_2d": [400, 300, 560, 520],
      "polygon": [[410, 340], [430, 510], [550, 490], [530, 320]],
      "blocked_by": "bottle",
      "grasp_point": [480, 415],
      "long_axis": [[410, 340], [430, 510]]
    }},
    {{
      "name": "screw",
      "box_2d": [500, 200, 560, 480],
      "polygon": [[505, 210], [515, 470], [555, 465], [545, 205]],
      "blocked_by": null,
      "grasp_point": [530, 340],
      "long_axis": [[520, 220], [540, 460]]
    }}
  ]
}}
""".strip()
