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
- Exclude the robot arm, gripper, and the bare table.
- The task instruction is the naming source of truth. If it refers to
  a region by color, pattern, or phrase (e.g. "the red region",
  "the grid", "left half of the mat"), detect that region as its own
  object and name it from those words (red_region, grid, …).
  Outline that named region only — not a larger parent tray/board.
- Also include movable objects the instruction mentions (screws,
  pieces, cups, …) and any play surface needed as a destination.
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


# Official Cosmos 3 Reasoner 2D grounding on a 720p (16:9) full frame.
# Do not list class names or a count — Nano copies them into a fake grid.
COSMOS_DETECTION_PROMPT = """
Think about where each pickable part actually is in this image, then locate it.
Task: "{instruction}"

Ignore the robot arm, gripper, cables, printed drawings, grid lines, text, and empty table.
A box must sit on the part's own pixels. If you are not sure of a location, omit that part.
Do not invent a row or grid of boxes on empty table.

label: required lowercase snake_case from the color and shape you see.
Never omit label. Never use object, item, or thing as the label.
bbox_2d: [x1, y1, x2, y2] (left, top, right, bottom), 0-1000, origin top-left.
point_2d: [x, y] on that part, not on the table.

Return a json list.
""".strip()

# Close-up tiles from a classical/open-vocab detector. Geometry is not from Cosmos.
COSMOS_NAMING_PROMPT = """
Each numbered tile is a close-up of ONE detector region.
Task: "{instruction}"

Name the real 3D object in that tile. lowercase snake_case from color and shape.
Use skip for the robot, cables, printed drawings, empty table, or junk.
Do not invent extra ids. One name per tile.
""".strip()
