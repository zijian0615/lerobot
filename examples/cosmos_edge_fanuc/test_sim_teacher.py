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

"""Run with the twin venv, which has mujoco:

    fanuc_lrmate200id_smc/.venv-twin/bin/python -m unittest cosmos_edge_fanuc.test_sim_teacher
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[1]
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from cosmos_edge_fanuc.sim_teacher import FPS, generate_episodes, load_task  # noqa: E402

NAMES = {"black_cube", "pen", "blue_bin", "orange_bin"}


class SimTeacherTest(unittest.TestCase):
    def test_episodes_pick_an_item_and_leave_it_at_a_bin(self) -> None:
        task = load_task()
        episodes = generate_episodes(4, seed=1)
        self.assertEqual(len(episodes), 4)
        for episode in episodes:
            self.assertEqual(episode["fps"], FPS)
            self.assertIn(episode["item"], task["items"])
            self.assertIn(episode["bin"], task["bins"])
            self.assertIn(task["bins"][episode["bin"]]["phrase"], episode["task"])
            grips = [frame["gripper"] for frame in episode["frames"]]
            self.assertEqual(grips[0], 0.0)
            self.assertEqual(grips[-1], 0.0)
            self.assertIn(1.0, grips)
            self.assertGreater(len(episode["frames"]), FPS)
            for frame in episode["frames"]:
                self.assertEqual(len(frame["joints_deg"]), 6)
                self.assertEqual({o["name"] for o in frame["objects"]}, NAMES)
            first = {o["name"]: o for o in episode["frames"][0]["objects"]}
            last = {o["name"]: o for o in episode["frames"][-1]["objects"]}
            item, target = last[episode["item"]], last[episode["bin"]]
            # the item ends over its bin, on the floor (cube) or on the rim (pen); everything else stays put
            self.assertLess(math.dist(item["pos"][:2], target["pos"][:2]), 0.015)
            self.assertGreater(item["pos"][2], 0.02)
            for name in NAMES - {episode["item"]}:
                self.assertEqual(first[name]["pos"], last[name]["pos"])


if __name__ == "__main__":
    unittest.main()
