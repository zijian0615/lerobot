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

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from cosmos_edge_fanuc.embodiment import (
    ACTION_DIM,
    CHUNK_STEPS,
    DOMAIN_ID,
    DROID_DOMAIN_ID,
    POLICY_PORT,
    REASONER_PORT,
    SO101_DOMAIN_ID,
    STATS_PATH,
    load_stats,
    model_limit_errors,
    reject_checkpoint,
    reject_port,
    stats_document,
)
from cosmos_edge_fanuc.joint_client import (
    ContractError,
    packets_from_chunk,
    parse_action_chunk,
    send_packets,
)

_HOME = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class ContractTest(unittest.TestCase):
    def test_domain_is_not_droid_or_so101(self) -> None:
        self.assertEqual(DOMAIN_ID, 23)
        self.assertNotEqual(DOMAIN_ID, DROID_DOMAIN_ID)
        self.assertNotEqual(DOMAIN_ID, SO101_DOMAIN_ID)
        self.assertEqual(POLICY_PORT, 8001)
        self.assertNotEqual(POLICY_PORT, REASONER_PORT)

    def test_stats_file_matches_code(self) -> None:
        on_disk = load_stats()
        fresh = stats_document()
        self.assertEqual(on_disk["domain_id"], DOMAIN_ID)
        self.assertEqual(on_disk["action"]["min"], fresh["action"]["min"])
        self.assertEqual(on_disk["action"]["max"], fresh["action"]["max"])
        self.assertEqual(len(on_disk["action"]["min"]), ACTION_DIM)
        self.assertEqual(on_disk["demonstrations"], 0)
        self.assertEqual(on_disk["action"]["min"][-1], 0.0)
        self.assertEqual(on_disk["action"]["max"][-1], 1.0)

    def test_home_pose_is_inside_the_urdf(self) -> None:
        self.assertEqual(model_limit_errors(_HOME), [])

    def test_so101_checkpoint_and_port_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            reject_checkpoint("cosmos_edge_policy_so101/iter_6500")
        with self.assertRaises(ValueError):
            reject_port(REASONER_PORT)

    def test_so101_width_is_refused(self) -> None:
        chunk = [[0.0] * 6 for _ in range(CHUNK_STEPS)]
        with self.assertRaises(ContractError):
            parse_action_chunk({"action": chunk})

    def test_fanuc_chunk_builds_joint_motion_and_pulses_gripper_once(self) -> None:
        chunk = [_HOME + [0.0] for _ in range(CHUNK_STEPS)]
        chunk[3][-1] = 1.0
        packets = packets_from_chunk(chunk, sequence_start=1, gripper_state=0.0)
        self.assertEqual(len(packets), CHUNK_STEPS)
        self.assertEqual(packets[0]["Instruction"], "FRC_JointMotion")
        self.assertNotIn("PortNumber", packets[0])
        self.assertEqual(packets[3]["PortNumber"], 4)
        self.assertEqual(packets[3]["JointAngle"]["J1"], 0.0)
        sent: list[dict] = []
        send_packets(packets[:1], sent.append)
        self.assertEqual(sent[0]["SequenceID"], 1)

    def test_serve_script_refuses_so101_without_a_framework(self) -> None:
        script = Path(__file__).with_name("serve_policy.sh")
        proc = subprocess.run(
            ["bash", str(script), "kabilanKB/cosmos_edge_policy_so101/iter_6500"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("SO-101", proc.stderr)

    def test_stats_json_is_utf8(self) -> None:
        json.loads(STATS_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
