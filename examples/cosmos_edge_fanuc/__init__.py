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

"""FANUC contract for a Cosmos Edge action policy.

This package does not load ``kabilanKB/cosmos_edge_policy_so101`` and does not
replace ``manipulation.run_fanuc_live``.
"""

from cosmos_edge_fanuc.embodiment import (
    ACTION_DIM,
    ARM_JOINT_DIM,
    CHUNK_STEPS,
    DOMAIN_ID,
    DOMAIN_NAME,
    POLICY_PORT,
    REASONER_PORT,
)

__all__ = [
    "ACTION_DIM",
    "ARM_JOINT_DIM",
    "CHUNK_STEPS",
    "DOMAIN_ID",
    "DOMAIN_NAME",
    "POLICY_PORT",
    "REASONER_PORT",
]
