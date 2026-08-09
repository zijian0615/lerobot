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

"""Geometry binding + per-arm execution for tabletop manipulation."""

from .coordination import OverlapBusy, OverlapGuard, overlap_from_workspaces
from .executor import ArmExecutor, ExecutionResult
from .orchestrator import ManipulationLoop, run_manipulation
from .solver import (
    GraspUnreachable,
    NoLegalPlacement,
    SolverConfig,
    solve,
)

__all__ = [
    "solve",
    "SolverConfig",
    "GraspUnreachable",
    "NoLegalPlacement",
    "ArmExecutor",
    "ExecutionResult",
    "ManipulationLoop",
    "run_manipulation",
    "OverlapGuard",
    "OverlapBusy",
    "overlap_from_workspaces",
]
