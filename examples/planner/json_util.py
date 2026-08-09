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

"""Small JSON helpers for planner VLM responses."""

from __future__ import annotations

import re

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def strip_markdown_fences(text: str) -> str:
    """Remove surrounding ``` / ```json fences if present."""
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    if "```" in stripped:
        inner = re.sub(r"^.*?```(?:json)?\s*", "", stripped, count=1, flags=re.DOTALL | re.IGNORECASE)
        inner = re.sub(r"\s*```.*$", "", inner, count=1, flags=re.DOTALL)
        return inner.strip()
    return stripped
