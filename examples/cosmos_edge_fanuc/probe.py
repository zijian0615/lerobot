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

"""Read-only check: can this machine host a FANUC Cosmos Edge policy server?"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

from cosmos_edge_fanuc.embodiment import POLICY_PORT, REASONER_PORT, STATS_PATH


def _listeners(port: int) -> list[str]:
    proc = subprocess.run(
        ["ss", "-ltnp"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = []
    for line in proc.stdout.splitlines():
        if f":{port}" in line:
            lines.append(line.strip())
    return lines


def _video_cards() -> list[dict[str, str]]:
    cards = []
    if shutil.which("v4l2-ctl") is None:
        return cards
    for node in sorted(Path("/dev").glob("video*")):
        proc = subprocess.run(
            ["v4l2-ctl", "-d", str(node), "--all"],
            check=False,
            capture_output=True,
            text=True,
        )
        bus = card = caps = ""
        for line in proc.stdout.splitlines():
            text = line.strip()
            if text.startswith("Bus info"):
                bus = text.split(":", 1)[-1].strip()
            elif text.startswith("Card type"):
                card = text.split(":", 1)[-1].strip()
            elif text.startswith("Device Caps"):
                caps = text
        cards.append({"node": str(node), "card": card, "bus": bus, "caps": caps})
    return cards


def collect() -> dict:
    py313 = shutil.which("python3.13")
    framework = Path.home() / "cosmos-framework"
    framework_python = framework / ".venv" / "bin" / "python"
    mem = Path("/proc/meminfo").read_text().splitlines()
    mem_total = next((line for line in mem if line.startswith("MemTotal")), "")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "machine": platform.machine(),
        "gpu": gpu.stdout.strip(),
        "mem_total": mem_total.split(":", 1)[-1].strip(),
        "python3": platform.python_version(),
        "python3_13": py313,
        "cuda_groups_need_python": "3.13",
        "cosmos_framework_dir": str(framework),
        "cosmos_framework_present": framework_python.is_file(),
        "reasoner_port": REASONER_PORT,
        "reasoner_listeners": _listeners(REASONER_PORT),
        "policy_port": POLICY_PORT,
        "policy_listeners": _listeners(POLICY_PORT),
        "videos": _video_cards(),
        "stats_path": str(STATS_PATH),
        "stats_exists": STATS_PATH.is_file(),
        "install_verdict": (
            "aarch64 is a cosmos-framework uv environment, and the policy-server "
            "group itself is only filelock + openpi-server. The CUDA groups that "
            "actually load the Edge weights require Python 3.13, which is not on "
            "this machine (only 3.12). Training extras torchao are published for "
            "x86_64 only. Do not uv-sync the train group here until 3.13 exists. "
            "Do not bind the policy to port 8000."
        ),
        "user": os.environ.get("USER", ""),
    }


def main() -> int:
    print(json.dumps(collect(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
