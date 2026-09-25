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

"""Pack rendered sim episodes into a LeRobot v3.0 dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from cosmos_edge_fanuc.sim_teacher import FPS

IMAGE_HW = (540, 960)
STATE_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "gripper"]


def _features() -> dict:
    return {
        "observation.images.overhead": {
            "dtype": "video",
            "shape": (IMAGE_HW[0], IMAGE_HW[1], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (IMAGE_HW[0], IMAGE_HW[1], 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (7,), "names": list(STATE_NAMES)},
        "action": {"dtype": "float32", "shape": (7,), "names": list(STATE_NAMES)},
    }


def _image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = np.asarray(Image.open(path).convert("RGB"))
    if image.shape != (IMAGE_HW[0], IMAGE_HW[1], 3):
        raise RuntimeError(f"{path} is {image.shape}, expected {IMAGE_HW}")
    return image


def _state(frame: dict) -> np.ndarray:
    values = list(frame["joints_deg"]) + [float(frame["gripper"])]
    return np.asarray(values, dtype=np.float32)


def export_dataset(traj_path: Path, rgb_root: Path, root: Path, repo_id: str) -> Path:
    if root.exists():
        raise FileExistsError(f"{root} already exists")
    episodes = json.loads(traj_path.read_text(encoding="utf-8"))
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=_features(),
        root=root,
        robot_type="fanuc",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
    )
    if dataset.meta.info.codebase_version != CODEBASE_VERSION:
        raise RuntimeError(f"expected {CODEBASE_VERSION}, writer used {dataset.meta.info.codebase_version}")
    try:
        for index, episode in enumerate(episodes):
            folder = rgb_root / f"episode_{index:03d}"
            frames = episode["frames"]
            if not episode.get("task"):
                raise RuntimeError(f"episode {index} has no task")
            states = [_state(frame) for frame in frames]
            actions = states[1:] + [states[-1]]
            for frame_index, (state, action) in enumerate(zip(states, actions, strict=True)):
                overhead = _image(folder / "overhead" / f"{frame_index:06d}.png")
                wrist = _image(folder / "wrist" / f"{frame_index:06d}.png")
                dataset.add_frame(
                    {
                        "observation.images.overhead": overhead,
                        "observation.images.wrist": wrist,
                        "observation.state": state,
                        "action": action,
                        "task": episode["task"],
                    }
                )
            dataset.save_episode()
            print(f"[export] episode {index} ({len(frames)} frames)", flush=True)
    except Exception:
        dataset.finalize()
        shutil.rmtree(root, ignore_errors=True)
        raise
    dataset.finalize()
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export sim RGB + joints to LeRobot v3")
    parser.add_argument("--traj", type=Path, required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/fanuc_sim_bin")
    args = parser.parse_args(argv)
    path = export_dataset(args.traj, args.rgb, args.root, args.repo_id)
    print(f"[export] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
