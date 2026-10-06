"""Pack record_screw_episode.py output into a LeRobot v3 dataset.

Camera keys name the viewpoint:
  observation.images.table  overhead
  observation.images.wrist  wrist
  observation.images.side   fixed side camera
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
from lerobot.datasets.lerobot_dataset import LeRobotDataset

STATE_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "gripper"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True,
                        help="one episode directory, or a directory of episode_* folders")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/fanuc_er4ia_screw")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    if (args.src / "frames.json").is_file():
        sources = [args.src]
    else:
        sources = sorted(p for p in args.src.iterdir() if (p / "frames.json").is_file())
    if not sources:
        raise FileNotFoundError(args.src)
    meta = json.loads((sources[0] / "frames.json").read_text())
    frames = meta["frames"]
    sample = {
        name: np.asarray(Image.open(sources[0] / name / frames[0]["images"][name]).convert("RGB"))
        for name in ("table", "wrist", "side")
    }
    features = {
        f"observation.images.{name}": {
            "dtype": "video",
            "shape": tuple(image.shape),
            "names": ["height", "width", "channels"],
        }
        for name, image in sample.items()
    }
    features["observation.state"] = {"dtype": "float32", "shape": (7,), "names": list(STATE_NAMES)}
    features["action"] = {"dtype": "float32", "shape": (7,), "names": list(STATE_NAMES)}
    if args.root.exists():
        shutil.rmtree(args.root)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(meta["fps"]),
        features=features,
        root=args.root,
        robot_type="fanuc_er4ia",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
    )
    if dataset.meta.info.codebase_version != CODEBASE_VERSION:
        raise RuntimeError(dataset.meta.info.codebase_version)
    total = 0
    for episode_index, src in enumerate(sources):
        episode = json.loads((src / "frames.json").read_text())
        frames = episode["frames"]
        states = [
            np.asarray(list(frame["joints_deg"]) + [float(frame["gripper"])], dtype=np.float32)
            for frame in frames
        ]
        actions = states[1:] + [states[-1]]
        for frame, state, action in zip(frames, states, actions, strict=True):
            row = {
                "observation.state": state,
                "action": action,
                "task": episode["task"],
            }
            for name in ("table", "wrist", "side"):
                row[f"observation.images.{name}"] = np.asarray(
                    Image.open(src / name / frame["images"][name]).convert("RGB")
                )
            dataset.add_frame(row)
        dataset.save_episode()
        total += len(frames)
        print(f"[pack] episode {episode_index} ({len(frames)} frames) from {src.name}", flush=True)
    dataset.finalize()
    print(f"[pack] {args.root} version {CODEBASE_VERSION} episodes {len(sources)} frames {total}")
    if args.push:
        dataset.push_to_hub(private=True if args.private else None)
        print(f"[pack] pushed {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
