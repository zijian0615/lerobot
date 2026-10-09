"""Write a 224x224 copy of a screw dataset for ACT.

    uv run python fanuc_er4ia_sim/resize_dataset.py --src-root ... --src-repo ... --dst-root ... --dst-repo ... [--push]
"""
import argparse
from pathlib import Path
import shutil

import numpy as np
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

CAMS = ("table", "wrist", "side")
SIZE = (224, 224)


def to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu()
    if array.ndim == 3 and array.shape[0] == 3:
        array = array.permute(1, 2, 0)
    array = array.numpy()
    if array.dtype != np.uint8:
        array = (array * 255).clip(0, 255).astype(np.uint8) if array.max() <= 1 else array.astype(np.uint8)
    return np.asarray(Image.fromarray(array).resize(SIZE, Image.BILINEAR))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", type=Path, default=Path("fanuc_er4ia_sim/data/fanuc_er4ia_screw_v3"))
    parser.add_argument("--src-repo", default="zijian2022/fanuc_er4ia_screw")
    parser.add_argument("--dst-root", type=Path, default=Path("fanuc_er4ia_sim/data/fanuc_er4ia_screw_224"))
    parser.add_argument("--dst-repo", default="zijian2022/fanuc_er4ia_screw_224")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    src = LeRobotDataset(args.src_repo, root=args.src_root)
    if args.dst_root.exists():
        shutil.rmtree(args.dst_root)
    features = {
        f"observation.images.{name}": {
            "dtype": "video",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
        }
        for name in CAMS
    }
    features["observation.state"] = src.meta.features["observation.state"]
    features["action"] = src.meta.features["action"]
    dst = LeRobotDataset.create(
        repo_id=args.dst_repo,
        fps=src.fps,
        features=features,
        root=args.dst_root,
        robot_type="fanuc_er4ia",
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
    )
    episode = None
    for index in range(len(src)):
        item = src[index]
        current = int(item["episode_index"])
        if episode is None:
            episode = current
        elif current != episode:
            dst.save_episode()
            episode = current
        row = {
            "observation.state": item["observation.state"].numpy().astype(np.float32),
            "action": item["action"].numpy().astype(np.float32),
            "task": item["task"],
        }
        for name in CAMS:
            row[f"observation.images.{name}"] = to_uint8(item[f"observation.images.{name}"])
        dst.add_frame(row)
        if index % 50 == 0:
            print(f"[resize] {index}/{len(src)}", flush=True)
    dst.save_episode()
    dst.finalize()
    print(f"[resize] {args.dst_root} frames {len(src)}")
    if args.push:
        dst.push_to_hub(private=True if args.private else None)
        print(f"[resize] pushed {args.dst_repo}")


if __name__ == "__main__":
    main()
