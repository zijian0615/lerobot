#!/usr/bin/env python
"""Push a local LeRobot dataset to the Hugging Face Hub.

Defaults to the newest local recording under ``zijian2022/``.

Examples:

```bash
# Push latest local zijian2022 dataset
python -m examples.manipulation.push_dataset

# Push a specific repo
python -m examples.manipulation.push_dataset --repo-id zijian2022/xarm_hts_demo_20260811_195721

# Private upload
python -m examples.manipulation.push_dataset --private
```
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME


def _newest_local_repo(user: str, prefix: str | None = None) -> str:
    root = HF_LEROBOT_HOME / user
    if not root.is_dir():
        raise SystemExit(f"No local datasets under {root}")

    candidates: list[Path] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if prefix and not path.name.startswith(prefix):
            continue
        if not (path / "meta" / "info.json").is_file():
            continue
        candidates.append(path)

    if not candidates:
        hint = f" matching prefix {prefix!r}" if prefix else ""
        raise SystemExit(f"No complete local datasets under {root}{hint}")

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return f"{user}/{newest.name}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--repo-id",
        default=None,
        help="Dataset repo id, e.g. zijian2022/xarm_hts_demo_YYYYMMDD_HHMMSS. "
        "Default: newest local dataset under --user.",
    )
    p.add_argument("--user", default="zijian2022", help="HF user/org used when --repo-id is omitted")
    p.add_argument(
        "--prefix",
        default="xarm_hts_demo",
        help="Only consider local dirs starting with this prefix when auto-picking latest "
        "(empty string = any). Default: xarm_hts_demo",
    )
    p.add_argument("--private", action="store_true", help="Create/update as a private dataset")
    p.add_argument("--no-videos", action="store_true", help="Skip uploading videos/")
    args = p.parse_args()

    prefix = args.prefix if args.prefix else None
    repo_id = args.repo_id or _newest_local_repo(args.user, prefix=prefix)

    print(f"Loading {repo_id} ...")
    ds = LeRobotDataset(repo_id)
    print(f"  root={ds.root}")
    print(f"  episodes={ds.num_episodes} frames={ds.num_frames}")
    print(f"  features={list(ds.features)}")
    print(f"Pushing to https://huggingface.co/datasets/{repo_id} ...")
    ds.push_to_hub(private=args.private or None, push_videos=not args.no_videos)
    print(f"Done: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
