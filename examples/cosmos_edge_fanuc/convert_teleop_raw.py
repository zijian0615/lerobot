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

"""Convert a FANUC teleop dataset from the sin/cos pose encoding (10-D) to the raw 7-D pose.

Old: j0 j1 j2 (mm), j3..j5 (W, P, R) as sin/cos pairs, j7.  New: j0 j1 j2, j3 = W in [0, 360), j4 = P, j5 = R (deg),
j7 -- the layout lerobot.robots.fanuc now records (pose.FANUC_RAW_NAMES). Tool-down W sits near 180 deg: [0, 360) keeps
it continuous where the controller's (-180, 180] range jumps by 360 deg. Videos are copied as they are; state/action
columns, info.json features and the global and per-episode stats are rewritten.

    uv run python examples/cosmos_edge_fanuc/convert_teleop_raw.py \\
        --src zijian2022/fanuc_teleop_20260923_030520 --dst zijian2022/fanuc_teleop_20260923_030520_raw --push
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.robots.fanuc.pose import FANUC_RAW_NAMES

OLD_NAMES = ["j0", "j1", "j2", "j3_sin", "j3_cos", "j4_sin", "j4_cos", "j5_sin", "j5_cos", "j7"]
KEYS = ("observation.state", "action")


def to_raw(values: np.ndarray) -> np.ndarray:
    """(N, 10) sin/cos rows -> (N, 7) raw rows."""
    v = np.asarray(values, dtype=np.float64)
    ang = np.degrees(np.arctan2(v[:, [3, 5, 7]], v[:, [4, 6, 8]]))
    w = np.mod(ang[:, 0], 360.0)
    return np.column_stack([v[:, 0:3], w, ang[:, 1], ang[:, 2], v[:, 9]]).astype(np.float32)


def _stats(arr: np.ndarray) -> dict:
    return {k: np.asarray(v) for k, v in get_feature_stats(arr, axis=0, keepdims=False).items()}


def convert(src_root: Path, dst_root: Path) -> dict:
    info = json.loads((src_root / "meta" / "info.json").read_text())
    for key in KEYS:
        names = info["features"][key]["names"]
        if list(names) != OLD_NAMES:
            raise ValueError(f"{key} names {names} are not the sin/cos layout {OLD_NAMES}")
    if dst_root.exists():
        raise FileExistsError(dst_root)
    shutil.copytree(src_root, dst_root, ignore=shutil.ignore_patterns(".cache"))

    per_episode: dict[int, dict[str, np.ndarray]] = {}
    everything: dict[str, list[np.ndarray]] = {k: [] for k in KEYS}
    for path in sorted((dst_root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        episodes = table.column("episode_index").to_numpy()
        for key in KEYS:
            raw = to_raw(np.stack(table.column(key).to_numpy(zero_copy_only=False)))
            everything[key].append(raw)
            for ep in np.unique(episodes):
                per_episode.setdefault(int(ep), {})[key] = raw[episodes == ep]
            column = pa.FixedSizeListArray.from_arrays(pa.array(raw.ravel(), pa.float32()), len(FANUC_RAW_NAMES))
            table = table.set_column(table.schema.get_field_index(key), key, column)
        pq.write_table(table.replace_schema_metadata(None), path)

    for key in KEYS:
        info["features"][key]["shape"] = [len(FANUC_RAW_NAMES)]
        info["features"][key]["names"] = list(FANUC_RAW_NAMES)
    (dst_root / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    stats = json.loads((dst_root / "meta" / "stats.json").read_text())
    for key in KEYS:
        stats[key] = {k: v.tolist() for k, v in _stats(np.concatenate(everything[key])).items()}
    (dst_root / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

    for path in sorted((dst_root / "meta" / "episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        eps = table.column("episode_index").to_numpy()
        for key in KEYS:
            ep_stats = [_stats(per_episode[int(e)][key]) for e in eps]
            for stat in ep_stats[0]:
                name = f"stats/{key}/{stat}"
                if name not in table.column_names:
                    continue
                dtype = table.schema.field(name).type
                values = pa.array([s[stat].tolist() for s in ep_stats], type=dtype)
                table = table.set_column(table.schema.get_field_index(name), name, values)
        pq.write_table(table, path)

    state = np.concatenate(everything["observation.state"])
    jumps = {n: float(np.abs(np.diff(state[:, i])).max()) for i, n in enumerate(FANUC_RAW_NAMES)}
    return {"frames": int(len(state)), "episodes": len(per_episode), "max_step_state": jumps}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source repo id (sin/cos layout)")
    ap.add_argument("--dst", required=True, help="new repo id for the raw layout")
    ap.add_argument("--work", default="", help="work folder (default: ./<dst name>)")
    ap.add_argument("--push", action="store_true", help="push --dst to the Hub (private)")
    a = ap.parse_args(argv)
    from huggingface_hub import snapshot_download

    work = Path(a.work or a.dst.split("/")[-1])
    src_root = Path(snapshot_download(a.src, repo_type="dataset", local_dir=str(work) + "_src"))
    report = convert(src_root, work)
    print(f"[convert] {a.src} -> {work}: {report}", flush=True)
    if a.push:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        LeRobotDataset(a.dst, root=work).push_to_hub(private=True)
        print(f"[convert] pushed {a.dst} (private)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
