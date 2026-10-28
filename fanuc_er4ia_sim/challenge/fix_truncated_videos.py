"""Repair recorded videos whose camera frames arrived truncated (the grey lower part of a cut-short MJPG frame).

Before 2026-10-18 22:44 the table camera ran 1080p MJPG at 30 fps beside its depth stream, and ~2.8 % of its frames
came in cut short; robot_server.py now runs it at 15 fps and drops such frames as they arrive. This repairs what was
recorded before: each truncated frame is replaced by the last whole one, and the video is encoded again as it was
made (rollout videos: H.264 as recorder.py writes them; dataset videos: AV1 with LeRobot's defaults), with the same
frame count and rate. Originals go to runs/_truncated_originals/ first.

    uv run python fanuc_er4ia_sim/challenge/fix_truncated_videos.py --dry-run
    uv run python fanuc_er4ia_sim/challenge/fix_truncated_videos.py
    uv run python fanuc_er4ia_sim/challenge/fix_truncated_videos.py --run r20261020-5fea53 --jumps --dry-run

--jumps also replaces frames whose brightness or colour cast jumps far from the recent good frames (a stream that
came up corrupt: cam_watch.py), as the server now does live.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
BACKUP = RUNS / "_truncated_originals"
ROLLOUT = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
DATASET = ["-c:v", "libsvtav1", "-g", "2", "-crf", "30", "-pix_fmt", "yuv420p"]


def truncated(bgr: np.ndarray) -> bool:  # as robot_server.truncated: flat grey (128) bottom rows
    tail = bgr[-8:]
    return float(tail.std()) < 3.0 and abs(float(tail.mean()) - 128.0) < 12.0


class Jumps:
    """Frames far from the recent good ones in brightness or colour cast (the rules of cam_watch.CameraWatch.bad)."""

    def __init__(self, jump: float = 25.0, cast: float = 15.0):
        import collections

        self.good = collections.deque(maxlen=30)
        self.jump, self.cast = jump, cast

    def bad(self, bgr: np.ndarray) -> bool:
        import statistics

        small = bgr[::4, ::4].astype(np.float32)
        b, g, r = small[..., 0].mean(), small[..., 1].mean(), small[..., 2].mean()
        lum, cast = float((r + g + b) / 3), float(g - (r + b) / 2)
        if len(self.good) >= 5:
            if (abs(lum - statistics.median(x[0] for x in self.good)) > self.jump
                    or abs(cast - statistics.median(x[1] for x in self.good)) > self.cast):
                return True
        self.good.append((lum, cast))
        return False


def probe(f: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames", "-show_entries",
                          "stream=codec_name,width,height,r_frame_rate,nb_read_frames", "-of", "json", str(f)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"][0]


def repair(f: Path, dry: bool, jumps: bool = False, limits: tuple[float, float] = (25.0, 15.0)) -> tuple[int, int]:
    """(frames, replaced); writes the repaired video unless dry or nothing to replace."""
    info = probe(f)
    w, h = int(info["width"]), int(info["height"])
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(f), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, check=True).stdout  # ffmpeg decodes AV1 too (this OpenCV does not)
    video = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    frames, bad, last, jump = [], [], None, Jumps(*limits) if jumps else None
    if jump is not None:  # start from the whole video's median, so its first frames can be judged too
        small = video[:, ::8, ::8].astype(np.float32)
        lum = small.mean(axis=(1, 2, 3))
        cast = small[..., 1].mean(axis=(1, 2)) - (small[..., 0].mean(axis=(1, 2)) + small[..., 2].mean(axis=(1, 2))) / 2
        jump.good.extend([(float(np.median(lum)), float(np.median(cast)))] * 5)
    for img in video:
        if (truncated(img) or (jump is not None and jump.bad(img))) and (last is not None or jump is not None):
            bad.append(len(frames))
            if last is not None:
                img = last
        else:
            last = img
        frames.append(img)
    if not bad or dry:
        return len(frames), len(bad)
    before = info
    fps = before["r_frame_rate"]
    args = DATASET if "datasets" in f.parts else ROLLOUT
    backup = BACKUP / f.relative_to(RUNS)
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(f, backup)
    with tempfile.TemporaryDirectory() as tmp:
        for k, img in enumerate(frames):
            cv2.imwrite(f"{tmp}/{k:06d}.png", img)
        out = Path(tmp) / f.name
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", fps, "-i", f"{tmp}/%06d.png", *args, str(out)],
                       check=True, timeout=1800)
        after = probe(out)
        same = {k: (before[k], after[k]) for k in ("codec_name", "width", "height", "r_frame_rate", "nb_read_frames")
                if before[k] != after[k]}
        if same:
            raise SystemExit(f"{f}: the repaired video differs {same}; original kept")
        shutil.move(str(out), f)
    return len(frames), len(bad)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="count damaged frames, change nothing")
    p.add_argument("--run", action="append", help="only this run (repeatable)")
    p.add_argument("--jumps", action="store_true", help="also frames whose brightness or colour jumps (see above)")
    p.add_argument("--camera", default=None, help="only this camera (table, side, wrist)")
    p.add_argument("--jump", type=float, default=25.0, help="with --jumps: brightness change (0..255) that marks a frame")
    p.add_argument("--cast", type=float, default=15.0, help="with --jumps: colour-cast change that marks a frame")
    args = p.parse_args()
    files = sorted(f for f in (*RUNS.glob("rollouts/*/*/*.mp4"), *RUNS.glob("datasets/*/videos/*/*/*.mp4"))
                   if "_truncated_originals" not in f.parts
                   and (not args.run or any(r in f.parts for r in args.run))
                   and (not args.camera or args.camera in f.stem or f"observation.images.{args.camera}" in f.parts))
    total = fixed = 0
    for f in files:
        n, b = repair(f, args.dry_run, args.jumps, (args.jump, args.cast))
        total += b
        if b:
            fixed += 1
            print(f"{'would repair' if args.dry_run else 'repaired'} {f.relative_to(RUNS)}: {b} of {n} frames", flush=True)
    print(f"{len(files)} videos checked, {fixed} with truncated frames, {total} frames "
          f"{'to replace' if args.dry_run else 'replaced'}" + ("" if args.dry_run else f"; originals in {BACKUP}"))


if __name__ == "__main__":
    sys.exit(main())
