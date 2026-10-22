"""frames/f00000.png ... -> an H.264 mp4 browsers play (imageio-ffmpeg's ffmpeg).

    ~/loto-agent-venv/bin/python showcase/encode.py <frames dir> <out.mp4> [--seconds 40]
"""
import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--seconds", type=float, default=40.0, help="longest video length")
    ap.add_argument("--min-fps", type=float, default=15.0)
    a = ap.parse_args()
    n = len(list(Path(a.frames).glob("f*.png")))
    if not n:
        raise SystemExit(f"no frames in {a.frames}")
    fps = max(a.min_fps, n / a.seconds)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                    "-framerate", f"{fps:.3f}", "-i", str(Path(a.frames) / "f%05d.png"),
                    "-vf", "scale=960:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "27", "-preset", "slow", "-movflags", "+faststart", a.out],
                   check=True)
    print(f"{a.out}: {n} frames at {fps:.1f} fps")


if __name__ == "__main__":
    main()
