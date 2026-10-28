"""Record every trial on the cell: the three cameras and the robot state at 10 Hz.

While a trial runs, a thread samples the cameras (JPEG frames on disk) and the state and keeps the last action
played. When the trial ends, a worker thread, one trial at a time, so the cell is never held up:

1. writes the web bundle runs/rollouts/<run_id>/<trial_id>/: <camera>.mp4 (H.264, plays in any browser),
   trace.json (time, joints, fingertip pose, action) and meta.json;
2. appends the trial as an episode to the run's LeRobotDataset in runs/datasets/<run_id> (lerobot-dataset-viz,
   training), with the operator's instruction as the task; datasets/<run_id>/manipbench.json maps episodes to
   trials;
3. calls on_done(run_id, trial_id), which marks the trial as recorded and pushes the bundle to the relay.

A failure in steps 1-3 is logged and never affects the trial or its score.
"""
from __future__ import annotations

import json
import logging
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("recorder")

FPS = 10
SIZE = (640, 360)  # (w, h) of every recorded camera; the cell's cameras are all ~16:9


class TrialRecorder:
    def __init__(self, root: Path, observe: Callable, on_done: Callable[[str, str], None] | None = None,
                 lerobot: bool = True):
        self.root, self.observe, self.on_done, self.lerobot = root, observe, on_done, lerobot
        self.lock = threading.Lock()
        self.cur: dict | None = None
        self.jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True, name="recorder-worker").start()

    # ---- called by the cell ----
    def start(self, run_id: str, trial_id: str, instruction: str, cams: list[str]) -> None:
        self.stop("superseded")  # never two at once
        # Frames go to a folder of this recording's own: the worker may still be writing a voided attempt of the same
        # trial from its folder (it did on 2026-10-22, and deleting the folder lost both). The worker writes the
        # bundles in order, so the trial's folder ends up with the latest attempt's videos.
        tmp = self.root / "rollouts" / run_id / trial_id / f"frames-{time.time_ns() // 1_000_000}"
        for c in cams:
            (tmp / c).mkdir(parents=True, exist_ok=True)
        cur = {"run_id": run_id, "trial_id": trial_id, "instruction": instruction, "cams": list(cams), "dir": tmp.parent,
               "frames": tmp, "trace": [], "action": None, "kind": None, "t0": time.time(), "stop": threading.Event()}
        with self.lock:
            self.cur = cur
        threading.Thread(target=self._sample, args=(cur,), daemon=True, name="recorder").start()
        log.info("recording %s", trial_id)

    def action(self, row, kind: str) -> None:
        with self.lock:
            if self.cur is not None:
                self.cur["action"], self.cur["kind"] = [float(v) for v in row], kind

    def stop(self, reason: str) -> None:
        with self.lock:
            cur, self.cur = self.cur, None
        if cur is None:
            return
        cur["stop"].set()
        cur["reason"] = reason
        self.jobs.put(cur)

    # ---- internals ----
    def _sample(self, cur: dict) -> None:
        period, t_next, k = 1.0 / FPS, time.perf_counter(), 0
        while not cur["stop"].is_set():
            try:
                joints, pose, images = self.observe(cur["cams"])
                for c, img in images.items():
                    small = cv2.resize(np.asarray(img), SIZE, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(cur["frames"] / c / f"{k:05d}.jpg"), cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
                with self.lock:
                    act, kind = cur["action"], cur["kind"]
                cur["trace"].append({"t": round(time.time() - cur["t0"], 3), "q": [round(float(v), 3) for v in joints],
                                     "tcp": None if pose is None else [round(float(v), 2) for v in pose],
                                     "a": act, "kind": kind})
                k += 1
            except Exception as e:  # noqa: BLE001  (a camera hiccup must not stop the recording)
                log.warning("recorder frame %d: %s", k, e)
            t_next += period
            time.sleep(max(0.0, t_next - time.perf_counter()))

    def _worker(self) -> None:
        while True:
            cur = self.jobs.get()
            time.sleep(0.3)  # let the sampler write its last frame
            try:
                n = self._bundle(cur)
            except Exception as e:  # noqa: BLE001
                log.error("recording %s not written: %s", cur["trial_id"], e)
                continue
            if not n:
                continue
            ok = True
            if self.lerobot:
                try:
                    self._lerobot(cur)
                except Exception as e:  # noqa: BLE001  (the videos are written: the replay still goes up)
                    ok = False
                    log.error("recording %s: dataset episode not written (frames kept): %s", cur["trial_id"], e)
            if ok:
                shutil.rmtree(cur["frames"], ignore_errors=True)
            if self.on_done:
                try:
                    self.on_done(cur["run_id"], cur["trial_id"])
                except Exception as e:  # noqa: BLE001
                    log.error("recording %s: not published: %s", cur["trial_id"], e)

    def _bundle(self, cur: dict) -> int:
        n = len(cur["trace"])
        if n == 0:
            return 0
        for c in cur["cams"]:
            if not any((cur["frames"] / c).glob("*.jpg")):
                continue
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", str(cur["frames"] / c / "%05d.jpg"),
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(cur["dir"] / f"{c}.mp4")], check=True, timeout=300)
        (cur["dir"] / "trace.json").write_text(json.dumps(cur["trace"], separators=(",", ":")))
        (cur["dir"] / "meta.json").write_text(json.dumps({
            "run_id": cur["run_id"], "trial_id": cur["trial_id"], "instruction": cur["instruction"], "fps": FPS,
            "size": SIZE, "frames": n, "cameras": cur["cams"], "end_reason": cur.get("reason"),
            "recorded_at": cur["t0"]}, indent=1))
        log.info("recorded %s: %d frames, %s", cur["trial_id"], n, ", ".join(cur["cams"]))
        return n

    def _lerobot(self, cur: dict) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = self.root / "datasets" / cur["run_id"]
        repo_id = f"manipbench/{cur['run_id']}"
        w, h = SIZE
        features = {f"observation.images.{c}": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
                    for c in cur["cams"]}
        features |= {
            "observation.state": {"dtype": "float32", "shape": (7,), "names": ["J1", "J2", "J3", "J4", "J5", "J6", "gripper"]},
            "observation.tcp_pose": {"dtype": "float32", "shape": (6,), "names": ["x", "y", "z", "w", "p", "r"]},
            "action": {"dtype": "float32", "shape": (7,), "names": ["a0", "a1", "a2", "a3", "a4", "a5", "gripper"]},
        }
        if (root / "meta" / "info.json").exists() and not (root / "data").is_dir():
            # Started but no episode saved (e.g. the server stopped while writing the first one): resume() would look
            # for the dataset on the Hub and fail for every later episode. Set it aside and start afresh.
            aside = self.root / "datasets" / "_broken" / f"{cur['run_id']}-{int(time.time())}"
            aside.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root), str(aside))
            log.warning("dataset %s had no episodes saved; moved to %s", cur["run_id"], aside)
        ds = (LeRobotDataset.resume(repo_id, root=root) if (root / "meta" / "info.json").exists()
              else LeRobotDataset.create(repo_id, FPS, features, root=root, robot_type="fanuc_er4ia"))
        try:
            last_a = None
            for k, fr in enumerate(cur["trace"]):
                last_a = fr["a"] or last_a or fr["q"]
                frame = {"task": cur["instruction"] or "",
                         "observation.state": np.asarray(fr["q"], np.float32),
                         "observation.tcp_pose": np.asarray(fr["tcp"] if fr["tcp"] else [np.nan] * 6, np.float32),
                         "action": np.asarray(last_a, np.float32)}
                for c in cur["cams"]:
                    img = cv2.imread(str(cur["frames"] / c / f"{k:05d}.jpg"))
                    frame[f"observation.images.{c}"] = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None
                                                        else np.zeros((h, w, 3), np.uint8))
                ds.add_frame(frame)
            # No process pool: forking this many-threaded server can deadlock the child on a lock held at fork time
            # (it did on 2026-10-22, and every recording after it waited behind it).
            ds.save_episode(parallel_encoding=False)
            episode = ds.meta.total_episodes - 1
        finally:
            ds.finalize()
        idx = root / "manipbench.json"
        rows = json.loads(idx.read_text()) if idx.exists() else []
        rows.append({"episode_index": episode, "trial_id": cur["trial_id"], "action_kind": cur["trace"][-1]["kind"],
                     "end_reason": cur.get("reason")})
        idx.write_text(json.dumps(rows, indent=1))
        log.info("dataset %s: episode %d = %s", repo_id, episode, cur["trial_id"])
