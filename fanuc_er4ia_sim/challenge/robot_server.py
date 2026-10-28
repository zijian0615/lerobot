"""Evaluation server for the FANUC ER-4iA cell (Rockwell-NVIDIA Fall 2026, Dataset 3). Runs on the cell PC.

Teams run their policy on their own machine with the cell client (github.com/SIDCLab/PhysicalAI-Hackathon-2026) and reach
this server over HTTP. The server owns the arm, the cameras and every safety limit. The operator drives the
trials with operator_console.py and scores them.

    uv run python fanuc_er4ia_sim/challenge/robot_server.py --backend mock                 # nothing moves
    uv run python fanuc_er4ia_sim/challenge/robot_server.py --backend fanuc --bind 0.0.0.0  # the real cell

Team API (header `Authorization: Bearer <token>`), base /api/v1:
    GET  /run                     this token's run and its trials
    GET  /trials/<id>             one trial
    POST /trials/<id>/start       start a trial the operator marked ready (its "instruction" is the operator's prompt)
    GET  /time                    {"t": cell clock}
    POST /submit                  {"submission": {...}} (no token): a request the operator approves into a run;
                                  returns {"request_id", "secret"}
    GET  /requests/<id>           header X-Request-Secret: {"status"}, and {"run_id", "token"} once approved
    GET  /observation?cameras=table,wrist&width=640&height=480[&format=jpeg&quality=90][&depth=table]
                                  depth=table,wrist: also those cameras' depth, {"depth": {camera: 16-bit PNG, mm}} at the
                                  depth streams' own sizes (client/calibration/cameras.json "<camera>_depth"); never recorded
    POST /actions                 {"joints": [[J1..J6 deg, gripper], ...], "dt": 0.1}
                                  [, "obs_t": t, "replace": true]: planned from the observation at t while the
                                  previous chunk still played: replace the queue, skipping rows already in the past
                                  or {"tcp": [[x, y, z mm, w, p, r deg, gripper], ...], "dt": 0.1} (UTOOL1 in UF0)
Operator API (header `x-admin-token`), base /admin: see operator_console.py.
Operate page (http://127.0.0.1:9097 on the cell PC only: bound to localhost, never tunnelled): run and score trials
in the browser, the same steps as `operator_console.py run`. Its /op API accepts only Host 127.0.0.1/localhost and,
on POST, the X-Requested-With header, so no other web page in the cell PC's browser can drive the arm.
Organizer API (session cookie from POST /org/login with the organizer password, set with `operator_console.py
set-password`), base /org: the site's Organizer view. Read the cell, the requests and every run; approve, reject,
hide, show, reveal. Nothing that moves the arm: running and scoring trials stays on /admin, in the lab.
Lab only (the relay does not forward them): GET /leaderboard, the full leaderboard (rebuilt from the scored runs on
every request; also written to leaderboard/index.html after every score), and GET /rollouts/<run>/<trial>/<file>.
Public, through the relay: GET /site, the site. Until the operator runs `operator_console.py reveal` it has no
leaderboard: each team sees only its own runs, under My results, with the link from `run.py --status`
(https://<site>/#key=<view_key>). The page loads GET /results/<view_key>.json and the videos from
GET /results/<view_key>/rollouts/<run>/<trial>/<file>. GET /site.json: {"reveal"}.
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import logging
import os
import re
import secrets
import signal
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
TASK = json.loads((HERE / "task_card.json").read_text())
RUNS = HERE / "runs"
log = logging.getLogger("robot_server")

CAMERAS = ("table", "side", "wrist")
JOINT_LIMITS_DEG = [(-170, 170), (-110, 120), (-69, 205), (-190, 190), (-120, 120), (-360, 360)]
DT_RANGE_S = (0.1, 1.0)  # 10 Hz is the twin datasets' rate; the speed cap (--max-speed-mm-s) applies regardless
MAX_CHUNK = 100
MAX_QUEUE = 100
# The work surface in UF0, as the controller's TCP (UTOOL1, Z 223 mm, at the fingertips) sees it.
SURFACE_Z_MM = -335.1
Z_MIN_HARD_MM = SURFACE_Z_MM + 1.0  # never let an operator set the TCP floor closer than ~1 mm to the surface
TCP_BOX_MM = {"x": (150.0, 650.0), "y": (-450.0, 450.0), "z": (Z_MIN_HARD_MM, 300.0)}  # UF0; the driver floor still applies
OPEN = ("waiting", "preparing", "ready", "running")  # trial statuses that are not final
REJECT_STREAK_FAULT = 10  # this many motions refused in a row by the controller is treated as a fault


# --------------------------------------------------------------------------------------------------------- #
#  Backends: the only code that touches hardware                                                           #
# --------------------------------------------------------------------------------------------------------- #

# Training start pose (frame 0 of the twin episodes), sim joint convention, gripper open.
HOME_Q = [-0.0065, 3.2909, -11.8967, -0.0007, -74.8218, -0.0286, 0.0]
NATIVE = {"table": (1920, 1080), "side": (1920, 1080), "wrist": (1280, 720)}


DEPTH_CAMERAS = ("table", "wrist")  # cameras with a depth stream: Orbbec Gemini 2 (table) and Gemini 305 (wrist)
DEPTH_HEADER = struct.Struct("<4sIdHH")


class DepthSource:
    """One camera's depth: depth_reader.py (Orbbec's SDK) in its own process, so a crash in the SDK costs the depth
    stream, never the arm. Restarted when it stops; keeps the latest frame (H x W uint16 mm, 0 = no depth).

    On demand: the stream runs only while someone wants it (demand(): a run that asked for this camera's depth is being
    set up or played, or a depth request came in), and stops IDLE_S after the last demand. All the cameras share one
    USB 2.0 bus, and an idle depth stream (the wrist's alone is ~12 MB/s) only crowds the colour streams there."""

    IDLE_S = 120.0
    REST_S = 2.0  # after a stop, before the next start (a restart right after a kill can stall the wrist camera)

    def __init__(self, camera: str, fps: int = 15):
        self.camera = camera
        self.cmd = ["uv", "run", "-q", "--no-project", "--python", "3.12", "--with", "pyorbbecsdk2",
                    "python", str(HERE / "depth_reader.py"), "--camera", camera, "--fps", str(fps)]
        self.frame, self.t, self.proc, self.running = None, 0.0, None, False
        self.want_until, self.stopped_at = 0.0, 0.0
        self.lock = threading.Lock()

    def demand(self, seconds: float | None = None) -> None:
        """Keep the stream on for at least `seconds` more (IDLE_S by default); start it if it is off."""
        with self.lock:
            self.want_until = max(self.want_until, time.time() + (seconds or self.IDLE_S))
            if self.running:
                return
            self.running, self.frame = True, None
        threading.Thread(target=self._run, daemon=True, name=f"depth-{self.camera}").start()

    def _run(self) -> None:
        time.sleep(max(0.0, self.stopped_at + self.REST_S - time.time()))
        log.info("%s depth stream: starting (on demand)", self.camera)
        backoff = 2.0
        while self.running and time.time() < self.want_until:
            started = time.time()
            # its own process group: `uv run` starts the reader as a child, and both have to go on a restart or a stop
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
            n = 0
            try:
                while self.running and time.time() < self.want_until:
                    head = self.proc.stdout.read(DEPTH_HEADER.size)
                    if len(head) < DEPTH_HEADER.size:
                        break
                    tag, _seq, t, w, h = DEPTH_HEADER.unpack(head)
                    data = self.proc.stdout.read(w * h * 2)
                    if tag != b"DPTH" or len(data) < w * h * 2:
                        break
                    with self.lock:
                        self.frame, self.t = np.frombuffer(data, np.uint16).reshape(h, w), t
                    if n == 0:
                        log.info("depth stream up: %s camera %dx%d", self.camera, w, h)
                    n += 1
            finally:
                self._kill()
            if self.running and time.time() < self.want_until:
                backoff = 2.0 if time.time() - started > 60 else min(backoff * 2, 60.0)
                log.warning("%s depth stream stopped after %d frames; restarting in %.0f s", self.camera, n, backoff)
                time.sleep(backoff)
        with self.lock:
            self.running, self.stopped_at = False, time.time()
        log.info("%s depth stream: stopped (not wanted for %.0f s)", self.camera, self.IDLE_S)
        if time.time() < self.want_until:  # wanted again while it was stopping
            self.demand(self.want_until - time.time())

    def get(self, max_age: float = 1.0, wait: float = 8.0) -> np.ndarray:
        """The latest frame; starts the stream if it is off, and waits up to `wait` s for a fresh frame."""
        self.demand()
        t_end = time.time() + wait
        while True:
            with self.lock:
                frame, t = self.frame, self.t
            if frame is not None and time.time() - t <= max_age:
                return frame
            if time.time() > t_end:
                raise RuntimeError(f"the {self.camera} camera's depth is not available right now; try again shortly")
            time.sleep(0.05)

    def _kill(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait()

    def stop(self) -> None:
        self.running, self.want_until = False, 0.0
        self._kill()


def mock_depth(camera: str = "table", q_deg=None) -> np.ndarray:
    """The clear table as the calibrated depth stream would see it (mm), for the mock backend. The wrist camera's pose
    comes from the joints (q_deg, the twin's convention) through the flange calibration."""
    try:
        cam = json.loads((HERE / "client" / "calibration" / "cameras.json").read_text())[f"{camera}_depth"]
    except (OSError, KeyError, ValueError):
        return np.full((480, 848) if camera == "wrist" else (400, 640), 1000 if camera == "table" else 350, np.uint16)
    w, h = cam["size"]
    if "T_flange_cam" in cam:
        sys.path.insert(0, str(SIM))
        import fk  # noqa: PLC0415

        R, p = fk.fk(np.radians(np.asarray(q_deg if q_deg is not None else HOME_Q, float)[:6]))
        T_bf = np.eye(4)
        T_bf[:3, :3], T_bf[:3, 3] = R, p
        T = T_bf @ np.array(cam["T_flange_cam"], float)
    else:
        T = np.array(cam["T_base_cam"], float)
    v, u = np.mgrid[0:h, 0:w]
    ray = np.stack([(u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"], np.ones((h, w))], -1) @ T[:3, :3].T
    s = (-0.0051 - T[2, 3]) / ray[..., 2]  # the ray parameter at the work surface = the depth (Z) in metres
    return np.where((s > 0) & (s < 3), s * 1000, 0).astype(np.uint16)


class MockBackend:
    """No hardware, nothing moves: joints follow the commands, images are synthetic test cards.

    With `episodes`, a recorded twin episode is replayed instead (one frame per executed action)."""

    def __init__(self, episodes: Path | None = None):
        self.episodes = []
        if episodes is not None:
            self.episodes = sorted(p for p in episodes.iterdir() if (p / "frames.json").exists())
            if not self.episodes:
                raise SystemExit(f"no recorded episodes under {episodes}")
        self.load(0)

    def load(self, index: int) -> None:
        self.i = 0
        self.tcp_cmd = None
        if not self.episodes:
            self.q = list(HOME_Q)
            return
        self.ep = self.episodes[index % len(self.episodes)]
        self.frames = json.loads((self.ep / "frames.json").read_text())["frames"]
        f = self.frames[0]
        self.q = [*map(float, f["joints_deg"]), float(f["gripper"])]

    def start(self) -> None:
        if self.episodes:
            log.info("mock backend: replaying %d recorded episodes under %s", len(self.episodes), self.ep.parent)
        else:
            log.info("mock backend: synthetic images")
        self.depth = mock_depth("table")

    def observe_depth(self, cams) -> dict:
        return {c: self.depth if c == "table" else mock_depth(c, self.q) for c in cams}

    def _card(self, cam: str) -> np.ndarray:
        """Native-size RGB test card: a per-camera colour, the camera name, step and joints, a moving bar."""
        w, h = NATIVE[cam]
        base = {"table": (40, 90, 160), "side": (60, 140, 80), "wrist": (170, 90, 50)}[cam]
        img = np.empty((h, w, 3), np.uint8)
        img[:] = base
        img[:, :, 0] = np.clip(base[0] + np.linspace(-30, 30, w), 0, 255).astype(np.uint8)
        x = int((self.i * 12) % w)
        img[:, max(0, x - 6): x + 6] = 255
        s = w / 640
        cv2.putText(img, f"MOCK {cam}", (int(20 * s), int(60 * s)), cv2.FONT_HERSHEY_SIMPLEX, 1.6 * s, (255, 255, 255),
                    max(1, int(3 * s)))
        cv2.putText(img, f"step {self.i}", (int(20 * s), int(110 * s)), cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s,
                    (255, 255, 255), max(1, int(2 * s)))
        cv2.putText(img, " ".join(f"{v:.1f}" for v in self.q), (int(20 * s), int(150 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (255, 255, 255), max(1, int(2 * s)))
        return img

    def tcp(self):
        """UTOOL1 pose in UF0 (mm, deg): the last TCP command, else the FK of the joints, as the controller reports it."""
        if self.tcp_cmd is not None:
            return list(self.tcp_cmd)
        from lerobot.robots.fanuc.er4ia_fk import controller_tcp_pose
        from lerobot.robots.fanuc.pose import wpr_from_rotation_matrix

        rot, pos = controller_tcp_pose(np.radians(self.q[:6]), (0.0, 0.0, 223.0, 0.0, 0.0, 0.0), None)
        return [*(np.asarray(pos) * 1000).tolist(), *wpr_from_rotation_matrix(np.asarray(rot).tolist())]

    def observe(self, cams):
        images = {}
        for c in cams:
            if self.episodes:
                img = cv2.imread(str(self.ep / c / f"{min(self.i, len(self.frames) - 1):06d}.png"))
                images[c] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                images[c] = self._card(c)
        return list(self.q), self.tcp(), images

    def send(self, a) -> None:
        self.q, self.tcp_cmd = list(a), None
        self.i += 1

    def send_tcp(self, a) -> None:
        self.tcp_cmd = list(a[:6])
        self.q[6] = float(a[6])
        self.i += 1

    def home(self, layout_index: int = 0) -> None:
        self.load(layout_index)
        time.sleep(0.5)

    def faulted(self) -> str | None:
        return None

    z_min_mm: float | None = None

    def set_z_min(self, z_mm: float) -> None:
        self.z_min_mm = z_mm

    def recover(self) -> None:
        pass

    def stop(self) -> None:
        pass


# camera: (USB vendor, product, interface of its colour stream); a by-path changes with the USB port it is plugged into
USB_CAMS = {"table": ("2bc5", "0670", 4), "wrist": ("2bc5", "0840", 4)}


def find_camera(cam: str, path: str) -> str:
    """The configured device if it exists, else the same camera's colour stream on whatever port it is on now."""
    if Path(path).exists() or cam not in USB_CAMS:
        return path
    vendor, product, iface = USB_CAMS[cam]
    for p in sorted(Path("/dev/v4l/by-path").glob(f"*-usb-0:*:1.{iface}-video-index0")):
        usb = Path("/sys/class/video4linux") / Path(os.path.realpath(p)).name / "device" / ".."
        try:
            if (usb / "idVendor").read_text().strip() == vendor and (usb / "idProduct").read_text().strip() == product:
                log.warning("%s camera: %s is gone; using %s (same camera, another USB port)", cam, path, p)
                return str(p)
        except OSError:
            continue
    return path


def truncated(img: np.ndarray) -> bool:
    """A camera's MJPG frame cut short on the USB bus: the decoder fills the missing rows with flat grey (128)."""
    tail = img[-8:]
    return float(tail.std()) < 3.0 and abs(float(tail.mean()) - 128.0) < 12.0


class FanucBackend:
    """The real ER-4iA through the lerobot joint-mode RMI driver. Cameras open once for the server's lifetime."""

    def __init__(self, host: str, z_min_mm: float, cams: dict[str, str], modes: dict[str, tuple[int, int, int]],
                 z_offset_mm: float = 0.0, gripper_wait_mm: float | None = 3.0, limits: dict | None = None):
        sys.path.insert(0, str(SIM))
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.fanuc import Fanuc, FanucConfig

        cfg = {
            name: OpenCVCameraConfig(index_or_path=path, width=modes[name][0], height=modes[name][1], fps=modes[name][2],
                                     fourcc="MJPG", **({"exposure": 65} if name == "side" else {}))
            for name, path in cams.items()
        }
        self.depth: dict[str, DepthSource] = {}  # set by main() from --depth
        self.rot_step_deg = float((limits or {}).get("max_rot_step_deg", 5.0))
        self.robot = Fanuc(FanucConfig(host=host, motion="joint", joint_z_min_mm=z_min_mm, joint_z_offset_mm=z_offset_mm,
                                       joint_gripper_wait_mm=gripper_wait_mm,
                                       **{f"joint_{k}": v for k, v in (limits or {}).items() if k != "max_rot_step_deg"},
                                       twin_udp_host="",
                                       cameras=cfg))

    def start(self) -> None:
        # The wrist Gemini 305 only streams on the first open after USB enumeration: open once, keep open.
        self.robot.connect()
        # The table camera's stream sometimes comes up corrupt (a shared USB 2.0 bus): check it now, then keep watching,
        # and reopen it when it goes bad. Never the wrist camera, which does not survive a reopen.
        from cam_watch import CameraWatch  # noqa: PLC0415

        self.watch = {c: CameraWatch(c, self.robot.cameras[c], log) for c in ("table",) if c in self.robot.cameras}
        for w in self.watch.values():
            w.startup()
            w.start()

    def observe(self, cams):
        from lerobot.robots.fanuc.er4ia_fk import sim_from_controller_deg

        r = self.robot
        q = [float(v) for v in sim_from_controller_deg(r._latest_joints)]
        g = r._latest_gripper_state if r._latest_gripper_state is not None else (r._commanded_gripper or 0.0)
        images = {c: self._whole(c, self._read(c)) for c in cams}
        return q + [float(g)], None if r._latest_pose is None else [float(v) for v in r._latest_pose], images

    def _read(self, cam: str):
        """The camera's latest frame; while a watched camera is being reopened, its last whole one."""
        try:
            return self.robot.cameras[cam].read_latest()
        except (RuntimeError, TimeoutError):
            if cam in getattr(self, "watch", {}) and cam in self.__dict__.get("last_whole", {}):
                return self.last_whole[cam]
            raise

    def _whole(self, cam: str, img: np.ndarray) -> np.ndarray:
        """The frame, or the camera's last whole one if this one arrived damaged: truncated (grey lower part), or, on a
        watched camera, with a sudden brightness or colour jump (teams and recordings never see them). Counted per
        camera in /admin/status."""
        good = self.__dict__.setdefault("last_whole", {})
        bad = self.__dict__.setdefault("truncated", collections.Counter())
        w = getattr(self, "watch", {}).get(cam)
        if img is not None and (w.bad(img) if w else truncated(img)):
            bad[cam] += 1
            if bad[cam] in (1, 10, 100) or bad[cam] % 1000 == 0:
                log.warning("%s camera: %d damaged frames so far, replaced by the last whole one", cam, bad[cam])
            if cam in good:
                return good[cam]
        elif img is not None:
            good[cam] = img
        return img

    def send(self, a) -> None:
        names = [f"J{i}.pos" for i in range(1, 7)]
        self.robot.send_action({**dict(zip(names, map(float, a[:6]), strict=True)), "gripper.pos": float(a[6])})

    def send_tcp(self, a) -> None:
        self.robot.send_tcp_action(a[:6], gripper=float(a[6]), max_rot_step_deg=self.rot_step_deg)

    def home(self, layout_index: int = 0) -> None:
        from move_home import go_home

        offset = self.robot.config.joint_z_offset_mm
        self.robot.config.joint_z_offset_mm = 0.0  # home is a fixed pose; the policy offset would keep it 3 mm off forever
        try:
            go_home(self.robot)
        finally:
            self.robot.config.joint_z_offset_mm = offset

    def faulted(self) -> str | None:
        ack = self.robot._last_ack or {}
        if ack.get("Communication") == "FRC_SystemFault":
            return json.dumps(ack)
        if self.robot.reject_streak >= REJECT_STREAK_FAULT:  # the controller refuses every motion: the arm is not moving
            return json.dumps({"rejected_in_a_row": self.robot.reject_streak, **ack})
        return None

    @property
    def z_min_mm(self) -> float | None:
        return self.robot.config.joint_z_min_mm

    def set_z_min(self, z_mm: float) -> None:
        self.robot.config.joint_z_min_mm = z_mm  # read on every joint-mode send

    def recover(self) -> None:
        self.robot.recover_after_fault()
        self.robot._last_ack = None

    def observe_depth(self, cams) -> dict:
        missing = [c for c in cams if c not in self.depth]
        if missing:
            raise RuntimeError(f"depth is off for {missing} on this cell (robot_server.py --depth)")
        return {c: self.depth[c].get() for c in cams}

    def stop(self) -> None:
        for src in self.depth.values():
            src.stop()
        self.robot.disconnect()


# --------------------------------------------------------------------------------------------------------- #
#  Runs and trials (one JSON file per run)                                                                 #
# --------------------------------------------------------------------------------------------------------- #

class Store:
    def __init__(self, root: Path = RUNS):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.runs = {p.stem: json.loads(p.read_text()) for p in root.glob("r*.json")}
        for run in self.runs.values():  # a previous server stopped mid-trial: close what it left open
            dirty = False
            if not run.get("view_key"):  # runs from before My results
                run["view_key"], dirty = secrets.token_hex(16), True
            dirty = self.mark_episodes(run) or dirty
            for t in run["trials"]:
                if t["status"] == "running":
                    t["status"] = "failed"
                    t["stats"].update(t_end=t["stats"].get("t_end") or time.time(), end_reason="server restart")
                    dirty = True
                elif t["status"] in ("preparing", "ready"):
                    t["status"], dirty = "waiting", True
            if dirty:
                self.save(run)

    @property
    def site(self) -> dict:
        """runs/site.json: {"reveal": bool}. False (the default): the public site shows each team only its own runs."""
        f = self.root / "site.json"
        return json.loads(f.read_text()) if f.is_file() else {"reveal": False}

    def set_site(self, **kw) -> dict:
        site = {**self.site, **kw}
        (self.root / "site.json").write_text(json.dumps(site))
        return site

    def mark_episodes(self, run: dict) -> bool:
        """Set each trial's "episode", its index in the run's LeRobotDataset (datasets/<run>/manipbench.json, written by
        recorder.py; a voided and re-run trial keeps its latest recording). True if anything changed."""
        idx = self.root / "datasets" / run["run_id"] / "manipbench.json"
        try:
            episodes = {r["trial_id"]: r["episode_index"] for r in json.loads(idx.read_text())}
        except (OSError, ValueError, KeyError):
            return False
        changed = False
        for t in run["trials"]:
            if t["trial_id"] in episodes and t.get("episode") != episodes[t["trial_id"]]:
                t["episode"], changed = episodes[t["trial_id"]], True
        return changed

    def by_key(self, key: str) -> dict | None:
        """The run this view key opens, or None."""
        return next((r for r in self.runs.values() if key and secrets.compare_digest(r.get("view_key", ""), key)), None)

    def save(self, run: dict) -> None:
        path = self.root / f"{run['run_id']}.json"
        path.with_suffix(".tmp").write_text(json.dumps(run, indent=1))
        path.with_suffix(".tmp").replace(path)

    def create(self, submission: dict, backend: str) -> dict:
        with self.lock:
            run_id = f"r{time.strftime('%Y%m%d')}-{secrets.token_hex(3)}"
            import task_spec

            card = submission.get("task_card") if isinstance(submission.get("task_card"), dict) else {}
            layouts = card.get("layouts") or [{"id": "L1"}]
            n = int(submission.get("n_trials") or (len(layouts) * int(card.get("trials_per_layout") or 5) if card else 10))
            trials = [{"trial_id": f"{run_id}-t{k:02d}", "index": k, "layout": layouts[k % len(layouts)].get("id", "L1"),
                       "layout_index": k % len(layouts), "repeat": k // len(layouts), "status": "waiting",
                       "outcome": None, "stats": {}} for k in range(n)]
            run = {"run_id": run_id, "token": secrets.token_hex(16), "view_key": secrets.token_hex(16),
                   "created": time.time(), "backend": backend,
                   "task_id": task_spec.spec(submission)["task_id"], "submission": submission, "trials": trials}
            self.runs[run_id] = run
            self.save(run)
            return run

    def trial(self, trial_id: str) -> tuple[dict, dict]:
        run = self.runs.get(trial_id.rsplit("-t", 1)[0])
        for t in (run or {}).get("trials", []):
            if t["trial_id"] == trial_id:
                return run, t
        raise KeyError(trial_id)


ROLLOUT_TYPES = {".mp4": "video/mp4", ".json": "application/json"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def rollout_file(root: Path, parts: list[str]) -> Path | None:
    """runs/rollouts/<run_id>/<trial_id>/<name>.mp4|json, or None: no other path can be reached."""
    run_id, trial_id, name = parts
    stem, _, ext = name.rpartition(".")
    if not (SAFE_NAME.match(run_id) and SAFE_NAME.match(trial_id) and SAFE_NAME.match(stem) and f".{ext}" in ROLLOUT_TYPES):
        return None
    f = root / run_id / trial_id / name
    return f if f.is_file() else None


# --------------------------------------------------------------------------------------------------------- #
#  Submission requests: a team submits submission.yaml with its client; the operator approves it into a run #
# --------------------------------------------------------------------------------------------------------- #

MAX_SUBMISSION = 16 * 1024
SUBMIT_PER_HOUR = 5  # per source IP
REQUIRED = ("team", "policy", "model_type", "task_id", "task")


def check_submission(sub) -> dict:
    """The fields a team sends, checked for shape and size (the operator still reviews the content)."""
    if not isinstance(sub, dict):
        raise ValueError("submission must be a JSON object")
    missing = [k for k in REQUIRED if not str(sub.get(k) or "").strip() or str(sub.get(k)).startswith("<")]
    if missing:
        raise ValueError(f"fill in {', '.join(missing)} in submission.yaml")
    if len(json.dumps(sub)) > MAX_SUBMISSION:
        raise ValueError("submission.yaml is too large")
    for k in ("objects", "layout"):
        v = sub.get(k)
        if v is not None and not isinstance(v, (dict, str)):
            raise ValueError(f"{k} must be {{\"type\": ..., \"note\": ...}}")
    if not SAFE_NAME.match(str(sub["task_id"]).replace(".", "-")):
        raise ValueError("task_id may use letters, digits, '-' and '_' only")
    if sub.get("placement") is not None:  # submission/placement.yaml: each rollout's objects and prompt
        import rollout_setup  # noqa: PLC0415

        sub["placement"] = rollout_setup.validate(sub["placement"])
        rollout_setup.check_count(sub)
    return sub


class Requests:
    """runs/requests/<id>.json: pending -> approved (with its run) | rejected. The team holds a secret for it."""

    def __init__(self, root: Path):
        self.root = root / "requests"
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.recent: dict[str, list[float]] = collections.defaultdict(list)

    def _path(self, rid: str) -> Path:
        if not SAFE_NAME.match(rid):
            raise KeyError(rid)
        return self.root / f"{rid}.json"

    def get(self, rid: str) -> dict:
        f = self._path(rid)
        if not f.is_file():
            raise KeyError(rid)
        return json.loads(f.read_text())

    def save(self, req: dict) -> None:
        self._path(req["id"]).write_text(json.dumps(req, indent=1))

    def submit(self, sub: dict, ip: str) -> tuple[dict, str]:
        with self.lock:
            now = time.time()
            self.recent[ip] = [t for t in self.recent[ip] if now - t < 3600] + [now]
            if len(self.recent[ip]) > SUBMIT_PER_HOUR:
                raise RuntimeError("too many submissions from this address; try again later")
            secret = secrets.token_hex(16)
            req = {"id": "q" + secrets.token_hex(3), "created": now, "ip": ip, "status": "pending",
                   "secret_sha256": hashlib.sha256(secret.encode()).hexdigest(), "submission": check_submission(sub)}
            self.save(req)
        log.info("submission request %s from %s: %s / %s", req["id"], ip, sub.get("team"), sub.get("policy"))
        return req, secret

    def check_secret(self, req: dict, secret: str) -> bool:
        return bool(secret) and secrets.compare_digest(req["secret_sha256"], hashlib.sha256(secret.encode()).hexdigest())

    def all(self) -> list[dict]:
        return sorted((json.loads(f.read_text()) for f in self.root.glob("q*.json")), key=lambda r: r["created"])


VIEW_KEY = re.compile(r"^[0-9a-f]{32}$")


def _build():
    import importlib

    sys.path.insert(0, str(HERE / "leaderboard"))
    import build

    return importlib.reload(build)  # edits to the leaderboard code show up without restarting the cell


class Organizer:
    """The site's Organizer view: one password (runs/organizer.json, PBKDF2), 12-hour sessions kept in memory (a
    restart signs everyone out), and at most 5 wrong passwords per address per 15 minutes."""

    SESSION_S = 12 * 3600
    ITERATIONS = 200_000

    def __init__(self, root: Path):
        self.file = root / "organizer.json"
        self.sessions: dict[str, float] = {}
        self.failures: dict[str, list[float]] = collections.defaultdict(list)
        self.lock = threading.Lock()

    def set_password(self, password: str) -> None:
        if len(password) < 10:
            raise ValueError("the organizer password needs at least 10 characters")
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, self.ITERATIONS)
        self.file.write_text(json.dumps({"salt": salt.hex(), "hash": digest.hex(), "iterations": self.ITERATIONS}))
        os.chmod(self.file, 0o600)
        with self.lock:
            self.sessions.clear()  # a new password signs everyone out

    def login(self, password: str, ip: str) -> str:
        """A session token, or ValueError / RuntimeError (too many tries)."""
        now = time.time()
        with self.lock:
            self.failures[ip] = [t for t in self.failures[ip] if now - t < 900]
            if len(self.failures[ip]) >= 5:
                raise RuntimeError("too many wrong passwords; try again in 15 minutes")
        try:
            rec = json.loads(self.file.read_text())
        except (OSError, ValueError):
            raise RuntimeError("no organizer password is set (operator_console.py set-password)") from None
        digest = hashlib.pbkdf2_hmac("sha256", str(password).encode(), bytes.fromhex(rec["salt"]), rec["iterations"])
        with self.lock:
            if not secrets.compare_digest(digest.hex(), rec["hash"]):
                self.failures[ip].append(now)
                raise ValueError("wrong password")
            token = secrets.token_urlsafe(32)
            self.sessions = {k: v for k, v in self.sessions.items() if v > now}
            self.sessions[token] = now + self.SESSION_S
        log.info("organizer signed in from %s", ip)
        return token

    def valid(self, token: str) -> bool:
        with self.lock:
            return bool(token) and self.sessions.get(token, 0) > time.time()

    def logout(self, token: str) -> None:
        with self.lock:
            self.sessions.pop(token, None)


ORG_COOKIE = "mb_org"
# (method, path pattern) the Organizer view may call on the operator API: nothing that moves the arm
ORG_ADMIN = [("GET", r"status"), ("GET", r"requests"), ("POST", r"requests/q[0-9a-f]+/(approve|reject)"),
             ("POST", r"runs/r[0-9a-z-]+/(hide|show)"), ("GET", r"site"), ("POST", r"site")]


def org_runs(store: Store) -> list[dict]:
    """Every run for the Organizer view, newest first, with its team's results link key."""
    out = []
    for r in sorted(store.runs.values(), key=lambda r: r["created"], reverse=True):
        ts, sub = r["trials"], r["submission"]
        scored = [t for t in ts if t.get("outcome")]
        out.append({"run_id": r["run_id"], "created": r["created"], "team": sub.get("team", "?"),
                    "policy": sub.get("policy", "?"), "task_id": r.get("task_id"), "backend": r.get("backend"),
                    "hidden": bool(r.get("hidden")), "view_key": r["view_key"], "trials": len(ts),
                    "scored": len(scored), "successes": sum(bool(t["outcome"]["success"]) for t in scored),
                    "open": sum(t["status"] in ("waiting", "preparing", "ready", "running") for t in ts)})
    return out


def org_results(store: Store) -> dict:
    """Every run's results (hidden and mock included), each row loading its videos with its own view key."""
    runs = list(store.runs.values())
    d = _build().data(runs, include_mock=True, examples=False, include_hidden=True)
    by_id = {r["run_id"]: r for r in runs}
    for row in d["rows"]:
        run = by_id[row["run_id"]]
        row |= {"hidden": bool(run.get("hidden")), "key": run["view_key"], "base": f"results/{run['view_key']}/"}
    return d


def leaderboard_html(store: Store, include_mock: bool = False) -> str:
    """The real-robot runs; a mock server (a rehearsal) lists its own mock runs too."""
    return _build().build(list(store.runs.values()), include_mock)


def site_html(store: Store, include_mock: bool = False) -> str:
    """The public site: the leaderboard once revealed, else the page without it (teams load their own runs)."""
    if store.site.get("reveal"):
        return leaderboard_html(store, include_mock)
    b = _build()
    shown = [r for r in store.runs.values() if r.get("showcase") and not r.get("hidden")]  # examples, for everyone
    d = b.data(shown, include_mock=True, examples=False)  # together, so each task's clip is its latest success
    keys = {r["run_id"]: r["view_key"] for r in shown}
    rows = [row | {"base": f"results/{keys[row['run_id']]}/", "key": keys[row["run_id"]], "showcase": True} for row in d["rows"]]
    return b.page({"tasks": {}, "rows": [], "showcase": {"tasks": d["tasks"], "rows": rows},
                   "built": time.strftime("%Y-%m-%d %H:%M")}, "private")


def results_json(run: dict) -> bytes:
    """One run's results, for its team's My results."""
    return json.dumps(_build().run_data(run)).encode()


def public(t: dict) -> dict:
    """What a team sees about a trial. `instruction` is what the operator asked for this trial (None until ready);
    `scored` False on an ended trial means the operator may still void it and run it again."""
    return {**{k: t[k] for k in ("trial_id", "index", "layout", "status")}, "instruction": t.get("instruction"),
            "scored": t.get("outcome") is not None}


# --------------------------------------------------------------------------------------------------------- #
#  The cell: one robot, one active trial, an action queue played at each chunk's dt                       #
# --------------------------------------------------------------------------------------------------------- #

class Cell:
    def __init__(self, backend, store: Store):
        self.backend, self.store = backend, store
        self.lock = threading.Condition()
        self.queue: collections.deque = collections.deque()
        self.active: str | None = None  # trial_id that owns the robot
        self.state = "idle"  # idle | homing | running | fault | stopped
        self.last_obs_t: float | None = None
        self.recorder = None  # recorder.TrialRecorder, set by main() unless --no-record
        self.push: dict | None = None  # where recorded rollouts are copied (the relay VM), set by main()
        self.cameras: tuple = CAMERAS
        self.depth_cameras: tuple = ()  # cameras whose depth teams may request (main(): --depth)
        self.sync_lock = threading.Lock()  # one dataset rsync at a time
        self.max_trial_s: float | None = None  # None: a trial runs until the operator scores or stops it
        threading.Thread(target=self._play, daemon=True).start()

    # ---- operator ----
    def prepare(self, trial_id: str) -> None:
        run, t = self.store.trial(trial_id)
        with self.lock:
            if self.active and self.active != trial_id:
                raise RuntimeError(f"{self.active} still owns the robot")
            if self.state in ("fault", "stopped"):
                raise RuntimeError(f"robot is {self.state}; recover first")
            self.active, self.state = trial_id, "homing"
            t["status"] = "preparing"
            self.store.save(run)
        depth = getattr(self.backend, "depth", None)
        if isinstance(depth, dict):  # start the depth streams this run asked for now, so they are up when it starts
            for c in run["submission"].get("depth") or []:
                if c in depth:
                    depth[c].demand()
        threading.Thread(target=self._home_then, args=(t["layout_index"],), daemon=True).start()

    def ready(self, trial_id: str, instruction: str | None = None) -> None:
        run, t = self.store.trial(trial_id)
        with self.lock:
            if t["status"] != "preparing" or self.state != "idle" or self.active != trial_id:
                raise RuntimeError(f"trial {t['status']}, robot {self.state}: wait for home and layout first")
            t["status"] = "ready"
            t["instruction"] = (instruction or "").strip() or TASK["instruction"]
            self.store.save(run)

    def finish(self, trial_id: str, outcome: dict) -> dict:
        run, t = self.store.trial(trial_id)
        with self.lock:
            if t["status"] == "running":
                self._end(run, t, "done", "operator")
            stage = int(outcome.get("stage", 0))
            top = int(outcome.get("success_stage", 4))  # the run's task card decides how many stages there are
            st = t["stats"]
            t_success = outcome.get("t_success")
            t["outcome"] = {
                "stage": stage,
                "success_stage": top,
                "success": stage >= top and not outcome.get("intervention") and not st.get("fault"),
                "intervention": bool(outcome.get("intervention")),
                "fault": st.get("fault"),
                "cycle_time_s": round(t_success - st["t_start"], 2) if t_success and st.get("t_start") else None,
                "notes": outcome.get("notes", ""),
                "scored_at": time.time(),
            }
            self.store.save(run)
            self.publish()
            return t

    def recorded(self, run_id: str, trial_id: str) -> None:
        """The recorder wrote this trial's rollout: push it to the relay, mark it, refresh the leaderboard."""
        if self.push:  # {"host": "ubuntu@...", "key": ..., "dir": "cell_relay/rollouts"}
            src = self.store.root / "rollouts" / run_id / trial_id
            self._push([f for f in sorted(src.iterdir()) if f.is_file()], f"{self.push['dir']}/{run_id}/{trial_id}",
                       f"rollout {trial_id}")
        run, t = self.store.trial(trial_id)
        try:
            cams = json.loads((self.store.root / "rollouts" / run_id / trial_id / "meta.json").read_text())["cameras"]
        except Exception:  # noqa: BLE001
            cams = list(self.cameras)
        with self.lock:
            t["video"] = cams  # the cameras recorded (see recorder.py)
            self.store.mark_episodes(run)
            self.store.save(run)
        self.sync_dataset(run_id)
        self.publish()

    def sync_dataset(self, run_id: str) -> None:
        """rsync the run's LeRobotDataset to the relay (for the dataset viewer), in the background, one at a time."""
        src = self.store.root / "datasets" / run_id
        if not self.push or not src.is_dir():
            return

        def go():
            with self.sync_lock:
                remote = f"{Path(self.push['dir']).parent / 'datasets'}/{run_id}/"
                ssh = f"ssh -i {self.push['key']} -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10"
                try:
                    subprocess.run(["rsync", "-a", "--delete", "--mkpath", "-e", ssh, f"{src}/", f"{self.push['host']}:{remote}"],
                                   check=True, timeout=900, capture_output=True)
                    log.info("dataset %s synced to %s", run_id, self.push["host"])
                except Exception as e:  # noqa: BLE001
                    log.warning("dataset %s not synced: %s", run_id, getattr(e, "stderr", b"") or e)

        threading.Thread(target=go, daemon=True).start()

    def publish(self) -> None:
        """Rewrite leaderboard/index.html and each run's results/<view_key>.json from the runs, and push the changed
        results to the relay (best effort: a scoring never fails because of it)."""
        try:
            (HERE / "leaderboard" / "index.html").write_text(
                leaderboard_html(self.store, isinstance(self.backend, MockBackend)))
        except Exception as e:  # noqa: BLE001
            log.warning("leaderboard not rebuilt: %s", e)
        changed = []
        try:
            out = self.store.root / "results"
            out.mkdir(exist_ok=True)
            for run in list(self.store.runs.values()):
                f, body = out / f"{run['view_key']}.json", results_json(run)
                if not f.is_file() or f.read_bytes() != body:
                    f.write_bytes(body)
                    changed.append(f)
        except Exception as e:  # noqa: BLE001
            log.warning("results not rebuilt: %s", e)
        if changed and self.push:
            remote = str(Path(self.push["dir"]).parent / "results")
            threading.Thread(target=self._push, args=(changed, remote, "results"), daemon=True).start()

    def _push(self, files: list[Path], remote: str, what: str) -> bool:
        """scp these files into remote (a folder on the relay VM)."""
        ssh = ["-i", str(self.push["key"]), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        try:
            subprocess.run(["ssh", *ssh, self.push["host"], f"mkdir -p {remote}"], check=True, timeout=30)
            subprocess.run(["scp", "-q", *ssh, *map(str, files), f"{self.push['host']}:{remote}/"], check=True, timeout=300)
            log.info("%s pushed to %s", what, self.push["host"])
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("%s not pushed: %s", what, e)
            return False

    def abort(self, trial_id: str) -> None:
        run, t = self.store.trial(trial_id)
        with self.lock:
            if t["status"] in OPEN:
                self._end(run, t, "cancelled", "operator abort")

    def stop(self) -> None:
        with self.lock:
            self.queue.clear()
            self.state = "stopped"
            if self.active:
                run, t = self.store.trial(self.active)
                if t["status"] == "running":
                    self._end(run, t, "failed", "stop", home=False)

    def retry(self, trial_id: str, reason: str = "") -> None:
        """Void a finished attempt (kept under "attempts", never scored) and put the trial back in the queue."""
        run, t = self.store.trial(trial_id)
        with self.lock:
            if t["status"] in OPEN:
                raise RuntimeError(f"trial is {t['status']}; only a finished trial can be retried")
            t.setdefault("attempts", []).append({k: t.get(k) for k in ("status", "instruction", "stats", "outcome")}
                                                | {"voided_at": time.time(), "reason": reason})
            t.update(status="waiting", instruction=None, outcome=None, stats={})
            if self.active == trial_id and self.state not in ("fault", "stopped"):
                self.active = None
            self.store.save(run)
        self.publish()

    def recover(self) -> None:
        with self.lock:
            self.backend.recover()
            self.state = "idle"
            self.active = None

    # ---- team ----
    def start(self, trial_id: str) -> None:
        run, t = self.store.trial(trial_id)
        with self.lock:
            if t["status"] != "ready":
                raise RuntimeError(f"trial is {t['status']}, not ready")
            self.queue.clear()
            t["status"] = "running"
            t["stats"].update(t_start=time.time(), actions=0, chunks=0, latency_s=[])
            t.pop("video", None)
            self.state = "running"
            self.store.save(run)
        if self.recorder:
            self.recorder.start(run["run_id"], trial_id, t.get("instruction") or "", list(self.cameras))

    def post(self, chunk: list, dt: float, kind: str = "joints", obs_t: float | None = None,
             replace: bool = False) -> int:
        """Queue a chunk; returns how many of its leading rows were skipped.

        replace: the chunk was planned from the observation taken at obs_t (the cell clock), while the previous one
        was still playing. Drop what is still queued, and skip the chunk's rows whose time has already passed
        (row i is meant for obs_t + i * dt), so the arm carries on from where the new plan is now.
        """
        with self.lock:
            if self.state != "running" or not self.active:
                raise RuntimeError(f"no trial is running (robot {self.state})")
            skip = 0
            if replace:
                self.queue.clear()
                if obs_t is not None and 0 <= time.time() - obs_t < 30:
                    skip = min(int((time.time() - obs_t) / dt), len(chunk) - 1)  # keep at least the last row
                chunk = chunk[skip:]
            if len(self.queue) + len(chunk) > MAX_QUEUE:
                raise RuntimeError("action queue is full")
            st = self.store.trial(self.active)[1]["stats"]
            st["chunks"] += 1
            st["skipped"] = st.get("skipped", 0) + skip
            if self.last_obs_t:
                st["latency_s"].append(round(time.time() - (obs_t or self.last_obs_t), 4))
            self.queue.extend((a, dt, kind) for a in chunk)
            self.lock.notify_all()
            return skip

    # ---- internals ----
    def _end(self, run, t, status: str, reason: str, home: bool = True) -> None:
        self.queue.clear()
        if self.recorder:
            self.recorder.stop(reason)
        t["status"] = status
        t["stats"].update(t_end=time.time(), end_reason=reason)
        self.store.save(run)
        if home and self.state == "running":  # a trial that never ran is already at home
            self.state = "homing"
            threading.Thread(target=self._home_then, args=(t["layout_index"], True), daemon=True).start()
        elif self.state not in ("fault", "stopped", "homing"):
            self.active = None  # a fault / stop keeps the trial attached until the operator recovers

    def _home_then(self, layout_index: int, release: bool = False) -> None:
        try:
            self.backend.home(layout_index)
        except BaseException as e:  # go_home raises SystemExit on a refused path
            log.error("homing failed: %s", e)
            with self.lock:
                self.state = "fault"
            return
        with self.lock:
            if self.state == "homing":
                self.state = "idle"
            if release or (self.active and self.store.trial(self.active)[1]["status"] not in OPEN):
                self.active = None

    def _play(self) -> None:
        t_next = time.perf_counter()
        while True:
            with self.lock:
                while not (self.queue and self.state == "running"):
                    self.lock.wait(0.1)
                    self._watch()
                    t_next = time.perf_counter()
                action, dt, kind = self.queue.popleft()
            try:
                (self.backend.send_tcp if kind == "tcp" else self.backend.send)(action)
            except Exception as e:
                log.error("send failed: %s", e)
            # Never let the bookkeeping end this thread: it is the only one that plays actions. (2026-10-27: a trial
            # voided while a gripper row was waiting for the arm had no "actions" left, and the KeyError stopped
            # every later action until a restart.)
            try:
                if self.recorder:
                    self.recorder.action(action, kind)
                with self.lock:
                    if self.active:
                        st = self.store.trial(self.active)[1]["stats"]
                        st["actions"] = st.get("actions", 0) + 1
                    self._watch()
            except Exception as e:  # noqa: BLE001
                log.error("after a send: %s", e)
            t_next += dt
            if t_next < time.perf_counter():  # a send that blocked (the gripper waiting for the arm): resume the
                t_next = time.perf_counter() + dt  # 10 Hz pace from now, never fire the queued rows in a burst
            time.sleep(max(0.0, t_next - time.perf_counter()))

    def _watch(self) -> None:
        """Called with the lock held: end the running trial on a controller fault or the time limit."""
        if not self.active:
            return
        run, t = self.store.trial(self.active)
        if t["status"] != "running":
            return
        fault = self.backend.faulted()
        if fault:
            t["stats"]["fault"] = fault
            self.state = "fault"
            self._end(run, t, "failed", "controller fault", home=False)
        elif self.max_trial_s and time.time() - t["stats"]["t_start"] > self.max_trial_s:
            self._end(run, t, "done", "time limit")


# --------------------------------------------------------------------------------------------------------- #
#  HTTP                                                                                                    #
# --------------------------------------------------------------------------------------------------------- #

def validate_tcp(rows, dt) -> tuple[list, float]:
    a = np.asarray(rows, dtype=float)
    if a.ndim != 2 or a.shape[1] != 7 or not 1 <= len(a) <= MAX_CHUNK:
        raise ValueError(f"tcp must have shape (N, 7) [x, y, z mm, w, p, r deg, gripper], 1 <= N <= {MAX_CHUNK}; got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("tcp contains NaN or inf")
    for k, (name, (lo, hi)) in enumerate(TCP_BOX_MM.items()):
        if a[:, k].min() < lo or a[:, k].max() > hi:
            raise ValueError(f"tcp {name} outside [{lo}, {hi}] mm (UF0)")
    if np.abs(a[:, 3:6]).max() > 360:
        raise ValueError("tcp w, p, r must be within +-360 deg")
    a[:, 6] = np.clip(a[:, 6], 0.0, 1.0)
    dt = float(dt)
    if not DT_RANGE_S[0] <= dt <= DT_RANGE_S[1]:
        raise ValueError(f"dt must be in [{DT_RANGE_S[0]}, {DT_RANGE_S[1]}] s")
    return a.tolist(), dt


def validate(joints, dt) -> tuple[list, float]:
    a = np.asarray(joints, dtype=float)
    if a.ndim != 2 or a.shape[1] != 7 or not 1 <= len(a) <= MAX_CHUNK:
        raise ValueError(f"joints must have shape (N, 7) with 1 <= N <= {MAX_CHUNK}, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("joints contain NaN or inf")
    for k, (lo, hi) in enumerate(JOINT_LIMITS_DEG):
        if a[:, k].min() < lo or a[:, k].max() > hi:
            raise ValueError(f"J{k + 1} outside [{lo}, {hi}] deg")
    a[:, 6] = np.clip(a[:, 6], 0.0, 1.0)
    dt = float(dt)
    if not DT_RANGE_S[0] <= dt <= DT_RANGE_S[1]:
        raise ValueError(f"dt must be in [{DT_RANGE_S[0]}, {DT_RANGE_S[1]}] s")
    return a.tolist(), dt


def encode_image(rgb: np.ndarray, w: int, h: int, fmt: str = "png", quality: int = 90) -> str:
    """Resize (area) and encode for the wire: lossless PNG (default) or JPEG, ~10x smaller for remote clients."""
    small = cv2.cvtColor(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
    ok, buf = (cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality]) if fmt == "jpeg"
               else cv2.imencode(".png", small))
    return base64.b64encode(buf.tobytes()).decode()


def make_handler(cell: Cell, store: Store, admin_token: str, requests_q: "Requests", org: Organizer):
    from calendar_book import Calendar  # noqa: PLC0415  (slot booking: the site's Calendar tab)

    cal = Calendar(store.root)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

        def _send(self, code: int, body, headers: dict | None = None) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _org_token(self) -> str:
            for part in self.headers.get("Cookie", "").split(";"):
                k, _, v = part.strip().partition("=")
                if k == ORG_COOKIE:
                    return v
            return ""

        def _calendar(self, method: str, parts: list[str]):
            """The site's Calendar tab: open slots for everyone; look up, book and cancel by request id."""
            if method == "GET" and parts == []:
                return self._send(200, cal.view(), {"Cache-Control": "no-store"})
            cal.limit(self._client_ip())
            if method == "GET" and len(parts) == 2 and parts[0] == "requests":
                req = requests_q.get(parts[1])
                sub = req.get("submission", {})
                return self._send(200, {"request_id": req["id"], "status": req["status"], "team": sub.get("team", "?"),
                                        "policy": sub.get("policy", "?"), "booking": cal.booking(req["id"])})
            body = self._body() if method == "POST" else {}
            if method == "POST" and parts == ["book"]:
                req = requests_q.get(str(body.get("request_id", "")).strip())
                t = cal.book(req, body.get("t"))
                log.info("request %s (%s) booked %s", req["id"], req["submission"].get("team"), t)
                return self._send(200, {"request_id": req["id"], "booking": t})
            if method == "POST" and parts == ["cancel"]:
                rid = str(body.get("request_id", "")).strip()
                requests_q.get(rid)
                cal.cancel(rid)
                log.info("request %s cancelled its booking", rid)
                return self._send(200, {"request_id": rid, "booking": None})
            return self._send(404, {"error": "not found"})

        def _org(self, method: str, parts: list[str]):
            """The Organizer view's API (see ORG_ADMIN): a session cookie, and a custom header on every POST so
            another site cannot post here from a visitor's browser."""
            nocache = {"Cache-Control": "no-store"}
            if method == "POST" and self.headers.get("X-Requested-With") != "manipbench":
                return self._send(403, {"error": "missing X-Requested-With"})
            if method == "POST" and parts == ["login"]:
                try:
                    token = org.login(self._body().get("password", ""), self._client_ip())
                except ValueError as e:
                    return self._send(401, {"error": str(e)})
                except RuntimeError as e:
                    return self._send(429, {"error": str(e)})
                return self._send(200, {"ok": True}, nocache | {"Set-Cookie": (
                    f"{ORG_COOKIE}={token}; Path=/; Max-Age={Organizer.SESSION_S}; HttpOnly; Secure; SameSite=Strict")})
            token = self._org_token()
            if method == "GET" and parts == ["session"]:
                return self._send(200, {"organizer": org.valid(token)}, nocache)
            if not org.valid(token):
                return self._send(401, {"error": "sign in as organizer"})
            if method == "POST" and parts == ["logout"]:
                org.logout(token)
                return self._send(200, {"ok": True}, nocache | {
                    "Set-Cookie": f"{ORG_COOKIE}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Strict"})
            if method == "GET" and parts == ["runs"]:
                return self._send(200, org_runs(store), nocache)
            if method == "GET" and parts == ["results"]:
                return self._send(200, org_results(store), nocache)
            if method == "GET" and parts == ["calendar"]:
                return self._send(200, cal.org_view(requests_q, store), nocache)
            if method == "POST" and parts == ["calendar", "open"]:
                body = self._body()
                return self._send(200, cal.set_open(body.get("times", []), bool(body.get("open", True))))
            if method == "POST" and parts == ["calendar", "cancel"]:
                cal.cancel(str(self._body().get("request_id", "")))
                return self._send(200, {"ok": True})
            path = "/".join(parts)
            if any(m == method and re.fullmatch(pat, path) for m, pat in ORG_ADMIN):
                return self._admin(method, parts)
            return self._send(404, {"error": "not found"})

        def _send_html(self, html: str) -> None:
            data = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, f: Path | None) -> None:
            if f is None:
                return self._send(404, {"error": "not found"})
            self._send_bytes(f.read_bytes(), ROLLOUT_TYPES[f.suffix])

        def _send_bytes(self, data: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def _client_ip(self) -> str:
            """The team's address: the relay's X-Forwarded-For when the request came through the tunnel."""
            peer = self.client_address[0]
            fwd = self.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            return fwd if fwd and (peer.startswith("127.") or peer == self.server.server_address[0]) else peer

        def _run(self) -> dict | None:
            auth = self.headers.get("Authorization", "")
            token = auth[7:] if auth.startswith("Bearer ") else ""
            return next((r for r in store.runs.values() if token and secrets.compare_digest(r["token"], token)), None)

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def _route(self, method: str) -> None:
            url = urlparse(self.path)
            q = parse_qs(url.query)
            parts = [p for p in url.path.split("/") if p]
            try:
                if parts[:1] == ["org"]:
                    return self._org(method, parts[1:])
                if parts[:1] == ["admin"]:
                    if not secrets.compare_digest(self.headers.get("x-admin-token", ""), admin_token):
                        return self._send(403, {"error": "bad admin token"})
                    return self._admin(method, parts[1:])
                if method == "GET" and len(parts) == 4 and parts[0] == "rollouts":
                    return self._send_file(rollout_file(store.root / "rollouts", parts[1:]))
                if method == "GET" and parts == ["leaderboard"]:
                    return self._send_html(leaderboard_html(store, isinstance(cell.backend, MockBackend)))
                if method == "GET" and parts == ["site"]:
                    return self._send_html(site_html(store, isinstance(cell.backend, MockBackend)))
                if method == "GET" and parts == ["site.json"]:
                    return self._send(200, {"reveal": bool(store.site.get("reveal"))})
                if method == "GET" and parts[:1] == ["results"] and len(parts) in (2, 6):
                    key = parts[1].removesuffix(".json") if len(parts) == 2 else parts[1]
                    run = store.by_key(key) if VIEW_KEY.match(key) else None
                    if run is None or (len(parts) == 2) != parts[1].endswith(".json"):
                        return self._send(404, {"error": "not found"})
                    if len(parts) == 2:
                        return self._send_bytes(results_json(run), "application/json")
                    if parts[2] != "rollouts" or parts[3] != run["run_id"]:
                        return self._send(404, {"error": "not found"})
                    return self._send_file(rollout_file(store.root / "rollouts", parts[3:]))
                if parts[:2] != ["api", "v1"]:
                    return self._send(404, {"error": "not found"})
                if method == "POST" and parts[2:] == ["submit"]:
                    if int(self.headers.get("Content-Length") or 0) > MAX_SUBMISSION:
                        return self._send(413, {"error": "submission.yaml is too large"})
                    req, secret = requests_q.submit(self._body().get("submission"), self._client_ip())
                    return self._send(200, {"request_id": req["id"], "secret": secret, "status": req["status"]})
                if parts[2:3] == ["calendar"]:
                    return self._calendar(method, parts[3:])
                if method == "GET" and len(parts) == 4 and parts[2] == "requests":
                    req = requests_q.get(parts[3])
                    if not requests_q.check_secret(req, self.headers.get("X-Request-Secret", "")):
                        return self._send(403, {"error": "wrong secret for this request"})
                    out = {"request_id": req["id"], "status": req["status"], "reason": req.get("reason", "")}
                    if req["status"] == "approved":
                        run = store.runs[req["run_id"]]
                        out |= {"run_id": run["run_id"], "token": run["token"], "view_key": run["view_key"]}
                    return self._send(200, out)
                run = self._run()
                if run is None:
                    return self._send(401, {"error": "unknown token"})
                return self._api(method, parts[2:], q, run)
            except (KeyError, IndexError):
                return self._send(404, {"error": "not found"})
            except (ValueError, RuntimeError) as e:
                return self._send(400, {"error": str(e)})

        def _api(self, method, parts, q, run):
            mine = cell.active if cell.active and cell.active.startswith(run["run_id"] + "-") else None
            if method == "GET" and parts == ["run"]:
                return self._send(200, {"run_id": run["run_id"], "view_key": run["view_key"],
                                        "trials": [public(t) for t in run["trials"]]})
            if method == "GET" and parts == ["time"]:
                return self._send(200, {"t": time.time()})
            if parts[:1] == ["trials"] and parts[1].startswith(run["run_id"] + "-"):
                if method == "GET" and len(parts) == 2:
                    return self._send(200, public(store.trial(parts[1])[1]))
                if method == "POST" and parts[2:] == ["start"]:
                    cell.start(parts[1])
                    return self._send(200, {"ok": True})
            if method == "GET" and parts == ["observation"]:
                cams = [c for c in ",".join(q.get("cameras", ["table,side,wrist"])).split(",") if c]
                bad = [c for c in cams if c not in cell.cameras]
                if bad:
                    raise ValueError(f"cameras {bad} not available on this cell; available {list(cell.cameras)}")
                w, h = int(q.get("width", ["640"])[0]), int(q.get("height", ["360"])[0])
                if not (16 <= w <= 1920 and 16 <= h <= 1080):
                    raise ValueError("width/height out of range")
                fmt, quality = q.get("format", ["png"])[0], int(q.get("quality", ["90"])[0])
                if fmt not in ("png", "jpeg") or not 50 <= quality <= 100:
                    raise ValueError("format must be png or jpeg, quality 50-100")
                depth = [c for c in ",".join(q.get("depth", [""])).split(",") if c]
                bad = [c for c in depth if c not in cell.depth_cameras]
                if bad:
                    raise ValueError(f"no depth for {bad} on this cell; depth is available for {list(cell.depth_cameras)}")
                joints, pose, images = cell.backend.observe(cams)
                depths = cell.backend.observe_depth(depth) if depth else {}
                cell.last_obs_t = time.time()
                trial = public(store.trial(mine)[1]) if mine else None
                out = {"t": cell.last_obs_t, "status": cell.state if mine else "idle", "trial": trial,
                       "queued": len(cell.queue) if mine else 0, "joints": joints, "tcp_pose": pose,
                       "format": fmt, "images": {c: encode_image(im, w, h, fmt, quality) for c, im in images.items()}}
                if depth:  # 16-bit PNG, millimetres, at the depth stream's own size: lossless, and its calibration holds
                    out["depth"] = {c: base64.b64encode(cv2.imencode(".png", d)[1].tobytes()).decode() for c, d in depths.items()}
                return self._send(200, out)
            if method == "POST" and parts == ["actions"]:
                if not mine:
                    raise RuntimeError("none of your trials is running")
                body = self._body()
                if ("joints" in body) == ("tcp" in body):
                    raise ValueError('send exactly one of "joints" or "tcp"')
                obs_t = body.get("obs_t")
                obs_t = float(obs_t) if isinstance(obs_t, (int, float)) else None
                if "tcp" in body:
                    chunk, dt = validate_tcp(body["tcp"], body.get("dt", 0.1))
                    skipped = cell.post(chunk, dt, "tcp", obs_t, bool(body.get("replace")))
                else:
                    chunk, dt = validate(body["joints"], body.get("dt", 0.1))
                    skipped = cell.post(chunk, dt, "joints", obs_t, bool(body.get("replace")))
                return self._send(200, {"ok": True, "queued": len(cell.queue), "skipped": skipped})
            return self._send(404, {"error": "not found"})

        def _admin(self, method, parts):
            if method == "GET" and parts == ["status"]:
                active = public(store.trial(cell.active)[1]) if cell.active else None
                return self._send(200, {"state": cell.state, "queued": len(cell.queue), "active": active,
                                        "truncated_frames": dict(getattr(cell.backend, "truncated", {})),
                                        "camera_reopens": [e for w in getattr(cell.backend, "watch", {}).values()
                                                           for e in w.events],
                                        "depth_streams": {c: d.running for c, d in getattr(cell.backend, "depth", {}).items()}
                                        if isinstance(getattr(cell.backend, "depth", None), dict) else {},
                                        "backend": type(cell.backend).__name__, "z_min_mm": cell.backend.z_min_mm,
                                        "z_offset_mm": getattr(getattr(cell.backend, "robot", None), "config", None)
                                        and cell.backend.robot.config.joint_z_offset_mm,
                                        "motion_limits": getattr(cell.backend, "limits", None),
                                        "surface_z_mm": SURFACE_Z_MM, "z_min_hard_mm": Z_MIN_HARD_MM})
            if method == "POST" and parts == ["config"]:
                z = float(self._body()["z_min_mm"])
                if not Z_MIN_HARD_MM <= z <= 0.0:
                    raise ValueError(f"z_min_mm must be in [{Z_MIN_HARD_MM}, 0]")
                with cell.lock:
                    if cell.state == "running":
                        raise RuntimeError("change the floor between trials, not during one")
                    cell.backend.set_z_min(z)
                log.info("TCP floor set to %.1f mm", z)
                return self._send(200, {"ok": True, "z_min_mm": z})
            if method == "GET" and parts == ["runs"]:
                return self._send(200, list(store.runs.values()))
            if method == "POST" and parts == ["runs"]:
                return self._send(200, store.create(self._body(), type(cell.backend).__name__))
            if method == "GET" and parts == ["requests"]:
                return self._send(200, [{k: v for k, v in r.items() if k != "secret_sha256"} for r in requests_q.all()])
            if method == "POST" and len(parts) == 3 and parts[0] == "requests" and parts[2] in ("approve", "reject"):
                req = requests_q.get(parts[1])
                if req["status"] != "pending":
                    raise RuntimeError(f"request {req['id']} is already {req['status']}")
                body = self._body()
                if parts[2] == "approve":
                    sub = dict(req["submission"])
                    asked = sub.get("rollouts")  # what the team asked for; the operator's --trials wins
                    sub["n_trials"] = int(body.get("trials") or (asked if isinstance(asked, int) and 1 <= asked <= 50 else 10))
                    run = store.create(sub, type(cell.backend).__name__)
                    req.update(status="approved", run_id=run["run_id"], decided=time.time())
                    log.info("request %s approved: run %s", req["id"], run["run_id"])
                else:
                    req.update(status="rejected", reason=str(body.get("reason", "")), decided=time.time())
                    cal.release(req["id"])  # a rejected request gives its slot back
                requests_q.save(req)
                return self._send(200, {k: v for k, v in req.items() if k != "secret_sha256"})
            if method == "POST" and len(parts) == 3 and parts[0] == "runs" and parts[2] in ("hide", "show"):
                run = store.runs[parts[1]]
                run["hidden"] = parts[2] == "hide"
                store.save(run)
                cell.publish()
                return self._send(200, {"ok": True, "run_id": run["run_id"], "hidden": run["hidden"]})
            if method == "POST" and parts == ["organizer-password"]:
                org.set_password(str(self._body().get("password", "")))
                log.info("organizer password changed")
                return self._send(200, {"ok": True})
            if method == "GET" and parts == ["depth"]:  # for depth_calibrate.py: one depth frame and the joints with it
                cam = parse_qs(urlparse(self.path).query).get("camera", ["table"])[0]
                if cam not in cell.depth_cameras:
                    raise ValueError(f"no depth for {cam}; on: {list(cell.depth_cameras)}")
                joints, _pose, _ = cell.backend.observe([])
                d = cell.backend.observe_depth([cam])[cam]
                return self._send(200, {"t": time.time(), "joints": joints,
                                        "depth": base64.b64encode(cv2.imencode(".png", d)[1].tobytes()).decode()})
            if method == "GET" and parts == ["site"]:
                return self._send(200, store.site)
            if method == "POST" and parts == ["site"]:
                site = store.set_site(reveal=bool(self._body().get("reveal")))
                log.info("public leaderboard %s", "revealed" if site["reveal"] else "hidden")
                return self._send(200, site)
            if method == "POST" and parts == ["stop"]:
                cell.stop()
                return self._send(200, {"ok": True})
            if method == "POST" and parts == ["recover"]:
                cell.recover()
                return self._send(200, {"ok": True})
            if method == "POST" and len(parts) == 3 and parts[0] == "trials":
                trial_id, verb = parts[1], parts[2]
                if verb == "finish":
                    return self._send(200, cell.finish(trial_id, self._body()))
                if verb == "ready":
                    cell.ready(trial_id, self._body().get("instruction"))
                    return self._send(200, {"ok": True})
                if verb == "retry":
                    cell.retry(trial_id, self._body().get("reason", ""))
                    return self._send(200, {"ok": True})
                {"prepare": cell.prepare, "abort": cell.abort}[verb](trial_id)
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})

    return Handler


OP_HOSTS = ("127.0.0.1", "localhost")


def make_op_handler(base: type, cell: Cell, store: Store, port: int) -> type:
    """The Operate page and its /op API, on a localhost-only port (see the docstring). Each step reuses the operator
    API (/admin) and operator_console's own scoring, so the page and the terminal console behave alike."""
    import operator_console as oc  # noqa: PLC0415  (its stages, setup text and score rules)
    import rollout_setup  # noqa: PLC0415
    import task_spec  # noqa: PLC0415

    page = HERE / "operator.html"

    def run_view(run: dict) -> dict:
        stages, top = oc.scoring(run)
        sub = run["submission"]
        setup = {}
        for lay in dict.fromkeys(t["layout"] for t in run["trials"]):
            card = task_spec.spec(sub)["card"]
            setup[lay] = oc.setup_text(run, lay) if card else "\n".join(task_spec.setup_lines(sub)[1:])
        placement, source = rollout_setup.for_run(sub)
        place = {t["trial_id"]: rollout_setup.entry(placement, t["index"]) for t in run["trials"]}
        return {"run_id": run["run_id"], "team": sub.get("team", "?"), "policy": sub.get("policy", "?"),
                "placement": source,
                "task_id": run.get("task_id"), "task": sub.get("task") or sub.get("instruction") or "",
                "action": sub.get("action"), "cameras": sub.get("cameras"), "stages": stages, "success_stage": top,
                "setup": setup, "hidden": bool(run.get("hidden")),
                "trials": [{k: t.get(k) for k in ("trial_id", "index", "layout", "status", "instruction", "outcome")}
                           | {"end_reason": t.get("stats", {}).get("end_reason"), "t_start": t.get("stats", {}).get("t_start"),
                              "actions": t.get("stats", {}).get("actions"), "attempts": len(t.get("attempts", [])),
                              "prompt": place[t["trial_id"]]["prompt"], "place": rollout_setup.lines(place[t["trial_id"]])}
                           for t in run["trials"]]}

    class OpHandler(base):
        def _route(self, method: str) -> None:
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            host = self.headers.get("Host", "").rsplit(":", 1)[0].strip("[]")
            if host not in OP_HOSTS:  # a page that rebinds its own name to 127.0.0.1 still sends its own Host
                return self._send(403, {"error": "open the Operate page as http://127.0.0.1:%d" % port})
            if method == "POST" and self.headers.get("X-Requested-With") != "manipbench":
                return self._send(403, {"error": "missing X-Requested-With"})
            try:
                if method == "GET" and parts in ([], ["operator"]):
                    return self._send_html(page.read_text())
                if parts[:1] != ["op"]:
                    return self._send(404, {"error": "not found"})
                return self._op(method, parts[1:], parse_qs(url.query))
            except (KeyError, IndexError, StopIteration):
                return self._send(404, {"error": "not found"})
            except (ValueError, RuntimeError) as e:
                return self._send(400, {"error": str(e)})

        def _op(self, method, parts, q):
            if method == "GET" and parts == ["state"]:
                active = public(store.trial(cell.active)[1]) if cell.active else None
                runs = [{"run_id": r["run_id"], "team": r["submission"].get("team", "?"),
                         "policy": r["submission"].get("policy", "?"), "task_id": r.get("task_id"), "created": r["created"],
                         "hidden": bool(r.get("hidden")), "trials": len(r["trials"]),
                         "scored": sum(bool(t.get("outcome")) for t in r["trials"]),
                         "open": sum(t.get("outcome") is None and t["status"] != "cancelled" for t in r["trials"])}
                        for r in sorted(store.runs.values(), key=lambda r: r["created"], reverse=True)]
                return self._send(200, {"state": cell.state, "active": active, "queued": len(cell.queue),
                                        "backend": type(cell.backend).__name__, "z_min_mm": cell.backend.z_min_mm,
                                        "cameras": list(cell.cameras), "time": time.time(), "runs": runs,
                                        "camera_reopens": [e for w in getattr(cell.backend, "watch", {}).values()
                                                           for e in w.events]})
            if method == "GET" and len(parts) == 2 and parts[0] == "runs":
                return self._send(200, run_view(store.runs[parts[1]]))
            if method == "GET" and len(parts) == 3 and parts[0] == "trials" and parts[2] == "overlay":
                run, t = store.trial(parts[1])  # where this rollout's objects go, drawn on the table image
                placement, _ = rollout_setup.for_run(run["submission"])
                return self._send(200, {"trial_id": t["trial_id"], "shapes": rollout_setup.overlay(
                    rollout_setup.entry(placement, t["index"]))})
            if method == "GET" and parts == ["align"]:  # a fixed camera against the view it was calibrated in
                import cam_align  # noqa: PLC0415

                cam = q.get("cam", ["side"])[0]
                if cam not in cell.cameras:
                    raise ValueError(f"no camera {cam}")
                return self._send(200, {"camera": cam, **cam_align.measure(cam, cell.backend.observe([cam])[2][cam])})
            if method == "GET" and len(parts) == 2 and parts[0] == "align-ref":  # that view, for a ghost overlay
                import cam_align  # noqa: PLC0415

                f = cam_align.ref_path(parts[1].removesuffix(".jpg"))
                if not f.is_file():
                    raise KeyError(parts[1])
                img = cv2.imread(str(f))
                img = cv2.resize(img, (1280, int(img.shape[0] * 1280 / img.shape[1])), interpolation=cv2.INTER_AREA)
                data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return None
            if method == "GET" and parts == ["frame.jpg"]:
                cam = q.get("cam", ["table"])[0]
                if cam not in cell.cameras:
                    raise ValueError(f"no camera {cam}")
                img = cell.backend.observe([cam])[2][cam]
                w = max(160, min(1280, int(q.get("w", ["960"])[0])))
                img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])))
                data = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return None
            if method != "POST":
                return self._send(404, {"error": "not found"})
            if parts in (["stop"], ["recover"]):
                return self._admin("POST", parts)
            if parts == ["config"]:
                return self._admin("POST", ["config"])
            if len(parts) == 3 and parts[0] == "trials" and parts[2] in ("prepare", "ready", "retry", "abort"):
                return self._admin("POST", parts)
            if len(parts) == 3 and parts[0] == "trials" and parts[2] == "score":
                run, t = store.trial(parts[1])
                if t["status"] in ("waiting", "preparing", "ready"):
                    raise RuntimeError(f"trial is {t['status']}: nothing to score yet")
                body = self._body()
                stages, top = oc.scoring(run)
                stage = int(body["stage"])
                if stage not in [s_["stage"] for s_ in stages if s_["stage"] <= top]:
                    raise ValueError(f"stage {stage} is not a stage of this run")
                outcome = {"stage": stage, "success_stage": top, "intervention": bool(body.get("intervention")),
                           "notes": str(body.get("notes", ""))[:500]}
                if stage >= top and not outcome["intervention"] and t["status"] == "running":
                    outcome["t_success"] = time.time()  # the cycle time ends when the operator stamps success
                res = cell.finish(parts[1], outcome)["outcome"]
                log.info("operate: %s scored stage %d (success %s)", parts[1], res["stage"], res["success"])
                return self._send(200, {"ok": True, "outcome": res})
            return self._send(404, {"error": "not found"})

    return OpHandler


def main() -> None:
    global SURFACE_Z_MM, Z_MIN_HARD_MM  # --short-fingers-mm moves them
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["mock", "fanuc"], default="mock")
    p.add_argument("--episodes", type=Path, help="mock only: replay recorded twin episodes (<dir>/episode_*/frames.json) "
                   "instead of synthetic images")
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9098)
    p.add_argument("--runs", type=Path, default=RUNS, help="where runs and scores are kept")
    p.add_argument("--host", default="172.30.109.22", help="FANUC controller")
    p.add_argument("--z-min-mm", type=float, default=-330.0,
                   help=f"TCP floor in UF0 (surface is about {SURFACE_Z_MM}; not below {Z_MIN_HARD_MM}); change it live with "
                        "operator_console.py floor")
    p.add_argument("--short-fingers-mm", type=float, default=0.0,
                   help="fingers this much shorter than the standard ones are fitted (UTOOL1 unchanged): the surface, "
                        "the hard floor and the TCP box move this much lower, so the operator may lower the floor that "
                        "far for that team (0-40). The floor itself stays at --z-min-mm until the operator changes it")
    p.add_argument("--z-offset-mm", type=float, default=0.0,
                   help="shift every TCP target in UF0 Z (mm, e.g. -3 when a policy stops short); the floor still applies")
    p.add_argument("--depth", default="table,wrist",
                   help="cameras whose depth teams may request: table, wrist (empty: off). Served on request, never recorded")
    p.add_argument("--depth-fps", type=int, default=15, help="the depth stream's frame rate (5, 10, 15 or 30)")
    p.add_argument("--operator-port", type=int, default=9097,
                   help="the Operate page, on 127.0.0.1 only (0: off). Never tunnel this port: it moves the arm")
    p.add_argument("--no-record", action="store_true", help="do not record trials (cameras + state, 10 Hz)")
    p.add_argument("--no-dataset", action="store_true", help="record the web videos only, not the LeRobotDataset")
    p.add_argument("--push-to", default="azureuser@172.208.64.66:cell_relay/rollouts",
                   help="copy each recorded rollout here (host:dir) for hackathon.sidclab.org; empty to keep it local")
    p.add_argument("--push-key", default="~/.ssh/cell_relay_admin")
    p.add_argument("--gripper-wait-mm", type=float, default=3.0,
                   help="open/close the gripper only once the fingertip is within this of the row's target (0: at once)")
    # Motion limits for every action (joints and tcp). The defaults are the ones the cell has always run with.
    p.add_argument("--max-speed-mm-s", type=int, default=50, help="TCP speed cap, mm/s")
    p.add_argument("--max-step-mm", type=float, default=8.0, help="how far a target may move from the previous one")
    p.add_argument("--max-lead-mm", type=float, default=40.0, help="how far a target may run ahead of the arm")
    p.add_argument("--max-step-deg", type=float, default=5.0, help="joint actions: how far any joint may move per step")
    p.add_argument("--max-rot-step-deg", type=float, default=5.0, help="tcp actions: how far the tool may turn per step")
    p.add_argument("--max-trial-s", type=float, default=0.0,
                   help="end a trial after this many seconds (0: no limit, the operator ends it)")
    p.add_argument("--cameras", default="table,side,wrist", help="cameras to open and serve, comma-separated")
    p.add_argument("--table-fps", type=int, default=30, choices=(5, 10, 15, 30),
                   help="the table camera's colour rate. At 30 fps ~2.5%% of its 1080p MJPG frames arrive truncated (the "
                        "server replaces them with the last whole one); at 15 fps whole frames come out dark, bright or "
                        "green instead, which nothing filters, so keep 30")
    p.add_argument("--wrist-mode", default="848x480@15",
                   help="wrist capture WxH@fps. Over USB 2.0 the Gemini 305 does 1280x720 at 10 fps only; 848x480@15 keeps "
                        "the 16:9 view")
    p.add_argument("--cam-table", default="/dev/v4l/by-path/platform-NVDA8000:01-usb-0:1.2:1.4-video-index0")
    p.add_argument("--cam-wrist", default="/dev/v4l/by-path/platform-NVDA8000:01-usb-0:1.1.2:1.4-video-index0")
    p.add_argument("--cam-side",
                   default="/dev/v4l/by-id/usb-SHENZHEN_AONI_ELECTRONIC_CO.__LTD_Full_HD_webcam_20200729001-video-index0")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    admin_token = os.environ.get("CELL_ADMIN_TOKEN") or secrets.token_urlsafe(12)
    (HERE / ".admin_token").write_text(admin_token)
    os.chmod(HERE / ".admin_token", 0o600)
    cameras = tuple(c for c in args.cameras.split(",") if c)
    if not cameras or any(c not in CAMERAS for c in cameras):
        raise SystemExit(f"--cameras must be a subset of {','.join(CAMERAS)}")
    if not 0.0 <= args.short_fingers_mm <= 40.0:
        raise SystemExit("--short-fingers-mm must be in [0, 40]")
    if args.short_fingers_mm:
        SURFACE_Z_MM -= args.short_fingers_mm
        Z_MIN_HARD_MM = SURFACE_Z_MM + 1.0
        TCP_BOX_MM["z"] = (Z_MIN_HARD_MM, TCP_BOX_MM["z"][1])
        log.warning("short fingers (%.1f mm): surface at about %.1f mm, floor may go down to %.1f mm (now %.1f)",
                    args.short_fingers_mm, SURFACE_Z_MM, Z_MIN_HARD_MM, args.z_min_mm)
    if not Z_MIN_HARD_MM <= args.z_min_mm <= 0.0:
        raise SystemExit(f"--z-min-mm must be in [{Z_MIN_HARD_MM}, 0]")
    if args.backend == "mock":
        backend = MockBackend(args.episodes)
    else:
        paths = {c: find_camera(c, p) for c, p in {"table": args.cam_table, "side": args.cam_side, "wrist": args.cam_wrist}.items()}
        wh, fps = args.wrist_mode.split("@")
        modes = {c: (*NATIVE[c], 30) for c in CAMERAS}
        modes["table"] = (*NATIVE["table"], args.table_fps)
        modes["wrist"] = (*map(int, wh.split("x")), int(fps))
        if not -10.0 <= args.z_offset_mm <= 10.0:
            raise SystemExit("--z-offset-mm must be within +-10 mm")
        limits = {"tcp_speed_mm_s": args.max_speed_mm_s, "max_step_mm": args.max_step_mm, "max_lead_mm": args.max_lead_mm,
                  "max_step_deg": args.max_step_deg, "max_rot_step_deg": args.max_rot_step_deg}
        if not (10 <= args.max_speed_mm_s <= 250 and 1 <= args.max_step_mm <= 25 and 5 <= args.max_lead_mm <= 100
                and 1 <= args.max_step_deg <= 15 and 1 <= args.max_rot_step_deg <= 15):
            raise SystemExit("motion limits out of range: speed 10-250 mm/s, step 1-25 mm, lead 5-100 mm, "
                             "joint/rotation step 1-15 deg")
        backend = FanucBackend(args.host, args.z_min_mm, {c: paths[c] for c in cameras}, modes, args.z_offset_mm,
                               args.gripper_wait_mm or None, limits)
        backend.limits = limits
        log.info("motion limits: %s", limits)
    depth_cams = tuple(c for c in args.depth.split(",") if c)
    if any(c not in DEPTH_CAMERAS or c not in cameras for c in depth_cams):
        raise SystemExit(f"--depth must be a subset of {','.join(c for c in DEPTH_CAMERAS if c in cameras)} (or empty)")
    if isinstance(backend, FanucBackend):
        backend.depth = {c: DepthSource(c, args.depth_fps) for c in depth_cams}
    backend.start()
    store = Store(args.runs)
    cell = Cell(backend, store)
    cell.max_trial_s = args.max_trial_s or None
    cell.cameras = cameras
    cell.depth_cameras = depth_cams
    if not args.no_record:
        from recorder import TrialRecorder

        if args.push_to:
            host, rdir = args.push_to.split(":", 1)
            cell.push = {"host": host, "dir": rdir, "key": Path(args.push_key).expanduser()}
        cell.recorder = TrialRecorder(store.root, lambda cams: backend.observe(cams), cell.recorded,
                                      lerobot=not args.no_dataset)
    cell.publish()  # every run's results/<view_key>.json, pushed to the relay
    for run_id in sorted(store.runs):  # and every dataset (rsync sends only what changed)
        cell.sync_dataset(run_id)
    handler = make_handler(cell, store, admin_token, Requests(store.root), Organizer(store.root))
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    if args.operator_port:
        op = ThreadingHTTPServer(("127.0.0.1", args.operator_port), make_op_handler(handler, cell, store, args.operator_port))
        threading.Thread(target=op.serve_forever, daemon=True).start()
        log.info("Operate page on http://127.0.0.1:%d (this PC only)", args.operator_port)
    log.info("listening on http://%s:%d (%s backend, cameras %s); admin token in %s", args.bind, args.port,
             args.backend, ",".join(cameras), HERE / ".admin_token")
    def _shutdown(signum, frame):  # closing the terminal / kill: still release RMI cleanly in the finally below
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGHUP, _shutdown)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cell.stop()
        backend.stop()


if __name__ == "__main__":
    main()
