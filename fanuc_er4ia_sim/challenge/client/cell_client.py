"""Client for the FANUC ER-4iA cell. You should not need to edit this file."""
from __future__ import annotations

import base64
import logging
import time

import cv2
import numpy as np
import requests

log = logging.getLogger("cell")
OPEN = ("waiting", "preparing", "ready", "running")


class CellClient:
    def __init__(self, url: str, token: str, ca: str | None = None):
        """ca: a certificate file to trust (the relay's self-signed one), else the system CAs."""
        self.api = url.rstrip("/") + "/api/v1"
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {token}"
        if ca:
            self.s.verify = ca

    def _call(self, method: str, path: str, **kw) -> dict:
        for attempt in range(3):
            try:
                r = self.s.request(method, f"{self.api}/{path}", timeout=10, **kw)
                break
            except requests.ConnectionError:
                if attempt == 2:
                    raise
                time.sleep(1)
        try:
            body = r.json()
        except ValueError:
            body = {"error": f"HTTP {r.status_code}"}
        if r.status_code == 502:
            raise RuntimeError("the cell is offline; try again later")
        if r.status_code != 200:
            raise RuntimeError(f"{method} /{path}: {body.get('error', body)}")
        return body

    def submit(self, submission: dict) -> dict:
        """Send submission.yaml; returns {"request_id", "secret"}. The operator approves it into a run."""
        return self._call("POST", "submit", json={"submission": submission})

    def request_status(self, request_id: str, secret: str) -> dict:
        """{"status": pending | approved | rejected, "reason"}, and {"run_id", "token"} once approved."""
        return self._call("GET", f"requests/{request_id}", headers={"X-Request-Secret": secret})

    def trials(self) -> list[dict]:
        return self._call("GET", "run")["trials"]

    def start(self, trial_id: str) -> None:
        self._call("POST", f"trials/{trial_id}/start")

    def clock(self, n: int = 5) -> tuple[float, float]:
        """(cell clock - local clock, round trip) in seconds, median of n /time calls."""
        offs, rtts = [], []
        for _ in range(n):
            t1 = time.time()
            t = float(self._call("GET", "time")["t"])
            t2 = time.time()
            offs.append(t - (t1 + t2) / 2)
            rtts.append(t2 - t1)
        return float(np.median(offs)), float(np.median(rtts))

    def observe(self, cameras, size=(640, 360), fmt: str = "png", quality: int = 90, depth=()) -> dict:
        """{"t", "status", "trial", "queued", "joints" (7,), "tcp_pose", "images": {camera: HxWx3 RGB uint8}}
        and, with depth=["table"], "depth": {"table": 400x640 uint16 mm, 0 = no depth}.

        fmt "jpeg" cuts the transfer ~10x (lossy); "png" is lossless. Depth is always lossless."""
        params = {"cameras": ",".join(cameras), "width": size[0], "height": size[1], "format": fmt, "quality": quality}
        if depth:
            params["depth"] = ",".join(depth)
        obs = self._call("GET", "observation", params=params)
        obs["joints"] = np.asarray(obs["joints"], dtype=np.float32)
        obs["images"] = {c: cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(b), np.uint8), cv2.IMREAD_COLOR),
                                         cv2.COLOR_BGR2RGB) for c, b in obs["images"].items()}
        if "depth" in obs:
            obs["depth"] = {c: cv2.imdecode(np.frombuffer(base64.b64decode(b), np.uint8), cv2.IMREAD_UNCHANGED)
                            for c, b in obs["depth"].items()}
        return obs

    def act(self, joints, dt: float = 0.1, obs_t: float | None = None) -> None:
        """obs_t: the observation this chunk was planned from, while the previous one still played (PREFETCH):
        the cell replaces what is still queued and skips the rows whose time has passed."""
        joints = np.asarray(joints, dtype=float)
        if not np.all(np.isfinite(joints)):
            raise RuntimeError("POST /actions: joints contain NaN or inf")
        self._call("POST", "actions", json={"joints": joints.tolist(), "dt": dt, **_replace(obs_t)})

    def act_tcp(self, rows, dt: float = 0.1, obs_t: float | None = None) -> None:
        """rows: (N, 7) [x, y, z mm, w, p, r deg, gripper], UTOOL1 (fingertip) in UF0, the frame of obs["tcp_pose"]."""
        rows = np.asarray(rows, dtype=float)
        if not np.all(np.isfinite(rows)):
            raise RuntimeError("POST /actions: tcp contains NaN or inf")
        self._call("POST", "actions", json={"tcp": rows.tolist(), "dt": dt, **_replace(obs_t)})


PREFETCH_STATS = {"chunks": 0, "ahead": 0, "short": 0}  # for run.py --mock's summary


def _replace(obs_t: float | None) -> dict:
    return {} if obs_t is None else {"obs_t": obs_t, "replace": True}


def run_trial(cell: CellClient, policy, trial: dict) -> None:
    offset, rtt = cell.clock()
    log.info("cell clock offset %+.3f s, round trip %.0f ms", offset, rtt * 1000)
    cell.start(trial["trial_id"])
    policy.reset()
    log.info("trial %s (layout %s) running", trial["trial_id"], trial["layout"])
    dt = getattr(policy, "DT", 0.1)
    prefetch = getattr(policy, "PREFETCH", False)
    took = None  # seconds from an observation to its actions reaching the cell (act + network), measured
    last = 0  # actions in the last chunk
    while True:
        obs = cell.observe(policy.CAMERAS, policy.IMAGE_SIZE, getattr(policy, "IMAGE_FORMAT", "png"),
                           depth=getattr(policy, "DEPTH", ()))
        if obs["trial"] is None or obs["trial"]["status"] != "running":
            log.info("trial %s ended", trial["trial_id"])
            return
        # PREFETCH: plan the next actions while the last ones still play. True starts when what is left covers
        # the time act() takes (measured); a number starts when that many actions are left.
        if prefetch is True:
            lead = 0 if took is None else int(np.ceil(took / dt)) + 2
            if last < 2 * lead:  # too few actions to cover act(): planning ahead would only cut them short
                lead = 0
        else:
            lead = int(prefetch or 0)
        if obs["status"] != "running" or obs["queued"] > lead:
            time.sleep(0.01)  # wait until the last actions have (almost) played out
            continue
        t0 = time.time()
        chunk = np.asarray(policy.act(obs), dtype=float)
        if chunk.ndim == 1:  # one target
            chunk = chunk[None]
        obs_t = obs["t"] if lead else None  # the cell drops what is left and skips rows already in the past
        last = len(chunk)
        PREFETCH_STATS["chunks"] += 1
        PREFETCH_STATS["ahead"] += bool(lead)
        PREFETCH_STATS["short"] += bool(prefetch is True and took is not None and lead == 0)
        try:
            if getattr(policy, "ACTION", "joints") == "tcp":
                cell.act_tcp(chunk, dt, obs_t)
            else:
                cell.act(chunk, dt, obs_t)
            took = time.time() - t0 if took is None else 0.7 * took + 0.3 * (time.time() - t0)
        except RuntimeError as e:  # the trial can end between observe and act
            log.warning("%s", e)


def run_all(cell: CellClient, policy) -> None:
    """Run every trial of the run as the operator marks it ready. Returns when every trial is over and scored (an
    ended trial not scored yet may still be voided and run again)."""
    while True:
        trials = cell.trials()
        if not any(t["status"] in OPEN or (t.get("scored") is False and t["status"] != "cancelled") for t in trials):
            log.info("all trials finished")
            return
        for t in trials:
            if t["status"] == "ready":
                run_trial(cell, policy, t)
        time.sleep(0.5)
