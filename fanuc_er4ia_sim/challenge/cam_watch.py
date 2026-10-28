"""Watch a camera whose stream can come up corrupt, and reopen it when it does (the table camera).

All three cameras share one USB 2.0 bus. When a stream starts, it reserves its bandwidth on that bus; sometimes the
table camera's colour stream gets too little, and from then on most of its MJPG frames arrive damaged (grey lower
part, wrong brightness, a green or purple cast) until the camera is opened again. This happens when the server opens
the cameras, and can happen later when another stream on the bus restarts.

CameraWatch samples the camera a few times a second. A frame is bad if it is truncated (flat grey rows at the bottom)
or if its brightness or colour cast jumps far from the recent median of good frames. The server replaces bad frames
with the last good one (teams and recordings never see them), and when more than a quarter of the recent samples are
bad, the watch closes and reopens the camera, at most once every COOLDOWN_S. startup() does the same check right after
the cameras open. Only for a camera that survives being reopened (not the wrist Gemini 305).
"""
from __future__ import annotations

import collections
import statistics
import threading
import time

import numpy as np

WINDOW = 60          # samples in the reopen decision (about 12 s at PERIOD_S)
BAD_RATIO = 0.25     # reopen when more than this share of WINDOW is bad
PERIOD_S = 0.2
COOLDOWN_S = 30.0
JUMP = 25.0          # brightness (0..255) away from the recent median of good frames
CAST = 15.0          # green-minus-red/blue away from the recent median


def stats(img: np.ndarray) -> tuple[float, float, bool]:
    """(brightness, colour cast, truncated) of an RGB frame, on a coarse grid (cheap at 1080p)."""
    small = img[::16, ::16].astype(np.float32)
    r, g, b = small[..., 0].mean(), small[..., 1].mean(), small[..., 2].mean()
    tail = img[-8:]
    cut = float(tail.std()) < 3.0 and abs(float(tail.mean()) - 128.0) < 12.0
    return float((r + g + b) / 3), float(g - (r + b) / 2), cut


class CameraWatch:
    def __init__(self, name: str, camera, log):
        self.name, self.camera, self.log = name, camera, log
        self.lock = threading.Lock()
        self.good = collections.deque(maxlen=30)      # (brightness, cast) of recent good frames
        self.window = collections.deque(maxlen=WINDOW)
        self.last_reopen = 0.0
        self.events: list[dict] = []                  # reopens, for /admin/status and the Operate page
        self.running = False

    def bad(self, img: np.ndarray) -> bool:
        """Is this frame damaged? Also feeds the recent-good baseline."""
        lum, cast, cut = stats(img)
        with self.lock:
            if cut:
                return True
            if len(self.good) >= 5:
                ml = statistics.median(x[0] for x in self.good)
                mc = statistics.median(x[1] for x in self.good)
                if abs(lum - ml) > JUMP or abs(cast - mc) > CAST:
                    return True
            self.good.append((lum, cast))
            return False

    def _sample(self) -> bool | None:
        try:
            img = self.camera.read_latest(max_age_ms=1000)
        except Exception:  # noqa: BLE001  (no frame, or the camera is being reopened)
            return None
        return self.bad(img) if img is not None else None

    def bad_share(self, seconds: float) -> float:
        """Sample for a while: the share of bad frames (no frame counts as bad)."""
        n = bad = 0
        t_end = time.time() + seconds
        while time.time() < t_end:
            b = self._sample()
            n += 1
            bad += b is not False
            time.sleep(PERIOD_S)
        return bad / max(n, 1)

    def reopen(self, reason: str) -> None:
        with self.lock:
            self.good.clear()
            self.window.clear()
        self.log.warning("%s camera: %s; reopening it", self.name, reason)
        try:
            self.camera.disconnect()
        except Exception as e:  # noqa: BLE001
            self.log.warning("%s camera: disconnect failed: %s", self.name, e)
        time.sleep(1.0)
        self.camera.connect()
        self.last_reopen = time.time()
        self.events = (self.events + [{"t": self.last_reopen, "camera": self.name, "reason": reason}])[-20:]

    def startup(self, tries: int = 3, seconds: float = 4.0) -> None:
        """Right after the cameras open: reopen until the stream is clean (or give up and say so)."""
        for k in range(tries):
            share = self.bad_share(seconds)
            if share <= BAD_RATIO:
                self.log.info("%s camera: stream clean at start (%.0f%% bad frames)", self.name, 100 * share)
                return
            self.reopen(f"{100 * share:.0f}% bad frames at start (try {k + 1} of {tries})")
        share = self.bad_share(seconds)
        self.log.log(30 if share > BAD_RATIO else 20, "%s camera: %.0f%% bad frames after %d reopens", self.name,
                     100 * share, tries)

    def start(self) -> None:
        self.running = True
        threading.Thread(target=self._run, daemon=True, name=f"watch-{self.name}").start()

    def _run(self) -> None:
        while self.running:
            b = self._sample()
            with self.lock:
                self.window.append(b is not False)
                share = sum(self.window) / len(self.window)
                full = len(self.window) >= WINDOW // 2
            if full and share > BAD_RATIO and time.time() - self.last_reopen > COOLDOWN_S:
                try:
                    self.reopen(f"{100 * share:.0f}% bad frames in the last {len(self.window) * PERIOD_S:.0f} s")
                except Exception as e:  # noqa: BLE001  (keep watching; try again after the cooldown)
                    self.last_reopen = time.time()
                    self.log.error("%s camera: reopen failed: %s", self.name, e)
            time.sleep(PERIOD_S)
