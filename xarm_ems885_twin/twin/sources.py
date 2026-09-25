"""Joint-state sources for the xArm twin. All READ-ONLY: nothing here sends a motion, mode, state or gripper command.

    XArmSource   one xArm over the SDK. Joints come from the controller's report stream (`arm.angles`), so reading costs
                 no command traffic; the gripper position is polled only with read_gripper=True (a read, but it goes
                 through the controller's command socket, so it is off by default).
    UdpSource    datagrams {"arm", "q" [6 rad], "gripper_pos" | null, "t"} published by publish_joints.py
    DemoSource   synthetic motion around each arm's home pose, no robot needed

Every source: .start()  .latest() -> {arm: ArmSample}  .status  .count  .close()
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from typing import NamedTuple

from xarm_model import gripper_pos_to_drive

DEFAULT_UDP_PORT = 5015


class ArmSample(NamedTuple):
    q: tuple                 # J1..J6 [rad], as reported by the controller
    gripper_pos: float | None  # SDK gripper position (850 open ... 0 closed) or None if unknown
    t_recv: float            # time.monotonic() when received

    def drive(self, default=0.0):
        return default if self.gripper_pos is None else gripper_pos_to_drive(self.gripper_pos)


class _Threaded:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: dict[str, ArmSample] = {}
        self._stop = threading.Event()
        self._thread = None
        self.status = "not started"
        self.count = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def latest(self):
        with self._lock:
            return dict(self._latest)

    def _put(self, arm, q, gripper_pos):
        with self._lock:
            self._latest[arm] = ArmSample(tuple(float(v) for v in q[:6]), gripper_pos, time.monotonic())
            self.count += 1

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        raise NotImplementedError


def _connect_xarm(ip):
    from xarm.wrapper import XArmAPI

    # is_radian only changes the units the SDK reports in; no command is sent to the arm here.
    return XArmAPI(ip, is_radian=True)


def read_xarm_once(ip, read_gripper=False, settle_s=0.5):
    """(q[6] rad, gripper drive rad) from one short read-only connection."""
    arm = _connect_xarm(ip)
    try:
        time.sleep(settle_s)                       # first report frames
        q = list(arm.angles[:6])
        drive = 0.0
        if read_gripper:
            code, pos = arm.get_gripper_position()
            if code == 0 and pos is not None:
                drive = gripper_pos_to_drive(pos)
        return q, drive
    finally:
        arm.disconnect()


class XArmSource(_Threaded):
    """Polls one arm's reported joints (and optionally the gripper) and keeps the latest sample. Reconnects on failure."""

    def __init__(self, arm_name, ip, rate_hz=30.0, read_gripper=False, gripper_rate_hz=5.0):
        super().__init__()
        self.arm_name, self.ip = arm_name, ip
        self.period = 1.0 / rate_hz
        self.read_gripper, self.gripper_period = read_gripper, 1.0 / gripper_rate_hz

    def _run(self):
        while not self._stop.is_set():
            try:
                self.status = f"{self.arm_name}: connecting {self.ip}"
                arm = _connect_xarm(self.ip)
            except Exception as e:  # noqa: BLE001 - keep retrying; the arm may be off or unreachable
                self.status = f"{self.arm_name}: connect failed ({e}); retry in 5 s"
                self._stop.wait(5.0)
                continue
            self.status = f"{self.arm_name}: connected {self.ip}"
            gpos, t_grip = None, 0.0
            try:
                while not self._stop.is_set() and arm.connected:
                    t0 = time.monotonic()
                    if self.read_gripper and t0 - t_grip >= self.gripper_period:
                        t_grip = t0
                        code, pos = arm.get_gripper_position()
                        gpos = float(pos) if code == 0 and pos is not None else gpos
                    self._put(self.arm_name, arm.angles, gpos)
                    self._stop.wait(max(0.0, self.period - (time.monotonic() - t0)))
                self.status = f"{self.arm_name}: disconnected"
            finally:
                try:
                    arm.disconnect()
                except Exception:  # noqa: BLE001
                    pass
            self._stop.wait(1.0)


class MultiSource:
    """Several single-arm sources behind the same interface."""

    def __init__(self, sources):
        self.sources = sources

    def start(self):
        for s in self.sources:
            s.start()
        return self

    def latest(self):
        out = {}
        for s in self.sources:
            out.update(s.latest())
        return out

    @property
    def status(self):
        return " | ".join(s.status for s in self.sources)

    @property
    def count(self):
        return sum(s.count for s in self.sources)

    def close(self):
        for s in self.sources:
            s.close()


def encode(arm, q, gripper_pos=None):
    return json.dumps({"arm": arm, "q": [float(v) for v in q[:6]],
                       "gripper_pos": None if gripper_pos is None else float(gripper_pos),
                       "t": time.time()}).encode()


class UdpSource(_Threaded):
    def __init__(self, port=DEFAULT_UDP_PORT, bind="0.0.0.0"):
        super().__init__()
        self.port, self.bind = port, bind
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((bind, port))
        self._sock.settimeout(0.2)
        self.status = f"udp {bind}:{port} waiting"

    def _run(self):
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data)
                self._put(str(msg["arm"]), msg["q"], msg.get("gripper_pos"))
                self.status = f"udp :{self.port} from {addr[0]}"
            except (ValueError, KeyError, TypeError) as e:
                self.status = f"udp :{self.port} bad datagram ({e})"

    def close(self):
        super().close()
        self._sock.close()


class Publisher:
    def __init__(self, host, port=DEFAULT_UDP_PORT):
        self._addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def publish(self, arm, q, gripper_pos=None):
        self._sock.sendto(encode(arm, q, gripper_pos), self._addr)

    def close(self):
        self._sock.close()


class DemoSource:
    """Each arm swings J1/J2/J3/J5 around its home pose; the gripper opens and closes."""

    def __init__(self, homes):
        self.homes = {arm: list(q) for arm, q in homes.items()}
        self.t0 = time.monotonic()
        self.status, self.count = "demo", 0

    def start(self):
        return self

    def latest(self):
        t = time.monotonic() - self.t0
        out = {}
        for k, (arm, h) in enumerate(self.homes.items()):
            ph = t * 0.6 + k * math.pi / 2
            q = list(h)
            q[0] += 0.5 * math.sin(ph)
            q[1] += 0.25 * math.sin(0.7 * ph)
            q[2] += 0.3 * math.sin(0.9 * ph + 1.0)
            q[4] += 0.3 * math.sin(1.3 * ph)
            gpos = 425 + 425 * math.cos(0.5 * ph)
            out[arm] = ArmSample(tuple(q), gpos, time.monotonic())
        self.count += 1
        return out

    def close(self):
        pass


def add_source_args(ap):
    ap.add_argument("--source", choices=("udp", "xarm", "demo"), default="udp",
                    help="udp: joints published by publish_joints.py (default); xarm: read the arms directly "
                         "(only where 192.168.x.x is reachable); demo: synthetic motion")
    ap.add_argument("--udp-port", type=int, default=DEFAULT_UDP_PORT)
    ap.add_argument("--arms", default=None, help="comma-separated arms for --source xarm (default: all in the chain)")
    ap.add_argument("--read-gripper", action="store_true",
                    help="--source xarm: also poll the gripper position (5 Hz, read over the command socket)")


def make_source(a, chain):
    if a.source == "demo":
        return DemoSource({arm: info.get("home_joints") or [0.0] * 6 for arm, info in chain.arms.items()}).start()
    if a.source == "udp":
        return UdpSource(a.udp_port).start()
    arms = a.arms.split(",") if a.arms else list(chain.arms)
    return MultiSource([XArmSource(arm, chain.arms[arm]["ip"], read_gripper=a.read_gripper) for arm in arms]).start()
