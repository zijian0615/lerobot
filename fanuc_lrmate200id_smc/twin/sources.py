"""Joint-state sources for the digital twin. All of them are READ-ONLY (no motion instructions are ever sent).

    RmiJointSource  polls FRC_ReadJointAngles over FANUC RMI (same protocol as fanuc_replay_live.py)
    UdpJointSource  receives {"joints_deg": [J1..J6]} datagrams published by another process
    DemoSource      synthetic motion, no robot needed
Every source offers  .start()  .latest() -> JointSample | None  .status  .close()
"""
import json
import math
import socket
import threading
import time
from typing import NamedTuple, Optional

import numpy as np

from joint_map import J3_MODES, model_to_fanuc


class JointSample(NamedTuple):
    joints_deg: tuple            # J1..J6 as reported by the controller (degrees)
    t_recv: float                # time.monotonic() when received
    tag: Optional[int] = None    # controller TimeTag if present


class RmiError(RuntimeError):
    pass


def parse_joint_response(resp):
    """Extract (J1..J6) from an FRC_ReadJointAngles response. Tolerant about the key name."""
    if resp.get("ErrorID", 0) not in (0, None):
        raise RmiError(f"controller error {resp.get('ErrorID')}: {resp}")
    cand = None
    for k in ("JointAngle", "JointAngles", "Joint"):
        if isinstance(resp.get(k), dict):
            cand = resp[k]
            break
    if cand is None:
        cand = resp
    up = {str(k).upper(): v for k, v in cand.items()}
    try:
        return tuple(float(up[f"J{i}"]) for i in range(1, 7))
    except KeyError:
        raise RmiError(f"no J1..J6 in response: {resp}") from None


class _LineSocket:
    """CRLF/LF framed JSON over TCP (same idea as LineSocket in fanuc_replay_live.py)."""

    def __init__(self, sock):
        self.sock, self._buf = sock, b""

    def send_json(self, obj):
        self.sock.sendall((json.dumps(obj) + "\r\n").encode())

    def read_json(self):
        while True:
            for sep in (b"\r\n", b"\n"):
                i = self._buf.find(sep)
                if i != -1:
                    line, self._buf = self._buf[:i].strip(), self._buf[i + len(sep):]
                    if line:
                        return json.loads(line.decode(errors="replace"))
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("socket closed by controller")
            self._buf += chunk


class _ThreadedSource:
    def __init__(self):
        self._lock = threading.Lock()
        self._sample = None
        self._status = "not started"
        self._stop = threading.Event()
        self._thread = None
        self.count = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def latest(self):
        with self._lock:
            return self._sample

    @property
    def status(self):
        return self._status

    def _publish(self, joints, tag=None):
        with self._lock:
            self._sample = JointSample(tuple(joints), time.monotonic(), tag)
            self.count += 1

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run(self):
        raise NotImplementedError


class RmiJointSource(_ThreadedSource):
    def __init__(self, host, port=16001, group=1, rate_hz=30.0, init=True, timeout=2.0):
        super().__init__()
        self.host, self.port, self.group = host, port, group
        self.period, self.init, self.timeout = 1.0 / rate_hz, init, timeout
        self.first_raw = None        # first joint response, printed by twin.py to help verify field names
        self._sess_sock = None
        self._sess_ls = None

    def _frc_connect(self):
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as s:
            s.sendall(b'{"Communication": "FRC_Connect"}\r\n')
            s.settimeout(self.timeout)
            resp = json.loads(s.recv(4096).decode())
        if resp.get("ErrorID", 0) != 0:
            raise RmiError(f"FRC_Connect failed: {resp}")
        return int(resp["PortNumber"])

    def _open(self):
        self._status = "connecting"
        sock = socket.create_connection((self.host, self._frc_connect()), timeout=self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(self.timeout)
        ls = _LineSocket(sock)
        if self.init:
            ls.send_json({"Command": "FRC_Initialize", "GroupMask": self.group})
            resp = ls.read_json()
            if resp.get("ErrorID", -1) != 0:
                raise RmiError(f"FRC_Initialize failed: {resp}")
        self._sess_sock, self._sess_ls = sock, ls
        return sock, ls

    def _end_session(self, sock, ls):
        """FANUC requires FRC_Abort or FRC_Disconnect before dropping TCP, else RMI_MOVE stays selected."""
        if sock is None:
            return
        try:
            if self.init and ls is not None:
                try:
                    ls.send_json({"Command": "FRC_Abort"})
                except OSError:
                    pass
                try:
                    ls.send_json({"Communication": "FRC_Disconnect"})
                except OSError:
                    pass
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        finally:
            try:
                sock.close()
            except OSError:
                pass
            if self._sess_sock is sock:
                self._sess_sock = self._sess_ls = None

    def _request(self, ls, command, **kw):
        ls.send_json({"Command": command, "Group": self.group, **kw})
        while True:                                   # skip unrelated messages
            resp = ls.read_json()
            if resp.get("Command") == command:
                return resp

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            sock = ls = None
            try:
                sock, ls = self._open()
                self._status = "connected"
                backoff = 1.0
                while not self._stop.is_set():
                    t0 = time.monotonic()
                    resp = self._request(ls, "FRC_ReadJointAngles")
                    if self.first_raw is None:
                        self.first_raw = resp
                    self._publish(parse_joint_response(resp), resp.get("TimeTag"))
                    time.sleep(max(0.0, self.period - (time.monotonic() - t0)))
            except Exception as e:                    # noqa: BLE001 - report and retry
                self._status = f"error: {type(e).__name__}: {e}"
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 5.0)
            finally:
                self._end_session(sock, ls)

    def close(self):
        self._stop.set()
        sock, ls = self._sess_sock, self._sess_ls
        if sock is not None:
            self._end_session(sock, ls)
        super().close()

    def read_cartesian_once(self):
        """One-off  (joints_deg, (X, Y, Z, W, P, R))  read for --check-cartesian. Uses its own short session."""
        sock, ls = self._open()
        try:
            j = parse_joint_response(self._request(ls, "FRC_ReadJointAngles"))
            c = self._request(ls, "FRC_ReadCartesianPosition")
            if c.get("ErrorID", 0) != 0:
                raise RmiError(f"FRC_ReadCartesianPosition failed: {c}")
            p = {k.upper(): v for k, v in c["Position"].items()}
            return j, tuple(float(p[k]) for k in "XYZWPR")
        finally:
            self._end_session(sock, ls)


class UdpJointSource(_ThreadedSource):
    """Datagram: {"joints_deg": [J1..J6]}  (see JointStatePublisher). Lets the process that owns the RMI
    session (replay/teleop script) feed the twin, because RMI normally allows a single client."""

    def __init__(self, port=5005, bind="0.0.0.0"):
        super().__init__()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((bind, port))
        self._sock.settimeout(0.2)
        self.port = self._sock.getsockname()[1]
        self._status = f"listening on udp/{self.port}"

    def _run(self):
        while not self._stop.is_set():
            try:
                data, _ = self._sock.recvfrom(4096)
                self._publish(json.loads(data)["joints_deg"][:6])
                self._status = f"receiving on udp/{self.port}"
            except socket.timeout:
                continue
            except (ValueError, KeyError, TypeError):
                self._status = "bad datagram ignored"

    def close(self):
        super().close()
        self._sock.close()


class JointStatePublisher:
    """Use inside a script that already owns the RMI session:  pub.publish(joints_deg_tuple)."""

    def __init__(self, host="127.0.0.1", port=5005):
        self._sock, self._addr = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, port)

    def publish(self, joints_deg):
        self._sock.sendto(json.dumps({"joints_deg": [float(x) for x in joints_deg[:6]]}).encode(), self._addr)


class DemoSource:
    """Synthetic smooth motion (defined in model space, reported in FANUC convention). No robot needed."""

    CENTER = np.array([0.0, 0.45, 0.25, 0.0, -0.45, 0.0])
    AMP = np.array([0.8, 0.35, 0.35, 1.0, 0.5, 1.5])
    FREQ = np.array([0.10, 0.13, 0.17, 0.11, 0.19, 0.15])

    def __init__(self, j3_mode="coupled"):
        self.j3_mode, self._t0, self.count = j3_mode, time.monotonic(), 0

    def start(self):
        return self

    def latest(self):
        t = time.monotonic()
        q = self.CENTER + self.AMP * np.sin(2 * math.pi * self.FREQ * (t - self._t0))
        self.count += 1
        return JointSample(tuple(model_to_fanuc(q, self.j3_mode)), t)

    status = "demo motion"

    def close(self):
        pass


def add_source_args(ap):
    """Command-line options shared by twin.py and omni_twin.py."""
    ap.add_argument("--source", choices=["rmi", "udp", "demo"], default="rmi")
    ap.add_argument("--host", default="172.30.109.22")
    ap.add_argument("--port", type=int, default=16001, help="RMI main port (FRC_Connect)")
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--rate", type=float, default=30.0, help="RMI polling rate [Hz]")
    ap.add_argument("--no-init", action="store_true", help="skip FRC_Initialize (try if init disturbs another RMI client)")
    ap.add_argument("--udp-port", type=int, default=5005)
    ap.add_argument("--j3-mode", choices=list(J3_MODES), default="coupled")


def make_source(a):
    if a.source == "demo":
        return DemoSource(a.j3_mode).start()
    if a.source == "udp":
        return UdpJointSource(a.udp_port).start()
    return RmiJointSource(a.host, a.port, a.group, a.rate, init=not a.no_init).start()
