"""Wire format between the agent's program and the simulator: length-prefixed
JSON over TCP, numpy arrays as base64.

Replaces multiprocessing.connection, whose pickle would let the program
execute code in the simulator's process (and container) by sending a crafted
message. Here the simulator side only ever parses JSON.

    frame   = 4-byte big-endian length + UTF-8 JSON
    ndarray = {"__nd__": base64(raw bytes), "dtype": "<f8", "shape": [...]}

The first frame a client sends is {"key": <run key>}; the server drops the
connection unless it matches.

Standalone on purpose (stdlib + numpy only): it is copied into the agent's
container alongside robot_api.py.
"""

from __future__ import annotations

import base64
import hmac
import json
import socket
import struct
import time

import numpy as np

MAX_FRAME = 256 * 1024 * 1024        # a few full-resolution images


class WireError(ConnectionError):
    pass


def _default(o):
    if isinstance(o, np.ndarray):
        a = np.ascontiguousarray(o)
        return {"__nd__": base64.b64encode(a.tobytes()).decode("ascii"),
                "dtype": a.dtype.str, "shape": list(a.shape)}
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return list(o)
    raise TypeError(f"cannot send {type(o).__name__}")


def _hook(d):
    if "__nd__" in d:
        dtype = np.dtype(d["dtype"])
        if dtype.kind not in "biufc":          # plain numbers only, no objects
            raise WireError(f"refusing dtype {dtype}")
        raw = base64.b64decode(d["__nd__"])
        return np.frombuffer(raw, dtype=dtype).reshape(d["shape"]).copy()
    return d


class Conn:
    def __init__(self, sock: socket.socket):
        self.sock = sock

    def send(self, obj) -> None:
        data = json.dumps(obj, default=_default, allow_nan=True).encode()
        if len(data) > MAX_FRAME:
            raise WireError(f"frame too large ({len(data)} bytes)")
        self.sock.sendall(struct.pack(">I", len(data)) + data)

    def _read(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(min(n - len(buf), 1 << 20))
            if not chunk:
                raise EOFError("connection closed")
            buf += chunk
        return bytes(buf)

    def recv(self):
        (n,) = struct.unpack(">I", self._read(4))
        if n > MAX_FRAME:
            raise WireError(f"frame too large ({n} bytes)")
        return json.loads(self._read(n), object_hook=_hook)

    def poll(self, timeout: float) -> bool:
        """Is a frame (or EOF) ready to read within `timeout` seconds?"""
        import select
        r, _, _ = select.select([self.sock], [], [], timeout)
        return bool(r)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


def parse_addr(addr: str) -> tuple[str, int]:
    host, port = addr.rsplit(":", 1)
    return host, int(port)


def connect(addr: str, key: str, timeout: float = 0.0) -> Conn:
    """Connect and authenticate. Retries for up to `timeout` seconds while the
    server is not listening yet (e.g. the simulator is still starting)."""
    deadline = time.time() + timeout
    while True:
        try:
            sock = socket.create_connection(parse_addr(addr), timeout=30)
            break
        except OSError:
            if time.time() >= deadline:
                raise
            time.sleep(1.0)
    sock.settimeout(None)
    conn = Conn(sock)
    conn.send({"key": key})
    return conn


class Server:
    """Single-client listener: accept() returns the first connection that
    presents the right key; others are dropped."""

    def __init__(self, addr: str, key: str):
        self.key = key
        self.sock = socket.create_server(parse_addr(addr), reuse_port=False)
        self.address = self.sock.getsockname()[:2]

    def accept(self, timeout: float) -> Conn | None:
        deadline = time.time() + timeout
        while True:
            left = deadline - time.time()
            if left <= 0:
                return None
            self.sock.settimeout(left)
            try:
                s, _ = self.sock.accept()
            except (socket.timeout, OSError):
                return None
            s.settimeout(10)
            conn = Conn(s)
            try:
                hello = conn.recv()
                ok = (isinstance(hello, dict) and isinstance(hello.get("key"), str)
                      and hmac.compare_digest(hello["key"], self.key))
            except (OSError, EOFError, ValueError, WireError, struct.error):
                ok = False
            if ok:
                s.settimeout(None)
                return conn
            conn.close()

    def close(self) -> None:
        self.sock.close()
