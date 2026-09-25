"""Fan out the robot's joint datagrams to several twins (the robot driver publishes to a single UDP port).

    .venv-twin/bin/python twin/udp_relay.py                       # 5005 -> 5006 (twin.py) and 5007 (omni_twin.py)
    .venv-twin/bin/python twin/udp_relay.py --listen 5005 --to 127.0.0.1:5006 --to 10.0.0.7:5007
"""
import argparse
import socket
import threading


class UdpRelay:
    def __init__(self, listen=5005, targets=(("127.0.0.1", 5006), ("127.0.0.1", 5007)), bind="0.0.0.0"):
        self.targets = list(targets)
        self._rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx.bind((bind, listen))
        self._rx.settimeout(0.2)
        self.port = self._rx.getsockname()[1]
        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop = threading.Event()
        self._thread = None
        self.count = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                data, _ = self._rx.recvfrom(4096)
            except socket.timeout:
                continue
            for t in self.targets:
                self._tx.sendto(data, t)
            self.count += 1

    def close(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._rx.close()
        self._tx.close()


def _target(s):
    host, _, port = s.rpartition(":")
    return host or "127.0.0.1", int(port)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", type=int, default=5005)
    ap.add_argument("--to", action="append", type=_target, help="host:port to forward to (repeatable)")
    a = ap.parse_args(argv)
    relay = UdpRelay(a.listen, a.to or [("127.0.0.1", 5006), ("127.0.0.1", 5007)]).start()
    print(f"[relay] udp/{relay.port} -> {relay.targets}. Ctrl+C to stop.", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        relay.close()


if __name__ == "__main__":
    main()
