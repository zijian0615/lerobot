"""Fake FANUC RMI controller for testing the twin without a robot.

    python mock_rmi_server.py --port 16001        # then:  twin.py --source rmi --host 127.0.0.1

It speaks the subset the twin uses (FRC_Connect, FRC_Initialize, FRC_ReadJointAngles,
FRC_ReadCartesianPosition, FRC_Abort, FRC_Disconnect), follows the message format of
fanuc_replay_live.py, and records motion instructions it should never receive.
"""
import argparse
import json
import socket
import threading
import time

from sources import DemoSource


class MockRmiController:
    def __init__(self, host="127.0.0.1", port=0, j3_mode="coupled", single_session=True, cartesian_fn=None):
        self.host, self.single_session, self.cartesian_fn = host, single_session, cartesian_fn
        self._demo = DemoSource(j3_mode)
        self._main = self._listen(port)
        self._sess = self._listen(0)
        self.port = self._main.getsockname()[1]
        self.session_port = self._sess.getsockname()[1]
        self._active = 0
        self._stop = threading.Event()
        self.last_joints = None
        self.n_reads = 0
        self.forbidden = []          # motion instructions / FRC_Abort seen
        self.refused = 0

    def _listen(self, port):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        s.bind((self.host, port))
        s.listen(4)
        s.settimeout(0.2)
        return s

    def start(self):
        threading.Thread(target=self._accept_loop, args=(self._main, self._serve_connect), daemon=True).start()
        threading.Thread(target=self._accept_loop, args=(self._sess, self._serve_session), daemon=True).start()
        return self

    def stop(self):
        self._stop.set()
        for s in (self._main, self._sess):
            s.close()

    def _accept_loop(self, srv, handler):
        while not self._stop.is_set():
            try:
                conn, _ = srv.accept()
            except (socket.timeout, OSError):
                continue
            threading.Thread(target=handler, args=(conn,), daemon=True).start()

    @staticmethod
    def _lines(conn):
        buf = b""
        conn.settimeout(0.5)
        while True:
            try:
                chunk = conn.recv(4096)
            except socket.timeout:
                yield None
                continue
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    yield json.loads(line.decode())

    @staticmethod
    def _send(conn, obj):
        try:
            conn.sendall((json.dumps(obj) + "\r\n").encode())
        except OSError:
            pass

    def _serve_connect(self, conn):
        with conn:
            for msg in self._lines(conn):
                if msg and msg.get("Communication") == "FRC_Connect":
                    self._send(conn, {"Communication": "FRC_Connect", "ErrorID": 0, "PortNumber": self.session_port,
                                      "MajorVersion": 1, "MinorVersion": 0})
                    return

    def _serve_session(self, conn):
        if self.single_session and self._active:
            self.refused += 1
            conn.close()
            return
        self._active += 1
        try:
            with conn:
                for msg in self._lines(conn):
                    if self._stop.is_set():
                        return
                    if msg is None:
                        continue
                    cmd = msg.get("Command")
                    if "Instruction" in msg:
                        self.forbidden.append(msg)
                        self._send(conn, {"Command": cmd, "ErrorID": 0})
                    elif cmd == "FRC_Abort":
                        self._send(conn, {"Command": cmd, "ErrorID": 0})
                    elif msg.get("Communication") == "FRC_Disconnect":
                        self._send(conn, {"Communication": "FRC_Disconnect", "ErrorID": 0})
                        return
                    elif cmd == "FRC_Initialize":
                        self._send(conn, {"Command": cmd, "ErrorID": 0, "GroupMask": msg.get("GroupMask", 1)})
                    elif cmd == "FRC_ReadJointAngles":
                        j = self._demo.latest().joints_deg
                        self.last_joints, self.n_reads = j, self.n_reads + 1
                        self._send(conn, {"Command": cmd, "ErrorID": 0, "TimeTag": self.n_reads, "Group": 1,
                                          "JointAngle": {f"J{i + 1}": j[i] for i in range(6)} | {"J7": 0.0, "J8": 0.0, "J9": 0.0}})
                    elif cmd == "FRC_ReadCartesianPosition" and self.cartesian_fn:
                        x = self.cartesian_fn(self._demo.latest().joints_deg)
                        self._send(conn, {"Command": cmd, "ErrorID": 0, "TimeTag": self.n_reads, "Group": 1,
                                          "Position": dict(zip("XYZWPR", x))})
                    else:
                        self._send(conn, {"Command": cmd, "ErrorID": 2556957})
        finally:
            self._active -= 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=16001)
    ap.add_argument("--j3-mode", default="coupled", choices=["coupled", "direct"])
    a = ap.parse_args()
    srv = MockRmiController(a.host, a.port, a.j3_mode).start()
    print(f"mock RMI controller on {a.host}:{srv.port} (session port {srv.session_port}); Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.stop()
