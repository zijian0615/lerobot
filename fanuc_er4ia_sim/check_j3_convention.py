"""Check which J3 convention the real controller reports, against the sim FK.

Read-only. It never sends a motion instruction, a port write, or a frame change.

The sim chain (fk.py) treats J3 as relative to the upper arm. FANUC usually
reports J3 against the ground (J2/J3 interaction). Each candidate maps the
controller joints onto the sim q. The right one makes fk.py agree with the
controller's Cartesian pose at every sample, whatever J2 is.

Take one sample per pose, robot standing still, controller in AUTO and the
pendant disabled (RMI needs both). Vary J2 by 20 deg or more between samples:

    uv run python fanuc_er4ia_sim/check_j3_convention.py read --label a
    # jog to another pose, disable the pendant, then
    uv run python fanuc_er4ia_sim/check_j3_convention.py read --label b
    uv run python fanuc_er4ia_sim/check_j3_convention.py analyze

Readings copied off the pendant work too (JOINT screen, then the WORLD/USER screen):

    uv run python fanuc_er4ia_sim/check_j3_convention.py add J1 J2 J3 J4 J5 J6 X Y Z W P R
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fk  # noqa: E402

from lerobot.robots.fanuc.er4ia_fk import controller_tcp_pose  # noqa: E402

DEFAULT_SAMPLES = Path(__file__).with_name("data") / "j3_samples.json"

# Controller joints (deg) -> sim q (deg).
CONVENTIONS = {
    "sim q3 = J3": lambda j: j,
    "sim q3 = J3 + J2": lambda j: j + np.array([0, 0, j[1], 0, 0, 0]),
    "sim q3 = J3 - J2": lambda j: j - np.array([0, 0, j[1], 0, 0, 0]),
}


def wpr_matrix(w, p, r):
    """FANUC WPR, R = Rz(R) Ry(P) Rx(W), as in robots/fanuc/pose.py."""
    return Rotation.from_euler("ZYX", [r, p, w], degrees=True).as_matrix()


# ------------------------------------------------------------------ reading


class ReadOnlyRmi:
    def __init__(self, host: str, port: int, group: int):
        self.group = group
        with socket.create_connection((host, port), timeout=5.0) as s:
            s.sendall(b'{"Communication": "FRC_Connect"}\r\n')
            reply = json.loads(s.recv(4096).decode())
        if reply.get("ErrorID", -1) != 0:
            raise RuntimeError(f"FRC_Connect failed: {reply}")
        self.sock = socket.create_connection((host, int(reply["PortNumber"])), timeout=5.0)
        self.buf = b""
        self.initialized = False

    def _send(self, payload):
        self.sock.sendall((json.dumps(payload) + "\r\n").encode())

    def _recv(self, command, timeout=5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            while b"\n" not in self.buf:
                chunk = self.sock.recv(65536)
                if not chunk:
                    raise ConnectionError("controller closed the socket")
                self.buf += chunk
            line, self.buf = self.buf.split(b"\n", 1)
            msg = json.loads(line.rstrip(b"\r"))
            if msg.get("Command") == command:
                return msg
        raise TimeoutError(command)

    def ask(self, command):
        self._send({"Command": command, "Group": self.group})
        return self._recv(command)

    def initialize(self):
        # Starts RMI_MOVE on the controller. It does not move the arm.
        self._send({"Command": "FRC_Initialize", "GroupMask": self.group})
        reply = self._recv("FRC_Initialize")
        if int(reply.get("ErrorID", -1)) not in (0, 7015):
            raise RuntimeError(f"FRC_Initialize failed: {reply} (AUTO mode, pendant disabled, fault reset?)")
        self.initialized = True

    def close(self):
        try:
            if self.initialized:
                self._send({"Command": "FRC_Abort"})
            self._send({"Communication": "FRC_Disconnect"})
        finally:
            self.sock.close()


def read_pair(rmi: ReadOnlyRmi):
    joints = rmi.ask("FRC_ReadJointAngles")
    cart = rmi.ask("FRC_ReadCartesianPosition")
    if joints.get("ErrorID", -1) != 0 or cart.get("ErrorID", -1) != 0:
        return None
    angles = {k.upper(): v for k, v in (joints.get("JointAngle") or {}).items()}
    pos = cart["Position"]
    return {
        "joints_deg": [float(angles[f"J{i}"]) for i in range(1, 7)],
        "xyzwpr": [float(pos[k]) for k in ("X", "Y", "Z", "W", "P", "R")],
        "configuration": cart.get("Configuration", {}),
    }


def cmd_read(args):
    rmi = ReadOnlyRmi(args.host, args.port, args.group)
    try:
        first = read_pair(rmi)
        if first is None:
            if args.no_init:
                raise RuntimeError("reads rejected without FRC_Initialize; rerun without --no-init")
            rmi.initialize()
            first = read_pair(rmi)
        time.sleep(0.3)
        second = read_pair(rmi)
    finally:
        rmi.close()
    if first is None or second is None:
        raise RuntimeError("controller rejected the reads")
    drift = max(abs(a - b) for a, b in zip(first["joints_deg"], second["joints_deg"], strict=True))
    if drift > 0.05:
        raise RuntimeError(f"arm is moving ({drift:.2f} deg between reads); stop it and read again")
    sample = {"label": args.label, "source": "rmi", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **second}
    append(args.samples, sample)


def cmd_add(args):
    v = args.values
    sample = {"label": args.label, "source": "pendant", "joints_deg": v[:6], "xyzwpr": v[6:], "configuration": {}}
    append(args.samples, sample)


def append(path: Path, sample):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = json.loads(path.read_text()) if path.is_file() else []
    samples.append(sample)
    path.write_text(json.dumps(samples, indent=1))
    cfg = sample["configuration"]
    print(f"[read] #{len(samples)} {sample['label'] or ''} J={np.round(sample['joints_deg'], 2).tolist()} "
          f"xyzwpr={np.round(sample['xyzwpr'], 1).tolist()} "
          f"UF={cfg.get('UFrameNumber')} UT={cfg.get('UToolNumber')} -> {path}")


# ------------------------------------------------------------------ analysis


def controller_pose(sample):
    x, y, z, w, p, r = sample["xyzwpr"]
    return wpr_matrix(w, p, r), np.array([x, y, z]) / 1000.0


def angle_deg(rotation):
    return float(np.degrees(Rotation.from_matrix(rotation).magnitude()))


def pairwise_rotation_error(samples, convention):
    """Relative rotation angle between two poses. Frame and tool rotation cancel out."""
    errors = []
    for a, b in itertools.combinations(samples, 2):
        fa, _ = fk.fk(np.radians(convention(np.array(a["joints_deg"]))))
        fb, _ = fk.fk(np.radians(convention(np.array(b["joints_deg"]))))
        ca, _ = controller_pose(a)
        cb, _ = controller_pose(b)
        errors.append(abs(angle_deg(fa.T @ fb) - angle_deg(ca.T @ cb)))
    return errors


def model_errors(samples, convention, utool):
    """Per-sample error of the driver's model (lerobot er4ia_fk) against the controller pose."""
    errors = []
    for sample in samples:
        rotation, position = controller_tcp_pose(np.radians(convention(np.array(sample["joints_deg"]))), utool)
        cr, cp = controller_pose(sample)
        errors.append((float(np.linalg.norm(position - cp) * 1000.0), angle_deg(rotation.T @ cr)))
    return errors


def cmd_analyze(args):
    samples = json.loads(args.samples.read_text())
    if len(samples) < 2:
        raise SystemExit("need at least 2 samples with different J2")
    frames = {(s["configuration"].get("UFrameNumber"), s["configuration"].get("UToolNumber")) for s in samples}
    if len(frames) > 1:
        print(f"[warn] samples use different UFrame/UTool {frames}; the model check assumes one", flush=True)
    if any(uf not in (0, None) for uf, _ in frames):
        print("[warn] the model check assumes UFrame 0 (World)", flush=True)
    j2 = [s["joints_deg"][1] for s in samples]
    print(f"{len(samples)} samples, J2 spread {max(j2) - min(j2):.1f} deg, UTOOL {args.utool}")
    if max(j2) - min(j2) < 15:
        print("[warn] J2 barely changes between samples; the conventions will look alike")

    print(f"\n{'convention':<18} {'pair rot err':>13} {'model pos err max':>19} {'model rot err max':>19}")
    for name, convention in CONVENTIONS.items():
        pair = pairwise_rotation_error(samples, convention)
        model = model_errors(samples, convention, args.utool)
        row = f"{name:<18} {max(pair):>9.2f} deg"
        row += f" {max(e[0] for e in model):>15.2f} mm {max(e[1] for e in model):>15.3f} deg"
        print(row)
    print("\npair rot err needs no frame or tool and only picks the J3 convention.")
    print("The model columns check the full driver chain (World, faceplate, UTOOL) per sample.")
    print("The right convention has both near 0; tens of mm or degrees means it is wrong.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    sub = parser.add_subparsers(dest="cmd", required=True)
    read = sub.add_parser("read", help="read joints + Cartesian pose over RMI (no motion)")
    read.add_argument("--host", default="172.30.109.22")
    read.add_argument("--port", type=int, default=16001)
    read.add_argument("--group", type=int, default=1)
    read.add_argument("--label", default="")
    read.add_argument("--no-init", action="store_true", help="never send FRC_Initialize")
    add = sub.add_parser("add", help="add a sample copied off the pendant")
    add.add_argument("values", type=float, nargs=12, metavar="V")
    add.add_argument("--label", default="")
    analyze = sub.add_parser("analyze", help="compare the conventions")
    analyze.add_argument("--utool", type=float, nargs=6, default=[0.0, 0.0, 223.0, 0.0, 0.0, 0.0],
                         metavar=("X", "Y", "Z", "W", "P", "R"), help="FRC_ReadUToolData of the sampled UTOOL")
    args = parser.parse_args()
    {"read": cmd_read, "add": cmd_add, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
