#!/usr/bin/env python3
"""
Forensic helper: detect suspicious Uxbus MOVE_SERVO_CART_AA packets after a stop.

Use when the arm halted on joint/pose jump while replay/teleop was running.
Compares wire captures against expected smooth trajectory and flags:
  - frozen pose while trans_id advances (classic replay attack)
  - backward / large forward jumps in commanded TCP pose
  - mismatch vs dataset action timeline (optional)

Examples
--------
# From MITM / proxy log:
python uxbus_forensic_check.py --jsonl uxbus_capture.jsonl

# From tcpdump (hex per line, e.g. tshark -T fields -e data):
python uxbus_forensic_check.py --hex-lines wire.hex

# Cross-check against dataset episode:
python uxbus_forensic_check.py --jsonl uxbus_capture.jsonl \\
    --dataset-csv episode0_actions.csv --jump-mm 15
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from uxbus_codec import MOVE_SERVO_CART_AA, decode_move_if_present, parse_frame


@dataclass
class WireSample:
    index: int
    trans_id: int
    x_mm: float
    y_mm: float
    z_mm: float
    source: str
    hex: str = ""


def _parse_hex_line(line: str) -> bytes | None:
    line = line.strip().replace(":", "").replace(" ", "")
    if not line:
        return None
    try:
        return bytes.fromhex(line)
    except ValueError:
        return None


def _frames_from_hex_lines(path: Path) -> list[WireSample]:
    samples: list[WireSample] = []
    idx = 0
    for line in path.read_text().splitlines():
        raw = _parse_hex_line(line)
        if raw is None:
            continue
        fr = parse_frame(raw)
        if fr is None:
            continue
        mv = decode_move_if_present(fr)
        if mv is None:
            continue
        samples.append(
            WireSample(
                index=idx,
                trans_id=fr.trans_id,
                x_mm=mv.x_mm,
                y_mm=mv.y_mm,
                z_mm=mv.z_mm,
                source="hex",
                hex=fr.hex,
            )
        )
        idx += 1
    return samples


def _samples_from_jsonl(path: Path) -> list[WireSample]:
    samples: list[WireSample] = []
    idx = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        hex_str = entry.get("on_wire_hex") or entry.get("hex")
        if not hex_str:
            continue
        raw = bytes.fromhex(hex_str.replace(" ", ""))
        fr = parse_frame(raw)
        if fr is None:
            continue
        mv = decode_move_if_present(fr)
        if mv is None:
            continue
        samples.append(
            WireSample(
                index=idx,
                trans_id=fr.trans_id,
                x_mm=mv.x_mm,
                y_mm=mv.y_mm,
                z_mm=mv.z_mm,
                source=entry.get("action", "jsonl"),
                hex=fr.hex,
            )
        )
        idx += 1
    return samples


def _load_dataset_xyz(path: Path) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((float(row["j0"]), float(row["j1"]), float(row["j2"])))
    return rows


def analyze(samples: list[WireSample], jump_mm: float, dataset: list[tuple[float, float, float]] | None) -> int:
    if not samples:
        print("No MOVE_SERVO_CART_AA (funcode=93) samples found.")
        return 1

    print(f"Loaded {len(samples)} cartesian-servo command sample(s).\n")

    findings = 0

    # 1) Consecutive jump on wire
    print("[A] Consecutive wire-command jumps (could trigger controller joint/pose jump stop)\n")
    for i in range(1, len(samples)):
        a, b = samples[i - 1], samples[i]
        dx = b.x_mm - a.x_mm
        dy = b.y_mm - a.y_mm
        dz = b.z_mm - a.z_mm
        dist = (dx * dx + dy * dy + dz * dz) ** 0.5
        if dist > jump_mm or dx < -1.0:
            findings += 1
            print(
                f"  ! sample {i-1}->{i}: Δ=({dx:+.1f},{dy:+.1f},{dz:+.1f}) mm, |Δ|={dist:.1f} mm "
                f"(trans {a.trans_id}->{b.trans_id})"
            )
            if dx < -5:
                print("      ^ backward X jump — uncommon in normal replay; check for stale replay")
    if findings == 0:
        print("  (no large consecutive jumps on wire)\n")
    else:
        print()

    # 2) Frozen pose while trans_id advances
    print("[B] Frozen TCP target while trans_id still increments (replay-attack signature)\n")
    frozen_hits = 0
    run_start = 0
    for i in range(1, len(samples)):
        prev, cur = samples[i - 1], samples[i]
        same_pose = (
            abs(cur.x_mm - prev.x_mm) < 0.05
            and abs(cur.y_mm - prev.y_mm) < 0.05
            and abs(cur.z_mm - prev.z_mm) < 0.05
        )
        tid_advances = cur.trans_id != prev.trans_id
        if same_pose and tid_advances:
            if frozen_hits == 0 or i - run_start > 1:
                frozen_hits += 1
                print(
                    f"  ! samples {i-1}->{i}: pose locked at "
                    f"({cur.x_mm:.1f},{cur.y_mm:.1f},{cur.z_mm:.1f}) mm "
                    f"but trans_id {prev.trans_id}->{cur.trans_id}"
                )
            run_start = i
    if frozen_hits == 0:
        print("  (no frozen-pose / advancing-trans_id pattern)\n")
    else:
        findings += frozen_hits
        print("  => Strong indicator someone replayed a stale funcode=93 PDU body.\n")

    # 3) Optional dataset cross-check
    if dataset:
        print("[C] Wire command vs dataset action (what PC *should* have sent)\n")
        n = min(len(samples), len(dataset))
        mism = 0
        for i in range(n):
            wx, wy, wz = samples[i].x_mm, samples[i].y_mm, samples[i].z_mm
            dx, dy, dz = dataset[i]
            err = ((wx - dx) ** 2 + (wy - dy) ** 2 + (wz - dz) ** 2) ** 0.5
            if err > jump_mm:
                mism += 1
                if mism <= 8:
                    print(
                        f"  ! index {i}: wire=({wx:.1f},{wy:.1f},{wz:.1f}) "
                        f"dataset=({dx:.1f},{dy:.1f},{dz:.1f}) err={err:.1f} mm"
                    )
        if mism == 0:
            print("  wire matches dataset within threshold\n")
        else:
            findings += 1
            print(f"  => {mism}/{n} frame(s) on wire ≠ dataset — packet tampering or wrong capture\n")

    # 4) jsonl explicit replace markers
    replaced = sum(1 for s in samples if s.source == "replaced_with_stale")
    if replaced:
        findings += 1
        print(f"[D] Proxy log marks {replaced} frame(s) as replaced_with_stale\n")

    print("=" * 60)
    if findings:
        print("VERDICT: SUSPICIOUS — wire commands show patterns consistent with packet attack.")
        print("Next: diff pcap vs benign baseline; check xArm Studio error log / error_code.")
    else:
        print("VERDICT: INCONCLUSIVE on wire alone — jump stop may be benign (EMA, bad dataset,")
        print("         mode switch, or real kinematic limit). Check controller error_code + report :30002.")
    print("=" * 60)
    return 0 if findings == 0 else 2


def main() -> int:
    p = argparse.ArgumentParser(description="Forensic Uxbus packet tampering check")
    p.add_argument("--jsonl", type=Path, help="Proxy capture log (uxbus_capture.jsonl)")
    p.add_argument("--hex-lines", type=Path, help="File with one hex payload per line")
    p.add_argument("--dataset-csv", type=Path, help="CSV with j0,j1,j2 columns")
    p.add_argument("--jump-mm", type=float, default=20.0, help="Jump threshold mm")
    args = p.parse_args()

    if not args.jsonl and not args.hex_lines:
        p.error("Provide --jsonl or --hex-lines")

    if args.jsonl:
        samples = _samples_from_jsonl(args.jsonl)
    else:
        samples = _frames_from_hex_lines(args.hex_lines)  # type: ignore[arg-type]

    dataset = _load_dataset_xyz(args.dataset_csv) if args.dataset_csv else None
    return analyze(samples, args.jump_mm, dataset)


if __name__ == "__main__":
    sys.exit(main())
