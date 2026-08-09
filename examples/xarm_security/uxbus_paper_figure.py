#!/usr/bin/env python3
"""Generate publication-ready tables / cleartext packet decode for replay-attack demo."""

from __future__ import annotations

import struct

from uxbus_codec import (
    MOVE_SERVO_CART_AA,
    MoveServoCartAa,
    UxbusFrame,
    build_move_servo_cart_aa,
    decode_move_if_present,
    PRIVATE_PROTO,
)


def _traj(n: int, x0: float, step: float) -> list[MoveServoCartAa]:
    return [
        MoveServoCartAa(x0 + i * step, 0.0, 400.0, 3.14, 0.0, 0.0, 300.0, 2000.0, 0, 0)
        for i in range(n)
    ]


def decode_cleartext(raw: bytes) -> dict:
    fr = None
    from uxbus_codec import parse_frame

    fr = parse_frame(raw)
    assert fr is not None
    mv = decode_move_if_present(fr)
    assert mv is not None
    return {
        "trans_id": fr.trans_id,
        "proto_id": f"0x{fr.proto_id:04X}",
        "length": struct.unpack(">H", raw[4:6])[0],
        "funcode": fr.funcode,
        "funcode_hex": f"0x{fr.funcode:02X}",
        "x_mm": mv.x_mm,
        "y_mm": mv.y_mm,
        "z_mm": mv.z_mm,
        "rx_rad": mv.rx_rad,
        "ry_rad": mv.ry_rad,
        "rz_rad": mv.rz_rad,
        "speed": mv.speed,
        "acc": mv.acc,
        "raw_hex": fr.hex,
    }


def packet_layout_markdown() -> str:
    return """
### Figure A — Uxbus `MOVE_SERVO_CART_AA` cleartext layout (TCP payload)

| Byte offset | Size | Field | Example (benign frame 1) |
|-------------|------|-------|--------------------------|
| 0–1 | 2 | Transaction ID (BE u16) | `0x0001` |
| 2–3 | 2 | Protocol ID | `0x0002` (UFACTORY private) |
| 4–5 | 2 | Length (= PDU+1) | `0x0026` (38) |
| 6 | 1 | Function code | `0x5D` (93 = cartesian servo, axis-angle) |
| 7–10 | 4 | **x** (float32 LE, mm) | `43 96 00 00` → **300.0** |
| 11–14 | 4 | **y** (float32 LE, mm) | `00 00 00 00` → **0.0** |
| 15–18 | 4 | **z** (float32 LE, mm) | `00 00 C8 43` → **400.0** |
| 19–22 | 4 | rx (float32 LE, rad) | … → **3.14** |
| 23–26 | 4 | ry (float32 LE, rad) | … → **0.0** |
| 27–30 | 4 | rz (float32 LE, rad) | … → **0.0** |
| 31–34 | 4 | speed (float32 LE) | … → **300.0** |
| 35–38 | 4 | acceleration (float32 LE) | … → **2000.0** |
| 39–42 | 4 | tool_coord (int32 LE) | `00 00 00 00` |
| 43 | 1 | relative flag (u8) | `00` |
"""


def main() -> None:
    benign = [build_move_servo_cart_aa(i + 1, p) for i, p in enumerate(_traj(3, 300, 2))]
    stale_pdu = benign[0].pdu
    intended = build_move_servo_cart_aa(42, _traj(1, 308, 0)[0])
    attacked = UxbusFrame(42, PRIVATE_PROTO, MOVE_SERVO_CART_AA, stale_pdu)

    b0 = decode_cleartext(benign[0].raw)
    b1 = decode_cleartext(benign[1].raw)
    intend = decode_cleartext(intended.raw)
    wire = decode_cleartext(attacked.raw)

    print("# Uxbus Replay Attack — Paper Tables (copy into article)\n")
    print(packet_layout_markdown())

    print("\n### Table 1 — Benign replay: commanded TCP movement evolves on the wire\n")
    print("| Frame | Trans. ID | x (mm) | y (mm) | z (mm) | Δx vs prev (mm) | PDU x bytes (LE) |")
    print("|-------|-----------|--------|--------|--------|-----------------|------------------|")
    prev_x = None
    for i, fr in enumerate(benign):
        d = decode_cleartext(fr.raw)
        dx = "—" if prev_x is None else f"{d['x_mm'] - prev_x:+.1f}"
        x_bytes = " ".join(f"{b:02x}" for b in fr.raw[7:11])
        print(f"| {i} | {d['trans_id']} | {d['x_mm']:.1f} | {d['y_mm']:.1f} | {d['z_mm']:.1f} | {dx} | `{x_bytes}` |")
        prev_x = d["x_mm"]

    print("\n### Table 2 — Replay attack at one timestep: PC intent vs bytes on TCP:502\n")
    print("| Source | Trans. ID | x (mm) | y (mm) | z (mm) | x PDU bytes | Interpretation |")
    print("|--------|-----------|--------|--------|--------|-------------|----------------|")
    ix = " ".join(f"{b:02x}" for b in intended.raw[7:11])
    wx = " ".join(f"{b:02x}" for b in attacked.raw[7:11])
    print(
        f"| PC / dataset intent | {intend['trans_id']} | {intend['x_mm']:.1f} | "
        f"{intend['y_mm']:.1f} | {intend['z_mm']:.1f} | `{ix}` | next replay step |"
    )
    print(
        f"| **On wire (attacked)** | {wire['trans_id']} | {wire['x_mm']:.1f} | "
        f"{wire['y_mm']:.1f} | {wire['z_mm']:.1f} | `{wx}` | **stale replay of frame 0** |"
    )
    print(
        f"\n**Evidence:** same `trans_id={wire['trans_id']}` header freshness, but PDU x bytes "
        f"`{wx}` decode to **{wire['x_mm']:.1f} mm** instead of **{intend['x_mm']:.1f} mm** "
        f"(Δ = {wire['x_mm'] - intend['x_mm']:+.1f} mm)."
    )

    print("\n### Table 3 — Full cleartext decode (single attacked packet)\n")
    print("| Layer | Field | Value |")
    print("|-------|-------|-------|")
    for k in (
        "trans_id", "proto_id", "length", "funcode_hex",
        "x_mm", "y_mm", "z_mm", "rx_rad", "ry_rad", "rz_rad", "speed", "acc",
    ):
        print(f"| Uxbus | {k} | {wire[k]} |")
    print(f"| Raw | full frame (hex) | `{wire['raw_hex']}` |")

    print("\n### Suggested figure caption\n")
    print(
        "> **Fig. X.** Cleartext decode of UFACTORY Uxbus `MOVE_SERVO_CART_AA` (funcode 0x5D) "
        "packets on TCP port 502. During a replay attack, the transaction ID advances normally "
        "while the PDU position field is replaced with a captured stale value (x: 308.0 mm intended "
        f"→ {wire['x_mm']:.1f} mm on wire), causing a backward command jump and controller safety stop."
    )

    print("\n### LaTeX snippet (Table 2)\n")
    print(r"""```latex
\begin{table}[t]
\centering
\caption{PC-intended vs on-wire TCP command at attack timestep.}
\begin{tabular}{lrrrrl}
\toprule
Source & Trans.\ ID & $x$ (mm) & $y$ (mm) & $z$ (mm) & PDU $x$ bytes \\
\midrule""")
    print(
        f"PC intent & {intend['trans_id']} & {intend['x_mm']:.1f} & {intend['y_mm']:.1f} & "
        f"{intend['z_mm']:.1f} & \\texttt{{{ix}}} \\\\"
    )
    print(
        f"On wire (attacked) & {wire['trans_id']} & {wire['x_mm']:.1f} & {wire['y_mm']:.1f} & "
        f"{wire['z_mm']:.1f} & \\texttt{{{wx}}} \\\\"
    )
    print(r"""\bottomrule
\end{tabular}
\end{table}
```""")


if __name__ == "__main__":
    main()
