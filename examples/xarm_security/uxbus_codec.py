"""Minimal UFACTORY Uxbus (private Modbus-TCP) frame codec for demos."""

from __future__ import annotations

import struct
from dataclasses import dataclass

PRIVATE_PROTO = 0x0002
STANDARD_PROTO = 0x0000
MOVE_SERVO_CART_AA = 93


@dataclass(frozen=True)
class UxbusFrame:
    trans_id: int
    proto_id: int
    funcode: int
    pdu: bytes

    @property
    def raw(self) -> bytes:
        length = len(self.pdu) + 1
        return (
            struct.pack(">HHH", self.trans_id, self.proto_id, length)
            + bytes([self.funcode & 0xFF])
            + self.pdu
        )

    @property
    def hex(self) -> str:
        return self.raw.hex(" ")


@dataclass(frozen=True)
class MoveServoCartAa:
    x_mm: float
    y_mm: float
    z_mm: float
    rx_rad: float
    ry_rad: float
    rz_rad: float
    speed: float
    acc: float
    tool_coord: int
    relative: int

    @classmethod
    def from_pdu(cls, pdu: bytes) -> MoveServoCartAa:
        if len(pdu) < 37:
            raise ValueError(f"MOVE_SERVO_CART_AA PDU needs 37 bytes, got {len(pdu)}")
        floats = struct.unpack("<8f", pdu[:32])
        tool_coord = struct.unpack("<i", pdu[32:36])[0]
        relative = pdu[36]
        return cls(*floats, tool_coord, relative)

    def to_pdu(self) -> bytes:
        return (
            struct.pack("<8f", self.x_mm, self.y_mm, self.z_mm, self.rx_rad, self.ry_rad, self.rz_rad, self.speed, self.acc)
            + struct.pack("<i", self.tool_coord)
            + bytes([self.relative & 0xFF])
        )

    def summary(self) -> str:
        return (
            f"pose=({self.x_mm:.1f}, {self.y_mm:.1f}, {self.z_mm:.1f}) mm, "
            f"rot=({self.rx_rad:.3f}, {self.ry_rad:.3f}, {self.rz_rad:.3f}) rad, "
            f"speed={self.speed:.1f}, acc={self.acc:.1f}"
        )


def build_move_servo_cart_aa(trans_id: int, pose: MoveServoCartAa) -> UxbusFrame:
    return UxbusFrame(
        trans_id=trans_id,
        proto_id=PRIVATE_PROTO,
        funcode=MOVE_SERVO_CART_AA,
        pdu=pose.to_pdu(),
    )


def parse_frame(raw: bytes) -> UxbusFrame | None:
    if len(raw) < 7:
        return None
    trans_id, proto_id, length = struct.unpack(">HHH", raw[:6])
    funcode = raw[6]
    pdu_len = length - 1
    expected = 7 + pdu_len
    if len(raw) < expected:
        return None
    return UxbusFrame(trans_id=trans_id, proto_id=proto_id, funcode=funcode, pdu=raw[7:expected])


def decode_move_if_present(frame: UxbusFrame) -> MoveServoCartAa | None:
    if frame.funcode != MOVE_SERVO_CART_AA or frame.proto_id != PRIVATE_PROTO:
        return None
    return MoveServoCartAa.from_pdu(frame.pdu)


def hex_diff(a: bytes, b: bytes) -> list[str]:
    max_len = max(len(a), len(b))
    lines: list[str] = []
    for i in range(max_len):
        av = a[i] if i < len(a) else None
        bv = b[i] if i < len(b) else None
        if av != bv:
            left = ".." if av is None else f"{av:02x}"
            right = ".." if bv is None else f"{bv:02x}"
            lines.append(f"  offset {i:02d}: {left} -> {right}")
    return lines
