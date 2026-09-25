# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenPI-style client for a FANUC Cosmos Edge policy, plus RMI joint packets.

``manipulation.run_fanuc_live`` does not call this module. Nothing here opens
an RMI socket. ``send_packets`` only runs when the caller passes a ``send_json``.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Sequence

from cosmos_edge_fanuc.embodiment import (
    ACTION_DIM,
    ARM_JOINT_DIM,
    CHUNK_STEPS,
    CONDITIONING_FPS,
    DOMAIN_NAME,
    GRIPPER_CLOSED_THRESHOLD,
    POLICY_PORT,
    VIEW_DESCRIPTION,
    model_limit_errors,
    reject_port,
)

SendJson = Callable[[dict[str, Any]], None]


class ContractError(ValueError):
    """Action chunk does not match the FANUC embodiment."""


def observation(
    *,
    prompt: str,
    image_rgb: Any,
    joints_deg: Sequence[float],
    gripper: float,
) -> dict[str, Any]:
    """One policy request. Joints are pendant degrees; gripper is 0 or 1."""
    if len(joints_deg) != ARM_JOINT_DIM:
        raise ContractError(f"joint_position must have {ARM_JOINT_DIM} values, got {len(joints_deg)}")
    errors = model_limit_errors(joints_deg)
    if errors:
        raise ContractError("joint observation outside the URDF: " + "; ".join(errors))
    grip = 1.0 if float(gripper) >= GRIPPER_CLOSED_THRESHOLD else 0.0
    return {
        "prompt": prompt,
        "observation/image": image_rgb,
        "observation/joint_position": [float(v) for v in joints_deg],
        "observation/gripper_position": [grip],
        "domain_name": DOMAIN_NAME,
        "view_description": VIEW_DESCRIPTION,
        "conditioning_fps": CONDITIONING_FPS,
    }


def parse_action_chunk(response: dict[str, Any]) -> list[list[float]]:
    """Return ``CHUNK_STEPS x ACTION_DIM`` raw pendant degrees plus gripper.

    A 6-wide chunk is the SO-101 contract and is rejected.
    """
    raw = response.get("action", response.get("actions"))
    if raw is None:
        raise ContractError(f"Policy response has no action: keys={sorted(response)}")
    rows = [list(map(float, row)) for row in raw]
    if len(rows) != CHUNK_STEPS:
        raise ContractError(f"Expected {CHUNK_STEPS} steps, got {len(rows)}")
    width = len(rows[0])
    if width == 6:
        raise ContractError(
            "Got a 6-D chunk (SO-101: 5 arm joints + gripper in LeRobot .pos). "
            "Refusing to send it to FANUC."
        )
    if any(len(row) != ACTION_DIM for row in rows):
        raise ContractError(f"Expected width {ACTION_DIM} (J1..J6 + gripper), got {width}")
    for step, row in enumerate(rows):
        errors = model_limit_errors(row[:ARM_JOINT_DIM])
        if errors:
            raise ContractError(f"step {step} outside the URDF: " + "; ".join(errors))
        if not 0.0 <= row[ARM_JOINT_DIM] <= 1.0:
            raise ContractError(f"step {step} gripper {row[ARM_JOINT_DIM]} is outside [0, 1]")
    return rows


def joint_motion_packet(
    joints_deg: Sequence[float],
    *,
    sequence_id: int,
    speed_percent: int = 10,
    gripper: float | None = None,
) -> dict[str, Any]:
    """One ``FRC_JointMotion``. Gripper is a digital port pulse, same ports as FanucConfig."""
    errors = model_limit_errors(joints_deg)
    if errors:
        raise ContractError("refusing JointMotion outside the URDF: " + "; ".join(errors))
    packet: dict[str, Any] = {
        "Instruction": "FRC_JointMotion",
        "SequenceID": int(sequence_id),
        "JointAngle": {f"J{i + 1}": float(joints_deg[i]) for i in range(ARM_JOINT_DIM)},
        "SpeedType": "Percent",
        "Speed": int(speed_percent),
        "TermType": "CNT",
        "TermValue": 50,
    }
    if gripper is not None:
        closed = float(gripper) >= GRIPPER_CLOSED_THRESHOLD
        packet.update(
            {
                "LCBType": "TA",
                "LCBValue": 10,
                "PortType": 2,
                "PortNumber": 4 if closed else 3,
                "PortValue": "ON",
            }
        )
    return packet


def packets_from_chunk(
    chunk: Sequence[Sequence[float]],
    *,
    sequence_start: int,
    speed_percent: int = 10,
    gripper_state: float | None = None,
) -> list[dict[str, Any]]:
    """Joint packets for one chunk. Gripper ports are attached only when the bit changes."""
    rows = parse_action_chunk({"action": chunk})
    packets = []
    previous = gripper_state
    for offset, row in enumerate(rows):
        grip = row[ARM_JOINT_DIM]
        discrete = 1.0 if grip >= GRIPPER_CLOSED_THRESHOLD else 0.0
        include = previous is None or discrete != (1.0 if previous >= GRIPPER_CLOSED_THRESHOLD else 0.0)
        packets.append(
            joint_motion_packet(
                row[:ARM_JOINT_DIM],
                sequence_id=sequence_start + offset,
                speed_percent=speed_percent,
                gripper=discrete if include else None,
            )
        )
        previous = discrete
    return packets


def send_packets(packets: Sequence[dict[str, Any]], send_json: SendJson) -> None:
    """Hand packets to an already-open RMI session. Does not connect by itself."""
    for packet in packets:
        if packet.get("Instruction") != "FRC_JointMotion":
            raise ContractError(f"Refusing non-joint packet: {packet.get('Instruction')}")
        send_json(packet)


def request_chunk(uri: str, obs: dict[str, Any], *, timeout_s: float = 120.0) -> list[list[float]]:
    """One OpenPI websocket round-trip. Does not talk to the robot."""
    reject_port(_port_of(uri))
    try:
        import msgpack
        import websockets.sync.client
    except ImportError as exc:
        raise RuntimeError(
            "Policy requests need the websockets and msgpack packages in this environment."
        ) from exc

    with websockets.sync.client.connect(uri, open_timeout=timeout_s, close_timeout=5) as ws:
        meta_raw = ws.recv(timeout=timeout_s)
        meta = msgpack.unpackb(meta_raw, raw=False) if isinstance(meta_raw, bytes) else json.loads(meta_raw)
        if isinstance(meta, dict):
            domain = str(meta.get("domain_name", meta.get("domain", ""))).lower()
            if "so101" in domain:
                raise ContractError(f"Server at {uri} advertised SO-101 ({domain!r})")
        ws.send(msgpack.packb(obs))
        reply_raw = ws.recv(timeout=timeout_s)
        reply = msgpack.unpackb(reply_raw, raw=False) if isinstance(reply_raw, bytes) else json.loads(reply_raw)
    if not isinstance(reply, dict):
        raise ContractError(f"Policy reply is {type(reply).__name__}, expected an object")
    return parse_action_chunk(reply)


def _port_of(uri: str) -> int:
    hostport = uri.split("://", 1)[-1].split("/", 1)[0]
    if ":" not in hostport:
        raise ContractError(f"Policy URI {uri!r} has no port")
    return int(hostport.rsplit(":", 1)[-1])


def _dry_run(prompt: str) -> dict[str, Any]:
    joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    obs = observation(prompt=prompt, image_rgb=None, joints_deg=joints, gripper=0.0)
    chunk = [joints + [0.0] for _ in range(CHUNK_STEPS)]
    chunk[3][ARM_JOINT_DIM] = 1.0
    packets = packets_from_chunk(chunk, sequence_start=1, gripper_state=0.0)
    printable = {key: value for key, value in obs.items() if key != "observation/image"}
    return {"observation": printable, "n_packets": len(packets), "first_packet": packets[0], "gripper_packet": packets[3]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FANUC Cosmos Edge joint client (no RMI connect)")
    parser.add_argument("--dry-run", action="store_true", help="Print one local chunk. Does not open sockets.")
    parser.add_argument("--prompt", default="pick the black cube and put it into the blue bin")
    parser.add_argument("--policy", default=f"ws://127.0.0.1:{POLICY_PORT}")
    args = parser.parse_args(argv)
    reject_port(_port_of(args.policy))
    if not args.dry_run:
        parser.error("Pass --dry-run. This CLI does not connect to the FANUC controller.")
    print(json.dumps(_dry_run(args.prompt), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
