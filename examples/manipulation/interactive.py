# Copyright 2026 The Hugging Face Inc. team. All rights reserved.
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

"""
Interactive loop: home → photograph → plan VLM → act or wait.

After every go-home the camera and both VLMs run automatically. The planner
only decides act / wait / done. No game-specific rules.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

PerceiveFn = Callable[[], tuple[Mapping[str, Any], Mapping[str, Any]]]
DecideFn = Callable[[Mapping[str, Any], str, Sequence[str], str | None], Mapping[str, Any]]
ExecuteFn = Callable[[list[dict[str, Any]], Mapping[str, Any]], str]
HomeFn = Callable[[], None]


def _object_names(view: Mapping[str, Any]) -> list[str]:
    return [str(o.get("name")) for o in (view.get("objects") or []) if o.get("name")]


def _object_index(view: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in view.get("objects") or []:
        if not isinstance(raw, Mapping):
            continue
        name = raw.get("name")
        if name:
            out[str(name)] = dict(raw)
    return out


def _footprint_area(obj: Mapping[str, Any]) -> float:
    wkt_s = obj.get("footprint_wkt")
    if not wkt_s:
        return 0.0
    try:
        from shapely import wkt

        return float(wkt.loads(str(wkt_s)).area)
    except Exception:  # noqa: BLE001
        return 0.0


def _remember_surfaces(
    sticky_sym: dict[str, dict[str, Any]],
    sticky_geo: dict[str, dict[str, Any]],
    symbolic: Mapping[str, Any],
    geometric: Mapping[str, Any],
) -> None:
    """Keep large surfaces (board / tray / stand) across flaky perception frames."""
    geos = [o for o in (geometric.get("objects") or []) if isinstance(o, Mapping) and o.get("name")]
    if not geos:
        return
    areas = [_footprint_area(g) for g in geos]
    piece = min(areas) if areas else 0.0
    # Boards / trays / stands are much larger than pieces; also keep anything ≥ 80 cm².
    thresh = max(piece * 6.0, 0.008)
    by_sym = _object_index(symbolic)
    for geo, area in zip(geos, areas, strict=True):
        if area < thresh:
            continue
        name = str(geo["name"])
        sticky_geo[name] = dict(geo)
        if name in by_sym:
            sticky_sym[name] = dict(by_sym[name])
        logger.info("interactive remember surface %s area=%.4f", name, area)


def _apply_sticky(
    sticky_sym: dict[str, dict[str, Any]],
    sticky_geo: dict[str, dict[str, Any]],
    symbolic: Mapping[str, Any],
    geometric: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _remember_surfaces(sticky_sym, sticky_geo, symbolic, geometric)
    present = set(_object_names(geometric)) | set(_object_names(symbolic))
    sym_objs = [dict(o) for o in (symbolic.get("objects") or []) if isinstance(o, Mapping)]
    geo_objs = [dict(o) for o in (geometric.get("objects") or []) if isinstance(o, Mapping)]
    for name, geo in sticky_geo.items():
        if name in present:
            continue
        geo_objs.append(dict(geo))
        if name in sticky_sym:
            sym_objs.append(dict(sticky_sym[name]))
        logger.info("interactive restore surface %s from earlier look", name)
    return {**dict(symbolic), "objects": sym_objs}, {**dict(geometric), "objects": geo_objs}


def run_interactive(
    instruction: str,
    *,
    arms: Sequence[str],
    perceive: PerceiveFn,
    decide: DecideFn,
    execute: ExecuteFn,
    go_home: HomeFn | None = None,
    wait_s: float = 2.0,
    on_phase: Callable[[str, str], None] | None = None,
) -> int:
    """
    ``home → perceive → plan`` after startup and after every robot move.

    On ``wait``, sleep ``wait_s`` (already home) and look again.
    """
    history_lines: list[str] = []
    sticky_sym: dict[str, dict[str, Any]] = {}
    sticky_geo: dict[str, dict[str, Any]] = {}
    last_act_names: list[str] | None = None
    snapshot_after_act = False
    look_i = 0

    def _emit(phase: str, detail: str = "") -> None:
        if on_phase is not None:
            on_phase(phase, detail)

    def _history(symbolic: Mapping[str, Any]) -> str:
        names = _object_names(symbolic)
        lines = list(history_lines[-8:])
        if last_act_names is not None:
            lines.append(f"objects right after last robot act: {last_act_names}")
        lines.append(f"current objects: {names}")
        multi = len(arms) >= 2
        if last_act_names is None:
            lines.append(
                "No robot act yet. If the instruction says the robot starts "
                "or is first player, status=act."
            )
        elif multi:
            lines.append(
                "Two robot arms are the players (the 'opponent' is the other "
                "arm, not a human). Scene names matching the last act is "
                "normal. status=act for the OTHER arm's next piece unless "
                "the board is full / no unused pieces remain (done). "
                "Do not wait for a person."
            )
        elif names == last_act_names:
            lines.append(
                "Objects match the last robot act. Wait only if a human "
                "opponent has not placed yet."
            )
        else:
            lines.append(
                "Current objects differ from after the last robot act. If this "
                "is a take-turns task vs a human, they likely moved — status=act "
                "unless the difference is only a rename of the same pieces."
            )
        lines.append("Judge THIS scene. Do not copy a previous wait.")
        return "\n".join(lines)

    def _look() -> tuple[str, str, list[dict[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
        nonlocal last_act_names, snapshot_after_act, look_i
        look_i += 1
        print(f"Look {look_i}: photograph + plan …")
        _emit("perceiving", f"look {look_i}")
        symbolic, geometric = perceive()
        symbolic, geometric = _apply_sticky(sticky_sym, sticky_geo, symbolic, geometric)
        names = _object_names(symbolic)
        if snapshot_after_act:
            last_act_names = names
            snapshot_after_act = False
            logger.info("interactive snapshot after act objects=%s", names)

        try:
            _emit("planning", instruction)
            decision = decide(symbolic, instruction, list(arms), _history(symbolic))
        except Exception as exc:  # noqa: BLE001
            logger.exception("interactive decide failed")
            print(f"  → wait: planner rejected the plan ({exc})")
            return "wait", str(exc), [], geometric, symbolic
        status = str(decision.get("status") or "wait")
        reason = str(decision.get("reason") or "")
        plan = list(decision.get("plan") or [])
        if plan:
            status = "act"
        elif status == "act":
            status = "wait"
        logger.info("interactive decide status=%s reason=%s steps=%d", status, reason, len(plan))
        print(f"  → {status}" + (f": {reason}" if reason else ""))
        return status, reason, plan, geometric, symbolic

    print("Interactive: go home → photograph → plan. Ctrl-C to quit.")
    if go_home is not None:
        _emit("go_home", "start")
        go_home()

    try:
        while True:
            status, reason, plan, geometric, symbolic = _look()
            if status == "done":
                _emit("done", reason)
                print(f"Done. {reason}".rstrip())
                return 0
            if status == "act":
                exec_status = execute(plan, geometric)
                placed = []
                acted_arms: list[str] = []
                for step in plan:
                    prim = step.get("primitive")
                    args = step.get("args") or {}
                    arm = step.get("arm")
                    if arm:
                        acted_arms.append(str(arm))
                    if prim == "Place":
                        placed.append(f"{args.get('object')}→{args.get('destination')}")
                arm_note = ",".join(dict.fromkeys(acted_arms)) or "?"
                history_lines.append(
                    f"robot act arm={arm_note}: "
                    f"{', '.join(placed) or reason or f'{len(plan)} step(s)'} "
                    f"→ {exec_status}"
                )
                geo_by_name = _object_index(geometric)
                sym_by_name = _object_index(symbolic)
                # Keep Place destinations even if the next photo drops them.
                for step in plan:
                    dest = (step.get("args") or {}).get("destination")
                    if step.get("primitive") != "Place" or not dest:
                        continue
                    if dest in geo_by_name:
                        sticky_geo[str(dest)] = dict(geo_by_name[str(dest)])
                    if dest in sym_by_name:
                        sticky_sym[str(dest)] = dict(sym_by_name[str(dest)])
                if exec_status != "success":
                    _emit("failed", exec_status)
                    print(f"Execute failed: {exec_status}")
                    return 1
                snapshot_after_act = True
                if go_home is not None:
                    _emit("go_home", "")
                    go_home()
                print("Home. Next look starts now.")
                continue
            _emit("waiting", reason or "human turn")
            print(f"Waiting {wait_s:.1f}s then look again. {reason}".rstrip())
            time.sleep(max(wait_s, 0.0))
    except KeyboardInterrupt:
        print("Stopped.")
        return 0
