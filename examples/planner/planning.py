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

"""Symbolic task planning: instruction + symbolic_view → primitive plan (no geometry)."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypedDict

from .json_util import strip_markdown_fences
from .prompts import DEFAULT_PLANNING_PROMPT

GEMINI_ROBOTICS_ER_MODEL = "gemini-robotics-er-2-preview"

ALLOWED_PRIMITIVES = frozenset({"Grasp", "Place", "LiftUp"})
SYMBOLIC_DESTINATIONS = frozenset({"free_space", "handover"})


class PlanStep(TypedDict):
    step: int
    arm: str
    primitive: str
    args: dict[str, Any]
    depends_on: list[int]


class PlanValidationError(Exception):
    """Typed validation failure for a planner VLM response."""

    def __init__(self, rule: str, message: str, *, step: int | None = None) -> None:
        self.rule = rule
        self.step = step
        self.message = message
        where = f" (step {step})" if step is not None else ""
        super().__init__(f"[{rule}]{where}: {message}")


PlannerVlmCaller = Callable[[dict[str, Any], str, list[str], str | None], list[PlanStep]]


def _object_names(symbolic_view: Mapping[str, Any]) -> set[str]:
    objects = symbolic_view.get("objects", [])
    if not isinstance(objects, list):
        raise PlanValidationError("scene", "symbolic_view['objects'] must be a list")
    names: set[str] = set()
    for obj in objects:
        if not isinstance(obj, Mapping) or "name" not in obj:
            raise PlanValidationError("scene", f"Invalid object entry: {obj!r}")
        names.add(str(obj["name"]))
    return names


def _in_workspace_map(symbolic_view: Mapping[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for obj in symbolic_view.get("objects", []):
        name = str(obj["name"])
        ws = obj.get("in_workspace", [])
        if ws is None:
            ws = []
        if not isinstance(ws, list):
            raise PlanValidationError("scene", f"in_workspace for {name!r} must be a list")
        out[name] = [str(a) for a in ws]
    return out


def _contains_number(value: Any, *, path: str) -> str | None:
    """Return a path string if ``value`` contains a numeric leaf (not bool)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return path
    if isinstance(value, Mapping):
        for k, v in value.items():
            hit = _contains_number(v, path=f"{path}.{k}")
            if hit:
                return hit
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            hit = _contains_number(v, path=f"{path}[{i}]")
            if hit:
                return hit
    return None


def _normalize_plan_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and "plan" in payload:
        payload = payload["plan"]
    if not isinstance(payload, list):
        raise PlanValidationError("parse", f"Plan payload must be a list, got {type(payload)!r}")
    return payload


def parse_plan_json_text(text: str) -> list[PlanStep]:
    """Parse VLM text into raw plan steps (no semantic validation)."""
    try:
        payload = json.loads(strip_markdown_fences(text))
    except json.JSONDecodeError as exc:
        raise PlanValidationError("parse", f"Malformed JSON: {exc}") from exc

    raw_steps = _normalize_plan_payload(payload)
    steps: list[PlanStep] = []
    for item in raw_steps:
        if not isinstance(item, Mapping):
            raise PlanValidationError("parse", f"Plan step must be an object, got {item!r}")
        try:
            step_no = int(item["step"])
            arm = str(item["arm"])
            primitive = str(item["primitive"])
            args = dict(item.get("args") or {})
            depends = [int(d) for d in (item.get("depends_on") or [])]
        except (KeyError, TypeError, ValueError) as exc:
            raise PlanValidationError("parse", f"Invalid step fields: {item!r} ({exc})") from exc
        steps.append(
            {
                "step": step_no,
                "arm": arm,
                "primitive": primitive,
                "args": args,
                "depends_on": depends,
            }
        )
    return steps


def _topo_ready_order(steps: Sequence[PlanStep]) -> list[PlanStep]:
    """Return steps in dependency order (ready set drained by ascending step id)."""
    by_id = {s["step"]: s for s in steps}
    if len(by_id) != len(steps):
        raise PlanValidationError("e", "Duplicate step numbers in plan")

    completed: set[int] = set()
    ordered: list[PlanStep] = []
    remaining = set(by_id)

    while remaining:
        ready = [
            sid
            for sid in remaining
            if all(dep in completed for dep in by_id[sid]["depends_on"])
        ]
        if not ready:
            raise PlanValidationError(
                "e",
                f"Dependency cycle or unsatisfiable depends_on among steps {sorted(remaining)}",
            )
        ready.sort()
        sid = ready[0]
        ordered.append(by_id[sid])
        remaining.remove(sid)
        completed.add(sid)
    return ordered


def validate_plan(
    plan: Sequence[Mapping[str, Any]],
    symbolic_view: Mapping[str, Any],
    arms: Sequence[str],
) -> list[PlanStep]:
    """
    Pure validation of a symbolic plan.

    Raises :class:`PlanValidationError` on the first failing rule.
    """
    arms_set = {str(a) for a in arms}
    if not arms_set:
        raise PlanValidationError("arms", "arms list must be non-empty")

    object_names = _object_names(symbolic_view)
    workspace = _in_workspace_map(symbolic_view)

    # Normalize into PlanStep list first.
    steps: list[PlanStep] = []
    for item in plan:
        if not isinstance(item, Mapping):
            raise PlanValidationError("parse", f"Plan step must be an object, got {item!r}")
        try:
            step_no = int(item["step"])
            arm = str(item["arm"])
            primitive = str(item["primitive"])
            args = dict(item.get("args") or {})
            depends = [int(d) for d in (item.get("depends_on") or [])]
        except (KeyError, TypeError, ValueError) as exc:
            raise PlanValidationError("parse", f"Invalid step fields: {item!r} ({exc})") from exc
        steps.append(
            {
                "step": step_no,
                "arm": arm,
                "primitive": primitive,
                "args": args,
                "depends_on": depends,
            }
        )

    if not steps:
        raise PlanValidationError("parse", "Plan is empty")

    step_ids = {s["step"] for s in steps}

    for s in steps:
        sid = s["step"]

        # (a) arm membership
        if s["arm"] not in arms_set:
            raise PlanValidationError(
                "a",
                f"Unknown arm {s['arm']!r}; allowed={sorted(arms_set)}",
                step=sid,
            )

        # (b) primitive
        if s["primitive"] not in ALLOWED_PRIMITIVES:
            raise PlanValidationError(
                "b",
                f"Unknown primitive {s['primitive']!r}; "
                f"allowed={sorted(ALLOWED_PRIMITIVES)}",
                step=sid,
            )

        # (d) no numbers in args
        num_path = _contains_number(s["args"], path="args")
        if num_path is not None:
            raise PlanValidationError(
                "d",
                f"Args must not contain numbers ({num_path}={s['args']!r})",
                step=sid,
            )

        # (c) object names
        args = s["args"]
        prim = s["primitive"]
        if prim == "Grasp":
            obj = args.get("object")
            if not isinstance(obj, str) or not obj:
                raise PlanValidationError("c", "Grasp requires string args.object", step=sid)
            if obj not in object_names:
                raise PlanValidationError(
                    "c",
                    f"Unknown object {obj!r} in Grasp",
                    step=sid,
                )
        elif prim == "Place":
            obj = args.get("object")
            dest = args.get("destination")
            if not isinstance(obj, str) or not obj:
                raise PlanValidationError("c", "Place requires string args.object", step=sid)
            if not isinstance(dest, str) or not dest:
                raise PlanValidationError(
                    "c", "Place requires string args.destination", step=sid
                )
            if obj not in object_names:
                raise PlanValidationError(
                    "c",
                    f"Unknown object {obj!r} in Place",
                    step=sid,
                )
            if dest not in SYMBOLIC_DESTINATIONS and dest not in object_names:
                raise PlanValidationError(
                    "c",
                    f"Unknown destination {dest!r} in Place "
                    f"(not an object and not free_space/handover)",
                    step=sid,
                )
        elif prim == "LiftUp":
            # Optional object name for clarity; if present must exist.
            obj = args.get("object")
            if obj is not None:
                if not isinstance(obj, str) or obj not in object_names:
                    raise PlanValidationError(
                        "c",
                        f"Unknown object {obj!r} in LiftUp",
                        step=sid,
                    )

        # (e) depends_on refer only to smaller step numbers
        for dep in s["depends_on"]:
            if dep not in step_ids:
                raise PlanValidationError(
                    "e",
                    f"depends_on references missing step {dep}",
                    step=sid,
                )
            if dep >= sid:
                raise PlanValidationError(
                    "e",
                    f"depends_on entry {dep} is not a smaller step number than {sid}",
                    step=sid,
                )

    # (e) acyclic + produce execution order
    ordered = _topo_ready_order(steps)

    # (f) gripper state + (g) reachability / handover exception
    holding: dict[str, str | None] = {a: None for a in arms_set}
    # Objects currently sitting at the symbolic handover spot.
    at_handover: set[str] = set()

    for s in ordered:
        sid = s["step"]
        arm = s["arm"]
        prim = s["primitive"]
        args = s["args"]

        if prim == "Grasp":
            obj = str(args["object"])
            if holding[arm] is not None:
                raise PlanValidationError(
                    "f",
                    f"Grasp requires empty gripper, but {arm!r} holds {holding[arm]!r}",
                    step=sid,
                )
            reachable = arm in workspace.get(obj, [])
            if not reachable and obj not in at_handover:
                raise PlanValidationError(
                    "g",
                    f"Object {obj!r} is not in workspace of {arm!r} and is not at handover",
                    step=sid,
                )
            holding[arm] = obj
            at_handover.discard(obj)

        elif prim == "Place":
            obj = str(args["object"])
            dest = str(args["destination"])
            if holding[arm] != obj:
                raise PlanValidationError(
                    "f",
                    f"Place requires {arm!r} to hold {obj!r}, "
                    f"but it holds {holding[arm]!r}",
                    step=sid,
                )
            holding[arm] = None
            if dest == "handover":
                at_handover.add(obj)
            else:
                at_handover.discard(obj)

        elif prim == "LiftUp":
            held = holding[arm]
            if held is None:
                raise PlanValidationError(
                    "f",
                    f"LiftUp requires {arm!r} to hold an object",
                    step=sid,
                )
            obj = args.get("object")
            if obj is not None and str(obj) != held:
                raise PlanValidationError(
                    "f",
                    f"LiftUp object {obj!r} does not match held {held!r}",
                    step=sid,
                )

    return [{**s, "depends_on": list(s["depends_on"]), "args": dict(s["args"])} for s in steps]


def build_planning_prompt(
    symbolic_view: Mapping[str, Any],
    instruction: str,
    arms: Sequence[str],
    *,
    prompt_template: str | None = None,
    feedback: str | None = None,
) -> str:
    template = prompt_template or DEFAULT_PLANNING_PROMPT
    feedback_block = ""
    if feedback:
        feedback_block = (
            "PREVIOUS PLAN WAS REJECTED\n"
            f"{feedback}\n"
            "Fix the violation and return a corrected plan.\n"
        )
    return template.format(
        scene_json=json.dumps(symbolic_view, indent=2, ensure_ascii=False),
        instruction=instruction,
        arms_json=json.dumps(list(arms), ensure_ascii=False),
        feedback_block=feedback_block,
    )


def call_planner_vlm(
    symbolic_view: Mapping[str, Any],
    instruction: str,
    arms: Sequence[str],
    *,
    prompt_template: str | None = None,
    feedback: str | None = None,
    model: str = GEMINI_ROBOTICS_ER_MODEL,
    api_key: str | None = None,
) -> list[PlanStep]:
    """
    Single swappable VLM entry point for planning (text-only).

    On malformed JSON: retry the API once, then raise.
    """
    from google import genai
    from google.genai import types

    prompt_text = build_planning_prompt(
        symbolic_view,
        instruction,
        arms,
        prompt_template=prompt_template,
        feedback=feedback,
    )
    key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=key) if key else genai.Client()

    last_error: Exception | None = None
    text = ""
    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt_text],
                config=types.GenerateContentConfig(temperature=0.2),
            )
            text = getattr(response, "text", None) or ""
            return parse_plan_json_text(text)
        except PlanValidationError as exc:
            last_error = exc
            if attempt == 0:
                continue
            raise PlanValidationError(
                "parse",
                f"Malformed plan JSON after retry. Last response was:\n{text[:2000]}",
            ) from last_error
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 0:
                continue
            raise RuntimeError(f"Planner VLM request failed: {exc}") from last_error

    raise RuntimeError("Unreachable")  # pragma: no cover


class Planner:
    """Wraps a planner VLM call + validation retries (no geometry)."""

    def __init__(
        self,
        *,
        vlm_caller: PlannerVlmCaller | None = None,
        prompt_template: str | None = None,
        max_attempts: int = 2,
    ) -> None:
        self.prompt_template = prompt_template
        self.max_attempts = max(1, int(max_attempts))
        self.vlm_caller: PlannerVlmCaller = vlm_caller or (
            lambda scene, instruction, arms, feedback: call_planner_vlm(
                scene,
                instruction,
                arms,
                prompt_template=self.prompt_template,
                feedback=feedback,
            )
        )

    def __call__(
        self,
        symbolic_view: Mapping[str, Any],
        instruction: str,
        arms: Sequence[str],
    ) -> list[PlanStep]:
        return self.run(symbolic_view, instruction, arms)

    def run(
        self,
        symbolic_view: Mapping[str, Any],
        instruction: str,
        arms: Sequence[str],
    ) -> list[PlanStep]:
        feedback: str | None = None
        last_exc: PlanValidationError | None = None

        for _attempt in range(self.max_attempts):
            raw_plan = self.vlm_caller(
                dict(symbolic_view),
                instruction,
                list(arms),
                feedback,
            )
            try:
                return validate_plan(raw_plan, symbolic_view, arms)
            except PlanValidationError as exc:
                last_exc = exc
                feedback = str(exc)
                continue

        assert last_exc is not None
        raise last_exc


def run_planning(
    symbolic_view: Mapping[str, Any],
    instruction: str,
    arms: Sequence[str],
    *,
    vlm_caller: PlannerVlmCaller | None = None,
    prompt_template: str | None = None,
    max_attempts: int = 2,
) -> list[PlanStep]:
    """Functional entry point wrapping :class:`Planner`."""
    return Planner(
        vlm_caller=vlm_caller,
        prompt_template=prompt_template,
        max_attempts=max_attempts,
    ).run(symbolic_view, instruction, arms)
