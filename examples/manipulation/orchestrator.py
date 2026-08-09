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

"""perceive → plan → solve → execute, with one-shot symbolic recovery."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .coordination import OverlapBusy, OverlapGuard
from .executor import ArmExecutor, ExecutionResult
from .solver import SolverConfig, solve

logger = logging.getLogger(__name__)

# perceive() -> (symbolic_view, geometric_view)
PerceiveFn = Callable[[], tuple[Mapping[str, Any], Mapping[str, Any]]]
# plan(symbolic_view, instruction, arms) -> plan list
PlanFn = Callable[[Mapping[str, Any], str, Sequence[str]], list[dict[str, Any]]]


def symbolic_views_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Equality on names / blocked_by / in_workspace only (order-insensitive)."""

    def _canon(view: Mapping[str, Any]) -> set[tuple]:
        rows = []
        for obj in view.get("objects", []):
            name = str(obj["name"])
            blocked = obj.get("blocked_by")
            blocked_s = None if blocked is None else str(blocked)
            ws = tuple(sorted(str(x) for x in (obj.get("in_workspace") or [])))
            rows.append((name, blocked_s, ws))
        return set(rows)

    return _canon(a) == _canon(b)


@dataclass
class LoopResult:
    status: str  # "success" | "fail"
    reason: str
    plan: list[dict[str, Any]] = field(default_factory=list)
    bound: list[dict[str, Any]] = field(default_factory=list)
    results: list[ExecutionResult] = field(default_factory=list)
    symbolic_view: dict[str, Any] = field(default_factory=dict)
    geometric_view: dict[str, Any] = field(default_factory=dict)


class ManipulationLoop:
    """
    Orchestrates perception, planning, solving, and per-arm execution.

    Recovery never asks a VLM for coordinates — only re-enters ``solve()``,
    or re-plans when the symbolic scene changes.
    """

    def __init__(
        self,
        *,
        perceive: PerceiveFn,
        plan_fn: PlanFn,
        executors: Mapping[str, ArmExecutor],
        solver_config: SolverConfig | Mapping[str, Any],
        arms: Sequence[str] | None = None,
        lookahead: bool = True,
        overlap_guard: OverlapGuard | None = None,
    ) -> None:
        self.perceive = perceive
        self.plan_fn = plan_fn
        self.executors = dict(executors)
        self.solver_config = solver_config
        self.arms = list(arms) if arms is not None else list(executors.keys())
        self.lookahead = lookahead
        self.overlap_guard = overlap_guard

    def run(self, instruction: str) -> LoopResult:
        symbolic_view, geometric_view = self.perceive()
        symbolic_view = _as_dict(symbolic_view)
        geometric_view = _as_dict(geometric_view)
        self._sync_overlap(geometric_view)

        plan = self.plan_fn(symbolic_view, instruction, self.arms)
        bound = solve(plan, geometric_view, self.solver_config, lookahead=self.lookahead)
        self._log_bound(bound)

        results: list[ExecutionResult] = []
        for i, step in enumerate(bound):
            result = self._execute_step(step)
            results.append(result)
            logger.info(
                "step=%s arm=%s prim=%s status=%s reason=%s params=%s",
                step["step"],
                step["arm"],
                step["primitive"],
                result["status"],
                result["reason"],
                json.dumps(step.get("params") or {}, sort_keys=True),
            )
            if result["status"] == "success":
                continue

            # ---- recovery (one retry) ----
            recovered = self._recover(
                instruction=instruction,
                failed_index=i,
                old_symbolic=symbolic_view,
                old_plan=plan,
                old_bound=bound,
                results_so_far=results,
            )
            return recovered

        return LoopResult(
            status="success",
            reason="",
            plan=plan,
            bound=bound,
            results=results,
            symbolic_view=symbolic_view,
            geometric_view=geometric_view,
        )

    def _sync_overlap(self, geometric_view: Mapping[str, Any]) -> None:
        from .coordination import overlap_from_workspaces

        workspaces = geometric_view.get("workspaces") or {}
        table = geometric_view.get("table_polygon")
        arms = None
        try:
            ov = overlap_from_workspaces(workspaces, arms=arms, table_polygon=table)
        except Exception as exc:  # noqa: BLE001
            logger.warning("overlap recompute failed: %s", exc)
            return
        if isinstance(geometric_view, dict):
            geometric_view["overlap"] = ov
        if self.overlap_guard is not None:
            self.overlap_guard.update_overlap(ov)
            if ov.is_empty:
                logger.info("overlap zone empty (no multi-arm intersection)")
            else:
                logger.info(
                    "overlap zone area=%.3f m^2 centroid=(%.3f, %.3f)",
                    float(ov.area),
                    float(ov.centroid.x),
                    float(ov.centroid.y),
                )

    def _execute_step(self, step: Mapping[str, Any]) -> ExecutionResult:
        arm = str(step["arm"])
        if arm not in self.executors:
            return {
                "step": int(step["step"]),
                "status": "fail",
                "reason": f"no_executor_for_arm:{arm}",
                "observed": {},
            }
        params = step.get("params") or {}
        pose = params.get("pose")
        xy = (float(pose[0]), float(pose[1])) if pose is not None and len(pose) >= 2 else None
        held = False
        try:
            if self.overlap_guard is not None:
                held = self.overlap_guard.acquire(arm, xy)
        except OverlapBusy as exc:
            return {
                "step": int(step["step"]),
                "status": "fail",
                "reason": f"overlap_busy:{exc.holder}",
                "observed": {"holder": exc.holder, "requester": exc.requester},
            }
        try:
            return self.executors[arm].execute(step)
        finally:
            if held and self.overlap_guard is not None:
                self.overlap_guard.release_after(arm)

    def _recover(
        self,
        *,
        instruction: str,
        failed_index: int,
        old_symbolic: Mapping[str, Any],
        old_plan: list[dict[str, Any]],
        old_bound: list[dict[str, Any]],
        results_so_far: list[ExecutionResult],
    ) -> LoopResult:
        new_symbolic, new_geometric = self.perceive()
        new_symbolic = _as_dict(new_symbolic)
        new_geometric = _as_dict(new_geometric)
        self._sync_overlap(new_geometric)

        if symbolic_views_equal(new_symbolic, old_symbolic):
            logger.info("Recovery: symbolic_view unchanged → re-solve from failed step")
            # Re-bind from the failed step onward with fresh geometry.
            # Prior steps' effects are reflected in the new geometric_view.
            remaining_plan = old_plan[failed_index:]
            try:
                new_bound_tail = solve(
                    remaining_plan,
                    new_geometric,
                    self.solver_config,
                    lookahead=self.lookahead,
                )
            except Exception as exc:  # noqa: BLE001
                return LoopResult(
                    status="fail",
                    reason=f"resolve_failed:{exc}",
                    plan=old_plan,
                    bound=old_bound,
                    results=results_so_far,
                    symbolic_view=new_symbolic,
                    geometric_view=new_geometric,
                )

            bound = list(old_bound[:failed_index]) + list(new_bound_tail)
            self._log_bound(new_bound_tail, prefix="resolve")
            results = list(results_so_far[:-1])  # drop failed attempt
            for step in new_bound_tail:
                result = self._execute_step(step)
                results.append(result)
                logger.info(
                    "retry step=%s status=%s reason=%s params=%s",
                    step["step"],
                    result["status"],
                    result["reason"],
                    json.dumps(step.get("params") or {}, sort_keys=True),
                )
                if result["status"] != "success":
                    return LoopResult(
                        status="fail",
                        reason=f"retry_failed:{result['reason']}",
                        plan=old_plan,
                        bound=bound,
                        results=results,
                        symbolic_view=new_symbolic,
                        geometric_view=new_geometric,
                    )
            return LoopResult(
                status="success",
                reason="recovered_resolve",
                plan=old_plan,
                bound=bound,
                results=results,
                symbolic_view=new_symbolic,
                geometric_view=new_geometric,
            )

        logger.info("Recovery: symbolic_view changed → re-plan + solve from scratch")
        try:
            new_plan = self.plan_fn(new_symbolic, instruction, self.arms)
            new_bound = solve(
                new_plan, new_geometric, self.solver_config, lookahead=self.lookahead
            )
        except Exception as exc:  # noqa: BLE001
            return LoopResult(
                status="fail",
                reason=f"replan_failed:{exc}",
                plan=old_plan,
                bound=old_bound,
                results=results_so_far,
                symbolic_view=new_symbolic,
                geometric_view=new_geometric,
            )

        self._log_bound(new_bound, prefix="replan")
        results: list[ExecutionResult] = []
        for step in new_bound:
            result = self._execute_step(step)
            results.append(result)
            logger.info(
                "replan step=%s status=%s reason=%s params=%s",
                step["step"],
                result["status"],
                result["reason"],
                json.dumps(step.get("params") or {}, sort_keys=True),
            )
            if result["status"] != "success":
                return LoopResult(
                    status="fail",
                    reason=f"replan_exec_failed:{result['reason']}",
                    plan=new_plan,
                    bound=new_bound,
                    results=results,
                    symbolic_view=new_symbolic,
                    geometric_view=new_geometric,
                )
        return LoopResult(
            status="success",
            reason="recovered_replan",
            plan=new_plan,
            bound=new_bound,
            results=results,
            symbolic_view=new_symbolic,
            geometric_view=new_geometric,
        )

    @staticmethod
    def _log_bound(bound: Sequence[Mapping[str, Any]], prefix: str = "bound") -> None:
        for step in bound:
            logger.info(
                "%s step=%s arm=%s prim=%s params=%s depends_on=%s",
                prefix,
                step.get("step"),
                step.get("arm"),
                step.get("primitive"),
                json.dumps(step.get("params") or {}, sort_keys=True),
                step.get("depends_on"),
            )


def _as_dict(view: Mapping[str, Any]) -> dict[str, Any]:
    # Shallow copy is enough; polygons stay shared references.
    return dict(view)


def run_manipulation(
    instruction: str,
    *,
    perceive: PerceiveFn,
    plan_fn: PlanFn,
    executors: Mapping[str, ArmExecutor],
    solver_config: SolverConfig | Mapping[str, Any],
    arms: Sequence[str] | None = None,
    lookahead: bool = True,
    overlap_guard: OverlapGuard | None = None,
) -> LoopResult:
    return ManipulationLoop(
        perceive=perceive,
        plan_fn=plan_fn,
        executors=executors,
        solver_config=solver_config,
        arms=arms,
        lookahead=lookahead,
        overlap_guard=overlap_guard,
    ).run(instruction)
