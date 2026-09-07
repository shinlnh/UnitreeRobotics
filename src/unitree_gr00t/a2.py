"""Frozen hierarchy contract for experiment A2.

A2 deliberately implements only the fixed-anchor portion of RoboCerebra's
public hierarchical evaluator.  It does not observe completion, re-plan, retry,
restore simulator state, or select a variable action prefix.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .a0 import BenchmarkCase, TaskDescription, parse_task_description

A2_ID = "A2"
A2_VARIANT = "GR00T-RC-fixed-hierarchy"
A2_PLANNER = "RoboCerebra-HPE-fixed-anchor-reimplementation"
A2_PLAN_SOURCE = "canonical task_description.txt step annotations"


class A2ContractError(ValueError):
    """Raised when the frozen A2 hierarchy contract is violated."""


@dataclass(frozen=True)
class FixedPlan:
    full_task_instruction: str
    subgoals: tuple[str, ...]
    subgoal_horizon_steps: int
    source: str
    sha256: str

    @property
    def max_steps(self) -> int:
        return len(self.subgoals) * self.subgoal_horizon_steps


@dataclass(frozen=True)
class PlannerDecision:
    subgoal_index: int
    subgoal_instruction: str
    segment_start_step: int
    segment_end_step: int
    steps_remaining_in_segment: int


@dataclass(frozen=True)
class HierarchyAudit:
    cases: int
    plans: int
    subgoals: int
    unique_plans: int
    subgoal_horizon_steps: int
    planner: str
    plan_source: str
    plan_sha256: tuple[str, ...]
    valid: bool
    issues: tuple[str, ...]


def _plan_sha256(task: TaskDescription, subgoal_horizon_steps: int) -> str:
    canonical = json.dumps(
        {
            "full_task_instruction": task.instruction,
            "subgoals": task.steps,
            "subgoal_horizon_steps": subgoal_horizon_steps,
            "source": A2_PLAN_SOURCE,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_fixed_plan(
    task: TaskDescription,
    subgoal_horizon_steps: int,
    *,
    source: str = A2_PLAN_SOURCE,
) -> FixedPlan:
    """Freeze canonical benchmark steps into a time-indexed, outcome-blind plan."""

    if subgoal_horizon_steps < 1:
        raise A2ContractError("A2 subgoal horizon must be positive")
    if not task.instruction.strip():
        raise A2ContractError("A2 plan requires a non-empty full-task instruction")
    if not task.steps or any(not step.strip() for step in task.steps):
        raise A2ContractError("A2 plan requires non-empty canonical subgoals")
    if source != A2_PLAN_SOURCE:
        raise A2ContractError(f"A2 plan source is not frozen: {source!r}")
    return FixedPlan(
        full_task_instruction=task.instruction,
        subgoals=task.steps,
        subgoal_horizon_steps=subgoal_horizon_steps,
        source=source,
        sha256=_plan_sha256(task, subgoal_horizon_steps),
    )


def select_subgoal(plan: FixedPlan, executed_steps: int) -> PlannerDecision:
    """Select solely by the fixed execution clock, never by observations or success."""

    if executed_steps < 0 or executed_steps >= plan.max_steps:
        raise A2ContractError(f"A2 planner step {executed_steps} is outside [0, {plan.max_steps})")
    index = executed_steps // plan.subgoal_horizon_steps
    start = index * plan.subgoal_horizon_steps
    end = start + plan.subgoal_horizon_steps
    return PlannerDecision(
        subgoal_index=index,
        subgoal_instruction=plan.subgoals[index],
        segment_start_step=start,
        segment_end_step=end,
        steps_remaining_in_segment=end - executed_steps,
    )


def select_fixed_prefix_length(
    decision: PlannerDecision,
    execution_horizon: int,
    remaining_episode_steps: int,
) -> int:
    """Truncate only at a predetermined anchor or the frozen episode end."""

    if execution_horizon < 1:
        raise A2ContractError("A2 execution horizon must be positive")
    if remaining_episode_steps < 1:
        raise A2ContractError("A2 remaining episode steps must be positive")
    return min(
        execution_horizon,
        decision.steps_remaining_in_segment,
        remaining_episode_steps,
    )


def audit_fixed_hierarchy(cases: list[BenchmarkCase], subgoal_horizon_steps: int) -> HierarchyAudit:
    issues: list[str] = []
    plans: list[FixedPlan] = []
    for case in cases:
        try:
            task = parse_task_description(case.path / "task_description.txt")
            plans.append(build_fixed_plan(task, subgoal_horizon_steps))
        except (OSError, ValueError) as exc:
            issues.append(f"{case.task_type}/{case.case_name}: {exc}")
    return HierarchyAudit(
        cases=len(cases),
        plans=len(plans),
        subgoals=sum(len(plan.subgoals) for plan in plans),
        unique_plans=len({plan.sha256 for plan in plans}),
        subgoal_horizon_steps=subgoal_horizon_steps,
        planner=A2_PLANNER,
        plan_source=A2_PLAN_SOURCE,
        plan_sha256=tuple(sorted({plan.sha256 for plan in plans})),
        valid=len(plans) == len(cases) and not issues,
        issues=tuple(issues),
    )


def plan_payload(plan: FixedPlan) -> dict[str, object]:
    """Return the immutable plan in a trace-safe JSON representation."""

    return asdict(plan)


def hierarchy_audit_payload(audit: HierarchyAudit) -> dict[str, object]:
    """Return stable JSON-native types so resume manifests compare exactly."""

    return asdict(audit) | {
        "plan_sha256": list(audit.plan_sha256),
        "issues": list(audit.issues),
    }
