"""Frozen contract for the B-retry naive retry evaluation control."""

from __future__ import annotations

from dataclasses import dataclass

from .b import B_PAPER

B_RETRY_ID = "B-retry"
B_RETRY_VARIANT = "GR00T-RC-SparkVLA-style-execution-naive-retry"
B_RETRY_METHOD = "B-plus-fixed-unconditional-single-subtask-retry"
B_RETRY_PAPER = B_PAPER
B_RETRY_TRIGGER = "first-confirmed-stop-per-subtask"
B_RETRY_MAX_RETRIES_PER_SUBTASK = 1
B_RETRY_DECISION_SCHEDULE = "B-retry-unconditional-single-retry-v1"


class BRetryContractError(ValueError):
    """Raised when the frozen naive retry contract is violated."""


@dataclass(frozen=True)
class RetryTransition:
    subgoal_index_before: int
    subgoal_index_after: int
    attempt_index_before: int
    attempt_index_after: int
    retry_triggered: bool
    subgoal_advanced: bool


def retry_contract_payload(config: object) -> dict[str, object]:
    """Validate and serialize the frozen config without accepting silent drift."""

    expected = {
        "experiment_id": B_RETRY_ID,
        "variant": B_RETRY_VARIANT,
        "method": B_RETRY_METHOD,
        "trigger": B_RETRY_TRIGGER,
        "max_retries_per_subtask": B_RETRY_MAX_RETRIES_PER_SUBTASK,
        "preserve_global_step_budget": True,
        "reset_selector_anchor_on_retry": True,
        "failure_detector": False,
        "recovery_memory": False,
        "recovery_policy": False,
    }
    actual = {key: getattr(config, key, None) for key in expected}
    if actual != expected:
        raise BRetryContractError("B-retry config differs from the frozen naive retry contract")
    return actual | {
        "unconditional": True,
        "observes_goal_predicates": False,
        "observes_injection_labels": False,
        "observes_task_outcomes": False,
        "state_restoration": False,
        "learned_parameters": 0,
    }


def confirmed_stop_transition(
    *,
    subgoal_index: int,
    subgoal_count: int,
    attempt_index: int,
    max_retries_per_subtask: int = B_RETRY_MAX_RETRIES_PER_SUBTASK,
) -> RetryTransition:
    """Apply the outcome-blind retry law after B commits a STOP.

    The first committed STOP repeats the current instruction once. The next
    committed STOP advances. No observation, predicate, injection, or outcome
    is an input to this function.
    """

    if subgoal_count < 1:
        raise BRetryContractError("subgoal count must be positive")
    if not 0 <= subgoal_index < subgoal_count:
        raise BRetryContractError("subgoal index is outside the plan")
    if max_retries_per_subtask != B_RETRY_MAX_RETRIES_PER_SUBTASK:
        raise BRetryContractError("B-retry is frozen to one retry per subtask")
    if not 0 <= attempt_index <= max_retries_per_subtask:
        raise BRetryContractError("attempt index is outside the retry contract")

    retry = attempt_index < max_retries_per_subtask
    return RetryTransition(
        subgoal_index_before=subgoal_index,
        subgoal_index_after=subgoal_index if retry else subgoal_index + 1,
        attempt_index_before=attempt_index,
        attempt_index_after=attempt_index + 1 if retry else attempt_index,
        retry_triggered=retry,
        subgoal_advanced=not retry,
    )
