"""Frozen contracts for Ours counterfactual temporal recovery research."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

OURS_ID = "Ours"
OURS_VARIANT = "GR00T-RC-SparkVLA-counterfactual-temporal-recovery"
OURS_METHOD = "counterfactual-temporal-recovery"
OURS_PARENT = "B"


class OursContractError(ValueError):
    """Raised when Ours crosses a frozen scientific boundary."""


class RecoveryOption(str, Enum):  # noqa: UP042 - Isaac-GR00T runtime is Python 3.10.
    ACCEPT_B = "ACCEPT_B"
    REOBSERVE = "REOBSERVE"
    RETRY_CURRENT = "RETRY_CURRENT"
    BACKTRACK_ONE = "BACKTRACK_ONE"
    ADVANCE = "ADVANCE"
    CONSENSUS_PREFIX = "CONSENSUS_PREFIX"


RECOVERY_OPTIONS = tuple(RecoveryOption)

# These values may appear in evaluator traces and offline labels, but never in
# the payload used by the deployable recovery controller.
_PROHIBITED_EXACT_KEYS = frozenset(
    {
        "case",
        "condition",
        "condition_start",
        "demonstration_id",
        "episode_outcome",
        "failure_label",
        "final_success",
        "goal",
        "goal_steps",
        "goal_state",
        "injection",
        "injection_label",
        "object_pose",
        "reached_success",
        "simulator_state",
        "success",
        "success_after",
        "success_before",
        "task_type",
        "task_goal",
        "trial",
    }
)
_PROHIBITED_KEY_PARTS = (
    "goal_predicate",
    "success_predicate",
    "injection",
    "object_pose",
    "simulator_state",
    "future_",
    "oracle_",
)


@dataclass(frozen=True)
class RecoveryOptionMask:
    options: tuple[str, ...]
    valid: tuple[bool, ...]

    def as_mapping(self) -> dict[str, bool]:
        return dict(zip(self.options, self.valid, strict=True))


@dataclass(frozen=True)
class RecoveryTransition:
    option: str
    subgoal_index_before: int
    subgoal_index_after: int
    attempt_index_before: int
    attempt_index_after: int
    reanchor: bool
    clear_stop_confirmation: bool
    consumes_simulator_step: bool


def recovery_decision_seed(
    episode_seed: int, decision_index: int, hypothesis_index: int = 0
) -> int:
    """Derive an order-independent inference seed for Ours."""

    if min(episode_seed, decision_index, hypothesis_index) < 0:
        raise OursContractError("recovery seed inputs must be non-negative")
    payload = f"Ours-decision-v1:{episode_seed}:{decision_index}:{hypothesis_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFF_FFFF


def counterfactual_branch_seed(
    base_seed: int,
    state_index: int,
    option: RecoveryOption | str,
    rollout_index: int,
) -> int:
    """Derive a stable seed for a training-only counterfactual branch."""

    if min(base_seed, state_index, rollout_index) < 0:
        raise OursContractError("counterfactual seed inputs must be non-negative")
    try:
        normalized = RecoveryOption(option).value
    except ValueError as exc:
        raise OursContractError(f"unknown recovery option: {option}") from exc
    payload = (
        f"Ours-counterfactual-v1:{base_seed}:{state_index}:{normalized}:{rollout_index}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFF_FFFF


def _normalized_key(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def prohibited_runtime_paths(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return forbidden signal paths present in a proposed runtime payload."""

    found: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                normalized = _normalized_key(key)
                child = f"{path}.{key}" if path else str(key)
                if normalized in _PROHIBITED_EXACT_KEYS or any(
                    part in normalized for part in _PROHIBITED_KEY_PARTS
                ):
                    found.append(child)
                visit(nested, child)
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")

    visit(payload, "")
    return tuple(sorted(set(found)))


def validate_runtime_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed if an inference payload contains privileged information."""

    paths = prohibited_runtime_paths(payload)
    if paths:
        raise OursContractError(
            "Ours runtime payload contains privileged evaluator signals: " + ", ".join(paths)
        )


def recovery_option_mask(
    *,
    subgoal_index: int,
    subgoal_count: int,
    stop_pending: bool,
    stop_committed: bool,
    recovery_triggered: bool,
    recovery_attempts: int,
    max_recovery_attempts: int,
    zero_progress_decisions: int,
    zero_progress_guard: int,
    remaining_steps: int,
    consensus_hypotheses: int,
) -> RecoveryOptionMask:
    """Build the fixed recovery action mask without evaluator-only signals."""

    counts = (
        subgoal_index,
        subgoal_count,
        recovery_attempts,
        max_recovery_attempts,
        zero_progress_decisions,
        zero_progress_guard,
        remaining_steps,
        consensus_hypotheses,
    )
    if min(counts) < 0 or subgoal_count < 1 or subgoal_index >= subgoal_count:
        raise OursContractError("invalid recovery option-mask state")
    if stop_pending and stop_committed:
        raise OursContractError("STOP cannot be pending and committed simultaneously")

    has_budget = remaining_steps > 0
    attempts_available = recovery_attempts < max_recovery_attempts
    recovery_boundary = recovery_triggered or stop_committed
    mapping = {
        RecoveryOption.ACCEPT_B: has_budget and not stop_committed,
        RecoveryOption.REOBSERVE: (
            has_budget
            and (stop_pending or recovery_triggered)
            and zero_progress_decisions < zero_progress_guard
        ),
        RecoveryOption.RETRY_CURRENT: has_budget and recovery_boundary and attempts_available,
        RecoveryOption.BACKTRACK_ONE: (
            has_budget and recovery_boundary and attempts_available and subgoal_index > 0
        ),
        RecoveryOption.ADVANCE: has_budget and stop_committed,
        RecoveryOption.CONSENSUS_PREFIX: (
            has_budget and recovery_triggered and attempts_available and consensus_hypotheses > 1
        ),
    }
    if not any(mapping.values()):
        raise OursContractError("recovery option mask has no valid action")
    return RecoveryOptionMask(
        options=tuple(option.value for option in RECOVERY_OPTIONS),
        valid=tuple(mapping[option] for option in RECOVERY_OPTIONS),
    )


def apply_recovery_transition(
    option: RecoveryOption | str,
    *,
    subgoal_index: int,
    subgoal_count: int,
    attempt_index: int,
    max_recovery_attempts: int,
) -> RecoveryTransition:
    """Apply an outcome-blind high-level transition from the live state."""

    if (
        subgoal_count < 1
        or not 0 <= subgoal_index < subgoal_count
        or attempt_index < 0
        or max_recovery_attempts < 0
    ):
        raise OursContractError("invalid recovery transition state")
    try:
        selected = RecoveryOption(option)
    except ValueError as exc:
        raise OursContractError(f"unknown recovery option: {option}") from exc

    if (
        selected
        in {
            RecoveryOption.RETRY_CURRENT,
            RecoveryOption.BACKTRACK_ONE,
            RecoveryOption.CONSENSUS_PREFIX,
        }
        and attempt_index >= max_recovery_attempts
    ):
        raise OursContractError("recovery attempt budget is exhausted")
    if selected is RecoveryOption.BACKTRACK_ONE and subgoal_index == 0:
        raise OursContractError("cannot backtrack before the first subtask")

    subgoal_after = subgoal_index
    attempt_after = attempt_index
    reanchor = False
    clear_stop = False
    if selected is RecoveryOption.RETRY_CURRENT:
        attempt_after += 1
        reanchor = True
        clear_stop = True
    elif selected is RecoveryOption.BACKTRACK_ONE:
        subgoal_after -= 1
        attempt_after += 1
        reanchor = True
        clear_stop = True
    elif selected is RecoveryOption.ADVANCE:
        subgoal_after += 1
        attempt_after = 0
        reanchor = subgoal_after < subgoal_count
        clear_stop = True
    elif selected is RecoveryOption.CONSENSUS_PREFIX:
        attempt_after += 1
        clear_stop = True
    elif selected is RecoveryOption.REOBSERVE:
        clear_stop = True

    return RecoveryTransition(
        option=selected.value,
        subgoal_index_before=subgoal_index,
        subgoal_index_after=subgoal_after,
        attempt_index_before=attempt_index,
        attempt_index_after=attempt_after,
        reanchor=reanchor,
        clear_stop_confirmation=clear_stop,
        consumes_simulator_step=False,
    )


def validate_recovery_config(config: Any) -> dict[str, Any]:
    """Validate the immutable Ours search boundary from typed project config."""

    identity = (
        str(config.experiment_id),
        str(config.variant),
        str(config.method),
        str(config.parent_experiment),
    )
    if identity != (OURS_ID, OURS_VARIANT, OURS_METHOD, OURS_PARENT):
        raise OursContractError("Ours identity or parent experiment differs from the freeze")
    if tuple(config.options) != tuple(option.value for option in RECOVERY_OPTIONS):
        raise OursContractError("recovery option order differs from the freeze")
    if (
        int(config.context_width) != 2048
        or int(config.temporal_width) < 1
        or int(config.temporal_layers) < 1
        or int(config.temporal_heads) < 1
        or int(config.temporal_width) % int(config.temporal_heads)
    ):
        raise OursContractError("invalid frozen temporal model dimensions")
    if int(config.max_parameters) < 1 or int(config.max_parameters) > 15_000_000:
        raise OursContractError("Ours new-parameter budget exceeds the freeze")
    if tuple(config.history_lengths) != (4, 8, 16):
        raise OursContractError("history-length search space differs from the freeze")
    if tuple(config.consensus_hypotheses) != (1, 4, 8):
        raise OursContractError("consensus search space differs from the freeze")
    if tuple(config.max_recovery_attempts) != (1, 2):
        raise OursContractError("recovery-attempt search space differs from the freeze")
    if (
        min(
            int(config.failure_confirmation_window),
            int(config.recovery_completion_confirmation_window),
            int(config.zero_progress_guard),
            int(config.max_search_variants),
        )
        < 1
    ):
        raise OursContractError("Ours confirmation and search counts must be positive")
    if not 0.0 <= float(config.false_recovery_rate_limit) <= 0.05:
        raise OursContractError("false-recovery limit is outside the freeze")
    if not 0.0 <= float(config.overhead_fraction_limit) <= 0.20:
        raise OursContractError("recovery overhead limit is outside the freeze")

    train = tuple(int(value) for value in config.train_base_seeds)
    development = tuple(int(value) for value in config.development_base_seeds)
    smoke = int(config.smoke_base_seed)
    final = int(config.final_base_seed)
    all_seeds = train + development + (smoke, final)
    if (
        train != (10007, 11007, 12007)
        or development != (20007, 21007, 22007, 23007)
        or smoke != 30007
        or final != 7
        or len(set(all_seeds)) != len(all_seeds)
        or min(all_seeds) < 0
        or max(all_seeds) >= 2**31
    ):
        raise OursContractError("Ours seed partitions differ from the freeze")

    return {
        "identity": {
            "experiment_id": OURS_ID,
            "variant": OURS_VARIANT,
            "method": OURS_METHOD,
            "parent_experiment": OURS_PARENT,
        },
        "runtime": {
            "observes_goal_predicates": False,
            "observes_injection_labels": False,
            "observes_task_outcomes": False,
            "observes_simulator_state": False,
            "state_restoration": False,
            "continuous_live_state": True,
        },
        "search": asdict(config)
        | {
            "checkpoint_dir": str(config.checkpoint_dir),
            "dataset_dir": str(config.dataset_dir),
        },
        "valid": True,
    }
