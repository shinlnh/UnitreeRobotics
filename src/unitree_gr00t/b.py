"""Frozen contracts for experiment B's unified STOP/action-prefix selector."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from .a1 import sha256_file

B_ID = "B"
B_VARIANT = "GR00T-RC-SparkVLA-style-execution"
B_METHOD = "SparkVLA-unified-stop-prefix-GR00T-reimplementation"
B_PAPER = "arXiv:2608.16172v1"
B_CHECKPOINT_PROVENANCE = "b_selector_provenance.json"


class BContractError(ValueError):
    """Raised when selector inputs or artifacts violate the frozen B contract."""


@dataclass(frozen=True)
class OrdinalTargets:
    priorities: tuple[int, ...]
    valid: tuple[bool, ...]
    stop_label: int
    rank_weight: float
    stop_weight: float
    comparable_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class StopConfirmationState:
    streak: int = 0


@dataclass(frozen=True)
class ConfirmedSelection:
    proposed_candidate: int
    executed_prefix_length: int
    stop_pending: bool
    stop_committed: bool
    state: StopConfirmationState


@dataclass(frozen=True)
class SelectorCheckpointAudit:
    checkpoint_dir: Path
    weights_sha256: str
    provenance_sha256: str
    action_horizon: int
    context_width: int
    method: str
    valid: bool


def a1_weight_hashes(contract: object) -> dict[str, str]:
    """Hash every frozen A1 weight shard named by its inspected contract."""

    root = Path(contract.checkpoint_dir)
    shards = tuple(contract.weight_shards)
    return {name: sha256_file(root / name) for name in shards}


def selector_decision_seed(episode_seed: int, decision_index: int) -> int:
    """Derive a request-local seed that is invariant to shard scheduling."""

    if episode_seed < 0 or decision_index < 0:
        raise BContractError("selector seed inputs cannot be negative")
    payload = f"B-decision-v1:{episode_seed}:{decision_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFF_FFFF


def verify_a1_weight_hashes(contract: object, expected: object) -> dict[str, str]:
    """Fail closed if the live A1 weights differ from B's training parent."""

    if not isinstance(expected, dict) or not expected:
        raise BContractError("B provenance is missing frozen A1 weight hashes")
    actual = a1_weight_hashes(contract)
    if actual != expected:
        raise BContractError("live A1 weight shards differ from B training provenance")
    return actual


def candidate_validity(
    *, decision_step: int, trajectory_steps: int, horizon: int, max_prefix: int | None = None
) -> tuple[bool, ...]:
    """Return validity for ``[STOP, prefix-1, ..., prefix-H]``."""

    if decision_step < 0 or trajectory_steps < 1 or horizon < 1:
        raise BContractError("decision step, trajectory length, and horizon must be valid")
    if decision_step >= trajectory_steps:
        raise BContractError("decision step must be inside the trajectory")
    selected_max = horizon if max_prefix is None else max_prefix
    if selected_max < 1 or selected_max > horizon:
        raise BContractError("max prefix must be inside [1, horizon]")
    return (True,) + tuple(
        n <= selected_max and decision_step + n - 1 < trajectory_steps
        for n in range(1, horizon + 1)
    )


def build_ordinal_targets(
    *,
    decision_step: int,
    trajectory_steps: int,
    horizon: int,
    successful: bool,
    completion_step: int | None,
    boundary_jitter: int = 0,
    near_boundary_steps: int = 80,
    unsuccessful_rank_weight: float = 0.1,
    stop_positive_weight: float = 3.0,
    near_boundary_stop_weight: float = 3.0,
    unsuccessful_stop_weight: float = 1.5,
) -> OrdinalTargets:
    """Implement SparkVLA Algorithm 1 without sampling hidden randomness.

    The caller samples and records ``boundary_jitter``. Keeping target creation
    deterministic makes every offline label independently replayable.
    """

    valid = candidate_validity(
        decision_step=decision_step, trajectory_steps=trajectory_steps, horizon=horizon
    )
    priorities = [0] * (horizon + 1)
    if not successful:
        if completion_step is not None:
            raise BContractError("unsuccessful trajectories cannot have a completion step")
        for n in range(1, horizon + 1):
            if valid[n]:
                priorities[n] = 1
        stop_label = 0
        rank_weight = unsuccessful_rank_weight
        stop_weight = unsuccessful_stop_weight
    else:
        if completion_step is None:
            raise BContractError("successful trajectories require a completion step")
        perturbed = completion_step + boundary_jitter
        if perturbed < 0 or perturbed > trajectory_steps:
            raise BContractError("perturbed completion boundary is outside the trajectory")
        rank_weight = 1.0
        if decision_step >= perturbed:
            priorities[0] = horizon + 1
            for n in range(1, horizon + 1):
                if valid[n]:
                    priorities[n] = horizon - n + 1
            stop_label = 1
            stop_weight = stop_positive_weight
        else:
            remaining = perturbed - decision_step
            for n in range(1, horizon + 1):
                if not valid[n]:
                    continue
                priorities[n] = (
                    n if remaining > horizon or n < remaining else 2 * horizon - (n - remaining)
                )
            stop_label = 0
            stop_weight = near_boundary_stop_weight if remaining <= near_boundary_steps else 1.0
    comparable_pairs = tuple(
        (left, right)
        for left in range(horizon + 1)
        for right in range(left + 1, horizon + 1)
        if valid[left] and valid[right] and priorities[left] != priorities[right]
    )
    return OrdinalTargets(
        priorities=tuple(priorities),
        valid=valid,
        stop_label=stop_label,
        rank_weight=rank_weight,
        stop_weight=stop_weight,
        comparable_pairs=comparable_pairs,
    )


def select_unified_candidate(scores: Sequence[float], valid: Sequence[bool]) -> int:
    """Select one candidate by masked argmax, with stable lower-index tie breaking."""

    if len(scores) < 2 or len(scores) != len(valid):
        raise BContractError("scores and validity must have the same H+1 length")
    best_index: int | None = None
    best_score = -math.inf
    for index, (score, is_valid) in enumerate(zip(scores, valid, strict=True)):
        value = float(score)
        if not math.isfinite(value):
            raise BContractError("selector scores must be finite")
        if is_valid and (best_index is None or value > best_score):
            best_index = index
            best_score = value
    if best_index is None:
        raise BContractError("selector has no valid candidate")
    return best_index


def confirm_stop(
    candidate: int,
    state: StopConfirmationState,
    *,
    confirmation_window: int,
) -> ConfirmedSelection:
    """Apply the frozen consecutive-decision guard to a unified candidate."""

    if candidate < 0:
        raise BContractError("candidate index cannot be negative")
    if confirmation_window < 1:
        raise BContractError("STOP confirmation window must be positive")
    if candidate != 0:
        return ConfirmedSelection(
            proposed_candidate=candidate,
            executed_prefix_length=candidate,
            stop_pending=False,
            stop_committed=False,
            state=StopConfirmationState(),
        )
    streak = state.streak + 1
    committed = streak >= confirmation_window
    return ConfirmedSelection(
        proposed_candidate=0,
        executed_prefix_length=0,
        stop_pending=not committed,
        stop_committed=committed,
        state=StopConfirmationState(0 if committed else streak),
    )


def inspect_selector_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_action_horizon: int,
    expected_context_width: int,
) -> tuple[SelectorCheckpointAudit, dict[str, object]]:
    """Validate identity, dimensions, and byte hashes of a frozen B selector."""

    root = Path(checkpoint_dir).expanduser().resolve()
    provenance_path = root / B_CHECKPOINT_PROVENANCE
    weights_path = root / "model.safetensors"
    if not provenance_path.is_file() or not weights_path.is_file():
        raise BContractError(f"selector checkpoint is incomplete: {root}")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BContractError(f"cannot read selector provenance: {exc}") from exc
    expected_identity = (B_ID, B_VARIANT, B_METHOD, B_PAPER)
    actual_identity = tuple(
        provenance.get(key) for key in ("experiment_id", "variant", "method", "paper")
    )
    if actual_identity != expected_identity:
        raise BContractError("selector provenance identity is not frozen B")
    action_horizon = int(provenance.get("action_horizon", -1))
    context_width = int(provenance.get("context_width", -1))
    if action_horizon != expected_action_horizon or context_width != expected_context_width:
        raise BContractError("selector checkpoint dimensions do not match configuration")
    weights_sha256 = sha256_file(weights_path)
    if provenance.get("weights_sha256") != weights_sha256:
        raise BContractError("selector weight hash does not match provenance")
    audit = SelectorCheckpointAudit(
        checkpoint_dir=root,
        weights_sha256=weights_sha256,
        provenance_sha256=sha256_file(provenance_path),
        action_horizon=action_horizon,
        context_width=context_width,
        method=str(provenance["method"]),
        valid=True,
    )
    return audit, provenance


def ordinal_targets_payload(targets: OrdinalTargets) -> dict[str, object]:
    return asdict(targets) | {
        "priorities": list(targets.priorities),
        "valid": list(targets.valid),
        "comparable_pairs": [list(pair) for pair in targets.comparable_pairs],
    }
