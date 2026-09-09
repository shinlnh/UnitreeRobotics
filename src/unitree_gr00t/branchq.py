"""Typed BranchQ-VLA lattice, Bellman targets, and conservative fallback."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .ours import OursContractError

BRANCHQ_ID = "Ours"
BRANCHQ_VARIANT = "GR00T-RC-BranchQ-VLA"
BRANCHQ_METHOD = "unified-counterfactual-branch-q-learning"
BRANCHQ_PARENT = "B"
BRANCHQ_DECISION_SCHEDULE = "joint-subgoal-hypothesis-prefix-lattice-v1"
BRANCHQ_PREFIX_LENGTHS = (1, 4, 8, 16)
BRANCHQ_SUBGOAL_DELTAS = (-1, 0, 1)


@dataclass(frozen=True)
class BranchCandidate:
    """One executable macro action in the unified BranchQ lattice."""

    candidate_id: str
    delta_subgoal: int
    target_subgoal_index: int
    hypothesis_index: int | None
    prefix_length: int
    stop: bool
    baseline: bool

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConformalCalibration:
    alpha: float
    quantile: float
    groups: int
    samples: int
    standard_deviation_floor: float


@dataclass(frozen=True)
class ConservativeBranchDecision:
    selected_index: int
    baseline_index: int
    selected_candidate_id: str
    baseline_candidate_id: str
    selected_mean: float
    selected_lower_bound: float
    baseline_mean: float
    baseline_upper_bound: float
    fallback: bool
    reason: str


def branchq_hypothesis_seed(
    episode_seed: int,
    decision_index: int,
    delta_subgoal: int,
    hypothesis_index: int,
) -> int:
    """Derive a stable seed for one lattice hypothesis independent of batching."""

    if min(episode_seed, decision_index, hypothesis_index) < 0:
        raise OursContractError("BranchQ seed inputs cannot be negative")
    if delta_subgoal not in BRANCHQ_SUBGOAL_DELTAS:
        raise OursContractError("BranchQ subgoal delta is outside {-1, 0, +1}")
    payload = (
        f"BranchQ-hypothesis-v1:{episode_seed}:{decision_index}:"
        f"{delta_subgoal}:{hypothesis_index}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFF_FFFF


def build_branch_lattice(
    *,
    subgoal_index: int,
    subgoal_count: int,
    hypothesis_count: int,
    prefix_lengths: Sequence[int] = BRANCHQ_PREFIX_LENGTHS,
    action_horizon: int = 16,
    remaining_steps: int,
    baseline_candidate: int,
) -> tuple[BranchCandidate, ...]:
    """Construct STOP plus every valid ``(delta, hypothesis, prefix)`` branch.

    Hypothesis zero at delta zero is the exact frozen-B proposal. Its selected
    prefix is inserted even when that length is absent from the registered
    coarse prefix grid, so the baseline is always represented byte-for-byte.
    """

    if (
        subgoal_count < 1
        or not 0 <= subgoal_index < subgoal_count
        or hypothesis_count < 1
        or action_horizon < 1
        or remaining_steps < 1
        or not 0 <= baseline_candidate <= min(action_horizon, remaining_steps)
    ):
        raise OursContractError("BranchQ lattice state is invalid")
    prefixes = tuple(int(value) for value in prefix_lengths)
    if (
        not prefixes
        or tuple(sorted(set(prefixes))) != prefixes
        or min(prefixes) < 1
        or max(prefixes) > action_horizon
    ):
        raise OursContractError("BranchQ prefix grid must be unique, sorted, and in horizon")

    candidates = [
        BranchCandidate(
            candidate_id="stop",
            delta_subgoal=1,
            target_subgoal_index=min(subgoal_index + 1, subgoal_count),
            hypothesis_index=None,
            prefix_length=0,
            stop=True,
            baseline=baseline_candidate == 0,
        )
    ]
    aliases = {-1: "m1", 0: "z0", 1: "p1"}
    for delta in BRANCHQ_SUBGOAL_DELTAS:
        target = subgoal_index + delta
        if not 0 <= target < subgoal_count:
            continue
        for hypothesis in range(hypothesis_count):
            lengths = [value for value in prefixes if value <= remaining_steps]
            if delta == 0 and hypothesis == 0 and baseline_candidate > 0:
                lengths = sorted(set(lengths) | {baseline_candidate})
            for length in lengths:
                candidates.append(
                    BranchCandidate(
                        candidate_id=f"dg{aliases[delta]}-h{hypothesis:02d}-l{length:02d}",
                        delta_subgoal=delta,
                        target_subgoal_index=target,
                        hypothesis_index=hypothesis,
                        prefix_length=length,
                        stop=False,
                        baseline=(
                            delta == 0
                            and hypothesis == 0
                            and length == baseline_candidate
                        ),
                    )
                )
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise OursContractError("BranchQ lattice contains duplicate candidate ids")
    if sum(candidate.baseline for candidate in candidates) != 1:
        raise OursContractError("BranchQ lattice must contain exactly one frozen-B candidate")
    return tuple(candidates)


def macro_discount(gamma: float, duration_steps: int) -> float:
    """Return the semi-Markov continuation discount for an action prefix."""

    if not 0.0 < gamma <= 1.0 or duration_steps < 0:
        raise OursContractError("BranchQ discount inputs are invalid")
    return float(gamma ** max(1, duration_steps))


def physical_macro_reward(
    *,
    completed_subtasks_before: int,
    completed_subtasks_after: int,
    subgoal_count: int,
    predicate_count_before: int,
    predicate_count_after: int,
    predicate_count: int,
    final_success: bool,
    continuation_discount: float,
    subtask_potential_weight: float = 0.25,
    predicate_potential_weight: float = 0.05,
) -> float:
    """Outcome-only terminal reward with bounded potential-based shaping.

    No step, policy-call, or option-specific bonus is present. The shaping term
    is ``discount * Phi(next) - Phi(current)``, preserving the underlying task
    objective while making sparse physical progress visible to Bellman backup.
    """

    counts = (
        completed_subtasks_before,
        completed_subtasks_after,
        subgoal_count,
        predicate_count_before,
        predicate_count_after,
        predicate_count,
    )
    if (
        min(counts) < 0
        or subgoal_count < 1
        or predicate_count < 1
        or completed_subtasks_before > subgoal_count
        or completed_subtasks_after > subgoal_count
        or predicate_count_before > predicate_count
        or predicate_count_after > predicate_count
        or not 0.0 <= continuation_discount <= 1.0
        or min(subtask_potential_weight, predicate_potential_weight) < 0.0
    ):
        raise OursContractError("BranchQ physical reward inputs are invalid")

    def potential(completed: int, predicates: int) -> float:
        return (
            subtask_potential_weight * completed / subgoal_count
            + predicate_potential_weight * predicates / predicate_count
        )

    return float(
        int(bool(final_success))
        + continuation_discount
        * potential(completed_subtasks_after, predicate_count_after)
        - potential(completed_subtasks_before, predicate_count_before)
    )


def semi_markov_double_q_target(
    *,
    reward: float,
    gamma: float,
    duration_steps: int,
    terminal: bool,
    next_online_values: Any,
    next_target_values: Any,
    next_valid: Any,
    np: Any,
) -> float:
    """Compute an n-step Double-Q target on the next finite branch lattice."""

    online = np.asarray(next_online_values, dtype=np.float32)
    target = np.asarray(next_target_values, dtype=np.float32)
    valid = np.asarray(next_valid, dtype=np.bool_)
    if (
        online.ndim != 1
        or online.shape != target.shape
        or online.shape != valid.shape
        or not np.isfinite(online).all()
        or not np.isfinite(target).all()
        or (not terminal and not valid.any())
        or not math.isfinite(float(reward))
    ):
        raise OursContractError("BranchQ Bellman target inputs are invalid")
    if terminal:
        return float(reward)
    selected = int(np.where(valid, online, -np.inf).argmax())
    return float(reward + macro_discount(gamma, duration_steps) * target[selected])


def calibrate_grouped_conformal_scale(
    ensemble_values: Any,
    targets: Any,
    group_ids: Sequence[Hashable],
    *,
    alpha: float,
    standard_deviation_floor: float,
    np: Any,
) -> ConformalCalibration:
    """Calibrate an episode-grouped normalized residual radius.

    Taking the maximum score inside each episode prevents dense episodes from
    dominating calibration and provides a simultaneous episode-level interval
    under the usual exchangeability assumption.
    """

    predictions = np.asarray(ensemble_values, dtype=np.float32)
    observed = np.asarray(targets, dtype=np.float32)
    if (
        predictions.ndim != 2
        or predictions.shape[0] < 2
        or observed.shape != (predictions.shape[1],)
        or len(group_ids) != predictions.shape[1]
        or not 0.0 < alpha < 1.0
        or standard_deviation_floor <= 0.0
        or not np.isfinite(predictions).all()
        or not np.isfinite(observed).all()
    ):
        raise OursContractError("BranchQ conformal calibration inputs are invalid")
    means = predictions.mean(axis=0)
    scales = predictions.std(axis=0, ddof=1) + standard_deviation_floor
    scores = np.abs(observed - means) / scales
    grouped: dict[Hashable, float] = {}
    for group, score in zip(group_ids, scores, strict=True):
        grouped[group] = max(grouped.get(group, -math.inf), float(score))
    ordered = sorted(grouped.values())
    rank = math.ceil((len(ordered) + 1) * (1.0 - alpha))
    quantile = math.inf if rank > len(ordered) else ordered[rank - 1]
    return ConformalCalibration(
        alpha=float(alpha),
        quantile=float(quantile),
        groups=len(ordered),
        samples=predictions.shape[1],
        standard_deviation_floor=float(standard_deviation_floor),
    )


def conservative_branch_selection(
    candidates: Sequence[BranchCandidate],
    ensemble_values: Any,
    *,
    conformal_scale: float,
    standard_deviation_floor: float,
    support_counts: Sequence[int] | None = None,
    minimum_support: int = 1,
    minimum_advantage: float = 0.0,
    np: Any,
) -> ConservativeBranchDecision:
    """Choose an alternative only when its LCB exceeds frozen B's UCB."""

    predictions = np.asarray(ensemble_values, dtype=np.float32)
    if (
        not candidates
        or predictions.ndim != 2
        or predictions.shape[1] != len(candidates)
        or predictions.shape[0] < 2
        or not np.isfinite(predictions).all()
        or conformal_scale < 0.0
        or standard_deviation_floor <= 0.0
        or minimum_support < 1
        or minimum_advantage < 0.0
    ):
        raise OursContractError("BranchQ conservative selection inputs are invalid")
    supports = (
        tuple(int(value) for value in support_counts)
        if support_counts is not None
        else (minimum_support,) * len(candidates)
    )
    if len(supports) != len(candidates) or min(supports) < 0:
        raise OursContractError("BranchQ support counts are invalid")
    baseline_indices = [index for index, candidate in enumerate(candidates) if candidate.baseline]
    if len(baseline_indices) != 1:
        raise OursContractError("BranchQ selection requires exactly one baseline candidate")
    baseline = baseline_indices[0]
    means = predictions.mean(axis=0)
    deviations = predictions.std(axis=0, ddof=1)
    widths = conformal_scale * (deviations + standard_deviation_floor)
    lower = means - widths
    upper = means + widths
    eligible = [
        index
        for index, candidate in enumerate(candidates)
        if not candidate.baseline and supports[index] >= minimum_support
    ]
    if not eligible:
        selected = baseline
        reason = "baseline-no-supported-alternative"
    else:
        selected = max(eligible, key=lambda index: (float(lower[index]), -index))
        if not float(lower[selected]) > float(upper[baseline]) + minimum_advantage:
            selected = baseline
            reason = "baseline-confidence-fallback"
        else:
            reason = "alternative-lcb-dominates-baseline-ucb"
    return ConservativeBranchDecision(
        selected_index=selected,
        baseline_index=baseline,
        selected_candidate_id=candidates[selected].candidate_id,
        baseline_candidate_id=candidates[baseline].candidate_id,
        selected_mean=float(means[selected]),
        selected_lower_bound=float(lower[selected]),
        baseline_mean=float(means[baseline]),
        baseline_upper_bound=float(upper[baseline]),
        fallback=selected == baseline,
        reason=reason,
    )
