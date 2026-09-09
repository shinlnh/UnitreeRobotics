"""Counterfactual reachability RL primitives for RESOLVE-VLA."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .ours import OursContractError

RESOLVE_ID = "Ours"
RESOLVE_VARIANT = "GR00T-RC-RESOLVE-VLA"
RESOLVE_METHOD = "counterfactual-rescue-bellman-recovery-policy"
RESOLVE_PARENT = "B"
RESOLVE_WORLDS = ("recovery", "deletion", "baseline")
RESOLVE_STATE_SCALARS = (
    "absolute_anchor_fraction",
    "remaining_budget_fraction",
    "active_subgoal_fraction",
    "completed_subtask_fraction",
    "recovery_elapsed_fraction",
    "selector_stop_probability",
)


@dataclass(frozen=True)
class ReachabilityTriplet:
    """One paired R/D/B target vector at a fixed physical horizon."""

    milestone_names: tuple[str, ...]
    recovery: tuple[float, ...]
    deletion: tuple[float, ...]
    baseline: tuple[float, ...]

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConstraintState:
    """Primal-dual state for separately reported recovery constraints."""

    names: tuple[str, ...]
    limits: tuple[float, ...]
    multipliers: tuple[float, ...]

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FiniteHorizonRDBValues:
    """Unique backward-induction solution of a fixed-policy R/D/B system."""

    recovery: Any
    deletion: Any
    baseline: Any


def _probabilities(values: Any, label: str, *, np: Any) -> Any:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.size < 1
        or not np.isfinite(array).all()
        or (array < 0.0).any()
        or (array > 1.0).any()
    ):
        raise OursContractError(f"RESOLVE {label} must contain finite probabilities")
    return array


def reachability_bellman_target(
    reached_during_macro: Any,
    next_reachability: Any,
    horizon_terminal: Any,
    *,
    np: Any,
) -> Any:
    """Apply the absorbing finite-horizon reachability Bellman operator.

    No discount, step bonus, or VLA-call penalty is mixed into the milestone.
    A terminal branch that has not reached the milestone bootstraps to zero.
    """

    reached = _probabilities(reached_during_macro, "macro reach", np=np)
    following = _probabilities(next_reachability, "next reachability", np=np)
    terminal = np.asarray(horizon_terminal, dtype=np.bool_)
    if reached.shape != following.shape or terminal.shape not in {reached.shape, reached.shape[:-1]}:
        raise OursContractError("RESOLVE Bellman target shapes are invalid")
    if terminal.shape == reached.shape[:-1]:
        terminal = terminal[..., None]
    return reached + (1.0 - reached) * (~terminal).astype(np.float64) * following


def counterfactual_rescue_bottleneck(
    recovery_value: Any,
    deletion_value: Any,
    baseline_value: Any,
    *,
    np: Any,
) -> Any:
    """Return the conjunctive long-horizon recovery/necessity advantage.

    ``min(Q_R-Q_B, Q_R-Q_D)`` is exactly ``Q_R-max(Q_B,Q_D)``. It is positive
    only if the learned recovery continuation beats pure B and the present
    macro is necessary relative to its one-position deletion world.
    """

    recovery = _probabilities(recovery_value, "recovery value", np=np)
    deletion = _probabilities(deletion_value, "deletion value", np=np)
    baseline = _probabilities(baseline_value, "baseline value", np=np)
    if recovery.shape != deletion.shape or recovery.shape != baseline.shape:
        raise OursContractError("RESOLVE bottleneck value shapes are invalid")
    return recovery - np.maximum(deletion, baseline)


def finite_horizon_rdb_fixed_point(
    *,
    recovery_reach: Any,
    deletion_reach: Any,
    baseline_reach: Any,
    recovery_transition: Any,
    deletion_transition: Any,
    baseline_transition: Any,
    np: Any,
) -> FiniteHorizonRDBValues:
    """Solve the coupled fixed-policy R/D/B reachability equations.

    Arrays of immediate reach have shape ``[anchor, state, milestone]`` and
    transition kernels have shape ``[anchor, state, next_state]``. R and D
    both bootstrap the *same* recovery continuation value; only their current
    physical transition differs. B bootstraps itself. Absolute anchor is part
    of state, so the finite system is acyclic and has one solution by backward
    induction even with undiscounted reachability.
    """

    reached_r = _probabilities(recovery_reach, "R immediate reach", np=np)
    reached_d = _probabilities(deletion_reach, "D immediate reach", np=np)
    reached_b = _probabilities(baseline_reach, "B immediate reach", np=np)
    transition_r = _probabilities(recovery_transition, "R transition", np=np)
    transition_d = _probabilities(deletion_transition, "D transition", np=np)
    transition_b = _probabilities(baseline_transition, "B transition", np=np)
    if (
        reached_r.ndim != 3
        or reached_r.shape != reached_d.shape
        or reached_r.shape != reached_b.shape
        or transition_r.ndim != 3
        or transition_r.shape != transition_d.shape
        or transition_r.shape != transition_b.shape
        or transition_r.shape[:2] != reached_r.shape[:2]
        or transition_r.shape[2] != reached_r.shape[1]
        or not np.allclose(transition_r.sum(axis=2), 1.0)
        or not np.allclose(transition_d.sum(axis=2), 1.0)
        or not np.allclose(transition_b.sum(axis=2), 1.0)
    ):
        raise OursContractError("RESOLVE finite-horizon RDB system is invalid")

    anchors, states, milestones = reached_r.shape
    value_r = np.zeros((anchors + 1, states, milestones), dtype=np.float64)
    value_d = np.zeros_like(value_r)
    value_b = np.zeros_like(value_r)
    for anchor in range(anchors - 1, -1, -1):
        next_r = transition_r[anchor] @ value_r[anchor + 1]
        next_d = transition_d[anchor] @ value_r[anchor + 1]
        next_b = transition_b[anchor] @ value_b[anchor + 1]
        value_r[anchor] = reached_r[anchor] + (1.0 - reached_r[anchor]) * next_r
        value_d[anchor] = reached_d[anchor] + (1.0 - reached_d[anchor]) * next_d
        value_b[anchor] = reached_b[anchor] + (1.0 - reached_b[anchor]) * next_b
    return FiniteHorizonRDBValues(
        recovery=value_r[:-1],
        deletion=value_d[:-1],
        baseline=value_b[:-1],
    )


def conservative_ensemble_bottleneck(
    recovery_values: Any,
    deletion_values: Any,
    baseline_values: Any,
    *,
    np: Any,
) -> Any:
    """Use recovery lower and counterfactual upper ensemble envelopes.

    The ensemble dimension is first. These envelopes are an operational guard,
    not a statistical confidence interval; final claims use grouped paired
    bootstrap certificates.
    """

    recovery = _probabilities(recovery_values, "ensemble recovery", np=np)
    deletion = _probabilities(deletion_values, "ensemble deletion", np=np)
    baseline = _probabilities(baseline_values, "ensemble baseline", np=np)
    if (
        recovery.ndim < 2
        or recovery.shape != deletion.shape
        or recovery.shape != baseline.shape
        or recovery.shape[0] < 2
    ):
        raise OursContractError("RESOLVE ensemble value shapes are invalid")
    return recovery.min(axis=0) - np.maximum(
        deletion.max(axis=0), baseline.max(axis=0)
    )


def reachability_triplet(
    *,
    milestone_names: tuple[str, ...],
    recovery: Any,
    deletion: Any,
    baseline: Any,
    np: Any,
) -> ReachabilityTriplet:
    """Validate and serialize one critic target without scalarizing milestones."""

    recovery_array = _probabilities(recovery, "triplet recovery", np=np)
    deletion_array = _probabilities(deletion, "triplet deletion", np=np)
    baseline_array = _probabilities(baseline, "triplet baseline", np=np)
    if (
        recovery_array.ndim != 1
        or recovery_array.shape != deletion_array.shape
        or recovery_array.shape != baseline_array.shape
        or len(milestone_names) != len(recovery_array)
        or len(set(milestone_names)) != len(milestone_names)
        or any(not item for item in milestone_names)
    ):
        raise OursContractError("RESOLVE reachability-triplet metadata are invalid")
    return ReachabilityTriplet(
        milestone_names=milestone_names,
        recovery=tuple(float(value) for value in recovery_array),
        deletion=tuple(float(value) for value in deletion_array),
        baseline=tuple(float(value) for value in baseline_array),
    )


def update_constraint_multipliers(
    state: ConstraintState,
    observed_costs: Any,
    *,
    learning_rate: float,
    np: Any,
) -> ConstraintState:
    """Perform projected dual ascent without folding costs into task labels."""

    limits = np.asarray(state.limits, dtype=np.float64)
    multipliers = np.asarray(state.multipliers, dtype=np.float64)
    observed = np.asarray(observed_costs, dtype=np.float64)
    if (
        not state.names
        or len(set(state.names)) != len(state.names)
        or any(not item for item in state.names)
        or limits.shape != (len(state.names),)
        or multipliers.shape != limits.shape
        or observed.shape != limits.shape
        or not np.isfinite(limits).all()
        or not np.isfinite(multipliers).all()
        or not np.isfinite(observed).all()
        or (limits < 0.0).any()
        or (multipliers < 0.0).any()
        or (observed < 0.0).any()
        or not math.isfinite(learning_rate)
        or learning_rate <= 0.0
    ):
        raise OursContractError("RESOLVE constraint update inputs are invalid")
    updated = np.maximum(0.0, multipliers + learning_rate * (observed - limits))
    return ConstraintState(
        names=state.names,
        limits=state.limits,
        multipliers=tuple(float(value) for value in updated),
    )
