"""Deletion-necessity primitives for adaptive MOSAIC-VLA recovery programs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .ours import OursContractError

MOSAIC_ID = "Ours"
MOSAIC_VARIANT = "GR00T-RC-MOSAIC-VLA"
MOSAIC_METHOD = "minimal-ordered-sufficient-ablation-identified-continuations"
MOSAIC_PARENT = "B"
MOSAIC_CERTIFICATE = "paired-one-deletion-bootstrap-v1"


@dataclass(frozen=True)
class ProgramArm:
    """One registered arm in a baseline/full/one-deletion experiment."""

    arm_id: str
    kind: str
    deleted_rule_index: int | None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeletionEffects:
    """Group-paired effects for a variable-depth recovery program.

    ``baseline_gain`` has shape ``[group, milestone]`` and
    ``necessity_gain`` has shape ``[group, rule, milestone]`` after conversion
    to an array. Keeping every group rather than only a mean is essential: the
    arms from one simulator snapshot are paired observations, not independent
    samples.
    """

    group_ids: tuple[str, ...]
    milestone_names: tuple[str, ...]
    baseline_gain: tuple[tuple[float, ...], ...]
    necessity_gain: tuple[tuple[tuple[float, ...], ...], ...]

    @property
    def groups(self) -> int:
        return len(self.group_ids)

    @property
    def depth(self) -> int:
        return len(self.necessity_gain[0])

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeletionCertificate:
    """Grouped bootstrap lower bounds for sufficiency and every deletion."""

    confidence: float
    bootstrap_samples: int
    groups: int
    milestone_names: tuple[str, ...]
    baseline_gain_mean: tuple[float, ...]
    baseline_gain_lower: tuple[float, ...]
    necessity_gain_mean: tuple[tuple[float, ...], ...]
    necessity_gain_lower: tuple[tuple[float, ...], ...]
    bottleneck_lower: tuple[float, ...]
    certified_milestones: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FixedAnchorArm:
    """Causal identifiers that must agree across all paired program arms."""

    arm_id: str
    snapshot_hash: str
    environment_rng_hash: str
    disturbance_rng_hash: str
    disturbance_cursor: int
    disturbance_toggle: bool
    start_step: int
    anchor_steps: tuple[int, ...]
    outcome_step: int
    global_step_budget: int


def program_arms(program_id: str, depth: int) -> tuple[ProgramArm, ...]:
    """Return exact B, the full program, and all one-rule deletions.

    A depth-``d`` certificate therefore consumes ``d+2`` distinct arms. This
    does not enumerate either the program space or the complete ablation cube.
    """

    if not program_id or program_id == MOSAIC_PARENT or depth < 1:
        raise OursContractError("MOSAIC program-arm inputs are invalid")
    arms = [
        ProgramArm(arm_id=f"{program_id}:baseline", kind="baseline", deleted_rule_index=None),
        ProgramArm(arm_id=f"{program_id}:full", kind="full", deleted_rule_index=None),
    ]
    arms.extend(
        ProgramArm(
            arm_id=f"{program_id}:delete-{index:02d}",
            kind="deletion",
            deleted_rule_index=index,
        )
        for index in range(depth)
    )
    return tuple(arms)


def _finite_outcomes(values: Any, label: str, *, np: Any) -> Any:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim < 2
        or 0 in array.shape
        or not np.isfinite(array).all()
        or (array < 0.0).any()
        or (array > 1.0).any()
    ):
        raise OursContractError(
            f"MOSAIC {label} outcomes must be non-empty finite probabilities"
        )
    return array


def paired_deletion_effects(
    *,
    group_ids: tuple[str, ...],
    milestone_names: tuple[str, ...],
    baseline: Any,
    full_program: Any,
    one_deletions: Any,
    np: Any,
) -> DeletionEffects:
    """Compute paired sufficiency and rule-necessity effects.

    Outcome shapes are ``baseline/full=[group, milestone]`` and
    ``one_deletions=[group, rule, milestone]``. A deletion arm retains rule
    order and replaces only its indexed rule with exact B.
    """

    baseline_array = _finite_outcomes(baseline, "baseline", np=np)
    full_array = _finite_outcomes(full_program, "full-program", np=np)
    deletion_array = _finite_outcomes(one_deletions, "one-deletion", np=np)
    if (
        baseline_array.ndim != 2
        or full_array.shape != baseline_array.shape
        or deletion_array.ndim != 3
        or deletion_array.shape[0] != baseline_array.shape[0]
        or deletion_array.shape[2] != baseline_array.shape[1]
        or deletion_array.shape[1] < 1
        or len(group_ids) != baseline_array.shape[0]
        or len(set(group_ids)) != len(group_ids)
        or any(not item for item in group_ids)
        or len(milestone_names) != baseline_array.shape[1]
        or len(set(milestone_names)) != len(milestone_names)
        or any(not item for item in milestone_names)
    ):
        raise OursContractError("MOSAIC paired outcome shapes or identifiers are invalid")

    baseline_gain = full_array - baseline_array
    necessity_gain = full_array[:, None, :] - deletion_array
    return DeletionEffects(
        group_ids=group_ids,
        milestone_names=milestone_names,
        baseline_gain=tuple(tuple(float(value) for value in row) for row in baseline_gain),
        necessity_gain=tuple(
            tuple(tuple(float(value) for value in row) for row in group)
            for group in necessity_gain
        ),
    )


def _effect_arrays(effects: DeletionEffects, *, np: Any) -> tuple[Any, Any]:
    baseline_gain = np.asarray(effects.baseline_gain, dtype=np.float64)
    necessity_gain = np.asarray(effects.necessity_gain, dtype=np.float64)
    if (
        baseline_gain.ndim != 2
        or necessity_gain.ndim != 3
        or necessity_gain.shape[0] != baseline_gain.shape[0]
        or necessity_gain.shape[2] != baseline_gain.shape[1]
        or len(effects.group_ids) != baseline_gain.shape[0]
        or len(effects.milestone_names) != baseline_gain.shape[1]
        or not np.isfinite(baseline_gain).all()
        or not np.isfinite(necessity_gain).all()
    ):
        raise OursContractError("MOSAIC deletion effects are malformed")
    return baseline_gain, necessity_gain


def deletion_minimal_certificate(
    effects: DeletionEffects,
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
    np: Any,
) -> DeletionCertificate:
    """Certify baseline superiority and every one-rule necessity effect.

    Whole episode/snapshot groups are resampled with common indices for all
    arms, rules, and milestones. This preserves the paired factorial structure.
    The returned lower endpoint is from a two-sided percentile interval at the
    requested confidence; certification requires its bottleneck to exceed zero.
    """

    baseline_gain, necessity_gain = _effect_arrays(effects, np=np)
    groups = baseline_gain.shape[0]
    if (
        groups < 2
        or not 0.5 < confidence < 1.0
        or bootstrap_samples < 100
        or seed < 0
    ):
        raise OursContractError("MOSAIC certificate inputs are invalid")

    generator = np.random.default_rng(seed)
    group_indices = generator.integers(0, groups, size=(bootstrap_samples, groups))
    sampled_baseline = baseline_gain[group_indices].mean(axis=1)
    sampled_necessity = necessity_gain[group_indices].mean(axis=1)
    lower_quantile = (1.0 - confidence) / 2.0
    baseline_lower = np.quantile(sampled_baseline, lower_quantile, axis=0)
    necessity_lower = np.quantile(sampled_necessity, lower_quantile, axis=0)
    bottleneck_lower = np.minimum(baseline_lower, necessity_lower.min(axis=0))
    certified = tuple(
        milestone
        for milestone, lower in zip(
            effects.milestone_names, bottleneck_lower, strict=True
        )
        if float(lower) > 0.0
    )
    return DeletionCertificate(
        confidence=float(confidence),
        bootstrap_samples=bootstrap_samples,
        groups=groups,
        milestone_names=effects.milestone_names,
        baseline_gain_mean=tuple(float(value) for value in baseline_gain.mean(axis=0)),
        baseline_gain_lower=tuple(float(value) for value in baseline_lower),
        necessity_gain_mean=tuple(
            tuple(float(value) for value in row) for row in necessity_gain.mean(axis=0)
        ),
        necessity_gain_lower=tuple(
            tuple(float(value) for value in row) for row in necessity_lower
        ),
        bottleneck_lower=tuple(float(value) for value in bottleneck_lower),
        certified_milestones=certified,
    )


def soft_causal_bottleneck(first: Any, second: Any, *, temperature: float, np: Any) -> Any:
    """Smooth the logical conjunction ``first>0 and second>0``.

    This is a stable soft minimum. A rule cannot receive a positive necessity
    weight merely because the full program succeeds: both its program-vs-B and
    full-vs-deletion effects must clear the temperature-dependent bottleneck.
    """

    first_array = np.asarray(first, dtype=np.float64)
    second_array = np.asarray(second, dtype=np.float64)
    if (
        first_array.shape != second_array.shape
        or not np.isfinite(first_array).all()
        or not np.isfinite(second_array).all()
        or not math.isfinite(temperature)
        or temperature <= 0.0
    ):
        raise OursContractError("MOSAIC causal-bottleneck inputs are invalid")
    return -temperature * np.logaddexp(
        -first_array / temperature,
        -second_array / temperature,
    )


def deletion_necessity_weights(
    effects: DeletionEffects,
    *,
    np: Any,
) -> Any:
    """Return exact deletion-necessity advantages for every paired group.

    ``min(full-B, full-deletion_i)`` is identically
    ``full-max(B, deletion_i)``. It is positive exactly when both sufficiency
    and the indexed rule's necessity hold for that physical milestone.
    """

    baseline_gain, necessity_gain = _effect_arrays(effects, np=np)
    repeated_baseline = np.broadcast_to(
        baseline_gain[:, None, :], necessity_gain.shape
    )
    return np.minimum(repeated_baseline, necessity_gain)


def deletion_necessity_odds_update(
    log_odds_against_baseline: Any,
    advantages: Any,
    *,
    step_size: float,
    np: Any,
) -> Any:
    """Apply one exponentiated mirror step in log-odds coordinates.

    The baseline is an explicit atom in the policy. Positive paired necessity
    raises the sampled intervention's odds against exact B; zero leaves them
    unchanged; harmful or insufficient programs lower them. ``advantages`` are
    stop-gradient physical counterfactual targets, not learned proxy rewards.
    """

    current = np.asarray(log_odds_against_baseline, dtype=np.float64)
    gain = np.asarray(advantages, dtype=np.float64)
    if (
        current.shape != gain.shape
        or not np.isfinite(current).all()
        or not np.isfinite(gain).all()
        or not math.isfinite(step_size)
        or step_size <= 0.0
    ):
        raise OursContractError("MOSAIC deletion-necessity odds inputs are invalid")
    updated = current + step_size * gain
    if not np.isfinite(updated).all():
        raise OursContractError("MOSAIC deletion-necessity odds update overflowed")
    return updated


def validate_fixed_anchor_arms(arms: tuple[FixedAnchorArm, ...]) -> None:
    """Reject causal comparisons whose exogenous tape or horizon differs."""

    if len(arms) < 3 or len({arm.arm_id for arm in arms}) != len(arms):
        raise OursContractError("MOSAIC fixed-anchor arm set is invalid")
    first = arms[0]
    if (
        not first.snapshot_hash
        or not first.environment_rng_hash
        or not first.disturbance_rng_hash
        or first.disturbance_cursor < 0
        or first.start_step < 0
        or first.global_step_budget < 1
        or first.outcome_step <= first.start_step
        or not first.anchor_steps
        or tuple(sorted(first.anchor_steps)) != first.anchor_steps
        or first.anchor_steps[0] < first.start_step
        or first.anchor_steps[-1] >= first.outcome_step
    ):
        raise OursContractError("MOSAIC fixed-anchor reference is invalid")
    reference = (
        first.snapshot_hash,
        first.environment_rng_hash,
        first.disturbance_rng_hash,
        first.disturbance_cursor,
        first.disturbance_toggle,
        first.start_step,
        first.anchor_steps,
        first.outcome_step,
        first.global_step_budget,
    )
    for arm in arms[1:]:
        candidate = (
            arm.snapshot_hash,
            arm.environment_rng_hash,
            arm.disturbance_rng_hash,
            arm.disturbance_cursor,
            arm.disturbance_toggle,
            arm.start_step,
            arm.anchor_steps,
            arm.outcome_step,
            arm.global_step_budget,
        )
        if candidate != reference:
            raise OursContractError(
                "MOSAIC paired arms differ in snapshot, random tape, anchors, or budget"
            )
