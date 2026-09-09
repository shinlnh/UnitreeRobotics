"""Classical factorial-curvature diagnostics for the MOSAIC-VLA hypothesis."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

from .ours import OursContractError

C2R_ID = "Ours"
C2R_VARIANT = "MOSAIC-curvature-diagnostic-v1"
C2R_METHOD = "paired-cross-temporal-factorial-diagnostic"
C2R_BASELINE = "B"
C2R_QUARTET = ("B->B", "u->B", "B->v", "u->v")


@dataclass(frozen=True)
class CurvatureQuartet:
    """One same-snapshot factorial experiment over ordered interventions."""

    snapshot_id: str
    episode_group: str
    task_case: str
    first_program_id: str
    second_program_id: str
    milestone_names: tuple[str, ...]
    milestone_levels: tuple[str, ...]
    baseline_baseline: tuple[float, ...]
    first_baseline: tuple[float, ...]
    baseline_second: tuple[float, ...]
    first_second: tuple[float, ...]

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CurvatureEffect:
    curvature: tuple[float, ...]
    irreducible_gain: tuple[float, ...]
    positive_milestones: tuple[str, ...]


@dataclass(frozen=True)
class CurvaturePilotAssessment:
    proceed: bool
    quartets: int
    positive_quartets: int
    independent_positive_groups: int
    positive_task_cases: int
    positive_subtask_quartets: int
    reasons: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def recovery_curvature_seed(base_seed: int, snapshot_id: str, replicate: int = 0) -> int:
    """Return one common-random-number seed shared by all quartet arms."""

    if base_seed < 0 or replicate < 0 or not snapshot_id:
        raise OursContractError("C2R seed inputs are invalid")
    payload = f"C2R-factorial-v1:{base_seed}:{snapshot_id}:{replicate}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFF_FFFF


def orthogonal_steering_basis(
    ambient_dimension: int,
    code_dimension: int,
    seed: int,
    *,
    np: Any,
) -> Any:
    """Create a deterministic orthonormal basis for frozen-VLA action noise."""

    if (
        ambient_dimension < 1
        or code_dimension < 1
        or code_dimension > ambient_dimension
        or seed < 0
    ):
        raise OursContractError("C2R steering-basis dimensions are invalid")
    generator = np.random.default_rng(seed)
    raw = generator.standard_normal((ambient_dimension, code_dimension))
    basis, _ = np.linalg.qr(raw, mode="reduced")
    # QR column signs are arbitrary. Canonicalizing the largest-magnitude entry
    # makes serialized bases stable across repeated construction.
    pivots = np.abs(basis).argmax(axis=0)
    signs = np.sign(basis[pivots, np.arange(code_dimension)])
    signs[signs == 0] = 1
    return (basis * signs).astype(np.float32)


def spherical_noise_steering(
    baseline_noise: Any,
    basis: Any,
    code: Any,
    *,
    maximum_angle: float,
    np: Any,
) -> Any:
    """Move frozen-VLA noise on its equal-norm sphere by a tangent code.

    Zero is byte-identical to baseline noise. Nonzero codes use the exponential
    map on the sphere, so the steering experiment changes direction without
    confounding it with the unusually large/small noise norms produced by an
    unconstrained additive perturbation.
    """

    baseline = np.asarray(baseline_noise, dtype=np.float64)
    matrix = np.asarray(basis, dtype=np.float64)
    coordinates = np.asarray(code, dtype=np.float64)
    ambient = baseline.size
    if (
        baseline.ndim < 1
        or matrix.ndim != 2
        or matrix.shape[0] != ambient
        or coordinates.shape != (matrix.shape[1],)
        or not np.isfinite(baseline).all()
        or not np.isfinite(matrix).all()
        or not np.isfinite(coordinates).all()
        or not 0.0 < maximum_angle <= math.pi / 2
    ):
        raise OursContractError("C2R spherical-steering inputs are invalid")
    baseline_flat = baseline.reshape(-1)
    baseline_norm = float(np.linalg.norm(baseline_flat))
    code_norm = float(np.linalg.norm(coordinates))
    if baseline_norm <= 0.0 or code_norm > maximum_angle + 1e-8:
        raise OursContractError("C2R spherical-steering norm is invalid")
    if code_norm == 0.0:
        return baseline.astype(np.float32, copy=True)
    baseline_unit = baseline_flat / baseline_norm
    tangent = matrix @ coordinates
    tangent = tangent - float(tangent @ baseline_unit) * baseline_unit
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-10:
        raise OursContractError("C2R steering direction is parallel to baseline noise")
    tangent_unit = tangent / tangent_norm
    steered = baseline_norm * (
        math.cos(code_norm) * baseline_unit + math.sin(code_norm) * tangent_unit
    )
    return steered.reshape(baseline.shape).astype(np.float32)


def _outcome_array(values: tuple[float, ...], width: int, label: str, *, np: Any) -> Any:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.shape != (width,)
        or not np.isfinite(array).all()
        or (array < 0.0).any()
        or (array > 1.0).any()
    ):
        raise OursContractError(f"C2R {label} outcomes must be finite probabilities")
    return array


def cross_temporal_recovery_effect(quartet: CurvatureQuartet, *, np: Any) -> CurvatureEffect:
    """Compute mixed curvature and strict irreducible gain for one quartet.

    ``curvature`` is the difference-in-differences physical effect. The stricter
    ``irreducible_gain`` is positive only when the joint ordered intervention
    exceeds the baseline and both proper one-intervention ablations.
    """

    width = len(quartet.milestone_names)
    if (
        width < 1
        or len(set(quartet.milestone_names)) != width
        or len(quartet.milestone_levels) != width
        or not quartet.snapshot_id
        or not quartet.episode_group
        or not quartet.task_case
        or not quartet.first_program_id
        or not quartet.second_program_id
        or quartet.first_program_id == "B"
        or quartet.second_program_id == "B"
        or any(level not in {"predicate", "subtask", "final"} for level in quartet.milestone_levels)
    ):
        raise OursContractError("C2R quartet metadata are invalid")
    bb = _outcome_array(quartet.baseline_baseline, width, "B->B", np=np)
    ub = _outcome_array(quartet.first_baseline, width, "u->B", np=np)
    bv = _outcome_array(quartet.baseline_second, width, "B->v", np=np)
    uv = _outcome_array(quartet.first_second, width, "u->v", np=np)
    curvature = uv - ub - bv + bb
    irreducible = uv - np.maximum.reduce([ub, bv, bb])
    positive = tuple(
        name
        for name, gain in zip(quartet.milestone_names, irreducible, strict=True)
        if float(gain) > 0.0
    )
    return CurvatureEffect(
        curvature=tuple(float(value) for value in curvature),
        irreducible_gain=tuple(float(value) for value in irreducible),
        positive_milestones=positive,
    )


def assess_curvature_pilot(
    quartets: tuple[CurvatureQuartet, ...],
    *,
    minimum_independent_groups: int = 3,
    minimum_task_cases: int = 2,
    require_subtask_gain: bool = True,
    np: Any,
) -> CurvaturePilotAssessment:
    """Apply the preregistered mechanism gates without a learned model."""

    if (
        not quartets
        or minimum_independent_groups < 1
        or minimum_task_cases < 1
    ):
        raise OursContractError("C2R pilot-assessment inputs are invalid")
    identities = [
        (item.snapshot_id, item.first_program_id, item.second_program_id) for item in quartets
    ]
    if len(set(identities)) != len(identities):
        raise OursContractError("C2R pilot contains duplicate factorial quartets")

    positive_groups: set[str] = set()
    positive_cases: set[str] = set()
    positive_quartets = 0
    positive_subtask_quartets = 0
    for quartet in quartets:
        effect = cross_temporal_recovery_effect(quartet, np=np)
        positive_indices = [
            index for index, gain in enumerate(effect.irreducible_gain) if gain > 0.0
        ]
        if not positive_indices:
            continue
        positive_quartets += 1
        positive_groups.add(quartet.episode_group)
        positive_cases.add(quartet.task_case)
        if any(quartet.milestone_levels[index] in {"subtask", "final"} for index in positive_indices):
            positive_subtask_quartets += 1

    reasons: list[str] = []
    if len(positive_groups) < minimum_independent_groups:
        reasons.append("insufficient-independent-positive-groups")
    if len(positive_cases) < minimum_task_cases:
        reasons.append("insufficient-positive-task-cases")
    if require_subtask_gain and positive_subtask_quartets < 1:
        reasons.append("no-irreducible-subtask-gain")
    return CurvaturePilotAssessment(
        proceed=not reasons,
        quartets=len(quartets),
        positive_quartets=positive_quartets,
        independent_positive_groups=len(positive_groups),
        positive_task_cases=len(positive_cases),
        positive_subtask_quartets=positive_subtask_quartets,
        reasons=tuple(reasons),
    )


def bilinear_curvature(codes_first: Any, matrix: Any, codes_second: Any, *, np: Any) -> Any:
    """Evaluate ``u^T M v`` for finite steering-code sets during a pilot."""

    first = np.asarray(codes_first, dtype=np.float64)
    curvature = np.asarray(matrix, dtype=np.float64)
    second = np.asarray(codes_second, dtype=np.float64)
    if (
        first.ndim != 2
        or second.ndim != 2
        or curvature.ndim != 2
        or first.shape[1] != curvature.shape[0]
        or second.shape[1] != curvature.shape[1]
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or not np.isfinite(curvature).all()
    ):
        raise OursContractError("C2R bilinear-curvature inputs are invalid")
    values = first @ curvature @ second.T
    if not np.isfinite(values).all() or math.prod(values.shape) < 1:
        raise OursContractError("C2R bilinear-curvature output is invalid")
    return values
