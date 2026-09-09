import numpy as np
import pytest

from unitree_gr00t.ours import OursContractError
from unitree_gr00t.recovery_curvature import (
    CurvatureQuartet,
    assess_curvature_pilot,
    bilinear_curvature,
    cross_temporal_recovery_effect,
    orthogonal_steering_basis,
    recovery_curvature_seed,
    spherical_noise_steering,
)


def quartet(
    *,
    snapshot: str = "snapshot-1",
    group: str = "episode-1",
    case: str = "Ideal/case1",
    bb=(0.0, 0.0),
    ub=(0.0, 0.0),
    bv=(0.0, 0.0),
    uv=(1.0, 1.0),
) -> CurvatureQuartet:
    return CurvatureQuartet(
        snapshot_id=snapshot,
        episode_group=group,
        task_case=case,
        first_program_id="u1",
        second_program_id="v1",
        milestone_names=("predicate-1", "subtask-1"),
        milestone_levels=("predicate", "subtask"),
        baseline_baseline=bb,
        first_baseline=ub,
        baseline_second=bv,
        first_second=uv,
    )


def test_curvature_is_factorial_difference_and_gain_is_irreducible() -> None:
    effect = cross_temporal_recovery_effect(quartet(), np=np)
    assert effect.curvature == (1.0, 1.0)
    assert effect.irreducible_gain == (1.0, 1.0)
    assert effect.positive_milestones == ("predicate-1", "subtask-1")

    # Both unary programs already work, so the joint arm is not a rescue that
    # requires their interaction. Difference-in-differences records saturation.
    additive = cross_temporal_recovery_effect(
        quartet(ub=(1.0, 1.0), bv=(1.0, 1.0), uv=(1.0, 1.0)),
        np=np,
    )
    assert additive.curvature == (-1.0, -1.0)
    assert additive.irreducible_gain == (0.0, 0.0)
    assert additive.positive_milestones == ()


def test_positive_curvature_is_not_automatically_irreducible_rescue() -> None:
    effect = cross_temporal_recovery_effect(
        quartet(bb=(1.0, 1.0), ub=(0.0, 0.0), bv=(0.0, 0.0), uv=(1.0, 1.0)),
        np=np,
    )
    assert effect.curvature == (2.0, 2.0)
    assert effect.irreducible_gain == (0.0, 0.0)


def test_common_random_seed_and_noise_basis_are_deterministic() -> None:
    seed = recovery_curvature_seed(10007, "snapshot-a")
    assert seed == recovery_curvature_seed(10007, "snapshot-a")
    assert seed != recovery_curvature_seed(10007, "snapshot-b")
    basis = orthogonal_steering_basis(12, 4, seed, np=np)
    repeated = orthogonal_steering_basis(12, 4, seed, np=np)
    assert np.array_equal(basis, repeated)
    assert np.allclose(basis.T @ basis, np.eye(4), atol=1e-6)


def test_spherical_noise_steering_preserves_norm_and_exact_zero() -> None:
    baseline = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    basis = orthogonal_steering_basis(12, 3, 19, np=np)
    zero = spherical_noise_steering(
        baseline,
        basis,
        np.zeros(3),
        maximum_angle=0.4,
        np=np,
    )
    positive = spherical_noise_steering(
        baseline,
        basis,
        np.asarray([0.2, 0.0, 0.0]),
        maximum_angle=0.4,
        np=np,
    )
    negative = spherical_noise_steering(
        baseline,
        basis,
        np.asarray([-0.2, 0.0, 0.0]),
        maximum_angle=0.4,
        np=np,
    )
    assert np.array_equal(zero, baseline)
    assert np.isclose(np.linalg.norm(positive), np.linalg.norm(baseline), rtol=1e-6)
    assert np.isclose(np.linalg.norm(negative), np.linalg.norm(baseline), rtol=1e-6)
    assert not np.allclose(positive, negative)


def test_pilot_gate_requires_independent_cross_case_subtask_evidence() -> None:
    positive = (
        quartet(snapshot="s1", group="g1", case="Ideal/case1"),
        quartet(snapshot="s2", group="g2", case="Ideal/case2"),
        quartet(snapshot="s3", group="g3", case="Mix/case1"),
    )
    accepted = assess_curvature_pilot(positive, np=np)
    assert accepted.proceed
    assert accepted.independent_positive_groups == 3
    assert accepted.positive_task_cases == 3
    assert accepted.positive_subtask_quartets == 3

    rejected = assess_curvature_pilot(positive[:2], np=np)
    assert not rejected.proceed
    assert "insufficient-independent-positive-groups" in rejected.reasons


def test_bilinear_curvature_scores_ordered_code_pairs() -> None:
    first = np.eye(2)
    second = np.eye(2)
    matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(bilinear_curvature(first, matrix, second, np=np), matrix)


def test_curvature_rejects_invalid_outcome_probabilities() -> None:
    with pytest.raises(OursContractError, match="finite probabilities"):
        cross_temporal_recovery_effect(quartet(uv=(1.1, 0.0)), np=np)
