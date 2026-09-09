import numpy as np
import pytest

from unitree_gr00t.ours import OursContractError
from unitree_gr00t.resolve import (
    ConstraintState,
    FiniteHorizonRDBValues,
    all_deletion_rescue_bottleneck,
    conservative_ensemble_bottleneck,
    counterfactual_rescue_bottleneck,
    finite_horizon_deletion_fixed_point,
    finite_horizon_deletion_operator,
    finite_horizon_rdb_fixed_point,
    finite_horizon_rdb_operator,
    reachability_bellman_target,
    reachability_triplet,
    update_constraint_multipliers,
)


def test_reachability_bellman_is_absorbing_and_horizon_aware() -> None:
    reached15 = reachability_bellman_target(
        np.asarray([[1.0, 0.0], [0.0, 0.0]]),
        np.asarray([[0.2, 0.8], [0.7, 0.9]]),
        np.asarray([False, True]),
        np=np,
    )
    assert np.array_equal(reached15, np.asarray([[1.0, 0.8], [0.0, 0.0]]))


def test_crb_requires_both_baseline_superiority_and_macro_necessity() -> None:
    recovery = np.asarray([0.9, 0.9, 0.7])
    deletion = np.asarray([0.2, 0.95, 0.4])
    baseline = np.asarray([0.3, 0.1, 0.8])
    bottleneck = counterfactual_rescue_bottleneck(recovery, deletion, baseline, np=np)
    assert np.allclose(bottleneck, np.asarray([0.6, -0.05, -0.1]))


def test_coupled_rdb_fixed_point_credits_delayed_necessary_recovery() -> None:
    # At anchor 0 no world reaches the milestone immediately. The recovery
    # macro moves to the recoverable state; replacing it with B moves to the
    # dead state. At anchor 1 only the recoverable state can finish.
    recovery_reach = np.asarray(
        [
            [[0.0], [0.0]],
            [[1.0], [0.0]],
        ]
    )
    deletion_reach = np.zeros_like(recovery_reach)
    baseline_reach = np.zeros_like(recovery_reach)
    recovery_transition = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    deletion_transition = np.asarray(
        [
            [[0.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    baseline_transition = deletion_transition.copy()
    values = finite_horizon_rdb_fixed_point(
        recovery_reach=recovery_reach,
        deletion_reach=deletion_reach,
        baseline_reach=baseline_reach,
        recovery_transition=recovery_transition,
        deletion_transition=deletion_transition,
        baseline_transition=baseline_transition,
        np=np,
    )
    assert values.recovery[0, 0, 0] == 1.0
    assert values.deletion[0, 0, 0] == 0.0
    assert values.baseline[0, 0, 0] == 0.0
    assert (
        counterfactual_rescue_bottleneck(
            values.recovery[0, 0],
            values.deletion[0, 0],
            values.baseline[0, 0],
            np=np,
        ).item()
        == 1.0
    )


def test_rdb_operator_forgets_initial_values_after_horizon_backups() -> None:
    recovery_reach = np.asarray(
        [
            [[0.0], [0.0]],
            [[1.0], [0.0]],
        ]
    )
    deletion_reach = np.zeros_like(recovery_reach)
    baseline_reach = np.zeros_like(recovery_reach)
    recovery_transition = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    deletion_transition = np.asarray(
        [
            [[0.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    baseline_transition = deletion_transition.copy()
    shape = (3, 2, 1)
    zero = FiniteHorizonRDBValues(np.zeros(shape), np.zeros(shape), np.zeros(shape))
    arbitrary = FiniteHorizonRDBValues(
        np.full(shape, 0.17), np.full(shape, 0.53), np.full(shape, 0.91)
    )
    arbitrary.recovery[-1] = 0.0
    arbitrary.deletion[-1] = 0.0
    arbitrary.baseline[-1] = 0.0
    kwargs = {
        "recovery_reach": recovery_reach,
        "deletion_reach": deletion_reach,
        "baseline_reach": baseline_reach,
        "recovery_transition": recovery_transition,
        "deletion_transition": deletion_transition,
        "baseline_transition": baseline_transition,
        "np": np,
    }
    for _ in range(2):
        zero = finite_horizon_rdb_operator(zero, **kwargs)
        arbitrary = finite_horizon_rdb_operator(arbitrary, **kwargs)
    assert np.array_equal(zero.recovery, arbitrary.recovery)
    assert np.array_equal(zero.deletion, arbitrary.deletion)
    assert np.array_equal(zero.baseline, arbitrary.baseline)
    fixed = finite_horizon_rdb_operator(zero, **kwargs)
    assert np.array_equal(fixed.recovery, zero.recovery)
    assert np.array_equal(fixed.deletion, zero.deletion)
    assert np.array_equal(fixed.baseline, zero.baseline)


def test_deletion_envelope_finds_state_adaptive_most_redundant_slot() -> None:
    # Deleting at one globally fixed slot has expected effects .7, .5, and .5.
    # A state-observing auditor does better: after the first transition it
    # deletes immediately in state 0 (effect .1), but defers in state 1 to the
    # last slot (effect .2).  The initial envelope is therefore .15.
    recovery = np.ones((3, 2, 1))
    delete_now_gap = np.asarray(
        [
            [[0.7], [0.7]],
            [[0.1], [0.9]],
            [[0.8], [0.2]],
        ]
    )
    deletion = recovery - delete_now_gap
    transition = np.asarray(
        [
            [[0.5, 0.5], [0.5, 0.5]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    values = finite_horizon_deletion_fixed_point(
        recovery_value=recovery,
        deletion_value=deletion,
        recovery_transition=transition,
        np=np,
    )
    assert values.necessity[1, :, 0] == pytest.approx([0.1, 0.2])
    assert values.necessity[0, 0, 0] == pytest.approx(0.15)
    score = all_deletion_rescue_bottleneck(
        recovery,
        np.full_like(recovery, 0.6),
        values.necessity,
        np=np,
    )
    assert score[0, 0, 0] == pytest.approx(0.15)


def test_deletion_operator_has_unique_finite_horizon_fixed_point() -> None:
    gap = np.asarray([[[0.8]], [[0.4]], [[0.6]]])
    recovery = np.ones_like(gap)
    deletion = recovery - gap
    transition = np.ones((3, 1, 1))
    fixed = finite_horizon_deletion_fixed_point(
        recovery_value=recovery,
        deletion_value=deletion,
        recovery_transition=transition,
        np=np,
    ).necessity
    assert fixed[:, 0, 0] == pytest.approx([0.4, 0.4, 0.6])

    arbitrary = np.asarray([[[-0.9]], [[0.9]], [[-0.2]]])
    for _ in range(3):
        arbitrary = finite_horizon_deletion_operator(
            arbitrary,
            delete_now_gap=gap,
            recovery_transition=transition,
            np=np,
        )
    assert np.array_equal(arbitrary, fixed)


def test_deletion_operator_is_sup_norm_nonexpansive() -> None:
    gap = np.asarray(
        [
            [[0.8], [0.5]],
            [[0.3], [0.7]],
            [[0.6], [0.4]],
        ]
    )
    transition = np.asarray(
        [
            [[0.25, 0.75], [0.6, 0.4]],
            [[0.5, 0.5], [0.1, 0.9]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    left = np.asarray(
        [
            [[-0.2], [0.3]],
            [[0.1], [0.9]],
            [[0.4], [-0.1]],
        ]
    )
    right = np.asarray(
        [
            [[0.7], [-0.1]],
            [[-0.4], [0.2]],
            [[0.8], [0.2]],
        ]
    )
    updated_left = finite_horizon_deletion_operator(
        left, delete_now_gap=gap, recovery_transition=transition, np=np
    )
    updated_right = finite_horizon_deletion_operator(
        right, delete_now_gap=gap, recovery_transition=transition, np=np
    )
    assert np.max(np.abs(updated_left - updated_right)) <= np.max(np.abs(left - right))


def test_all_deletion_bottleneck_rejects_redundant_future_macro() -> None:
    value = all_deletion_rescue_bottleneck(
        np.asarray([0.9, 0.9]),
        np.asarray([0.2, 0.2]),
        np.asarray([0.3, -0.05]),
        np=np,
    )
    assert np.allclose(value, np.asarray([0.3, -0.05]))


def test_conservative_ensemble_uses_recovery_lcb_and_counterfactual_ucb() -> None:
    recovery = np.asarray([[0.8, 0.9], [0.7, 0.85]])
    deletion = np.asarray([[0.2, 0.6], [0.3, 0.5]])
    baseline = np.asarray([[0.4, 0.2], [0.35, 0.3]])
    value = conservative_ensemble_bottleneck(recovery, deletion, baseline, np=np)
    assert np.allclose(value, np.asarray([0.3, 0.25]))


def test_triplet_keeps_physical_milestones_vector_valued() -> None:
    item = reachability_triplet(
        milestone_names=("predicate", "subtask", "final"),
        recovery=np.asarray([1.0, 1.0, 0.0]),
        deletion=np.asarray([1.0, 0.0, 0.0]),
        baseline=np.asarray([0.0, 0.0, 0.0]),
        np=np,
    )
    assert item.recovery == (1.0, 1.0, 0.0)
    assert item.deletion == (1.0, 0.0, 0.0)


def test_projected_dual_update_only_penalizes_violated_constraints() -> None:
    state = ConstraintState(
        names=("regression", "invalid-action", "budget"),
        limits=(0.05, 0.01, 32.0),
        multipliers=(0.0, 0.2, 0.1),
    )
    updated = update_constraint_multipliers(
        state,
        np.asarray([0.15, 0.0, 30.0]),
        learning_rate=0.5,
        np=np,
    )
    assert updated.multipliers == pytest.approx((0.05, 0.195, 0.0))


def test_resolve_rejects_out_of_range_values() -> None:
    with pytest.raises(OursContractError, match="probabilities"):
        counterfactual_rescue_bottleneck(
            np.asarray([1.1]), np.asarray([0.0]), np.asarray([0.0]), np=np
        )
