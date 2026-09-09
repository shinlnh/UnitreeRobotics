import numpy as np
import pytest

from unitree_gr00t.ours import OursContractError
from unitree_gr00t.resolve import (
    ConstraintState,
    conservative_ensemble_bottleneck,
    counterfactual_rescue_bottleneck,
    finite_horizon_rdb_fixed_point,
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
    bottleneck = counterfactual_rescue_bottleneck(
        recovery, deletion, baseline, np=np
    )
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
    assert counterfactual_rescue_bottleneck(
        values.recovery[0, 0],
        values.deletion[0, 0],
        values.baseline[0, 0],
        np=np,
    ).item() == 1.0


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
