import numpy as np
import pytest

from unitree_gr00t.branchq import (
    branchq_hypothesis_seed,
    build_branch_lattice,
    calibrate_grouped_conformal_scale,
    conservative_branch_selection,
    macro_discount,
    physical_macro_reward,
    semi_markov_double_q_target,
)
from unitree_gr00t.ours import OursContractError


def test_branchq_seed_is_stable_and_factor_specific() -> None:
    seed = branchq_hypothesis_seed(10007, 3, 0, 1)
    assert seed == branchq_hypothesis_seed(10007, 3, 0, 1)
    assert seed != branchq_hypothesis_seed(10007, 3, 1, 1)
    assert seed != branchq_hypothesis_seed(10007, 3, 0, 2)


def test_lattice_unifies_stop_subgoals_hypotheses_and_prefixes() -> None:
    lattice = build_branch_lattice(
        subgoal_index=1,
        subgoal_count=3,
        hypothesis_count=2,
        prefix_lengths=(1, 4),
        remaining_steps=16,
        baseline_candidate=3,
    )
    # STOP + 3 deltas * 2 hypotheses * 2 registered prefixes + exact B prefix.
    assert len(lattice) == 14
    assert lattice[0].stop
    baseline = [candidate for candidate in lattice if candidate.baseline]
    assert len(baseline) == 1
    assert baseline[0].candidate_id == "dgz0-h00-l03"
    assert {candidate.delta_subgoal for candidate in lattice if not candidate.stop} == {
        -1,
        0,
        1,
    }


def test_lattice_masks_invalid_subgoal_transitions_and_preserves_b_stop() -> None:
    lattice = build_branch_lattice(
        subgoal_index=0,
        subgoal_count=2,
        hypothesis_count=1,
        remaining_steps=3,
        baseline_candidate=0,
    )
    assert sum(candidate.baseline for candidate in lattice) == 1
    assert lattice[0].baseline and lattice[0].stop
    assert all(candidate.delta_subgoal >= 0 for candidate in lattice if not candidate.stop)
    assert all(candidate.prefix_length <= 3 for candidate in lattice)


def test_physical_reward_has_no_step_or_call_shortcut() -> None:
    discount = macro_discount(0.99, 8)
    reward = physical_macro_reward(
        completed_subtasks_before=1,
        completed_subtasks_after=2,
        subgoal_count=4,
        predicate_count_before=2,
        predicate_count_after=3,
        predicate_count=8,
        final_success=False,
        continuation_discount=discount,
    )
    expected = discount * (0.25 * 2 / 4 + 0.05 * 3 / 8) - (
        0.25 * 1 / 4 + 0.05 * 2 / 8
    )
    assert np.isclose(reward, expected)


def test_semi_markov_double_q_selects_online_and_evaluates_target() -> None:
    target = semi_markov_double_q_target(
        reward=0.2,
        gamma=0.9,
        duration_steps=2,
        terminal=False,
        next_online_values=np.asarray([1.0, 3.0, 2.0]),
        next_target_values=np.asarray([10.0, 4.0, 8.0]),
        next_valid=np.asarray([True, True, False]),
        np=np,
    )
    assert np.isclose(target, 0.2 + 0.9**2 * 4.0)


def test_grouped_conformal_uses_worst_residual_per_episode() -> None:
    predictions = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    calibration = calibrate_grouped_conformal_scale(
        predictions,
        np.asarray([0.0, 0.2, 1.0, 1.1], dtype=np.float32),
        ["episode-a", "episode-a", "episode-b", "episode-b"],
        alpha=0.5,
        standard_deviation_floor=0.1,
        np=np,
    )
    assert calibration.groups == 2
    assert calibration.samples == 4
    assert np.isclose(calibration.quantile, 2.0)


def test_conservative_selection_overrides_only_on_separated_bounds() -> None:
    lattice = build_branch_lattice(
        subgoal_index=0,
        subgoal_count=2,
        hypothesis_count=1,
        prefix_lengths=(1,),
        remaining_steps=16,
        baseline_candidate=1,
    )
    baseline = next(index for index, candidate in enumerate(lattice) if candidate.baseline)
    alternative = next(
        index
        for index, candidate in enumerate(lattice)
        if candidate.delta_subgoal == 1 and not candidate.stop
    )
    values = np.zeros((3, len(lattice)), dtype=np.float32)
    values[:, baseline] = [0.0, 0.1, -0.1]
    values[:, alternative] = [1.0, 1.1, 0.9]
    decision = conservative_branch_selection(
        lattice,
        values,
        conformal_scale=1.0,
        standard_deviation_floor=0.01,
        support_counts=[10] * len(lattice),
        minimum_support=5,
        np=np,
    )
    assert not decision.fallback
    assert decision.selected_index == alternative

    uncertain = values.copy()
    uncertain[:, alternative] = [-1.0, 1.0, 3.0]
    fallback = conservative_branch_selection(
        lattice,
        uncertain,
        conformal_scale=1.0,
        standard_deviation_floor=0.01,
        support_counts=[10] * len(lattice),
        minimum_support=5,
        np=np,
    )
    assert fallback.fallback
    assert fallback.reason == "baseline-confidence-fallback"


def test_conservative_selection_falls_back_outside_support() -> None:
    lattice = build_branch_lattice(
        subgoal_index=0,
        subgoal_count=1,
        hypothesis_count=1,
        prefix_lengths=(1,),
        remaining_steps=16,
        baseline_candidate=1,
    )
    values = np.zeros((2, len(lattice)), dtype=np.float32)
    values[:, 0] = 100.0
    decision = conservative_branch_selection(
        lattice,
        values,
        conformal_scale=0.0,
        standard_deviation_floor=0.01,
        support_counts=[0, 10],
        minimum_support=5,
        np=np,
    )
    assert decision.fallback
    assert decision.reason == "baseline-no-supported-alternative"


def test_branchq_rejects_duplicate_prefix_grid() -> None:
    with pytest.raises(OursContractError, match="prefix grid"):
        build_branch_lattice(
            subgoal_index=0,
            subgoal_count=1,
            hypothesis_count=1,
            prefix_lengths=(1, 1),
            remaining_steps=16,
            baseline_candidate=1,
        )
