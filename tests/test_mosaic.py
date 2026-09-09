import numpy as np
import pytest

from unitree_gr00t.mosaic import (
    FixedAnchorArm,
    deletion_minimal_certificate,
    deletion_necessity_odds_update,
    deletion_necessity_weights,
    paired_deletion_effects,
    program_arms,
    soft_causal_bottleneck,
    validate_fixed_anchor_arms,
)
from unitree_gr00t.ours import OursContractError


def effects_with_redundant_second_rule():
    return paired_deletion_effects(
        group_ids=("g0", "g1", "g2", "g3"),
        milestone_names=("subtask-1",),
        baseline=np.zeros((4, 1)),
        full_program=np.ones((4, 1)),
        one_deletions=np.asarray(
            [
                [[0.0], [1.0]],
                [[0.0], [1.0]],
                [[0.0], [1.0]],
                [[0.0], [1.0]],
            ]
        ),
        np=np,
    )


def fixed_arm(arm_id: str, **overrides) -> FixedAnchorArm:
    values = {
        "arm_id": arm_id,
        "snapshot_hash": "snapshot-hash",
        "environment_rng_hash": "env-rng-hash",
        "disturbance_rng_hash": "disturbance-rng-hash",
        "disturbance_cursor": 7,
        "disturbance_toggle": False,
        "start_step": 16,
        "anchor_steps": (16, 24),
        "outcome_step": 64,
        "global_step_budget": 48,
    }
    values.update(overrides)
    return FixedAnchorArm(**values)


def test_program_certificate_uses_exactly_d_plus_two_arms() -> None:
    arms = program_arms("program-7", depth=3)
    assert len(arms) == 5
    assert [arm.kind for arm in arms] == [
        "baseline",
        "full",
        "deletion",
        "deletion",
        "deletion",
    ]
    assert [arm.deleted_rule_index for arm in arms[2:]] == [0, 1, 2]


def test_paired_effects_separate_sufficiency_from_each_rule_necessity() -> None:
    effects = effects_with_redundant_second_rule()
    assert np.array_equal(np.asarray(effects.baseline_gain), np.ones((4, 1)))
    necessity = np.asarray(effects.necessity_gain)
    assert np.array_equal(necessity[:, 0, :], np.ones((4, 1)))
    assert np.array_equal(necessity[:, 1, :], np.zeros((4, 1)))


def test_deletion_necessity_does_not_credit_a_redundant_rule() -> None:
    advantages = deletion_necessity_weights(effects_with_redundant_second_rule(), np=np)
    assert np.all(advantages[:, 0, 0] == 1.0)
    assert np.all(advantages[:, 1, 0] == 0.0)

    # Smooth bottlenecks remain available for learned-effect diagnostics, but
    # DNOU uses the exact hard causal conjunction above.
    assert float(soft_causal_bottleneck(0.0, 1.0, temperature=0.1, np=np)) < 0.0


def test_odds_update_moves_only_with_signed_deletion_necessity() -> None:
    updated = deletion_necessity_odds_update(
        np.asarray([0.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, -1.0]),
        step_size=0.25,
        np=np,
    )
    assert np.array_equal(updated, np.asarray([0.25, 0.0, -0.25]))


def test_certificate_requires_baseline_gain_and_every_one_deletion_gain() -> None:
    redundant = deletion_minimal_certificate(
        effects_with_redundant_second_rule(),
        bootstrap_samples=500,
        seed=11,
        np=np,
    )
    assert redundant.baseline_gain_lower == (1.0,)
    assert redundant.necessity_gain_lower == ((1.0,), (0.0,))
    assert redundant.bottleneck_lower == (0.0,)
    assert redundant.certified_milestones == ()

    necessary = paired_deletion_effects(
        group_ids=tuple(f"g{index}" for index in range(6)),
        milestone_names=("subtask-1",),
        baseline=np.zeros((6, 1)),
        full_program=np.ones((6, 1)),
        one_deletions=np.zeros((6, 2, 1)),
        np=np,
    )
    certified = deletion_minimal_certificate(
        necessary,
        bootstrap_samples=500,
        seed=11,
        np=np,
    )
    assert certified.bottleneck_lower == (1.0,)
    assert certified.certified_milestones == ("subtask-1",)


def test_fixed_anchor_contract_includes_complete_exogenous_tape() -> None:
    arms = tuple(fixed_arm(item.arm_id) for item in program_arms("p1", depth=2))
    validate_fixed_anchor_arms(arms)

    altered_rng = list(arms)
    altered_rng[-1] = fixed_arm(altered_rng[-1].arm_id, disturbance_rng_hash="other")
    with pytest.raises(OursContractError, match="random tape"):
        validate_fixed_anchor_arms(tuple(altered_rng))

    shifted_anchor = list(arms)
    shifted_anchor[-1] = fixed_arm(shifted_anchor[-1].arm_id, anchor_steps=(16, 25))
    with pytest.raises(OursContractError, match="anchors"):
        validate_fixed_anchor_arms(tuple(shifted_anchor))


def test_invalid_outcome_shapes_are_rejected() -> None:
    with pytest.raises(OursContractError, match="shapes"):
        paired_deletion_effects(
            group_ids=("g0", "g1"),
            milestone_names=("m",),
            baseline=np.zeros((2, 1)),
            full_program=np.ones((2, 1)),
            one_deletions=np.zeros((3, 2, 1)),
            np=np,
        )
