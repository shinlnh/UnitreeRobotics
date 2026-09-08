from dataclasses import replace
from pathlib import Path

import pytest

from unitree_gr00t.config import load_config
from unitree_gr00t.ours import (
    OursContractError,
    apply_recovery_transition,
    counterfactual_branch_seed,
    prohibited_runtime_paths,
    recovery_decision_seed,
    recovery_option_mask,
    validate_recovery_config,
    validate_runtime_payload,
)

ROOT = Path(__file__).resolve().parents[1]


def test_recovery_seeds_are_stable_and_counterfactual_seeds_share_noise() -> None:
    assert recovery_decision_seed(7, 4, 0) == recovery_decision_seed(7, 4, 0)
    assert recovery_decision_seed(7, 4, 0) != recovery_decision_seed(7, 4, 1)
    assert recovery_decision_seed(7, 4, 0) != recovery_decision_seed(7, 5, 0)
    assert counterfactual_branch_seed(10007, 3) == counterfactual_branch_seed(10007, 3)
    assert counterfactual_branch_seed(10007, 3) != counterfactual_branch_seed(10007, 4)
    assert counterfactual_branch_seed(10007, 3, 0) != counterfactual_branch_seed(
        10007, 3, 1
    )


def test_runtime_payload_audit_rejects_nested_privileged_signals() -> None:
    allowed = {
        "context": [0.0],
        "history": [{"selector_scores": [1.0, 0.0], "executed_action": [0.0] * 7}],
        "subgoal_index": 2,
        "remaining_budget_fraction": 0.5,
    }
    validate_runtime_payload(allowed)
    assert prohibited_runtime_paths(allowed) == ()

    forbidden = {
        **allowed,
        "metadata": {
            "success_predicates_after": {"Pick": True},
            "injection_type": "drop",
        },
    }
    with pytest.raises(OursContractError, match="metadata.injection_type"):
        validate_runtime_payload(forbidden)

    for forbidden_key in ("injections", "goal_steps", "task_goal"):
        with pytest.raises(OursContractError):
            validate_runtime_payload({forbidden_key: []})


def test_option_mask_is_boundary_and_attempt_aware() -> None:
    mask = recovery_option_mask(
        subgoal_index=0,
        subgoal_count=4,
        stop_pending=False,
        stop_committed=True,
        recovery_triggered=True,
        recovery_attempts=0,
        max_recovery_attempts=2,
        zero_progress_decisions=0,
        zero_progress_guard=8,
        remaining_steps=100,
        consensus_hypotheses=4,
    ).as_mapping()
    assert not mask["ACCEPT_B"]
    assert mask["ADVANCE"]
    assert mask["RETRY_CURRENT"]
    assert not mask["BACKTRACK_ONE"]
    assert mask["CONSENSUS_PREFIX"]

    exhausted = recovery_option_mask(
        subgoal_index=2,
        subgoal_count=4,
        stop_pending=True,
        stop_committed=False,
        recovery_triggered=True,
        recovery_attempts=2,
        max_recovery_attempts=2,
        zero_progress_decisions=8,
        zero_progress_guard=8,
        remaining_steps=100,
        consensus_hypotheses=8,
    ).as_mapping()
    assert exhausted["ACCEPT_B"]
    assert not exhausted["REOBSERVE"]
    assert not exhausted["RETRY_CURRENT"]
    assert not exhausted["BACKTRACK_ONE"]
    assert not exhausted["CONSENSUS_PREFIX"]


def test_recovery_transition_never_restores_or_consumes_a_hidden_step() -> None:
    retry = apply_recovery_transition(
        "RETRY_CURRENT",
        subgoal_index=2,
        subgoal_count=5,
        attempt_index=0,
        max_recovery_attempts=2,
    )
    assert retry.subgoal_index_after == 2
    assert retry.attempt_index_after == 1
    assert retry.reanchor
    assert not retry.consumes_simulator_step

    backtrack = apply_recovery_transition(
        "BACKTRACK_ONE",
        subgoal_index=2,
        subgoal_count=5,
        attempt_index=1,
        max_recovery_attempts=2,
    )
    assert backtrack.subgoal_index_after == 1
    assert backtrack.attempt_index_after == 2
    assert backtrack.reanchor
    with pytest.raises(OursContractError, match="before the first"):
        apply_recovery_transition(
            "BACKTRACK_ONE",
            subgoal_index=0,
            subgoal_count=5,
            attempt_index=0,
            max_recovery_attempts=2,
        )


def test_recovery_config_freezes_identity_search_space_and_seed_partitions() -> None:
    config = load_config(ROOT / "configs" / "project.toml").robocerebra_recovery
    audit = validate_recovery_config(config)
    assert audit["valid"]
    assert not audit["runtime"]["observes_goal_predicates"]
    assert audit["runtime"]["continuous_live_state"]
    assert audit["search"]["development_base_seeds"] == (
        20007,
        21007,
        22007,
        23007,
    )

    with pytest.raises(OursContractError, match="seed partitions"):
        validate_recovery_config(replace(config, final_base_seed=20007))
    with pytest.raises(OursContractError, match="option order"):
        validate_recovery_config(replace(config, options=tuple(reversed(config.options))))
