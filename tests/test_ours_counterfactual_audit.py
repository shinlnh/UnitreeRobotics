import numpy as np

from unitree_gr00t.ours_counterfactual_audit import (
    residual_contract_checks,
    summarize_option_targets,
    summarize_residual_branch_mechanisms,
)


def test_counterfactual_audit_detects_diverse_strict_preferences() -> None:
    values = np.zeros((3, 6), dtype=np.float32)
    valid = np.zeros((3, 6), dtype=np.bool_)
    valid[:, [0, 4, 5]] = True
    values[0, [0, 4, 5]] = [2.0, 1.0, 0.0]
    values[1, [0, 4, 5]] = [0.0, 3.0, 1.0]
    values[2, [0, 4, 5]] = [1.0, 1.0, 1.0]

    audit = summarize_option_targets(values, valid, np=np)

    assert audit["labeled_states"] == 3
    assert audit["strict_preferences"] == 2
    assert audit["ties"] == 1
    assert audit["winning_options"] == {"ACCEPT_B": 1, "ADVANCE": 1}
    assert audit["distinct_winning_options"] == 2


def test_counterfactual_audit_rejects_single_valid_option_rows() -> None:
    values = np.zeros((2, 6), dtype=np.float32)
    valid = np.zeros((2, 6), dtype=np.bool_)
    valid[:, 5] = True

    audit = summarize_option_targets(values, valid, np=np)

    assert audit["labeled_states"] == 0
    assert audit["distinct_winning_options"] == 0


def test_residual_audit_requires_confirmed_b_retry_branch_contract() -> None:
    branches = [
        {
            "episode_index": 0,
            "sample_index": 4,
            "option": option,
            "source_stop_pending": True,
            "branch_seed": 1234,
        }
        for option in ("REOBSERVE", "RETRY_CURRENT", "ADVANCE", "CONSENSUS_PREFIX")
    ]
    manifest = {
        "counterfactual_branches_sha256": "branches",
        "counterfactual_sampling": {
            "residual_retry_baseline": True,
            "source_behavior_contract": {
                "decision_schedule": "counterfactual-residual-over-b-retry-v1",
                "residual_retry_baseline": True,
                "option_value_margin": 1_000_000.0,
                "recovery_triggers": 0,
                "capture_training_context": True,
                "collection_force_boundary_steps": None,
                "stagnation_boundary_steps": None,
            },
            "require_stop_pending": True,
            "continuation_policy": "B-retry-confirmed-stop-one-retry-per-subtask",
            "consensus_source_proposal_included": True,
            "randomness_coupling": "common-random-numbers-per-state-v1",
            "rollouts_per_option": 1,
            "state_count": 1,
            "branch_count": 4,
            "options": {
                "ACCEPT_B": 0,
                "REOBSERVE": 1,
                "RETRY_CURRENT": 1,
                "BACKTRACK_ONE": 0,
                "ADVANCE": 1,
                "CONSENSUS_PREFIX": 1,
            },
        },
    }

    checks = residual_contract_checks(
        manifest,
        branches,
        branches_sha256="branches",
    )

    assert checks
    assert all(checks.values())
    branches[0]["source_stop_pending"] = False
    assert not residual_contract_checks(
        manifest,
        branches,
        branches_sha256="branches",
    )["residual_confirmed_stop_sources"]
    branches[0]["source_stop_pending"] = True
    branches[0]["branch_seed"] = 4321
    assert not residual_contract_checks(
        manifest,
        branches,
        branches_sha256="branches",
    )["residual_common_random_numbers"]
    manifest["counterfactual_sampling"]["source_behavior_contract"][
        "recovery_triggers"
    ] = 1
    assert not residual_contract_checks(
        manifest,
        branches,
        branches_sha256="branches",
    )["residual_source_is_abstaining_b_retry"]
    manifest["counterfactual_sampling"]["return_target"] = "unknown"
    assert not residual_contract_checks(
        manifest,
        branches,
        branches_sha256="branches",
    )["residual_return_target"]


def test_residual_mechanism_audit_separates_physical_and_efficiency_gains() -> None:
    branches = []
    for state, returns, predicates in (
        (1, {"RETRY_CURRENT": 0.0, "REOBSERVE": 0.1}, {"RETRY_CURRENT": 0, "REOBSERVE": 0}),
        (2, {"RETRY_CURRENT": 0.0, "ADVANCE": 1.0}, {"RETRY_CURRENT": 0, "ADVANCE": 1}),
    ):
        for option, value in returns.items():
            branches.append(
                {
                    "episode_index": 0,
                    "sample_index": state,
                    "option": option,
                    "return_value": value,
                    "final_success": False,
                    "completed_subtasks_after": 0,
                    "predicate_count_after": predicates[option],
                }
            )

    result = summarize_residual_branch_mechanisms(branches)

    assert result["states"] == 2
    assert result["return_beneficial_override_states"] == 2
    assert result["physical_beneficial_override_states"] == 1
    assert result["efficiency_only_return_override_states"] == 1
    assert result["return_override_with_physical_regression_states"] == 0
    assert result["physical_winning_options"] == {"ADVANCE": 1}
