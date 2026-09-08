from argparse import Namespace
from pathlib import Path

import pytest

from unitree_gr00t.b_eval import B_EVALUATION, _run_manifest
from unitree_gr00t.b_retry import (
    B_RETRY_MAX_RETRIES_PER_SUBTASK,
    BRetryContractError,
    confirmed_stop_transition,
    retry_contract_payload,
)
from unitree_gr00t.b_retry_eval import B_RETRY_EVALUATION
from unitree_gr00t.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_naive_retry_first_stop_retries_and_second_advances() -> None:
    first = confirmed_stop_transition(subgoal_index=2, subgoal_count=4, attempt_index=0)
    second = confirmed_stop_transition(
        subgoal_index=first.subgoal_index_after,
        subgoal_count=4,
        attempt_index=first.attempt_index_after,
    )

    assert first.retry_triggered
    assert not first.subgoal_advanced
    assert first.subgoal_index_after == 2
    assert first.attempt_index_after == 1
    assert not second.retry_triggered
    assert second.subgoal_advanced
    assert second.subgoal_index_after == 3


def test_naive_retry_applies_identically_to_final_subtask() -> None:
    retry = confirmed_stop_transition(subgoal_index=3, subgoal_count=4, attempt_index=0)
    finish = confirmed_stop_transition(subgoal_index=3, subgoal_count=4, attempt_index=1)

    assert retry.subgoal_index_after == 3
    assert finish.subgoal_index_after == 4


def test_naive_retry_rejects_count_or_state_drift() -> None:
    with pytest.raises(BRetryContractError, match="frozen to one"):
        confirmed_stop_transition(
            subgoal_index=0,
            subgoal_count=1,
            attempt_index=0,
            max_retries_per_subtask=2,
        )
    with pytest.raises(BRetryContractError, match="attempt index"):
        confirmed_stop_transition(subgoal_index=0, subgoal_count=1, attempt_index=2)


def test_typed_retry_config_matches_frozen_contract() -> None:
    config = load_config(ROOT / "configs" / "project.toml")
    payload = retry_contract_payload(config.robocerebra_retry)

    assert payload["max_retries_per_subtask"] == B_RETRY_MAX_RETRIES_PER_SUBTASK
    assert payload["unconditional"]
    assert payload["learned_parameters"] == 0
    assert not payload["failure_detector"]
    assert not payload["recovery_memory"]
    assert not payload["recovery_policy"]


def test_b_default_evaluation_contract_remains_retry_free() -> None:
    assert not B_EVALUATION.retry
    assert B_EVALUATION.max_retries_per_subtask == 0
    assert B_RETRY_EVALUATION.retry
    assert B_RETRY_EVALUATION.max_retries_per_subtask == 1


def test_b_default_manifest_schema_does_not_gain_retry_fields(tmp_path: Path) -> None:
    args = _manifest_args(tmp_path)
    manifest = _run_manifest(
        args,
        _checkpoint_contract(tmp_path),
        {"experiment_id": "A1", "training_dataset_revision": "model"},
        _selector_audit(tmp_path),
        _selector_provenance(),
        [_case()],
        tmp_path / "run",
    )

    assert manifest["experiment_id"] == "B"
    assert manifest["retry"] is False
    assert "retry_contract" not in manifest
    assert "decision_schedule" not in manifest


def test_retry_manifest_keeps_b_budget_and_prohibits_outcome_inputs(tmp_path: Path) -> None:
    manifest = _run_manifest(
        _manifest_args(tmp_path),
        _checkpoint_contract(tmp_path),
        {"experiment_id": "A1", "training_dataset_revision": "model"},
        _selector_audit(tmp_path),
        _selector_provenance(),
        [_case()],
        tmp_path / "run",
        B_RETRY_EVALUATION,
    )

    assert manifest["experiment_id"] == "B-retry"
    assert manifest["retry"]
    assert not manifest["recovery"]
    assert manifest["retry_contract"] == {
        "trigger": "first-confirmed-stop-per-subtask",
        "max_retries_per_subtask": 1,
        "unconditional": True,
        "preserve_global_step_budget": True,
        "reset_selector_anchor_on_retry": True,
        "observes_goal_predicates": False,
        "observes_injection_labels": False,
        "observes_task_outcomes": False,
    }
    assert manifest["steps_per_subtask"] == 150


def _manifest_args(tmp_path: Path) -> Namespace:
    return Namespace(
        benchmark_dir=tmp_path / "bench",
        robocerebra_source=tmp_path / "source",
        benchmark_revision="benchmark",
        model_revision="model",
        dataset_revision="dataset",
        task_types=["Ideal"],
        trials=1,
        execution_horizon=16,
        stop_confirmation_window=2,
        control_frequency_hz=20,
        steps_per_subtask=150,
        initial_wait_steps=15,
        post_success_steps=80,
        seed=7,
        no_trace_images=True,
    )


def _checkpoint_contract(tmp_path: Path) -> object:
    return type("Contract", (), {"checkpoint_dir": tmp_path / "a1", "action_horizon": 16})()


def _selector_audit(tmp_path: Path) -> object:
    return type(
        "Selector",
        (),
        {
            "checkpoint_dir": tmp_path / "selector",
            "weights_sha256": "selector-weights",
            "provenance_sha256": "selector-provenance",
        },
    )()


def _selector_provenance() -> dict[str, object]:
    return {
        "a1_checkpoint_weight_shards_sha256": {"model.safetensors": "a1"},
        "training": {"selected_step": 6000},
    }


def _case() -> object:
    return type("Case", (), {"task_type": "Ideal", "case_name": "case1"})()
