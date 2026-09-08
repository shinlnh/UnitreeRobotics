import numpy as np
import pytest

from unitree_gr00t.ours import OursContractError
from unitree_gr00t.ours_counterfactual_rollout_prepare import (
    _apply_residual_stop_confirmation,
    _apply_stop_confirmation,
    _best_nonstop,
    _fresh_consensus_count,
    _medoid_index,
    _selected_rollout_positions,
    _valid_options,
    _validate_residual_source,
)


def test_counterfactual_stop_confirmation_continues_into_next_subgoal() -> None:
    assert _apply_stop_confirmation(
        target_subgoal=1,
        subgoal_count=3,
        stop_streak=0,
    ) == (1, 1, False)
    assert _apply_stop_confirmation(
        target_subgoal=1,
        subgoal_count=3,
        stop_streak=1,
    ) == (2, 0, True)
    assert _apply_stop_confirmation(
        target_subgoal=2,
        subgoal_count=3,
        stop_streak=1,
    ) == (3, 0, True)


def test_residual_counterfactual_continues_with_b_retry_per_subtask() -> None:
    assert _apply_residual_stop_confirmation(
        target_subgoal=1,
        subgoal_count=3,
        stop_streak=0,
        retry_attempt_index=0,
    ) == (1, 1, 0, False, False)
    assert _apply_residual_stop_confirmation(
        target_subgoal=1,
        subgoal_count=3,
        stop_streak=1,
        retry_attempt_index=0,
    ) == (1, 0, 1, True, False)
    assert _apply_residual_stop_confirmation(
        target_subgoal=1,
        subgoal_count=3,
        stop_streak=1,
        retry_attempt_index=1,
    ) == (2, 0, 0, False, True)


def test_rollout_options_respect_backtrack_boundary() -> None:
    first = {value.value for value in _valid_options(0, 3)}
    later = {value.value for value in _valid_options(1, 3)}
    pending = {value.value for value in _valid_options(1, 3, stop_pending=True)}
    assert "BACKTRACK_ONE" not in first
    assert "BACKTRACK_ONE" in later
    assert "ADVANCE" not in later
    assert "ADVANCE" in pending
    residual = {
        value.value
        for value in _valid_options(
            1,
            3,
            stop_pending=True,
            residual_retry_baseline=True,
        )
    }
    assert "ACCEPT_B" not in residual
    assert "RETRY_CURRENT" in residual
    assert "ADVANCE" in residual
    with pytest.raises(OursContractError, match="active subgoal"):
        _valid_options(3, 3)


def test_rollout_consensus_and_nonstop_selection_are_deterministic() -> None:
    chunks = [
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[1.0, 0.1]], dtype=np.float32),
        np.asarray([[-1.0, 0.0]], dtype=np.float32),
    ]
    assert _medoid_index(chunks, np) == 1
    scores = np.asarray([5.0, 1.0, 4.0], dtype=np.float32)
    valid = np.asarray([True, True, True])
    assert _best_nonstop(scores, valid, np) == 2
    assert _fresh_consensus_count(4) == 3


def test_rollout_sampling_uses_subgoal_relative_failure_onset() -> None:
    rows = [
        {
            "step_before": step,
            "new_subgoal_anchor": [0.0] if step in {0, 100} else None,
            "selector_candidate_before_recovery": 0,
        }
        for step in (0, 50, 75, 100, 174, 175)
    ]
    assert _selected_rollout_positions(
        rows,
        stride=1,
        maximum=8,
        min_elapsed_steps=75,
    ) == {2, 5}


def test_counterfactual_selection_can_focus_on_confirmed_stops() -> None:
    rows = [
        {
            "selector_candidate_before_recovery": 0,
            "step_before": 75,
            "new_subgoal_anchor": [0.0] if index == 0 else None,
            "active_subgoal_index_before": 0,
            "active_subgoal_index_after": 0,
            "stop_confirmation_streak_after": 1 if index in {0, 2} else 0,
        }
        for index in range(4)
    ]

    assert _selected_rollout_positions(
        rows,
        stride=1,
        maximum=8,
        min_elapsed_steps=0,
        require_stop_pending=True,
    ) == {1, 3}


def test_residual_source_requires_exact_abstaining_b_retry() -> None:
    manifest = {
        "base_seed": 10007,
        "decision_schedule": "counterfactual-residual-over-b-retry-v1",
        "residual_retry_baseline": True,
        "retry": True,
        "retry_contract": {
            "trigger": "first-confirmed-stop-per-subtask",
            "max_retries_per_subtask": 1,
            "unconditional": True,
            "preserve_global_step_budget": True,
        },
        "capture_training_context": True,
        "collection_force_boundary_steps": None,
        "stagnation_boundary_steps": None,
        "option_value_margin": 1_000_000.0,
    }
    summary = {"total_recovery_triggers": 0}

    _validate_residual_source(manifest, summary)
    for key, value in (
        ("capture_training_context", False),
        ("collection_force_boundary_steps", 150),
        ("option_value_margin", 999_999.0),
    ):
        invalid = dict(manifest)
        invalid[key] = value
        with pytest.raises(OursContractError, match="exact abstaining B-retry"):
            _validate_residual_source(invalid, summary)
    with pytest.raises(OursContractError, match="exact abstaining B-retry"):
        _validate_residual_source(manifest, {"total_recovery_triggers": 1})
