import numpy as np
import pytest

from unitree_gr00t.ours import OursContractError
from unitree_gr00t.ours_counterfactual_rollout_prepare import (
    _apply_stop_confirmation,
    _best_nonstop,
    _medoid_index,
    _selected_rollout_positions,
    _valid_options,
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


def test_rollout_options_respect_backtrack_boundary() -> None:
    assert "BACKTRACK_ONE" not in {value.value for value in _valid_options(0, 3)}
    assert "BACKTRACK_ONE" in {value.value for value in _valid_options(1, 3)}
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
