import numpy as np
import pytest

from unitree_gr00t.ours import OursContractError
from unitree_gr00t.ours_counterfactual_rollout_prepare import (
    _best_nonstop,
    _medoid_index,
    _valid_options,
)


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
