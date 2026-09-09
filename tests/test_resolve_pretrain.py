import numpy as np
import pytest

from unitree_gr00t.ours import OursContractError
from unitree_gr00t.resolve_pretrain import (
    bounded_model_actions,
    causal_history_indices,
    expert_action_chunks,
)


def test_bounded_model_actions_uses_distinct_gripper_domain() -> None:
    value = np.asarray([[2.0, -2.0, -1.0], [-0.5, 0.5, 2.0]])
    bounded = bounded_model_actions(value, gripper_index=2, np=np)
    assert bounded.tolist() == [[1.0, -1.0, 0.0], [-0.5, 0.5, 1.0]]


def test_expert_action_chunks_pad_only_after_episode_end() -> None:
    actions = np.arange(15, dtype=np.float32).reshape(5, 3)
    chunks = expert_action_chunks(actions, np.asarray([1, 4]), horizon=3, gripper_index=None, np=np)
    expected = np.clip(actions, -1.0, 1.0)
    assert np.array_equal(chunks[0], expected[[1, 2, 3]])
    assert np.array_equal(chunks[1], expected[[4, 4, 4]])


def test_causal_history_never_crosses_subgoal_or_episode() -> None:
    history = causal_history_indices(
        np.asarray([0, 0, 0, 0, 1]),
        np.asarray([0, 0, 1, 1, 0]),
        np.asarray([0, 8, 16, 24, 0]),
        history_length=3,
        np=np,
    )
    assert history.tolist() == [
        [0, 0, 0],
        [0, 0, 1],
        [2, 2, 2],
        [2, 2, 3],
        [4, 4, 4],
    ]


def test_decreasing_frame_order_is_rejected() -> None:
    with pytest.raises(OursContractError, match="not causal"):
        causal_history_indices(
            np.asarray([0, 0]),
            np.asarray([1, 1]),
            np.asarray([8, 0]),
            history_length=2,
            np=np,
        )
