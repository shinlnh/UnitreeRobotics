import numpy as np
import pytest

from unitree_gr00t.a0 import ACTION_KEYS
from unitree_gr00t.ours import OursContractError
from unitree_gr00t.resolve_runtime import replace_action_chunk


def test_replace_action_chunk_preserves_wire_shapes_and_metadata() -> None:
    action = {
        **{f"action.{key}": np.zeros((1, 3, 1), dtype=np.float32) for key in ACTION_KEYS},
        "metadata": "kept",
    }
    corrected = np.arange(21, dtype=np.float32).reshape(1, 3, 7)
    replaced = replace_action_chunk(action, corrected, np=np)
    assert replaced["metadata"] == "kept"
    assert np.array_equal(
        np.concatenate([replaced[f"action.{key}"] for key in ACTION_KEYS], axis=2),
        corrected,
    )
    assert all(np.count_nonzero(action[f"action.{key}"]) == 0 for key in ACTION_KEYS)


def test_replace_action_chunk_rejects_incompatible_base() -> None:
    with pytest.raises(OursContractError, match="incompatible"):
        replace_action_chunk(
            {f"action.{key}": np.zeros((1, 2, 1)) for key in ACTION_KEYS},
            np.zeros((1, 3, 7)),
            np=np,
        )
