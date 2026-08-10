from __future__ import annotations

import numpy as np
import pytest
from unitree_rl_groot.groot.command_filter import ActionChunk, VelocityCommandFilter, VelocityLimits


def test_velocity_filter_enforces_slew_limits_and_clamp() -> None:
    limits = VelocityLimits(
        minimum=np.array([-1.0, -1.0, -1.0]),
        maximum=np.array([1.0, 1.0, 1.0]),
        max_acceleration=np.array([0.5, 1.0, 2.0]),
        deadband=np.zeros(3),
    )
    command_filter = VelocityCommandFilter(limits)
    actual = command_filter.step(np.array([5.0, -5.0, 5.0]), dt=0.1)
    np.testing.assert_allclose(actual, [0.05, -0.1, 0.2], atol=1e-7)


def test_velocity_filter_deadband_and_rejects_nan() -> None:
    command_filter = VelocityCommandFilter()
    np.testing.assert_array_equal(command_filter.step(np.array([0.01, -0.01, 0.01]), 0.02), 0.0)
    with pytest.raises(ValueError, match="NaN"):
        command_filter.step(np.array([np.nan, 0.0, 0.0]), 0.02)


def test_action_chunk_batch_and_execution_horizon() -> None:
    source = np.arange(30, dtype=np.float32).reshape(1, 10, 3)
    chunk = ActionChunk.from_policy_action({"navigate_command": source}, execution_horizon=4)
    assert len(chunk) == 4
    np.testing.assert_array_equal(chunk.pop(), source[0, 0])
    assert len(chunk) == 3


def test_action_chunk_rejects_wrong_contract() -> None:
    with pytest.raises(KeyError):
        ActionChunk.from_policy_action({})
    with pytest.raises(ValueError, match="shape"):
        ActionChunk(np.zeros((5, 4), dtype=np.float32))
