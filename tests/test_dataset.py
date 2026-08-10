from __future__ import annotations

import numpy as np
import pytest
from unitree_rl_groot.groot.dataset import RawNavigationEpisode, numeric_stats


def _episode(frame_count: int = 5) -> RawNavigationEpisode:
    return RawNavigationEpisode(
        rgb=np.zeros((frame_count, 8, 8, 3), dtype=np.uint8),
        state=np.arange(frame_count * 4, dtype=np.float32).reshape(frame_count, 4),
        action=np.zeros((frame_count, 3), dtype=np.float32),
        timestamp=np.arange(frame_count, dtype=np.float32) / 10.0,
        language="walk forward",
        goal_xy=np.array([4.0, 1.0], dtype=np.float32),
        obstacles=np.array([[2.0, 0.0, 0.4]], dtype=np.float32),
        success=True,
    )


def test_raw_episode_round_trip_without_pickle(tmp_path) -> None:
    expected = _episode()
    path = expected.save(tmp_path / "episode_000000.npz")
    actual = RawNavigationEpisode.load(path)
    np.testing.assert_array_equal(actual.rgb, expected.rgb)
    np.testing.assert_array_equal(actual.state, expected.state)
    np.testing.assert_array_equal(actual.goal_xy, expected.goal_xy)
    np.testing.assert_array_equal(actual.obstacles, expected.obstacles)
    assert actual.language == expected.language
    assert actual.success is True


def test_episode_contract_rejects_unsynchronized_data() -> None:
    with pytest.raises(ValueError, match="action"):
        RawNavigationEpisode(
            rgb=np.zeros((5, 8, 8, 3), dtype=np.uint8),
            state=np.zeros((5, 4), dtype=np.float32),
            action=np.zeros((4, 3), dtype=np.float32),
            timestamp=np.arange(5, dtype=np.float32),
            language="walk",
        )


def test_numeric_stats_are_per_feature() -> None:
    stats = numeric_stats(np.array([[1.0, 10.0], [3.0, 14.0]]))
    assert stats["mean"] == [2.0, 12.0]
    assert stats["min"] == [1.0, 10.0]
    assert stats["max"] == [3.0, 14.0]
