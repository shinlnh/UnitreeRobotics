from __future__ import annotations

import numpy as np
import pytest
from unitree_rl_groot.groot.observations import build_navigation_observation, build_proprio


def test_proprio_order_and_navigation_shapes() -> None:
    proprio = build_proprio(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([5.0, 6.0, 7.0]),
        np.array([8.0, 9.0, 10.0]),
        np.array([11.0, 12.0, 13.0]),
    )
    np.testing.assert_array_equal(proprio, np.arange(1, 14, dtype=np.float32))
    observation = build_navigation_observation(np.zeros((8, 12, 4), dtype=np.uint8), proprio, "go left")
    assert observation["video"]["ego_view"].shape == (1, 1, 8, 12, 3)
    assert observation["state"]["proprio"].shape == (1, 1, 13)
    assert observation["language"]["annotation.human.task_description"] == [["go left"]]


def test_float_rgb_is_converted_to_uint8() -> None:
    observation = build_navigation_observation(np.full((2, 2, 3), 0.5, dtype=np.float32), np.ones(3), "stand")
    image = observation["video"]["ego_view"]
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image, 128)


def test_invalid_proprio_is_rejected() -> None:
    with pytest.raises(ValueError, match="same shape"):
        build_proprio(np.ones(2), np.ones(3), np.ones(3), np.ones(3), np.ones(3))
