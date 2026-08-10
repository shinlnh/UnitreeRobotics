from __future__ import annotations

import numpy as np
from unitree_rl_groot.groot.navigation import (
    CircularObstacle,
    NavigationLayout,
    NavigationLayoutSampler,
    WaypointFollower,
    plan_path,
    yaw_from_quat_xyzw,
)


def test_astar_routes_around_inflated_obstacle() -> None:
    layout = NavigationLayout(
        start_xy=np.array([0.0, 0.0]),
        start_yaw=0.0,
        goal_xy=np.array([4.0, 0.0]),
        obstacles=(CircularObstacle(np.array([2.0, 0.0]), 0.5),),
        instruction="reach the green goal",
    )
    path = plan_path(layout, resolution=0.1, robot_clearance=0.4)
    assert path.shape[1] == 2
    np.testing.assert_allclose(path[[0, -1]], [[0.0, 0.0], [4.0, 0.0]], atol=1e-5)
    segment_midpoints = (path[:-1] + path[1:]) / 2.0
    assert np.min(np.linalg.norm(segment_midpoints - np.array([2.0, 0.0]), axis=1)) > 0.7


def test_sampler_is_seeded_and_solvable() -> None:
    first = NavigationLayoutSampler(seed=7).sample(np.zeros(2), 0.0)
    second = NavigationLayoutSampler(seed=7).sample(np.zeros(2), 0.0)
    np.testing.assert_allclose(first.goal_xy, second.goal_xy)
    np.testing.assert_allclose(first.obstacle_array(), second.obstacle_array())
    assert len(plan_path(first)) >= 2


def test_waypoint_follower_turns_then_stops_at_goal() -> None:
    follower = WaypointFollower(np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    turning = follower.command(np.array([0.0, 0.0]), yaw=0.0)
    assert turning[0] == 0.0
    assert turning[2] > 0.0
    np.testing.assert_array_equal(follower.command(np.array([0.0, 2.0]), yaw=1.57), np.zeros(3))


def test_xyzw_quaternion_yaw() -> None:
    half = np.sqrt(0.5)
    assert np.isclose(yaw_from_quat_xyzw(np.array([0.0, 0.0, half, half])), np.pi / 2.0)
