"""Framework-neutral navigation layouts, A* planning, and expert control."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np


def _xy(value: np.ndarray | tuple[float, float], *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (2,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain two finite values")
    return result


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class CircularObstacle:
    """A conservative 2-D collision proxy for one navigation obstacle."""

    center: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        center = _xy(self.center, name="obstacle center").copy()
        if not math.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("obstacle radius must be positive and finite")
        center.setflags(write=False)
        object.__setattr__(self, "center", center)


@dataclass(frozen=True)
class NavigationLayout:
    """One randomized goal-reaching scene expressed in world coordinates."""

    start_xy: np.ndarray
    start_yaw: float
    goal_xy: np.ndarray
    obstacles: tuple[CircularObstacle, ...]
    instruction: str

    def __post_init__(self) -> None:
        start = _xy(self.start_xy, name="start_xy").copy()
        goal = _xy(self.goal_xy, name="goal_xy").copy()
        instruction = self.instruction.strip()
        if not math.isfinite(self.start_yaw):
            raise ValueError("start_yaw must be finite")
        if np.linalg.norm(goal - start) < 1.0:
            raise ValueError("navigation goal must be at least one metre from the start")
        if not instruction:
            raise ValueError("instruction must not be empty")
        start.setflags(write=False)
        goal.setflags(write=False)
        object.__setattr__(self, "start_xy", start)
        object.__setattr__(self, "goal_xy", goal)
        object.__setattr__(self, "start_yaw", wrap_angle(self.start_yaw))
        object.__setattr__(self, "obstacles", tuple(self.obstacles))
        object.__setattr__(self, "instruction", instruction)

    def obstacle_array(self) -> np.ndarray:
        """Return obstacle ``x, y, radius`` rows for dataset metadata."""

        if not self.obstacles:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(
            [[*obstacle.center.tolist(), obstacle.radius] for obstacle in self.obstacles],
            dtype=np.float32,
        )


class NavigationLayoutSampler:
    """Generate solvable scenes whose goal starts inside the forward camera sector."""

    _INSTRUCTIONS = (
        "Walk to the green goal while avoiding the obstacles.",
        "Navigate safely to the green marker.",
        "Go around the obstacles and stop at the green target.",
        "Reach the green goal without touching any obstacle.",
    )

    def __init__(
        self,
        *,
        obstacle_count: int = 8,
        obstacle_radius: float = 0.43,
        seed: int = 42,
    ) -> None:
        if obstacle_count < 0:
            raise ValueError("obstacle_count must be non-negative")
        if obstacle_radius <= 0.0:
            raise ValueError("obstacle_radius must be positive")
        self.obstacle_count = obstacle_count
        self.obstacle_radius = float(obstacle_radius)
        self.rng = np.random.default_rng(seed)

    def sample(self, start_xy: np.ndarray, start_yaw: float) -> NavigationLayout:
        """Sample a layout, retrying until the inflated A* grid is connected."""

        start = _xy(start_xy, name="start_xy")
        rotation = np.asarray(
            [
                [math.cos(start_yaw), -math.sin(start_yaw)],
                [math.sin(start_yaw), math.cos(start_yaw)],
            ],
            dtype=np.float32,
        )
        for _ in range(200):
            goal_local = np.asarray(
                [self.rng.uniform(4.5, 6.0), self.rng.uniform(-1.8, 1.8)],
                dtype=np.float32,
            )
            goal = start + rotation @ goal_local
            obstacles: list[CircularObstacle] = []
            attempts = 0
            while len(obstacles) < self.obstacle_count and attempts < 1_000:
                attempts += 1
                center_local = np.asarray(
                    [
                        self.rng.uniform(1.2, max(1.3, goal_local[0] - 0.8)),
                        self.rng.uniform(-2.25, 2.25),
                    ],
                    dtype=np.float32,
                )
                center = start + rotation @ center_local
                radius = self.obstacle_radius * self.rng.uniform(0.9, 1.1)
                if np.linalg.norm(center - start) < radius + 0.9:
                    continue
                if np.linalg.norm(center - goal) < radius + 0.8:
                    continue
                if any(
                    np.linalg.norm(center - item.center) < radius + item.radius + 0.18 for item in obstacles
                ):
                    continue
                obstacles.append(CircularObstacle(center=center, radius=radius))
            if len(obstacles) != self.obstacle_count:
                continue
            instruction = str(self.rng.choice(self._INSTRUCTIONS))
            layout = NavigationLayout(start, start_yaw, goal, tuple(obstacles), instruction)
            try:
                plan_path(layout)
            except RuntimeError:
                continue
            return layout
        raise RuntimeError("could not sample a solvable navigation layout")


def _segment_clear(
    start: np.ndarray,
    end: np.ndarray,
    obstacles: tuple[CircularObstacle, ...],
    clearance: float,
) -> bool:
    delta = end - start
    length_sq = float(np.dot(delta, delta))
    for obstacle in obstacles:
        if length_sq <= 1.0e-12:
            distance = float(np.linalg.norm(obstacle.center - start))
        else:
            projection = float(np.dot(obstacle.center - start, delta) / length_sq)
            closest = start + np.clip(projection, 0.0, 1.0) * delta
            distance = float(np.linalg.norm(obstacle.center - closest))
        if distance <= obstacle.radius + clearance:
            return False
    return True


def _simplify_path(
    points: np.ndarray,
    obstacles: tuple[CircularObstacle, ...],
    clearance: float,
) -> np.ndarray:
    simplified = [points[0]]
    index = 0
    while index < len(points) - 1:
        candidate = len(points) - 1
        while candidate > index + 1 and not _segment_clear(
            points[index], points[candidate], obstacles, clearance
        ):
            candidate -= 1
        simplified.append(points[candidate])
        index = candidate
    return np.asarray(simplified, dtype=np.float32)


def plan_path(
    layout: NavigationLayout,
    *,
    resolution: float = 0.2,
    robot_clearance: float = 0.55,
) -> np.ndarray:
    """Plan and line-of-sight simplify an inflated-grid A* path."""

    if resolution <= 0.0 or robot_clearance < 0.0:
        raise ValueError("resolution must be positive and clearance non-negative")
    points = [layout.start_xy, layout.goal_xy, *(obstacle.center for obstacle in layout.obstacles)]
    stacked = np.stack(points)
    margin = 1.2
    lower = stacked.min(axis=0) - margin
    upper = stacked.max(axis=0) + margin
    shape = np.ceil((upper - lower) / resolution).astype(int) + 1

    def index_of(point: np.ndarray) -> tuple[int, int]:
        raw = np.rint((point - lower) / resolution).astype(int)
        return int(raw[0]), int(raw[1])

    def point_of(index: tuple[int, int]) -> np.ndarray:
        return lower + resolution * np.asarray(index, dtype=np.float32)

    start = index_of(layout.start_xy)
    goal = index_of(layout.goal_xy)

    def blocked(index: tuple[int, int]) -> bool:
        point = point_of(index)
        return any(
            np.linalg.norm(point - obstacle.center) <= obstacle.radius + robot_clearance
            for obstacle in layout.obstacles
        )

    queue: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {start: 0.0}
    directions = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current == goal:
            break
        if current_cost > cost[current] + 1.0e-8:
            continue
        for dx, dy in directions:
            neighbor = current[0] + dx, current[1] + dy
            if not (0 <= neighbor[0] < shape[0] and 0 <= neighbor[1] < shape[1]):
                continue
            if neighbor not in {start, goal} and blocked(neighbor):
                continue
            step_cost = math.sqrt(2.0) if dx and dy else 1.0
            candidate_cost = current_cost + step_cost
            if candidate_cost >= cost.get(neighbor, math.inf):
                continue
            cost[neighbor] = candidate_cost
            came_from[neighbor] = current
            heuristic = math.hypot(goal[0] - neighbor[0], goal[1] - neighbor[1])
            heapq.heappush(queue, (candidate_cost + heuristic, candidate_cost, neighbor))
    if goal not in cost:
        raise RuntimeError("navigation layout has no collision-free path")

    indices = [goal]
    while indices[-1] != start:
        indices.append(came_from[indices[-1]])
    indices.reverse()
    raw = np.stack([point_of(index) for index in indices])
    raw[0] = layout.start_xy
    raw[-1] = layout.goal_xy
    return _simplify_path(raw, layout.obstacles, robot_clearance)


class WaypointFollower:
    """Convert a world-frame A* path into safe body-frame velocity commands."""

    def __init__(
        self,
        path: np.ndarray,
        *,
        max_speed: float = 0.85,
        max_lateral_speed: float = 0.45,
        max_yaw_rate: float = 1.0,
        waypoint_radius: float = 0.45,
        goal_radius: float = 0.5,
    ) -> None:
        points = np.asarray(path, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.all(np.isfinite(points)):
            raise ValueError("path must be a finite (N,2) array with at least two points")
        if min(max_speed, max_lateral_speed, max_yaw_rate, waypoint_radius, goal_radius) <= 0.0:
            raise ValueError("follower limits must be positive")
        self.path = points
        self.max_speed = float(max_speed)
        self.max_lateral_speed = float(max_lateral_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.waypoint_radius = float(waypoint_radius)
        self.goal_radius = float(goal_radius)
        self.index = 1

    def reached_goal(self, position_xy: np.ndarray) -> bool:
        return bool(np.linalg.norm(_xy(position_xy, name="position_xy") - self.path[-1]) <= self.goal_radius)

    def command(self, position_xy: np.ndarray, yaw: float) -> np.ndarray:
        position = _xy(position_xy, name="position_xy")
        if not math.isfinite(yaw):
            raise ValueError("yaw must be finite")
        if self.reached_goal(position):
            return np.zeros(3, dtype=np.float32)
        while self.index < len(self.path) - 1:
            if np.linalg.norm(self.path[self.index] - position) > self.waypoint_radius:
                break
            self.index += 1
        delta = self.path[self.index] - position
        distance = float(np.linalg.norm(delta))
        desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
        heading_error = wrap_angle(desired_yaw - yaw)
        yaw_rate = float(np.clip(1.8 * heading_error, -self.max_yaw_rate, self.max_yaw_rate))
        scale = min(1.0, distance / 1.0)
        forward = float(np.clip(0.9 * distance * math.cos(heading_error), -0.25, self.max_speed)) * scale
        lateral = (
            float(
                np.clip(
                    0.8 * distance * math.sin(heading_error), -self.max_lateral_speed, self.max_lateral_speed
                )
            )
            * scale
        )
        if abs(heading_error) > 1.45:
            forward = 0.0
        return np.asarray([forward, lateral, yaw_rate], dtype=np.float32)


def yaw_from_quat_xyzw(quaternion: np.ndarray) -> float:
    """Return planar yaw from an Isaac Lab ``(x, y, z, w)`` quaternion."""

    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
