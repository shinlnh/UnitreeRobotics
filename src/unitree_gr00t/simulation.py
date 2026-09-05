"""Small deterministic world used for fast end-to-end generalist smoke tests."""

from __future__ import annotations

import random
import time
from dataclasses import asdict
from math import dist, hypot

from .config import SafetyConfig
from .policy import Policy
from .safety import SafetySupervisor
from .tasks import TaskSpec
from .types import CartesianAction, EpisodeResult, Observation, SceneObject


class TabletopWorld:
    def __init__(self, task: TaskSpec, seed: int):
        self.task = task
        self.seed = seed
        self.sequence = 0
        self.eef = [0.50, 0.12, 0.75]
        self.gripper_closed = False
        self.held_object: str | None = None
        self.events: list[str] = []
        rng = random.Random(seed)

        red_x, red_y = rng.uniform(0.25, 0.45), rng.uniform(0.35, 0.62)
        blue_x, blue_y = rng.uniform(0.57, 0.76), rng.uniform(0.35, 0.62)
        self.objects = {
            "red_cube": SceneObject("red_cube", "cube", "#ef4444", red_x, red_y),
            "blue_cube": SceneObject("blue_cube", "cube", "#3b82f6", blue_x, blue_y),
            "green_bin": SceneObject("green_bin", "bin", "#22c55e", 0.18, 0.83, 0.02),
            "yellow_bin": SceneObject("yellow_bin", "bin", "#eab308", 0.82, 0.83, 0.02),
        }

    def observe(self) -> Observation:
        return Observation(
            sequence=self.sequence,
            timestamp=time.monotonic(),
            task=self.task,
            eef=tuple(self.eef),
            gripper_closed=self.gripper_closed,
            held_object=self.held_object,
            objects={name: SceneObject(**asdict(obj)) for name, obj in self.objects.items()},
        )

    def _attach_nearest(self) -> None:
        candidates = [obj for obj in self.objects.values() if obj.kind == "cube" and not obj.in_bin]
        if not candidates:
            return
        nearest = min(candidates, key=lambda obj: dist(self.eef, (obj.x, obj.y, obj.z + 0.025)))
        if dist(self.eef, (nearest.x, nearest.y, nearest.z + 0.025)) <= 0.07:
            nearest.held = True
            nearest.stacked_on = None
            self.held_object = nearest.name
            self.events.append(f"grasp:{nearest.name}")

    def _release(self) -> None:
        if not self.held_object:
            return
        held = self.objects[self.held_object]
        bins = [obj for obj in self.objects.values() if obj.kind == "bin"]
        nearest_bin = min(bins, key=lambda obj: hypot(held.x - obj.x, held.y - obj.y))
        if hypot(held.x - nearest_bin.x, held.y - nearest_bin.y) <= 0.13 and self.eef[2] < 0.30:
            held.in_bin = nearest_bin.name
            held.x, held.y, held.z = nearest_bin.x, nearest_bin.y, 0.04
            self.events.append(f"place:{held.name}:{nearest_bin.name}")
        else:
            cubes = [
                obj for obj in self.objects.values() if obj.kind == "cube" and obj.name != held.name
            ]
            nearest_cube = min(cubes, key=lambda obj: hypot(held.x - obj.x, held.y - obj.y))
            if (
                hypot(held.x - nearest_cube.x, held.y - nearest_cube.y) <= 0.08
                and self.eef[2] < 0.25
            ):
                held.stacked_on = nearest_cube.name
                held.x, held.y, held.z = nearest_cube.x, nearest_cube.y, nearest_cube.z + 0.08
                self.events.append(f"stack:{held.name}:{nearest_cube.name}")
            else:
                held.z = 0.04
                self.events.append(f"drop:{held.name}")
        held.held = False
        self.held_object = None

    def step(self, action: CartesianAction) -> None:
        was_closed = self.gripper_closed
        self.eef[0] += action.dx
        self.eef[1] += action.dy
        self.eef[2] += action.dz
        self.gripper_closed = action.close_gripper

        if self.held_object:
            held = self.objects[self.held_object]
            held.x, held.y = self.eef[0], self.eef[1]
            held.z = max(0.04, self.eef[2] - 0.025)

        if self.gripper_closed and not was_closed and not self.held_object:
            self._attach_nearest()
        elif not self.gripper_closed and was_closed:
            self._release()
        self.sequence += 1

    def succeeded(self) -> bool:
        target = self.objects[self.task.target]
        if self.task.kind == "pick":
            return target.held and target.z >= 0.60
        if self.task.kind == "place":
            return target.in_bin == self.task.destination
        if self.task.kind == "stack":
            return target.stacked_on == self.task.destination
        return False


class EpisodeRunner:
    def __init__(
        self,
        policy: Policy,
        safety_config: SafetyConfig,
        execution_horizon: int = 4,
    ):
        self.policy = policy
        self.safety_config = safety_config
        self.execution_horizon = execution_horizon

    def run(self, task: TaskSpec, seed: int) -> EpisodeResult:
        world = TabletopWorld(task, seed)
        supervisor = SafetySupervisor(self.safety_config)
        self.policy.reset()
        started = time.perf_counter()
        policy_calls = 0
        safety_clips = 0
        steps = 0

        while steps < self.safety_config.max_episode_steps and not world.succeeded():
            observation = world.observe()
            chunk = self.policy.get_action(observation)
            chunk.validate()
            policy_calls += 1
            for index in range(min(self.execution_horizon, chunk.horizon)):
                decision = supervisor.filter(chunk.cartesian_at(index), tuple(world.eef))
                safety_clips += int(decision.clipped)
                world.step(decision.action)
                steps += 1
                if world.succeeded() or steps >= self.safety_config.max_episode_steps:
                    break

        elapsed = time.perf_counter() - started
        final_objects = {
            name: {
                "kind": obj.kind,
                "color": obj.color,
                "x": round(obj.x, 4),
                "y": round(obj.y, 4),
                "z": round(obj.z, 4),
                "held": obj.held,
                "in_bin": obj.in_bin,
                "stacked_on": obj.stacked_on,
            }
            for name, obj in world.objects.items()
        }
        return EpisodeResult(
            task_id=task.id,
            instruction=task.instruction,
            seed=seed,
            success=world.succeeded(),
            steps=steps,
            policy_calls=policy_calls,
            safety_clips=safety_clips,
            elapsed_seconds=elapsed,
            final_eef=tuple(round(value, 4) for value in world.eef),
            final_objects=final_objects,
            events=world.events,
        )


def run_suite(
    tasks: list[TaskSpec],
    episodes: int,
    seed: int,
    policy: Policy,
    safety_config: SafetyConfig,
    execution_horizon: int = 4,
) -> list[EpisodeResult]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    runner = EpisodeRunner(policy, safety_config, execution_horizon)
    results: list[EpisodeResult] = []
    for task_index, task in enumerate(tasks):
        for episode_index in range(episodes):
            episode_seed = seed + task_index * 10_000 + episode_index
            results.append(runner.run(task, episode_seed))
    return results
