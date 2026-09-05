"""Policy interfaces and the dependency-free generalist smoke policy."""

from __future__ import annotations

from math import hypot
from typing import Protocol

from .types import ActionChunk, CartesianAction, Observation


class Policy(Protocol):
    def reset(self) -> None: ...

    def get_action(self, observation: Observation) -> ActionChunk: ...


class HeuristicGeneralistPolicy:
    """Deterministic CPU policy for validating the entire control harness.

    It consumes the same task/state concepts as the production path and emits
    SONIC-shaped action chunks, but it is intentionally not a learned GR00T
    model. Real inference is launched through the upstream PolicyServer.
    """

    def __init__(self, horizon: int = 40, execution_horizon: int = 4):
        if horizon < 1 or execution_horizon < 1 or execution_horizon > horizon:
            raise ValueError("Invalid policy/execution horizon")
        self.horizon = horizon
        self.execution_horizon = execution_horizon

    def reset(self) -> None:
        return None

    def _chunk(
        self,
        observation: Observation,
        target: tuple[float, float, float],
        close_gripper: bool,
    ) -> ActionChunk:
        dx = (target[0] - observation.eef[0]) / self.execution_horizon
        dy = (target[1] - observation.eef[1]) / self.execution_horizon
        dz = (target[2] - observation.eef[2]) / self.execution_horizon
        action = CartesianAction(dx, dy, dz, close_gripper)
        return ActionChunk.from_cartesian([action] * self.horizon)

    def get_action(self, observation: Observation) -> ActionChunk:
        task = observation.task
        target = observation.objects[task.target]
        eef_x, eef_y, eef_z = observation.eef

        if observation.held_object != task.target:
            planar_distance = hypot(eef_x - target.x, eef_y - target.y)
            if planar_distance > 0.025:
                return self._chunk(
                    observation,
                    (target.x, target.y, max(eef_z, 0.28)),
                    close_gripper=False,
                )
            contact_z = target.z + 0.025
            if abs(eef_z - contact_z) > 0.015:
                return self._chunk(
                    observation,
                    (target.x, target.y, contact_z),
                    close_gripper=False,
                )
            return self._chunk(observation, observation.eef, close_gripper=True)

        if task.kind == "pick":
            return self._chunk(
                observation,
                (eef_x, eef_y, 0.72),
                close_gripper=True,
            )

        assert task.destination is not None
        destination = observation.objects[task.destination]
        planar_distance = hypot(eef_x - destination.x, eef_y - destination.y)
        if planar_distance > 0.03:
            if eef_z < 0.62:
                return self._chunk(
                    observation,
                    (eef_x, eef_y, 0.68),
                    close_gripper=True,
                )
            return self._chunk(
                observation,
                (destination.x, destination.y, 0.68),
                close_gripper=True,
            )

        release_z = 0.13 if task.kind == "place" else destination.z + 0.09
        if abs(eef_z - release_z) > 0.015:
            return self._chunk(
                observation,
                (destination.x, destination.y, release_z),
                close_gripper=True,
            )
        return self._chunk(observation, observation.eef, close_gripper=False)
