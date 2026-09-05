"""Safety envelope used by the deterministic smoke simulator."""

from __future__ import annotations

from dataclasses import dataclass

from .config import SafetyConfig
from .types import CartesianAction


@dataclass(frozen=True)
class SafetyDecision:
    action: CartesianAction
    clipped: bool
    reasons: tuple[str, ...]


class SafetySupervisor:
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.estopped = False

    def emergency_stop(self) -> None:
        self.estopped = True

    def reset(self) -> None:
        self.estopped = False

    def filter(
        self,
        action: CartesianAction,
        eef: tuple[float, float, float],
    ) -> SafetyDecision:
        if self.estopped:
            return SafetyDecision(
                CartesianAction(0.0, 0.0, 0.0, action.close_gripper),
                True,
                ("emergency_stop",),
            )

        reasons: list[str] = []
        limit = self.config.max_cartesian_delta

        def clamp(value: float, low: float, high: float, reason: str) -> float:
            bounded = min(max(value, low), high)
            if bounded != value:
                reasons.append(reason)
            return bounded

        dx = clamp(action.dx, -limit, limit, "delta_x")
        dy = clamp(action.dy, -limit, limit, "delta_y")
        dz = clamp(action.dz, -limit, limit, "delta_z")

        target_x = clamp(eef[0] + dx, *self.config.workspace_x, "workspace_x")
        target_y = clamp(eef[1] + dy, *self.config.workspace_y, "workspace_y")
        target_z = clamp(eef[2] + dz, *self.config.workspace_z, "workspace_z")
        safe = CartesianAction(
            dx=target_x - eef[0],
            dy=target_y - eef[1],
            dz=target_z - eef[2],
            close_gripper=action.close_gripper,
        )
        return SafetyDecision(safe, bool(reasons), tuple(dict.fromkeys(reasons)))
