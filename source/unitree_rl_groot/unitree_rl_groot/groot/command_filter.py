"""Safety shaping between a high-level VLA and the low-level locomotion policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VelocityLimits:
    """Velocity and acceleration limits in body frame."""

    minimum: np.ndarray = field(default_factory=lambda: np.array([-0.5, -0.6, -1.2], dtype=np.float32))
    maximum: np.ndarray = field(default_factory=lambda: np.array([1.5, 0.6, 1.2], dtype=np.float32))
    max_acceleration: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 2.0], dtype=np.float32))
    deadband: np.ndarray = field(default_factory=lambda: np.array([0.03, 0.03, 0.04], dtype=np.float32))

    def __post_init__(self) -> None:
        for name in ("minimum", "maximum", "max_acceleration", "deadband"):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,), got {value.shape}")
            object.__setattr__(self, name, value)
        if np.any(self.minimum >= self.maximum):
            raise ValueError("every minimum must be less than its maximum")
        if np.any(self.max_acceleration <= 0.0):
            raise ValueError("max_acceleration must be positive")


class VelocityCommandFilter:
    """Clamp VLA output and enforce a per-axis slew-rate limit."""

    def __init__(self, limits: VelocityLimits | None = None) -> None:
        self.limits = limits or VelocityLimits()
        self._command = np.zeros(3, dtype=np.float32)

    @property
    def command(self) -> np.ndarray:
        return self._command.copy()

    def reset(self, command: np.ndarray | None = None) -> None:
        value = np.zeros(3, dtype=np.float32) if command is None else np.asarray(command, dtype=np.float32)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("reset command must contain three finite values")
        self._command = np.clip(value, self.limits.minimum, self.limits.maximum)

    def step(self, target: np.ndarray, dt: float) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (3,):
            raise ValueError(f"target must have shape (3,), got {target.shape}")
        if not np.all(np.isfinite(target)):
            raise ValueError("target contains NaN or infinity")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        target = np.clip(target, self.limits.minimum, self.limits.maximum)
        target = np.where(np.abs(target) < self.limits.deadband, 0.0, target)
        max_delta = self.limits.max_acceleration * np.float32(dt)
        self._command += np.clip(target - self._command, -max_delta, max_delta)
        return self.command


class ActionChunk:
    """Validated navigation actions returned by one GR00T inference call."""

    def __init__(self, commands: np.ndarray) -> None:
        array = np.asarray(commands, dtype=np.float32)
        if array.ndim == 3:
            if array.shape[0] != 1:
                raise ValueError("closed-loop navigation currently supports one Isaac Lab environment")
            array = array[0]
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"navigation chunk must have shape (H, 3) or (1, H, 3), got {array.shape}")
        if array.shape[0] == 0 or not np.all(np.isfinite(array)):
            raise ValueError("navigation chunk must be non-empty and finite")
        self._commands = array
        self._index = 0

    @classmethod
    def from_policy_action(
        cls,
        action: dict[str, Any],
        *,
        key: str = "navigate_command",
        execution_horizon: int | None = None,
    ) -> ActionChunk:
        if key not in action:
            raise KeyError(f"GR00T action has no {key!r}; keys={sorted(action)}")
        commands = np.asarray(action[key], dtype=np.float32)
        if execution_horizon is not None:
            if execution_horizon <= 0:
                raise ValueError("execution_horizon must be positive")
            axis = 1 if commands.ndim == 3 else 0
            slicer = [slice(None)] * commands.ndim
            slicer[axis] = slice(0, execution_horizon)
            commands = commands[tuple(slicer)]
        return cls(commands)

    def __len__(self) -> int:
        return self._commands.shape[0] - self._index

    def pop(self) -> np.ndarray:
        if not len(self):
            raise IndexError("action chunk is exhausted")
        command = self._commands[self._index]
        self._index += 1
        return command.copy()
