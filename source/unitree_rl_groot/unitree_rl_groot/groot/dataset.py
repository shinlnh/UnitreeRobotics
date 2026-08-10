"""Raw navigation episode schema shared by Isaac Lab and the converter."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RawNavigationEpisode:
    """One synchronized camera/state/command trajectory."""

    rgb: np.ndarray
    state: np.ndarray
    action: np.ndarray
    timestamp: np.ndarray
    language: str
    goal_xy: np.ndarray | None = None
    obstacles: np.ndarray | None = None
    success: bool | None = None

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        state = np.asarray(self.state, dtype=np.float32)
        action = np.asarray(self.action, dtype=np.float32)
        timestamp = np.asarray(self.timestamp, dtype=np.float32)
        language = str(self.language).strip()
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8 (T,H,W,3), got {rgb.dtype} {rgb.shape}")
        frame_count = rgb.shape[0]
        if frame_count < 2:
            raise ValueError("an episode needs at least two frames")
        if state.ndim != 2 or state.shape[0] != frame_count or state.shape[1] == 0:
            raise ValueError(f"state must be (T,D), got {state.shape}")
        if action.shape != (frame_count, 3):
            raise ValueError(f"action must be (T,3), got {action.shape}")
        if timestamp.shape != (frame_count,):
            raise ValueError(f"timestamp must be (T,), got {timestamp.shape}")
        if not all(np.all(np.isfinite(array)) for array in (state, action, timestamp)):
            raise ValueError("state, action, and timestamp must be finite")
        if timestamp[0] < 0.0 or np.any(np.diff(timestamp) <= 0.0):
            raise ValueError("timestamps must be non-negative and strictly increasing")
        if not language:
            raise ValueError("language instruction must not be empty")
        goal_xy = None if self.goal_xy is None else np.asarray(self.goal_xy, dtype=np.float32)
        obstacles = None if self.obstacles is None else np.asarray(self.obstacles, dtype=np.float32)
        if goal_xy is not None and (goal_xy.shape != (2,) or not np.all(np.isfinite(goal_xy))):
            raise ValueError(f"goal_xy must be finite shape (2,), got {goal_xy.shape}")
        if obstacles is not None and (
            obstacles.ndim != 2
            or obstacles.shape[1] != 3
            or not np.all(np.isfinite(obstacles))
            or np.any(obstacles[:, 2] <= 0.0)
        ):
            raise ValueError(f"obstacles must be finite (N,3) x/y/radius rows, got {obstacles.shape}")
        object.__setattr__(self, "rgb", np.ascontiguousarray(rgb))
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "goal_xy", goal_xy)
        object.__setattr__(self, "obstacles", obstacles)
        object.__setattr__(self, "success", None if self.success is None else bool(self.success))

    @classmethod
    def load(cls, path: str | Path) -> RawNavigationEpisode:
        with np.load(Path(path), allow_pickle=False) as payload:
            required = {"rgb", "state", "action", "timestamp", "language"}
            missing = required - set(payload.files)
            if missing:
                raise ValueError(f"{path}: missing keys {sorted(missing)}")
            language_value = np.asarray(payload["language"])
            if language_value.shape != () or language_value.dtype.kind not in "US":
                raise ValueError(f"{path}: language must be a scalar string")
            success = None
            if "success" in payload.files:
                success_value = np.asarray(payload["success"])
                if success_value.shape != () or success_value.dtype.kind != "b":
                    raise ValueError(f"{path}: success must be a scalar bool")
                success = bool(success_value.item())
            return cls(
                rgb=payload["rgb"],
                state=payload["state"],
                action=payload["action"],
                timestamp=payload["timestamp"],
                language=str(language_value.item()),
                goal_xy=payload["goal_xy"] if "goal_xy" in payload.files else None,
                obstacles=payload["obstacles"] if "obstacles" in payload.files else None,
                success=success,
            )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary = Path(file.name)
                payload = {
                    "rgb": self.rgb,
                    "state": self.state,
                    "action": self.action,
                    "timestamp": self.timestamp,
                    "language": np.asarray(self.language),
                }
                if self.goal_xy is not None:
                    payload["goal_xy"] = self.goal_xy
                if self.obstacles is not None:
                    payload["obstacles"] = self.obstacles
                if self.success is not None:
                    payload["success"] = np.asarray(self.success, dtype=np.bool_)
                np.savez_compressed(file, **payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, destination)
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
        return destination


def numeric_stats(array: np.ndarray) -> dict[str, list[float]]:
    """Compute LeRobot-v2-compatible per-feature summary statistics."""

    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("statistics input must be a non-empty finite 1D/2D array")
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }
