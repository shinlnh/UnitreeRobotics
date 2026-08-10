"""Framework-neutral construction of GR00T navigation observations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def build_proprio(
    joint_position: np.ndarray,
    joint_velocity: np.ndarray,
    base_linear_velocity: np.ndarray,
    base_angular_velocity: np.ndarray,
    projected_gravity: np.ndarray,
) -> np.ndarray:
    """Build the state vector used by the navigation embodiment.

    The order is part of the dataset contract and must remain stable between
    collection, fine-tuning, and closed-loop inference.
    """

    parts = [
        np.asarray(joint_position, dtype=np.float32).reshape(-1),
        np.asarray(joint_velocity, dtype=np.float32).reshape(-1),
        np.asarray(base_linear_velocity, dtype=np.float32).reshape(-1),
        np.asarray(base_angular_velocity, dtype=np.float32).reshape(-1),
        np.asarray(projected_gravity, dtype=np.float32).reshape(-1),
    ]
    expected_tail_sizes = (3, 3, 3)
    if tuple(part.size for part in parts[-3:]) != expected_tail_sizes:
        raise ValueError("base velocities and projected gravity must each contain three values")
    if parts[0].size == 0 or parts[0].shape != parts[1].shape:
        raise ValueError("joint position and velocity must be non-empty and have the same shape")
    result = np.concatenate(parts)
    if not np.all(np.isfinite(result)):
        raise ValueError("proprioception contains NaN or infinity")
    return result


def build_navigation_observation(
    rgb: np.ndarray,
    proprio: np.ndarray,
    instruction: str,
    *,
    video_key: str = "ego_view",
    state_key: str = "proprio",
    language_key: str = "annotation.human.task_description",
) -> dict[str, dict[str, np.ndarray | Sequence[Sequence[str]]]]:
    """Add the batch/time dimensions expected by the native GR00T policy."""

    frame = np.asarray(rgb)
    state = np.asarray(proprio, dtype=np.float32).reshape(-1)
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"rgb must have shape (H, W, 3|4), got {frame.shape}")
    if frame.dtype != np.uint8:
        if (
            np.issubdtype(frame.dtype, np.floating)
            and frame.size
            and frame.min() >= 0.0
            and frame.max() <= 1.0
        ):
            frame = np.rint(frame * 255.0).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    frame = np.ascontiguousarray(frame[..., :3])
    if state.size == 0 or not np.all(np.isfinite(state)):
        raise ValueError("proprio must be a non-empty finite vector")
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("instruction must not be empty")
    return {
        "video": {video_key: frame[None, None]},
        "state": {state_key: state[None, None]},
        "language": {language_key: [[instruction]]},
    }
