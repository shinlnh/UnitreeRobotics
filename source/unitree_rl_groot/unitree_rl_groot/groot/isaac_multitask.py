"""Pure NumPy bridge from GR00T N1.7 ``REAL_G1`` to Isaac Lab G1.

The policy predicts base-relative wrist poses plus hand and navigation targets.
Isaac Lab's locomanipulation environment accepts world-frame wrist poses, hands,
base velocity and base height.  World-frame composition stays in the Isaac
runner; this module owns the representation and joint-order conversions that
can be unit-tested without launching Kit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

REAL_G1_STATE_JOINTS: dict[str, tuple[str, ...]] = {
    "left_hand": (
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
    ),
    "right_hand": (
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
    ),
    "left_arm": (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ),
    "right_arm": (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ),
    "waist": ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"),
}

REAL_G1_ACTION_WIDTHS = {
    "left_wrist_eef_9d": 9,
    "right_wrist_eef_9d": 9,
    "left_hand": 7,
    "right_hand": 7,
    "left_arm": 7,
    "right_arm": 7,
    "waist": 3,
    "base_height_command": 1,
    "navigate_command": 3,
}

# PinkInverseKinematicsActionCfg interleaves the two hands by actuator level.
# GR00T stores seven consecutive joints per hand in the public dataset order.
POLICY_HANDS_TO_PINK = (0, 2, 4, 7, 9, 11, 1, 3, 5, 8, 10, 12, 6, 13)

NOMINAL_BASE_HEIGHT_M = 0.72
BASE_HEIGHT_LIMITS_M = (0.68, 0.76)
NAV_COMMAND_LIMITS = np.array((0.35, 0.20, 0.35), dtype=np.float32)

# Pelvis-relative workspaces keep each wrist on its own side of the body.  The
# source fruit scene needs forward reach, but never requires the arms to cross.
WRIST_WORKSPACES = {
    "left": np.array(((0.05, 0.60), (0.04, 0.50), (-0.08, 0.50)), dtype=np.float32),
    "right": np.array(((0.05, 0.60), (-0.50, -0.04), (-0.08, 0.50)), dtype=np.float32),
}

# Joint limits in the exact PinkInverseKinematicsAction hand order.
PINK_HAND_LIMITS = np.array(
    (
        (-1.571, 0.0),
        (-1.571, 0.0),
        (-1.047, 1.047),
        (0.0, 1.571),
        (0.0, 1.571),
        (-1.047, 1.047),
        (-1.745, 0.0),
        (-1.745, 0.0),
        (-0.724, 1.047),
        (0.0, 1.745),
        (0.0, 1.745),
        (-1.047, 0.724),
        (0.0, 1.745),
        (-1.745, 0.0),
    ),
    dtype=np.float32,
)


@dataclass(frozen=True)
class ActionSafetyReport:
    """What the safety envelope changed for one 32D Isaac action."""

    clipped_fields: tuple[str, ...]
    requested_wrist_step_m: tuple[float, float]
    applied_wrist_step_m: tuple[float, float]
    requested_wrist_rotation_deg: tuple[float, float]
    applied_base_height_m: float

    @property
    def was_clipped(self) -> bool:
        return bool(self.clipped_fields)


def _normalized_quaternion_xyzw(quaternion: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite XYZW quaternion, got {value}")
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        raise ValueError(f"{name} contains a zero quaternion")
    return value / norm


def _bounded_quaternion_step_xyzw(
    current: np.ndarray, target: np.ndarray, max_angle_rad: float
) -> tuple[np.ndarray, float, bool]:
    """Shortest-path SLERP with an angular step limit."""

    current_q = _normalized_quaternion_xyzw(current, "current wrist orientation")
    target_q = _normalized_quaternion_xyzw(target, "target wrist orientation")
    dot = float(np.dot(current_q, target_q))
    if dot < 0.0:
        target_q = -target_q
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    angle = 2.0 * float(np.arccos(dot))
    if angle <= max_angle_rad:
        return target_q.astype(np.float32), angle, False

    ratio = max_angle_rad / angle
    half_angle = float(np.arccos(dot))
    if half_angle < 1e-6:
        interpolated = current_q + ratio * (target_q - current_q)
    else:
        denominator = np.sin(half_angle)
        interpolated = (
            np.sin((1.0 - ratio) * half_angle) / denominator * current_q
            + np.sin(ratio * half_angle) / denominator * target_q
        )
    interpolated /= np.linalg.norm(interpolated)
    return interpolated.astype(np.float32), angle, True


class IsaacActionSafetyFilter:
    """Stateful safety envelope between GR00T and the Isaac whole-body controller.

    It bounds task-space reach and slew, prevents the two wrists from crossing,
    clips finger targets to the simulated hand limits, and rate-limits the
    locomotion command.  Inputs are always copied; the policy chunk is immutable.
    """

    def __init__(
        self,
        *,
        max_wrist_step_m: float = 0.012,
        max_wrist_rotation_deg: float = 7.5,
        max_hand_step_rad: float = 0.08,
        max_navigation_step: Sequence[float] = (0.04, 0.04, 0.05),
        max_height_step_m: float = 0.005,
    ) -> None:
        if min(max_wrist_step_m, max_wrist_rotation_deg, max_hand_step_rad, max_height_step_m) <= 0:
            raise ValueError("all scalar safety step limits must be positive")
        navigation_step = np.asarray(max_navigation_step, dtype=np.float32)
        if navigation_step.shape != (3,) or np.any(navigation_step <= 0):
            raise ValueError("max_navigation_step must contain three positive values")
        self.max_wrist_step_m = float(max_wrist_step_m)
        self.max_wrist_rotation_rad = float(np.deg2rad(max_wrist_rotation_deg))
        self.max_hand_step_rad = float(max_hand_step_rad)
        self.max_navigation_step = navigation_step
        self.max_height_step_m = float(max_height_step_m)
        self.reset()

    def reset(self) -> None:
        self._previous_navigation = np.zeros(3, dtype=np.float32)
        self._previous_height = float(NOMINAL_BASE_HEIGHT_M)

    def filter(
        self,
        action: np.ndarray,
        *,
        current_left_wrist: np.ndarray,
        current_right_wrist: np.ndarray,
        current_hands: np.ndarray,
    ) -> tuple[np.ndarray, ActionSafetyReport]:
        """Return a safe copied action and a report; never mutate ``action``."""

        requested = np.asarray(action, dtype=np.float32)
        if requested.shape != (32,) or not np.isfinite(requested).all():
            raise ValueError(f"Isaac action must be finite with shape (32,), got {requested.shape}")
        safe = np.array(requested, dtype=np.float32, copy=True)
        hands = np.asarray(current_hands, dtype=np.float32)
        if hands.shape != (14,) or not np.isfinite(hands).all():
            raise ValueError(f"current_hands must be finite with shape (14,), got {hands.shape}")

        clipped_fields: list[str] = []
        requested_steps: list[float] = []
        applied_steps: list[float] = []
        requested_rotations: list[float] = []
        wrist_specs = (
            ("left", 0, np.asarray(current_left_wrist, dtype=np.float32)),
            ("right", 7, np.asarray(current_right_wrist, dtype=np.float32)),
        )
        for side, offset, current_pose in wrist_specs:
            if current_pose.shape != (7,) or not np.isfinite(current_pose).all():
                raise ValueError(f"current_{side}_wrist must be finite with shape (7,)")
            workspace = WRIST_WORKSPACES[side]
            workspace_target = np.clip(safe[offset : offset + 3], workspace[:, 0], workspace[:, 1])
            if not np.allclose(workspace_target, safe[offset : offset + 3], atol=1e-7):
                clipped_fields.append(f"{side}_wrist_workspace")

            displacement = workspace_target - current_pose[:3]
            requested_step = float(np.linalg.norm(displacement))
            requested_steps.append(requested_step)
            if requested_step > self.max_wrist_step_m:
                displacement *= self.max_wrist_step_m / requested_step
                clipped_fields.append(f"{side}_wrist_position_slew")
            safe[offset : offset + 3] = current_pose[:3] + displacement
            applied_steps.append(float(np.linalg.norm(displacement)))

            bounded_quat, requested_angle, was_bounded = _bounded_quaternion_step_xyzw(
                current_pose[3:7], safe[offset + 3 : offset + 7], self.max_wrist_rotation_rad
            )
            safe[offset + 3 : offset + 7] = bounded_quat
            requested_rotations.append(float(np.rad2deg(requested_angle)))
            if was_bounded:
                clipped_fields.append(f"{side}_wrist_orientation_slew")

        hand_target = np.clip(safe[14:28], PINK_HAND_LIMITS[:, 0], PINK_HAND_LIMITS[:, 1])
        if not np.allclose(hand_target, safe[14:28], atol=1e-7):
            clipped_fields.append("hand_joint_limits")
        hand_delta = np.clip(hand_target - hands, -self.max_hand_step_rad, self.max_hand_step_rad)
        if not np.allclose(hand_delta, hand_target - hands, atol=1e-7):
            clipped_fields.append("hand_joint_slew")
        safe[14:28] = hands + hand_delta

        navigation = np.clip(safe[28:31], -NAV_COMMAND_LIMITS, NAV_COMMAND_LIMITS)
        if not np.allclose(navigation, safe[28:31], atol=1e-7):
            clipped_fields.append("navigation_limits")
        navigation_delta = np.clip(
            navigation - self._previous_navigation,
            -self.max_navigation_step,
            self.max_navigation_step,
        )
        if not np.allclose(navigation_delta, navigation - self._previous_navigation, atol=1e-7):
            clipped_fields.append("navigation_slew")
        safe[28:31] = self._previous_navigation + navigation_delta

        requested_height = float(safe[31])
        bounded_height = float(np.clip(requested_height, *BASE_HEIGHT_LIMITS_M))
        if not np.isclose(bounded_height, requested_height, atol=1e-7):
            clipped_fields.append("base_height_limits")
        applied_height = float(
            np.clip(
                bounded_height,
                self._previous_height - self.max_height_step_m,
                self._previous_height + self.max_height_step_m,
            )
        )
        if not np.isclose(applied_height, bounded_height, atol=1e-7):
            clipped_fields.append("base_height_slew")
        safe[31] = applied_height

        self._previous_navigation = safe[28:31].copy()
        self._previous_height = applied_height
        report = ActionSafetyReport(
            clipped_fields=tuple(dict.fromkeys(clipped_fields)),
            requested_wrist_step_m=(requested_steps[0], requested_steps[1]),
            applied_wrist_step_m=(applied_steps[0], applied_steps[1]),
            requested_wrist_rotation_deg=(requested_rotations[0], requested_rotations[1]),
            applied_base_height_m=applied_height,
        )
        return np.ascontiguousarray(safe), report


def joint_groups(joint_names: Sequence[str], joint_positions: np.ndarray) -> dict[str, np.ndarray]:
    """Select REAL_G1 state groups from an Isaac articulation by exact joint name."""

    positions = np.asarray(joint_positions, dtype=np.float32)
    if positions.ndim == 1:
        positions = positions[None, :]
    if positions.ndim != 2 or positions.shape[1] != len(joint_names):
        raise ValueError(
            f"joint_positions must have shape [batch, {len(joint_names)}], got {positions.shape}"
        )
    index = {name: position for position, name in enumerate(joint_names)}
    if len(index) != len(joint_names):
        raise ValueError("Isaac articulation contains duplicate joint names")

    result: dict[str, np.ndarray] = {}
    for group, names in REAL_G1_STATE_JOINTS.items():
        missing = [name for name in names if name not in index]
        if missing:
            raise KeyError(f"Isaac G1 is missing {group} joints: {missing}")
        result[group] = np.ascontiguousarray(positions[:, [index[name] for name in names]])
    return result


def pose7_xyzw_to_eef9d(pose: np.ndarray) -> np.ndarray:
    """Convert ``XYZ + quaternion XYZW`` to ``XYZ + first two rotation rows``."""

    value = np.asarray(pose, dtype=np.float64)
    if value.shape[-1] != 7:
        raise ValueError(f"pose must end in 7 values, got {value.shape}")
    quat = value[..., 3:7]
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("pose contains a zero quaternion")
    x, y, z, w = np.moveaxis(quat / norm, -1, 0)
    matrix = np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(*value.shape[:-1], 3, 3)
    return np.concatenate((value[..., :3], matrix[..., :2, :].reshape(*value.shape[:-1], 6)), axis=-1).astype(
        np.float32
    )


def _matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Convert one proper 3x3 rotation matrix to a normalized XYZW quaternion."""

    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat_wxyz = np.array(
            [
                0.25 * scale,
                (m[2, 1] - m[1, 2]) / scale,
                (m[0, 2] - m[2, 0]) / scale,
                (m[1, 0] - m[0, 1]) / scale,
            ]
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        quat_wxyz = np.array(
            [
                (m[2, 1] - m[1, 2]) / scale,
                0.25 * scale,
                (m[0, 1] + m[1, 0]) / scale,
                (m[0, 2] + m[2, 0]) / scale,
            ]
        )
    elif m[1, 1] > m[2, 2]:
        scale = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        quat_wxyz = np.array(
            [
                (m[0, 2] - m[2, 0]) / scale,
                (m[0, 1] + m[1, 0]) / scale,
                0.25 * scale,
                (m[1, 2] + m[2, 1]) / scale,
            ]
        )
    else:
        scale = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        quat_wxyz = np.array(
            [
                (m[1, 0] - m[0, 1]) / scale,
                (m[0, 2] + m[2, 0]) / scale,
                (m[1, 2] + m[2, 1]) / scale,
                0.25 * scale,
            ]
        )
    quat_wxyz /= np.linalg.norm(quat_wxyz)
    return quat_wxyz[[1, 2, 3, 0]]


def eef9d_to_pose7_xyzw(eef: np.ndarray) -> np.ndarray:
    """Convert one or more GR00T EEF9D values to ``XYZ + quaternion XYZW``."""

    value = np.asarray(eef, dtype=np.float64)
    if value.shape[-1] != 9:
        raise ValueError(f"EEF value must end in 9 values, got {value.shape}")
    flat = value.reshape(-1, 9)
    result = np.empty((flat.shape[0], 7), dtype=np.float32)
    result[:, :3] = flat[:, :3]
    for row_index, row in enumerate(flat):
        first = row[3:6]
        second = row[6:9]
        first_norm = np.linalg.norm(first)
        if first_norm < 1e-8:
            raise ValueError("EEF rot6d first row has zero norm")
        first = first / first_norm
        second = second - np.dot(second, first) * first
        second_norm = np.linalg.norm(second)
        if second_norm < 1e-8:
            raise ValueError("EEF rot6d rows are collinear")
        second = second / second_norm
        rotation = np.stack((first, second, np.cross(first, second)), axis=0)
        result[row_index, 3:7] = _matrix_to_quat_xyzw(rotation)
    return result.reshape(*value.shape[:-1], 7)


def make_policy_observation(
    *,
    rgb_history: Sequence[np.ndarray],
    left_wrist_pose: np.ndarray,
    right_wrist_pose: np.ndarray,
    groups: Mapping[str, np.ndarray],
    prompt: str,
) -> dict[str, object]:
    """Build the flat input accepted by ``Gr00tSimPolicyWrapper`` for REAL_G1."""

    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if len(rgb_history) < 1:
        raise ValueError("rgb_history must contain at least one image")
    current = np.asarray(rgb_history[-1], dtype=np.uint8)
    previous = np.asarray(rgb_history[0], dtype=np.uint8)
    if current.shape != previous.shape or current.ndim != 3 or current.shape[-1] != 3:
        raise ValueError(f"RGB frames must share shape [H, W, 3], got {previous.shape}, {current.shape}")

    result: dict[str, object] = {
        "video.ego_view": np.ascontiguousarray(np.stack((previous, current), axis=0)[None]),
        "annotation.human.task_description": [prompt],
        "state.left_wrist_eef_9d": pose7_xyzw_to_eef9d(left_wrist_pose)[None, None],
        "state.right_wrist_eef_9d": pose7_xyzw_to_eef9d(right_wrist_pose)[None, None],
    }
    for group_name in REAL_G1_STATE_JOINTS:
        if group_name not in groups:
            raise KeyError(f"missing REAL_G1 state group: {group_name}")
        values = np.asarray(groups[group_name], dtype=np.float32)
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        expected = len(REAL_G1_STATE_JOINTS[group_name])
        if values.shape != (expected,):
            raise ValueError(f"state.{group_name} must have shape ({expected},), got {values.shape}")
        result[f"state.{group_name}"] = np.ascontiguousarray(values[None, None])
    return result


def _unbatch_action(value: np.ndarray, width: int, key: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{key} must have shape [1, horizon, {width}], got {np.shape(value)}")
    return np.ascontiguousarray(array)


def real_g1_actions_to_isaac(action: Mapping[str, np.ndarray]) -> np.ndarray:
    """Translate a decoded REAL_G1 chunk into Isaac's base-relative 32D layout.

    Returned wrist quaternions are XYZW, Isaac Lab 3.0's native convention.
    The runner composes both wrist poses
    with the current pelvis pose before passing the result to Pink IK.
    """

    decoded: dict[str, np.ndarray] = {}
    for name, width in REAL_G1_ACTION_WIDTHS.items():
        key = f"action.{name}"
        if key not in action:
            raise KeyError(f"GR00T action is missing {key}")
        decoded[name] = _unbatch_action(action[key], width, key)
    horizons = {values.shape[0] for values in decoded.values()}
    if len(horizons) != 1:
        raise ValueError(f"GR00T action groups have inconsistent horizons: {sorted(horizons)}")
    if next(iter(horizons)) < 1:
        raise ValueError("GR00T returned an empty action horizon")

    hands = np.concatenate((decoded["left_hand"], decoded["right_hand"]), axis=1)
    pink_hands = hands[:, POLICY_HANDS_TO_PINK]
    result = np.concatenate(
        (
            eef9d_to_pose7_xyzw(decoded["left_wrist_eef_9d"]),
            eef9d_to_pose7_xyzw(decoded["right_wrist_eef_9d"]),
            pink_hands,
            decoded["navigate_command"],
            decoded["base_height_command"],
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    if result.shape[1] != 32 or not np.isfinite(result).all():
        raise ValueError(f"translated Isaac action is invalid: {result.shape}")
    return np.ascontiguousarray(result)
