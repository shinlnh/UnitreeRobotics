"""Runtime data contracts shared by the policy, safety layer, and simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tasks import TaskSpec


@dataclass
class SceneObject:
    name: str
    kind: str
    color: str
    x: float
    y: float
    z: float = 0.04
    held: bool = False
    in_bin: str | None = None
    stacked_on: str | None = None


@dataclass(frozen=True)
class Observation:
    sequence: int
    timestamp: float
    task: TaskSpec
    eef: tuple[float, float, float]
    gripper_closed: bool
    held_object: str | None
    objects: dict[str, SceneObject]

    def as_sonic_state(self) -> dict[str, Any]:
        """Return the state/language skeleton used by UNITREE_G1_SONIC.

        The production camera and true qpos values are supplied by GEAR-SONIC.
        This method exists to make the modality contract visible and testable.
        """

        return {
            "video": {"ego_view": None},
            "state": {
                "left_leg": [[0.0] * 6],
                "right_leg": [[0.0] * 6],
                "waist": [[0.0] * 3],
                "left_arm": [[0.0] * 7],
                "right_arm": [[0.0] * 7],
                "left_hand": [[0.0] * 7],
                "right_hand": [[0.0] * 7],
                "projected_gravity": [[0.0, 0.0, -1.0]],
            },
            "language": {
                "annotation.human.task_description": [[self.task.instruction]],
            },
        }


@dataclass(frozen=True)
class CartesianAction:
    dx: float
    dy: float
    dz: float
    close_gripper: bool


@dataclass(frozen=True)
class ActionChunk:
    """Mock analogue of a GR00T action chunk.

    Production UNITREE_G1_SONIC returns 40 x (64 + 7 + 7) values. The smoke
    simulator interprets the first three latent dimensions as Cartesian deltas;
    no claim is made that these synthetic latents are valid SONIC tokens.
    """

    motion_token: tuple[tuple[float, ...], ...]
    left_hand_joints: tuple[tuple[float, ...], ...]
    right_hand_joints: tuple[tuple[float, ...], ...]

    @classmethod
    def from_cartesian(cls, actions: list[CartesianAction]) -> ActionChunk:
        motion: list[tuple[float, ...]] = []
        left: list[tuple[float, ...]] = []
        right: list[tuple[float, ...]] = []
        for action in actions:
            motion.append((action.dx, action.dy, action.dz, *([0.0] * 61)))
            hand_value = 1.0 if action.close_gripper else 0.0
            left.append((hand_value, *([0.0] * 6)))
            right.append((0.0,) * 7)
        return cls(tuple(motion), tuple(left), tuple(right))

    @property
    def horizon(self) -> int:
        return len(self.motion_token)

    def validate(self) -> None:
        if self.horizon == 0:
            raise ValueError("Action chunk cannot be empty")
        if (
            len(self.left_hand_joints) != self.horizon
            or len(self.right_hand_joints) != self.horizon
        ):
            raise ValueError("All action modalities must have the same horizon")
        if any(len(token) != 64 for token in self.motion_token):
            raise ValueError("Each SONIC motion token must have 64 values")
        if any(len(hand) != 7 for hand in (*self.left_hand_joints, *self.right_hand_joints)):
            raise ValueError("Each hand action must have 7 joint values")

    def cartesian_at(self, index: int) -> CartesianAction:
        token = self.motion_token[index]
        return CartesianAction(
            dx=token[0],
            dy=token[1],
            dz=token[2],
            close_gripper=self.left_hand_joints[index][0] >= 0.5,
        )


@dataclass
class EpisodeResult:
    task_id: str
    instruction: str
    seed: int
    success: bool
    steps: int
    policy_calls: int
    safety_clips: int
    elapsed_seconds: float
    final_eef: tuple[float, float, float] = (0.0, 0.0, 0.0)
    final_objects: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: list[str] = field(default_factory=list)
