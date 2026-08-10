"""Deterministic mobile-manipulation supervisor around a learned GR00T policy.

GR00T proposes task-space manipulation targets.  This module owns the parts
that should not be left implicit in a long-horizon deployment: observable
mission phases, conservative geometric navigation, and upper-body gating while
the robot is walking.  It is simulator-independent NumPy so the safety logic is
unit-testable without launching Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from unitree_rl_groot.groot.fetch import FetchEvent, FetchExecutive, FetchMission, FetchPhase
from unitree_rl_groot.groot.isaac_multitask import NAV_COMMAND_LIMITS, NOMINAL_BASE_HEIGHT_M

# PinkInverseKinematicsAction interleaves both hands.  These positions select
# the seven joints belonging to one physical hand in that exact action order.
PINK_LEFT_HAND_INDICES = np.array((0, 1, 2, 6, 7, 8, 12), dtype=np.int64)
PINK_RIGHT_HAND_INDICES = np.array((3, 4, 5, 9, 10, 11, 13), dtype=np.int64)

# A learned wrist target is only meaningful near the physical entity that the
# executive selected.  These base-frame offsets form a conservative approach
# corridor around a fruit/placement target.  In particular, the upper Z bound
# prevents a slowly repeated VLA target from walking a hand above the shoulder
# while still allowing a tabletop object to be lifted and released.
GRASP_TARGET_OFFSETS_M = np.array(((-0.20, 0.20), (-0.18, 0.18), (-0.10, 0.22)), dtype=np.float32)
PLACE_TARGET_OFFSETS_M = np.array(((-0.22, 0.22), (-0.22, 0.22), (-0.12, 0.25)), dtype=np.float32)


def _pose7(value: np.ndarray, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float32)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError(f"{name} must be a finite XYZ+XYZW pose with shape (7,)")
    quaternion_norm = float(np.linalg.norm(pose[3:7]))
    if quaternion_norm < 1e-8:
        raise ValueError(f"{name} contains a zero quaternion")
    return pose


def _yaw_from_xyzw(pose: np.ndarray) -> float:
    x, y, z, w = pose[3:7] / np.linalg.norm(pose[3:7])
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _world_point_in_base(robot_pose: np.ndarray, point_world: np.ndarray) -> np.ndarray:
    """Transform a world XYZ point into the robot's full 3-D base frame."""

    quaternion = robot_pose[3:7] / np.linalg.norm(robot_pose[3:7])
    x, y, z, w = quaternion
    rotation_base_to_world = np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )
    return rotation_base_to_world.T @ (np.asarray(point_world, dtype=np.float32) - robot_pose[:3])


@dataclass(frozen=True)
class FetchControlDecision:
    """One observable supervisor decision consumed by the Isaac runner."""

    phase: FetchPhase
    prompt: str
    navigation_command: np.ndarray
    active_hand: str
    manipulation_target_base: np.ndarray
    transitions: tuple[str, ...]

    @property
    def walking(self) -> bool:
        return self.phase in (FetchPhase.NAVIGATE_TO_OBJECT, FetchPhase.CARRY_TO_DESTINATION)


class SafeFetchSupervisor:
    """Phase-aware safety shell for a single-object fetch-and-deliver task.

    Simulator poses are used as privileged safety/mission evidence.  They do
    not replace GR00T's RGB/language input or manipulation prediction; they
    make navigation deterministic and prevent unrelated arm motion while the
    locomotion policy is active.
    """

    def __init__(
        self,
        mission: FetchMission,
        *,
        initial_object_height_m: float,
        pickup_standoff_m: float = 0.68,
        destination_standoff_m: float = 0.62,
        lifted_height_m: float = 0.06,
        delivery_radius_m: float = 0.45,
    ) -> None:
        if min(pickup_standoff_m, destination_standoff_m, lifted_height_m, delivery_radius_m) <= 0:
            raise ValueError("all supervisor distances must be positive")
        if not np.isfinite(initial_object_height_m):
            raise ValueError("initial_object_height_m must be finite")
        self.executive = FetchExecutive(mission)
        self.initial_object_height_m = float(initial_object_height_m)
        self.pickup_standoff_m = float(pickup_standoff_m)
        self.destination_standoff_m = float(destination_standoff_m)
        self.lifted_height_m = float(lifted_height_m)
        self.delivery_radius_m = float(delivery_radius_m)

    @property
    def phase(self) -> FetchPhase:
        return self.executive.phase

    @staticmethod
    def _planar_navigation(
        robot_pose: np.ndarray,
        goal_xy: np.ndarray,
        *,
        standoff_m: float,
    ) -> np.ndarray:
        """Proportional holonomic command in the robot base frame."""

        displacement_world = np.asarray(goal_xy, dtype=np.float32) - robot_pose[:2]
        distance = float(np.linalg.norm(displacement_world))
        if distance <= standoff_m:
            return np.zeros(3, dtype=np.float32)

        yaw = _yaw_from_xyzw(robot_pose)
        cosine, sine = np.cos(yaw), np.sin(yaw)
        forward = float(cosine * displacement_world[0] + sine * displacement_world[1])
        lateral = float(-sine * displacement_world[0] + cosine * displacement_world[1])
        scale = max(0.0, 1.0 - standoff_m / distance)
        bearing = _wrap_angle(float(np.arctan2(lateral, forward)))
        command = np.array(
            (
                0.8 * forward * scale,
                0.8 * lateral * scale,
                1.2 * bearing,
            ),
            dtype=np.float32,
        )
        # Turn before translating aggressively when the goal is behind the robot.
        if abs(bearing) > np.deg2rad(55.0):
            command[:2] *= 0.25
        return np.clip(command, -NAV_COMMAND_LIMITS, NAV_COMMAND_LIMITS).astype(np.float32)

    def observe(
        self,
        *,
        robot_pose: np.ndarray,
        object_pose: np.ndarray,
        destination_pose: np.ndarray,
        official_success: bool = False,
    ) -> FetchControlDecision:
        """Update the executive from measurable evidence and produce control intent."""

        robot = _pose7(robot_pose, "robot_pose")
        target = _pose7(object_pose, "object_pose")
        destination = _pose7(destination_pose, "destination_pose")
        robot_object_distance = float(np.linalg.norm(target[:2] - robot[:2]))
        robot_destination_distance = float(np.linalg.norm(destination[:2] - robot[:2]))
        object_destination_distance = float(np.linalg.norm(destination[:2] - target[:2]))
        object_is_lifted = target[2] >= self.initial_object_height_m + self.lifted_height_m
        object_is_placed = (
            object_destination_distance <= self.delivery_radius_m
            and target[2] <= self.initial_object_height_m + 0.04
        )

        transitions: list[str] = []

        def advance(event: FetchEvent) -> None:
            previous = self.executive.phase
            current = self.executive.advance(event)
            transitions.append(f"{previous.value}->{current.value}")

        # Collapse evidence-only phases in one observation.  Learned motion is
        # only requested in the stable terminal phase of this loop.
        progressed = True
        while progressed and not self.executive.done:
            progressed = False
            phase = self.executive.phase
            if phase is FetchPhase.SEARCH:
                advance(FetchEvent.OBJECT_FOUND)
                progressed = True
            elif phase is FetchPhase.NAVIGATE_TO_OBJECT and robot_object_distance <= self.pickup_standoff_m:
                advance(FetchEvent.PICKUP_POSE_REACHED)
                progressed = True
            elif phase is FetchPhase.ALIGN_FOR_GRASP:
                advance(FetchEvent.PREGRASP_ALIGNED)
                progressed = True
            elif phase is FetchPhase.GRASP and object_is_lifted:
                advance(FetchEvent.GRASP_CLOSED)
                progressed = True
            elif phase is FetchPhase.VERIFY_GRASP and object_is_lifted:
                advance(FetchEvent.OBJECT_LIFTED)
                progressed = True
            elif (
                phase is FetchPhase.CARRY_TO_DESTINATION
                and robot_destination_distance <= self.destination_standoff_m
            ):
                advance(FetchEvent.DESTINATION_REACHED)
                progressed = True
            elif phase is FetchPhase.PLACE and object_is_placed:
                advance(FetchEvent.OBJECT_RELEASED)
                progressed = True
            elif phase is FetchPhase.VERIFY_DELIVERY and official_success:
                advance(FetchEvent.DELIVERY_CONFIRMED)
                progressed = True

        displacement_world = target[:2] - robot[:2]
        yaw = _yaw_from_xyzw(robot)
        object_lateral = -np.sin(yaw) * displacement_world[0] + np.cos(yaw) * displacement_world[1]
        active_hand = "left" if object_lateral >= 0.0 else "right"

        phase = self.executive.phase
        if phase is FetchPhase.NAVIGATE_TO_OBJECT:
            navigation = self._planar_navigation(robot, target[:2], standoff_m=self.pickup_standoff_m)
        elif phase is FetchPhase.CARRY_TO_DESTINATION:
            navigation = self._planar_navigation(
                robot, destination[:2], standoff_m=self.destination_standoff_m
            )
        else:
            navigation = np.zeros(3, dtype=np.float32)
        manipulation_target_world = destination[:3] if phase is FetchPhase.PLACE else target[:3]
        return FetchControlDecision(
            phase=phase,
            prompt=self.executive.prompt,
            navigation_command=navigation,
            active_hand=active_hand,
            manipulation_target_base=_world_point_in_base(robot, manipulation_target_world),
            transitions=tuple(transitions),
        )

    @staticmethod
    def supervise_action(
        action: np.ndarray,
        decision: FetchControlDecision,
        *,
        current_left_wrist: np.ndarray,
        current_right_wrist: np.ndarray,
        current_hands: np.ndarray,
        neutral_left_wrist: np.ndarray | None = None,
        neutral_right_wrist: np.ndarray | None = None,
        neutral_hands: np.ndarray | None = None,
    ) -> np.ndarray:
        """Gate upper-body motion by phase and override navigation deterministically."""

        requested = np.asarray(action, dtype=np.float32)
        left = _pose7(current_left_wrist, "current_left_wrist")
        right = _pose7(current_right_wrist, "current_right_wrist")
        hands = np.asarray(current_hands, dtype=np.float32)
        if requested.shape != (32,) or not np.isfinite(requested).all():
            raise ValueError("action must be finite with shape (32,)")
        if hands.shape != (14,) or not np.isfinite(hands).all():
            raise ValueError("current_hands must be finite with shape (14,)")
        neutral_left = (
            left if neutral_left_wrist is None else _pose7(neutral_left_wrist, "neutral_left_wrist")
        )
        neutral_right = (
            right if neutral_right_wrist is None else _pose7(neutral_right_wrist, "neutral_right_wrist")
        )
        neutral_fingers = hands if neutral_hands is None else np.asarray(neutral_hands, dtype=np.float32)
        if neutral_fingers.shape != (14,) or not np.isfinite(neutral_fingers).all():
            raise ValueError("neutral_hands must be finite with shape (14,)")

        supervised = np.array(requested, dtype=np.float32, copy=True)
        supervised[28:31] = decision.navigation_command
        # The source tabletop demonstrations do not require crouching. Keep
        # the locomotion policy at its trained nominal height instead of
        # allowing noisy VLA height samples to pump the lower body.
        supervised[31] = NOMINAL_BASE_HEIGHT_M

        if decision.walking or decision.phase in (
            FetchPhase.SEARCH,
            FetchPhase.RECOVER,
            FetchPhase.COMPLETE,
            FetchPhase.VERIFY_GRASP,
            FetchPhase.VERIFY_DELIVERY,
        ):
            # Base-relative holds keep both arms quiet while locomotion moves the base.
            supervised[0:7] = neutral_left
            supervised[7:14] = neutral_right
            supervised[14:28] = neutral_fingers
            return supervised

        # Small fruit is a one-hand task.  GR00T controls the hand nearest the
        # target; the other wrist and fingers return toward the warm-up posture.
        # The active wrist is additionally bound to the selected physical
        # target so repeated predictions cannot accumulate into a "clown pose".
        offsets = PLACE_TARGET_OFFSETS_M if decision.phase is FetchPhase.PLACE else GRASP_TARGET_OFFSETS_M
        corridor = decision.manipulation_target_base[:, None] + offsets
        if decision.active_hand == "left":
            supervised[0:3] = np.clip(supervised[0:3], corridor[:, 0], corridor[:, 1])
            supervised[7:14] = neutral_right
            supervised[14 + PINK_RIGHT_HAND_INDICES] = neutral_fingers[PINK_RIGHT_HAND_INDICES]
        elif decision.active_hand == "right":
            supervised[7:10] = np.clip(supervised[7:10], corridor[:, 0], corridor[:, 1])
            supervised[0:7] = neutral_left
            supervised[14 + PINK_LEFT_HAND_INDICES] = neutral_fingers[PINK_LEFT_HAND_INDICES]
        else:
            raise ValueError(f"unknown active hand: {decision.active_hand}")
        return supervised
