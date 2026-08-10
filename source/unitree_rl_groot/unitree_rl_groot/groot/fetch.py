"""Long-horizon fetch mission contract shared by collection and deployment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


def _label(value: str, name: str) -> str:
    result = " ".join(str(value).split())
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


class FetchPhase(str, Enum):  # noqa: UP042 -- SONIC runtime is Python 3.10
    """Observable phases of a fetch-and-deliver mission."""

    SEARCH = "search"
    NAVIGATE_TO_OBJECT = "navigate_to_object"
    ALIGN_FOR_GRASP = "align_for_grasp"
    GRASP = "grasp"
    VERIFY_GRASP = "verify_grasp"
    CARRY_TO_DESTINATION = "carry_to_destination"
    PLACE = "place"
    VERIFY_DELIVERY = "verify_delivery"
    RECOVER = "recover"
    COMPLETE = "complete"


class FetchEvent(str, Enum):  # noqa: UP042 -- SONIC runtime is Python 3.10
    """Perception/controller evidence accepted by :class:`FetchExecutive`."""

    OBJECT_FOUND = "object_found"
    PICKUP_POSE_REACHED = "pickup_pose_reached"
    PREGRASP_ALIGNED = "pregrasp_aligned"
    GRASP_CLOSED = "grasp_closed"
    OBJECT_LIFTED = "object_lifted"
    DESTINATION_REACHED = "destination_reached"
    OBJECT_RELEASED = "object_released"
    DELIVERY_CONFIRMED = "delivery_confirmed"
    FAILURE = "failure"
    RETRY_READY = "retry_ready"


@dataclass(frozen=True)
class FetchMission:
    """Semantic fetch request independent of a particular controller."""

    object_name: str
    source: str
    destination: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_name", _label(self.object_name, "object_name"))
        object.__setattr__(self, "source", _label(self.source, "source"))
        object.__setattr__(self, "destination", _label(self.destination, "destination"))

    @property
    def instruction(self) -> str:
        return (
            f"Find the {self.object_name} at the {self.source}, pick it up, "
            f"carry it to the {self.destination}, place it safely, and verify delivery."
        )

    def phase_prompt(self, phase: FetchPhase) -> str:
        prompts = {
            FetchPhase.SEARCH: f"Search carefully for the {self.object_name} at the {self.source}.",
            FetchPhase.NAVIGATE_TO_OBJECT: (
                f"Walk safely toward the {self.object_name} at the {self.source} and stop at grasping distance."
            ),
            FetchPhase.ALIGN_FOR_GRASP: (
                f"Align the torso and hands with the {self.object_name} without touching nearby objects."
            ),
            FetchPhase.GRASP: f"Grasp the {self.object_name} securely with an appropriate hand.",
            FetchPhase.VERIFY_GRASP: (
                f"Lift the {self.object_name} slightly and verify that the grasp is stable."
            ),
            FetchPhase.CARRY_TO_DESTINATION: (
                f"Carry the {self.object_name} safely to the {self.destination} while keeping a stable grasp."
            ),
            FetchPhase.PLACE: f"Place the {self.object_name} gently at the {self.destination} and release it.",
            FetchPhase.VERIFY_DELIVERY: (
                f"Verify that the {self.object_name} is stable inside or on the {self.destination}."
            ),
            FetchPhase.RECOVER: (
                "Stop safely, regain a stable posture, open the hands if necessary, and prepare to retry."
            ),
            FetchPhase.COMPLETE: "Stand still in a safe neutral posture; the fetch mission is complete.",
        }
        return prompts[phase]


class FetchExecutive:
    """Deterministic safety shell around learned whole-body fetch skills."""

    _TRANSITIONS = {
        (FetchPhase.SEARCH, FetchEvent.OBJECT_FOUND): FetchPhase.NAVIGATE_TO_OBJECT,
        (FetchPhase.NAVIGATE_TO_OBJECT, FetchEvent.PICKUP_POSE_REACHED): FetchPhase.ALIGN_FOR_GRASP,
        (FetchPhase.ALIGN_FOR_GRASP, FetchEvent.PREGRASP_ALIGNED): FetchPhase.GRASP,
        (FetchPhase.GRASP, FetchEvent.GRASP_CLOSED): FetchPhase.VERIFY_GRASP,
        (FetchPhase.VERIFY_GRASP, FetchEvent.OBJECT_LIFTED): FetchPhase.CARRY_TO_DESTINATION,
        (FetchPhase.CARRY_TO_DESTINATION, FetchEvent.DESTINATION_REACHED): FetchPhase.PLACE,
        (FetchPhase.PLACE, FetchEvent.OBJECT_RELEASED): FetchPhase.VERIFY_DELIVERY,
        (FetchPhase.VERIFY_DELIVERY, FetchEvent.DELIVERY_CONFIRMED): FetchPhase.COMPLETE,
    }

    def __init__(self, mission: FetchMission, *, max_retries: int = 3) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.mission = mission
        self.max_retries = max_retries
        self.phase = FetchPhase.SEARCH
        self.retries = 0
        self.failed = False

    @property
    def prompt(self) -> str:
        return self.mission.phase_prompt(self.phase)

    @property
    def done(self) -> bool:
        return self.phase is FetchPhase.COMPLETE or self.failed

    def advance(self, event: FetchEvent | str) -> FetchPhase:
        event = FetchEvent(event)
        if self.done:
            raise RuntimeError("fetch mission has already terminated")
        if event is FetchEvent.FAILURE:
            self.retries += 1
            if self.retries > self.max_retries:
                self.failed = True
                return self.phase
            self.phase = FetchPhase.RECOVER
            return self.phase
        if self.phase is FetchPhase.RECOVER:
            if event is not FetchEvent.RETRY_READY:
                raise ValueError("recovery only accepts retry_ready")
            self.phase = FetchPhase.SEARCH
            return self.phase
        try:
            self.phase = self._TRANSITIONS[(self.phase, event)]
        except KeyError as exc:
            raise ValueError(f"event {event.value} is invalid during phase {self.phase.value}") from exc
        return self.phase
