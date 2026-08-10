"""Language-conditioned task catalogue for the GR00T G1 fruit policy.

The catalogue deliberately mirrors the four commands in NVIDIA's
``PhysicalAI-Robotics-GR00T-Teleop-G1`` dataset.  A single policy is trained
on every entry; these are not four independently selected checkpoints.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def _ascii_words(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", " ".join(value.lower().split()))
    return " ".join(re.findall(r"[a-z0-9]+", normalized.encode("ascii", "ignore").decode()))


@dataclass(frozen=True)
class G1FruitTask:
    """One language-conditioned branch of the shared G1 policy."""

    object_name: str
    dataset_directory: str
    prompt: str
    aliases: tuple[str, ...]

    @property
    def all_aliases(self) -> tuple[str, ...]:
        return (self.object_name, *self.aliases)


G1_FRUIT_TASKS: tuple[G1FruitTask, ...] = (
    G1FruitTask(
        object_name="apple",
        dataset_directory="g1-pick-apple",
        prompt="Pick up the red apple and place it on the plate",
        aliases=("red apple", "tao", "tao do", "quả táo", "quả táo đỏ"),
    ),
    G1FruitTask(
        object_name="pear",
        dataset_directory="g1-pick-pear",
        prompt="Pick up the yellow pear and place it on the plate",
        aliases=("yellow pear", "le", "le vang", "quả lê", "quả lê vàng"),
    ),
    G1FruitTask(
        object_name="grapes",
        dataset_directory="g1-pick-grapes",
        prompt="Pick up the green grapes and place it on the plate",
        aliases=("grape", "green grape", "green grapes", "nho", "nho xanh", "chùm nho xanh"),
    ),
    G1FruitTask(
        object_name="starfruit",
        dataset_directory="g1-pick-starfruit",
        prompt="Pick up the yellow starfruit and place it on the plate",
        aliases=("star fruit", "yellow starfruit", "khe", "khe vang", "quả khế", "quả khế vàng"),
    ),
)

TASK_BY_OBJECT = {task.object_name: task for task in G1_FRUIT_TASKS}


def resolve_fruit_task(instruction: str) -> G1FruitTask:
    """Resolve the physical target in a free-form English or Vietnamese command.

    The original instruction is still sent to GR00T.  Resolution only binds the
    simulator success condition to exactly one object and rejects unsafe,
    ambiguous commands before any robot motion begins.
    """

    words = _ascii_words(instruction)
    if not words:
        raise ValueError("instruction must not be empty")
    padded = f" {words} "
    matches: list[G1FruitTask] = []
    for task in G1_FRUIT_TASKS:
        aliases = {_ascii_words(alias) for alias in task.all_aliases}
        if any(f" {alias} " in padded for alias in aliases):
            matches.append(task)
    if not matches:
        supported = ", ".join(TASK_BY_OBJECT)
        raise ValueError(f"instruction does not name a supported object ({supported})")
    if len(matches) > 1:
        names = ", ".join(task.object_name for task in matches)
        raise ValueError(f"instruction is ambiguous; it names multiple objects: {names}")
    return matches[0]


def canonical_prompt(object_name: str) -> str:
    """Return the exact training prompt for a canonical object name."""

    try:
        return TASK_BY_OBJECT[_ascii_words(object_name)].prompt
    except KeyError as exc:
        raise ValueError(f"unsupported object: {object_name}") from exc
