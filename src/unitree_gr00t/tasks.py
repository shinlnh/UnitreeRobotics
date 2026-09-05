"""Language-conditioned task catalog used by smoke tests and evaluation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

VALID_KINDS = {"pick", "place", "stack"}


@dataclass(frozen=True)
class TaskSpec:
    id: str
    kind: str
    target: str
    instruction: str
    destination: str | None = None


def load_tasks(path: str | Path = "configs/tasks.toml") -> list[TaskSpec]:
    task_path = Path(path).expanduser().resolve()
    if not task_path.is_file():
        raise FileNotFoundError(f"Task catalog not found: {task_path}")
    with task_path.open("rb") as handle:
        raw = tomllib.load(handle)

    tasks: list[TaskSpec] = []
    seen: set[str] = set()
    for item in raw.get("tasks", []):
        task = TaskSpec(
            id=str(item["id"]),
            kind=str(item["kind"]),
            target=str(item["target"]),
            destination=str(item["destination"]) if item.get("destination") else None,
            instruction=str(item["instruction"]),
        )
        if task.id in seen:
            raise ValueError(f"Duplicate task id: {task.id}")
        if task.kind not in VALID_KINDS:
            raise ValueError(f"Unsupported task kind {task.kind!r} for {task.id}")
        if task.kind in {"place", "stack"} and not task.destination:
            raise ValueError(f"Task {task.id} requires a destination")
        seen.add(task.id)
        tasks.append(task)

    if not tasks:
        raise ValueError(f"No tasks defined in {task_path}")
    return tasks


def select_tasks(tasks: list[TaskSpec], requested: str | None) -> list[TaskSpec]:
    if not requested or requested == "all":
        return tasks
    ids = [value.strip() for value in requested.split(",") if value.strip()]
    index = {task.id: task for task in tasks}
    missing = [task_id for task_id in ids if task_id not in index]
    if missing:
        raise ValueError(f"Unknown task id(s): {', '.join(missing)}")
    return [index[task_id] for task_id in ids]
