"""List Gym tasks registered by the project."""

import gymnasium as gym
import unitree_rl_groot.tasks  # noqa: F401

PREFIX = "Unitree-G1-"


def main() -> int:
    task_ids = sorted(spec.id for spec in gym.registry.values() if spec.id.startswith(PREFIX))
    if not task_ids:
        print(f"No tasks with prefix {PREFIX!r} were registered.")
        return 1
    for task_id in task_ids:
        print(task_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
