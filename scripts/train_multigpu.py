"""Multi-GPU training launcher."""

import sys

import unitree_rl_groot.tasks  # noqa: F401
from isaaclab_rl.entrypoints import run_train_multigpu_cli

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--external_callback" not in arguments:
        arguments = [
            "--external_callback",
            "unitree_rl_groot.tasks.register_tasks",
            *arguments,
        ]
    raise SystemExit(run_train_multigpu_cli(arguments))
