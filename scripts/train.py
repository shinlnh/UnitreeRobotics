"""Unified Isaac Lab 3.0 training entry point for this external project."""

import warp as wp

wp.config.enable_backward = False

import unitree_rl_groot.tasks  # noqa: E402, F401
from isaaclab_rl.entrypoints import run_train_cli  # noqa: E402


def main() -> int:
    return run_train_cli()


if __name__ == "__main__":
    from torch.distributed.elastic.multiprocessing.errors import record

    raise SystemExit(record(main)())
