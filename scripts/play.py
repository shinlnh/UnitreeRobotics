"""Unified checkpoint playback and automatic JIT/ONNX export entry point."""

import warp as wp

wp.config.enable_backward = False

import unitree_rl_groot.tasks  # noqa: E402, F401
from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402


def main() -> int:
    return run_play_cli()


if __name__ == "__main__":
    raise SystemExit(main())
