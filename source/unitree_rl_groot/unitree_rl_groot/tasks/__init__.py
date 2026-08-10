"""Task package and Gym registration entry point."""

import sys

from isaaclab_tasks.utils import import_packages

from unitree_rl_groot.rl import install_rsl_rl_finite_check

install_rsl_rl_finite_check()
import_packages(__name__, ["utils", ".mdp"])


def register_tasks() -> list[str]:
    """External-entrypoint callback used by spawned multi-GPU workers."""

    return sys.argv[1:]
