"""Random-action smoke test for stepping and reset behavior."""

import unitree_rl_groot.tasks  # noqa: F401
from isaaclab_rl.entrypoints import run_random_agent_cli

if __name__ == "__main__":
    raise SystemExit(run_random_agent_cli())
