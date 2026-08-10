"""Zero-action smoke test for environment initialization."""

import unitree_rl_groot.tasks  # noqa: F401
from isaaclab_rl.entrypoints import run_zero_agent_cli

if __name__ == "__main__":
    raise SystemExit(run_zero_agent_cli())
