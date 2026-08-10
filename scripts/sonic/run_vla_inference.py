#!/usr/bin/env python3
"""Run upstream SONIC inference with the project's lightweight GR00T client."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from types import ModuleType

from policy_client import PolicyClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SONIC_ROOT = Path(os.environ.get("SONIC_ROOT", PROJECT_ROOT / ".deps/GR00T-WholeBodyControl"))
UPSTREAM_RUNNER = SONIC_ROOT / "gear_sonic/scripts/run_vla_inference.py"


def main() -> None:
    if not UPSTREAM_RUNNER.is_file():
        raise FileNotFoundError(UPSTREAM_RUNNER)
    gr00t_module = ModuleType("gr00t")
    policy_module = ModuleType("gr00t.policy")
    client_module = ModuleType("gr00t.policy.server_client")
    client_module.PolicyClient = PolicyClient
    gr00t_module.policy = policy_module
    policy_module.server_client = client_module
    sys.modules["gr00t"] = gr00t_module
    sys.modules["gr00t.policy"] = policy_module
    sys.modules["gr00t.policy.server_client"] = client_module
    runpy.run_path(str(UPSTREAM_RUNNER), run_name="__main__")


if __name__ == "__main__":
    main()
