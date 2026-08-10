"""Strict environment-output validation for RSL-RL training."""

from __future__ import annotations

import torch
from tensordict import TensorDict


def check_finite_env_output(obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor) -> None:
    """Fail immediately on NaN or infinity before it can poison PPO state."""

    for key, tensor in obs.items():
        invalid = ~torch.isfinite(tensor)
        if invalid.any():
            raise ValueError(
                f"The observation group {key!r} contains {int(invalid.sum().item())} non-finite values. "
                "Check physics GPU capacities and environment reset logic."
            )
    for name, tensor in (("rewards", rewards), ("dones", dones)):
        invalid = ~torch.isfinite(tensor)
        if invalid.any():
            raise ValueError(f"The environment {name} contain {int(invalid.sum().item())} non-finite values.")


def install_rsl_rl_finite_check() -> None:
    """Replace RSL-RL's NaN-only rollout guard with the stricter guard."""

    from rsl_rl.runners import on_policy_runner

    on_policy_runner.check_nan = check_finite_env_output
