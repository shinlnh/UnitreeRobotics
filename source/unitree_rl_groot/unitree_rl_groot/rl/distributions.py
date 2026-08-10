"""Numerically robust action distributions for long-running PPO jobs."""

from __future__ import annotations

import math

import torch
from rsl_rl.modules.distribution import GaussianDistribution


class BoundedGaussianDistribution(GaussianDistribution):
    """Gaussian with finite gradients and an explicit deployable std range.

    RSL-RL's Gaussian clamps the value used for sampling, but a single non-finite
    optimizer update can still poison its learnable parameter. Rough-terrain PPO
    occasionally produces such an update during the rapid early curriculum. This
    subclass sanitizes the gradient before Adam sees it and repairs/clamps the
    parameter before every distribution update.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (0.05, 2.0),
        std_type: str = "log",
        learn_std: bool = True,
    ) -> None:
        super().__init__(
            output_dim,
            init_std=init_std,
            std_range=std_range,
            std_type=std_type,
            learn_std=learn_std,
        )
        if std_type == "log":
            self._fallback_param = math.log(init_std)
            self._parameter_bounds = tuple(math.log(value) for value in std_range)
            parameter = self.log_std_param
        else:
            self._fallback_param = init_std
            self._parameter_bounds = std_range
            parameter = self.std_param
        if learn_std:
            parameter.register_hook(self._finite_gradient)

    @staticmethod
    def _finite_gradient(gradient: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

    def _repair_parameter(self) -> None:
        parameter = self.log_std_param if self.std_type == "log" else self.std_param
        lower, upper = self._parameter_bounds
        with torch.no_grad():
            parameter.nan_to_num_(nan=self._fallback_param, posinf=upper, neginf=lower)
            parameter.clamp_(lower, upper)

    def update(self, mlp_output: torch.Tensor) -> None:
        self._repair_parameter()
        super().update(mlp_output)
