from __future__ import annotations

import torch
from tensordict import TensorDict
from unitree_rl_groot.rl import BoundedGaussianDistribution, check_finite_env_output


def test_bounded_gaussian_repairs_parameter_and_gradient() -> None:
    distribution = BoundedGaussianDistribution(3, init_std=0.8)
    distribution.log_std_param.data[:] = torch.tensor([float("nan"), float("inf"), float("-inf")])
    distribution.update(torch.zeros((2, 3)))

    assert torch.isfinite(distribution.log_std_param).all()
    assert torch.all(distribution.std >= 0.0499)
    assert torch.all(distribution.std <= 2.0001)

    (distribution.log_std_param * torch.tensor([float("nan"), float("inf"), 1.0])).sum().backward()
    assert torch.isfinite(distribution.log_std_param.grad).all()


def test_environment_guard_rejects_infinity() -> None:
    observations = TensorDict({"policy": torch.tensor([[0.0, float("inf")]])}, batch_size=[1])

    try:
        check_finite_env_output(observations, torch.zeros(1), torch.zeros(1, dtype=torch.bool))
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("infinite observation was accepted")
