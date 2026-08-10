from __future__ import annotations

import torch
from unitree_rl_groot.groot.isaac_runtime import _inference_actor_state


def test_inference_actor_state_converts_legacy_standard_deviation() -> None:
    payload = {
        "actor_state_dict": {
            "mlp.0.weight": torch.ones(2, 2),
            "distribution.std_param": torch.tensor([0.8, 0.01, float("inf"), float("nan")]),
        }
    }

    state = _inference_actor_state(payload)

    assert "distribution.std_param" not in state
    torch.testing.assert_close(
        state["distribution.log_std_param"].exp(),
        torch.tensor([0.8, 0.05, 2.0, 0.8]),
    )
    assert state["mlp.0.weight"] is payload["actor_state_dict"]["mlp.0.weight"]
    assert "distribution.std_param" in payload["actor_state_dict"]


def test_inference_actor_state_preserves_current_checkpoint() -> None:
    expected = torch.tensor([-0.2, 0.1])
    state = _inference_actor_state({"actor_state_dict": {"distribution.log_std_param": expected}})
    assert state["distribution.log_std_param"] is expected
