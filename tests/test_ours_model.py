import pytest

from unitree_gr00t.ours import OursContractError
from unitree_gr00t.ours_model import (
    TemporalRecoveryModelConfig,
    build_temporal_recovery_model,
    completion_progress_loss,
    count_trainable_parameters,
)

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("encoder", ["linear", "mlp", "gru", "transformer"])
def test_temporal_model_shapes_and_parameter_budget(encoder: str) -> None:
    config = TemporalRecoveryModelConfig(
        encoder=encoder,
        history_length=4,
        context_width=8,
        action_horizon=3,
        action_dim=2,
        selector_candidates=4,
        scalar_width=6,
        temporal_width=16,
        temporal_layers=1,
        temporal_heads=4,
        feedforward_width=32,
    )
    model = build_temporal_recovery_model(config)
    output = model(
        torch.zeros(2, 4, 8),
        torch.zeros(2, 4, 8),
        torch.zeros(2, 4, 3, 2),
        torch.zeros(2, 4, 8),
        torch.zeros(2, 4, 6),
    )
    assert output["completion_logit"].shape == (2,)
    assert output["option_values"].shape == (2, 6)
    assert count_trainable_parameters(model) > 0
    loss, metrics = completion_progress_loss(
        output,
        torch.tensor([False, True]),
        torch.tensor([0.2, 1.0]),
        target_option_values=torch.tensor(
            [[0.0, 16.0, 0.0, 0.0, 0.0, 0.0], [16.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
        target_option_valid=torch.tensor(
            [[True, True, False, False, False, False]] * 2
        ),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["completion_loss"] >= 0
    assert metrics["option_value_loss"] > 0
    assert metrics["option_rank_loss"] >= 0
    assert metrics["option_classification_loss"] > 0


def test_temporal_model_rejects_unregistered_encoder() -> None:
    with pytest.raises(OursContractError, match="unsupported"):
        build_temporal_recovery_model(TemporalRecoveryModelConfig(encoder="rnn"))


def test_temporal_model_rejects_invalid_dropout() -> None:
    with pytest.raises(OursContractError, match="dimensions"):
        build_temporal_recovery_model(TemporalRecoveryModelConfig(dropout=1.0))
