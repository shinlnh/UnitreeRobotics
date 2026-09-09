import pytest

from unitree_gr00t.resolve_pilot_train import (
    PilotModelConfig,
    build_program_scorer,
    causal_program_view,
    crb_from_world_probabilities,
    intervention_sequence_worlds,
    intervention_worlds,
)

torch = pytest.importorskip("torch")


def test_intervention_worlds_encode_ordered_deletions_and_handoffs() -> None:
    residuals = torch.ones(1, 2, 3, 2)
    codes = torch.ones(1, 2, 4)
    residual_worlds, code_worlds = intervention_worlds(residuals, codes, torch=torch)
    assert residual_worlds.shape == (1, 5, 2, 3, 2)
    assert code_worlds.shape == (1, 5, 2, 4)
    assert residual_worlds[0, 1, 0].count_nonzero() == 0
    assert residual_worlds[0, 1, 1].count_nonzero() > 0
    assert residual_worlds[0, 3].count_nonzero() == 0
    assert residual_worlds[0, 4, 0].count_nonzero() > 0
    assert residual_worlds[0, 4, 1].count_nonzero() == 0
    offsets = intervention_sequence_worlds(torch.tensor([[-2.0, 1.0]]), torch=torch)
    assert offsets[:, :, :].tolist() == [
        [[-2.0, 1.0], [0.0, 1.0], [-2.0, 0.0], [0.0, 0.0], [-2.0, 0.0]]
    ]


def test_program_scorer_shares_one_network_across_all_worlds() -> None:
    config = PilotModelConfig(
        program_depth=2,
        context_width=12,
        action_horizon=3,
        action_dim=2,
        code_dimension=4,
        model_width=24,
        transformer_layers=2,
        transformer_heads=4,
        feedforward_width=48,
    )
    model = build_program_scorer(config)
    logits = model(
        torch.zeros(5, 2, 12),
        torch.zeros(5, 2, 3, 2),
        torch.zeros(5, 2, 3, 2),
        torch.zeros(5, 2, 4),
        torch.ones(5, 2),
        torch.zeros(5, 2),
    )
    assert logits.shape == (5,)
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_program_crb_uses_the_worst_deletion_or_baseline() -> None:
    probabilities = torch.tensor([[0.9, 0.2, 0.8, 0.3, 0.4]])
    assert crb_from_world_probabilities(probabilities).item() == pytest.approx(0.1)


def test_causal_program_view_removes_all_future_observations() -> None:
    contexts = torch.randn(2, 3, 12)
    base = torch.randn(2, 3, 4, 2)
    residual = torch.randn(2, 3, 4, 2)
    changed_contexts = contexts.clone()
    changed_base = base.clone()
    changed_residual = residual.clone()
    changed_contexts[:, 1:] += 1000
    changed_base[:, 1:] -= 1000
    changed_residual[:, 1:] *= -1000
    first = causal_program_view(contexts, base, residual, torch=torch)
    second = causal_program_view(changed_contexts, changed_base, changed_residual, torch=torch)
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
    assert first[2][:, 1:].count_nonzero() == 0
