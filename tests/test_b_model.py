import pytest

from unitree_gr00t.b_model import SelectorModelConfig, build_selector, selector_loss

torch = pytest.importorskip("torch")


def test_selector_scores_unified_h_plus_one_candidates() -> None:
    config = SelectorModelConfig(
        action_horizon=4,
        context_width=8,
        scoring_width=16,
        scoring_layers=1,
        scoring_heads=4,
        feedforward_width=32,
    )
    model = build_selector(config)
    scores = model(
        torch.zeros(2, 4, 7),
        torch.zeros(2, 8),
        torch.zeros(2, 3, 8),
        torch.tensor([[True, False, False], [True, True, True]]),
        torch.tensor([[True, True, True, False, False], [True] * 5]),
    )
    assert scores.shape == (2, 5)
    assert scores[0, 3] == torch.finfo(scores.dtype).min


def test_selector_loss_is_finite_and_backpropagates() -> None:
    scores = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.2, 0.1]], requires_grad=True)
    loss, metrics = selector_loss(
        scores,
        priorities=torch.tensor([[0, 1, 2], [3, 2, 1]]),
        valid=torch.ones(2, 3, dtype=torch.bool),
        stop_labels=torch.tensor([0, 1]),
        rank_weights=torch.tensor([1.0, 1.0]),
        stop_weights=torch.tensor([1.0, 3.0]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert scores.grad is not None
    assert metrics["pairs"].item() == 6
