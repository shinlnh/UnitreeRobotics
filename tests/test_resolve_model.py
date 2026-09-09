import pytest

from unitree_gr00t.ours import OursContractError
from unitree_gr00t.resolve_model import (
    ResolveModelConfig,
    build_reachability_critics,
    build_recovery_actor,
    conservative_all_deletion_advantage,
    conservative_crb_from_logits,
    count_trainable_parameters,
    reachability_critic_loss,
    sample_recovery_action,
)

torch = pytest.importorskip("torch")


def small_config() -> ResolveModelConfig:
    return ResolveModelConfig(
        history_length=4,
        context_width=12,
        action_horizon=3,
        action_dim=2,
        scalar_width=5,
        model_width=16,
        transformer_layers=2,
        transformer_heads=4,
        feedforward_width=32,
        latent_codes=4,
        milestone_count=3,
        critic_ensemble=2,
        cost_count=3,
        maximum_residual_scale=4.0,
        gripper_index=None,
    )


def inputs(config: ResolveModelConfig, batch: int = 2):
    return (
        torch.zeros(batch, config.history_length, config.context_width),
        torch.zeros(
            batch,
            config.history_length,
            config.action_horizon,
            config.action_dim,
        ),
        torch.zeros(batch, config.history_length, config.scalar_width),
        torch.zeros(batch, config.history_length, dtype=torch.long),
        torch.arange(config.history_length, dtype=torch.long)[None].expand(batch, -1),
        torch.tensor([0, 2], dtype=torch.long),
    )


def test_recovery_actor_emits_a_new_bounded_action_and_latent_program() -> None:
    config = small_config()
    actor = build_recovery_actor(config)
    actor_outputs = actor(*inputs(config))
    assert actor_outputs["latent_logits"].shape == (2, config.latent_codes + 1)
    assert actor_outputs["residual_mean"].shape == (
        2,
        config.action_horizon,
        config.action_dim,
    )
    sampled = sample_recovery_action(
        actor_outputs,
        torch.zeros(2, config.action_horizon, config.action_dim),
        deterministic=True,
    )
    assert sampled["corrected_action"].shape == (
        2,
        config.action_horizon,
        config.action_dim,
    )
    assert sampled["corrected_action"].abs().max() <= 1.0
    assert sampled["handoff"].shape == (2,)
    assert sampled["handoff"].all()
    assert count_trainable_parameters(actor) > 0


def test_zero_logit_residual_is_exact_executable_baseline() -> None:
    config = small_config()
    actor = build_recovery_actor(config)
    outputs = actor(*inputs(config))
    assert torch.count_nonzero(outputs["residual_mean"]) == 0
    base = torch.tensor(
        [[[2.0, -2.0], [0.25, -0.5], [0.0, 0.75]]] * 2,
        dtype=torch.float32,
    )
    sampled = sample_recovery_action(outputs, base, deterministic=True)
    assert torch.allclose(sampled["corrected_action"], base.clamp(-1.0, 1.0))
    assert torch.allclose(sampled["residual_action"], torch.zeros_like(base))
    assert torch.isfinite(sampled["action_log_probability"]).all()
    sampled["corrected_action"].sum().backward()
    assert actor.residual_mean_head.weight.grad.abs().sum() > 0


def test_gripper_uses_zero_one_action_domain() -> None:
    config = ResolveModelConfig(
        history_length=2,
        context_width=8,
        action_horizon=2,
        action_dim=3,
        scalar_width=2,
        model_width=8,
        transformer_layers=1,
        transformer_heads=2,
        feedforward_width=16,
        latent_codes=2,
        milestone_count=2,
        critic_ensemble=2,
        cost_count=2,
        maximum_residual_scale=4.0,
        gripper_index=2,
    )
    actor = build_recovery_actor(config)
    args = (
        torch.zeros(1, 2, 8),
        torch.zeros(1, 2, 2, 3),
        torch.zeros(1, 2, 2),
        torch.zeros(1, 2, dtype=torch.long),
        torch.zeros(1, 2, dtype=torch.long),
        torch.zeros(1, dtype=torch.long),
    )
    outputs = actor(*args)
    outputs["residual_mean"] = torch.zeros_like(outputs["residual_mean"])
    outputs["residual_trust"] = torch.ones_like(outputs["residual_trust"])
    base = torch.tensor([[[2.0, -2.0, 1.5], [0.0, 0.0, -0.5]]])
    sampled = sample_recovery_action(outputs, base, deterministic=True)
    assert torch.allclose(
        sampled["corrected_action"],
        torch.tensor([[[1.0, -1.0, 1.0], [0.0, 0.0, 0.0]]]),
    )
    assert sampled["corrected_action"][..., 2].min() >= 0.0


def test_twin_critics_keep_recovery_deletion_and_baseline_separate() -> None:
    config = small_config()
    critics = build_reachability_critics(config)
    common = inputs(config)
    outputs = critics(
        *common,
        torch.zeros(2, config.action_horizon, config.action_dim),
        torch.tensor([1, 2], dtype=torch.long),
    )
    assert outputs["reachability_logits"].shape == (
        config.critic_ensemble,
        2,
        3,
        config.milestone_count,
    )
    assert outputs["cost_predictions"].shape == (
        config.critic_ensemble,
        2,
        config.cost_count,
    )
    assert outputs["necessity_values"].shape == (
        config.critic_ensemble,
        2,
        config.milestone_count,
    )
    # D is the exact-zero residual through the shared recovery continuation
    # head, so it cannot drift into an independently renamed critic.
    assert torch.equal(
        outputs["reachability_logits"][:, :, 0],
        outputs["reachability_logits"][:, :, 1],
    )
    loss, metrics = reachability_critic_loss(
        outputs,
        torch.zeros(2, 3, config.milestone_count),
        torch.zeros(2, config.cost_count),
        necessity_targets=torch.zeros(2, config.milestone_count),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["reachability_loss"] > 0
    assert count_trainable_parameters(critics) > count_trainable_parameters(
        build_recovery_actor(config)
    )


def test_conservative_crb_uses_recovery_lower_and_counterfactual_upper() -> None:
    probabilities = torch.tensor(
        [
            [[[0.9], [0.2], [0.4]]],
            [[[0.8], [0.3], [0.35]]],
        ]
    )
    outputs = {"reachability_logits": torch.logit(probabilities)}
    crb = conservative_crb_from_logits(outputs, torch.tensor([0]))
    assert crb.item() == pytest.approx(0.4)


def test_all_deletion_actor_advantage_requires_superiority_and_every_macro() -> None:
    probabilities = torch.tensor(
        [
            [[[0.9], [0.2], [0.4]], [[0.8], [0.2], [0.3]]],
            [[[0.8], [0.3], [0.35]], [[0.9], [0.2], [0.4]]],
        ]
    )
    outputs = {
        "reachability_logits": torch.logit(probabilities),
        "necessity_values": torch.tensor([[[0.25], [-0.10]], [[0.20], [0.05]]]),
    }
    advantage = conservative_all_deletion_advantage(outputs, torch.tensor([0, 0]))
    assert advantage.tolist() == pytest.approx([0.2, -0.1])


def test_invalid_recovery_model_configuration_is_rejected() -> None:
    with pytest.raises(OursContractError, match="dimensions"):
        build_recovery_actor(ResolveModelConfig(model_width=15, transformer_heads=8))
