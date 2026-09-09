"""Recovery transformer and R/D/B reachability critics for RESOLVE-VLA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .ours import OursContractError


@dataclass(frozen=True)
class ResolveModelConfig:
    history_length: int = 8
    context_width: int = 2048
    action_horizon: int = 16
    action_dim: int = 7
    scalar_width: int = 6
    model_width: int = 512
    transformer_layers: int = 8
    transformer_heads: int = 8
    feedforward_width: int = 2048
    latent_codes: int = 16
    maximum_program_depth: int = 8
    milestone_count: int = 16
    critic_ensemble: int = 2
    cost_count: int = 3
    dropout: float = 0.0
    maximum_residual_scale: float = 0.25
    minimum_log_scale: float = -5.0
    maximum_log_scale: float = 0.5

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("RESOLVE recovery models require PyTorch") from exc
    return torch


def _validate(config: ResolveModelConfig) -> None:
    if (
        min(
            config.history_length,
            config.context_width,
            config.action_horizon,
            config.action_dim,
            config.scalar_width,
            config.model_width,
            config.transformer_layers,
            config.transformer_heads,
            config.feedforward_width,
            config.latent_codes,
            config.maximum_program_depth,
            config.milestone_count,
            config.critic_ensemble,
            config.cost_count,
        )
        < 1
        or config.model_width % config.transformer_heads
        or config.critic_ensemble < 2
        or not 0.0 <= config.dropout < 1.0
        or not 0.0 < config.maximum_residual_scale <= 1.0
        or not config.minimum_log_scale < config.maximum_log_scale
    ):
        raise OursContractError("RESOLVE model dimensions are invalid")


def build_recovery_actor(config: ResolveModelConfig) -> Any:
    """Build a causal transformer that emits latent programs and action chunks."""

    _validate(config)
    torch = _torch()
    nn = torch.nn

    class RecoveryActor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = config.model_width
            self.config = config
            self.context_projection = nn.Sequential(
                nn.LayerNorm(config.context_width), nn.Linear(config.context_width, width)
            )
            self.action_projection = nn.Sequential(
                nn.LayerNorm(config.action_horizon * config.action_dim),
                nn.Linear(config.action_horizon * config.action_dim, width),
            )
            self.scalar_projection = nn.Sequential(
                nn.LayerNorm(config.scalar_width), nn.Linear(config.scalar_width, width)
            )
            # One extra state is the pre-recovery / no-program state.
            self.latent_embedding = nn.Embedding(config.latent_codes + 1, width)
            self.program_position_embedding = nn.Embedding(
                config.maximum_program_depth, width
            )
            self.milestone_embedding = nn.Embedding(config.milestone_count, width)
            self.position_embedding = nn.Embedding(config.history_length, width)
            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=config.transformer_heads,
                dim_feedforward=config.feedforward_width,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=config.transformer_layers)
            self.output_norm = nn.LayerNorm(width)
            # The extra categorical output is HANDOFF to exact B.
            self.latent_head = nn.Linear(width, config.latent_codes + 1)
            action_width = config.action_horizon * config.action_dim
            self.residual_mean_head = nn.Linear(width, action_width)
            self.residual_log_scale_head = nn.Linear(width, action_width)
            self.residual_trust_head = nn.Linear(width, action_width)

        def forward(
            self,
            contexts: Any,
            base_action_chunks: Any,
            scalars: Any,
            latent_history: Any,
            program_positions: Any,
            milestone_ids: Any,
        ) -> dict[str, Any]:
            batch, history, width = contexts.shape
            if (
                (batch, history, width)
                != (batch, config.history_length, config.context_width)
                or base_action_chunks.shape
                != (batch, history, config.action_horizon, config.action_dim)
                or scalars.shape != (batch, history, config.scalar_width)
                or latent_history.shape != (batch, history)
                or program_positions.shape != (batch, history)
                or milestone_ids.shape != (batch,)
                or latent_history.dtype not in (torch.int32, torch.int64)
                or program_positions.dtype not in (torch.int32, torch.int64)
                or milestone_ids.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("RESOLVE actor input shapes are invalid")
            positions = torch.arange(history, device=contexts.device)
            tokens = (
                self.context_projection(contexts)
                + self.action_projection(base_action_chunks.flatten(start_dim=2))
                + self.scalar_projection(scalars)
                + self.latent_embedding(latent_history)
                + self.program_position_embedding(program_positions)
                + self.milestone_embedding(milestone_ids)[:, None, :]
                + self.position_embedding(positions)[None, :, :]
            )
            causal_mask = torch.triu(
                torch.ones(history, history, dtype=torch.bool, device=tokens.device), diagonal=1
            )
            state = self.output_norm(self.temporal(tokens, mask=causal_mask)[:, -1])
            mean = self.residual_mean_head(state).view(
                batch, config.action_horizon, config.action_dim
            )
            log_scale = self.residual_log_scale_head(state).view_as(mean).clamp(
                config.minimum_log_scale, config.maximum_log_scale
            )
            trust = config.maximum_residual_scale * self.residual_trust_head(state).view_as(
                mean
            ).sigmoid()
            return {
                "state": state,
                "latent_logits": self.latent_head(state),
                "residual_mean": mean,
                "residual_log_scale": log_scale,
                "residual_trust": trust,
            }

    return RecoveryActor()


def build_reachability_critics(config: ResolveModelConfig) -> Any:
    """Build twin coupled recovery/deletion, B, and constraint critics."""

    _validate(config)
    torch = _torch()
    nn = torch.nn

    class HistoryEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = config.model_width
            self.context_projection = nn.Sequential(
                nn.LayerNorm(config.context_width), nn.Linear(config.context_width, width)
            )
            self.base_action_projection = nn.Sequential(
                nn.LayerNorm(config.action_horizon * config.action_dim),
                nn.Linear(config.action_horizon * config.action_dim, width),
            )
            self.scalar_projection = nn.Sequential(
                nn.LayerNorm(config.scalar_width), nn.Linear(config.scalar_width, width)
            )
            self.latent_embedding = nn.Embedding(config.latent_codes + 1, width)
            self.program_position_embedding = nn.Embedding(
                config.maximum_program_depth, width
            )
            self.milestone_embedding = nn.Embedding(config.milestone_count, width)
            self.position_embedding = nn.Embedding(config.history_length, width)
            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=config.transformer_heads,
                dim_feedforward=config.feedforward_width,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=config.transformer_layers)
            self.output_norm = nn.LayerNorm(width)

        def forward(
            self,
            contexts: Any,
            base_action_chunks: Any,
            scalars: Any,
            latent_history: Any,
            program_positions: Any,
            milestone_ids: Any,
        ) -> Any:
            batch, history, _ = contexts.shape
            positions = torch.arange(history, device=contexts.device)
            tokens = (
                self.context_projection(contexts)
                + self.base_action_projection(base_action_chunks.flatten(start_dim=2))
                + self.scalar_projection(scalars)
                + self.latent_embedding(latent_history)
                + self.program_position_embedding(program_positions)
                + self.milestone_embedding(milestone_ids)[:, None, :]
                + self.position_embedding(positions)[None, :, :]
            )
            mask = torch.triu(
                torch.ones(history, history, dtype=torch.bool, device=tokens.device), diagonal=1
            )
            return self.output_norm(self.temporal(tokens, mask=mask)[:, -1])

    class OneReachabilityCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = config.model_width
            action_width = config.action_horizon * config.action_dim
            self.history = HistoryEncoder()
            self.recovery_action = nn.Sequential(
                nn.LayerNorm(action_width), nn.Linear(action_width, width), nn.GELU()
            )
            self.next_latent = nn.Embedding(config.latent_codes + 1, width)
            # R and D share this head. D is exact B at the current macro with
            # the same latent continuation, not a freely learned third critic.
            self.recovery_continuation_head = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, config.milestone_count)
            )
            self.baseline_head = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, config.milestone_count)
            )
            self.cost_head = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, config.cost_count)
            )

        def forward(
            self,
            contexts: Any,
            base_action_chunks: Any,
            scalars: Any,
            latent_history: Any,
            program_positions: Any,
            milestone_ids: Any,
            residual_chunk: Any,
            next_latent_id: Any,
        ) -> tuple[Any, Any]:
            batch = contexts.shape[0]
            if (
                contexts.shape != (batch, config.history_length, config.context_width)
                or base_action_chunks.shape
                != (
                    batch,
                    config.history_length,
                    config.action_horizon,
                    config.action_dim,
                )
                or scalars.shape != (batch, config.history_length, config.scalar_width)
                or latent_history.shape != (batch, config.history_length)
                or program_positions.shape != (batch, config.history_length)
                or milestone_ids.shape != (batch,)
                or residual_chunk.shape
                != (batch, config.action_horizon, config.action_dim)
                or next_latent_id.shape != (batch,)
            ):
                raise ValueError("RESOLVE critic input shapes are invalid")
            state = self.history(
                contexts,
                base_action_chunks,
                scalars,
                latent_history,
                program_positions,
                milestone_ids,
            )
            latent = self.next_latent(next_latent_id)
            recovery_state = state + latent + self.recovery_action(
                residual_chunk.flatten(start_dim=1)
            )
            deletion_state = state + latent + self.recovery_action(
                torch.zeros_like(residual_chunk).flatten(start_dim=1)
            )
            reachability = torch.stack(
                (
                    self.recovery_continuation_head(recovery_state),
                    self.recovery_continuation_head(deletion_state),
                    self.baseline_head(state),
                ),
                dim=1,
            )
            return reachability, self.cost_head(recovery_state)

    class ReachabilityCriticEnsemble(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.members = nn.ModuleList(
                OneReachabilityCritic() for _ in range(config.critic_ensemble)
            )

        def forward(self, *inputs: Any) -> dict[str, Any]:
            values = [member(*inputs) for member in self.members]
            return {
                "reachability_logits": torch.stack([item[0] for item in values]),
                "cost_predictions": torch.stack([item[1] for item in values]),
            }

    return ReachabilityCriticEnsemble()


def sample_recovery_action(
    actor_outputs: dict[str, Any],
    base_action_chunk: Any,
    *,
    deterministic: bool,
) -> dict[str, Any]:
    """Sample a bounded residual chunk and a latent program transition."""

    torch = _torch()
    mean = actor_outputs["residual_mean"]
    log_scale = actor_outputs["residual_log_scale"]
    trust = actor_outputs["residual_trust"]
    if base_action_chunk.shape != mean.shape or log_scale.shape != mean.shape or trust.shape != mean.shape:
        raise ValueError("RESOLVE sampled-action shapes are invalid")
    distribution = torch.distributions.Normal(mean, log_scale.exp())
    raw = mean if deterministic else distribution.rsample()
    squashed = torch.tanh(raw)
    residual = trust * squashed
    corrected = base_action_chunk + residual
    log_probability = distribution.log_prob(raw) - torch.log1p(-squashed.square() + 1e-6)
    log_probability = log_probability.flatten(start_dim=1).sum(dim=1)
    latent_distribution = torch.distributions.Categorical(logits=actor_outputs["latent_logits"])
    next_latent = (
        actor_outputs["latent_logits"].argmax(dim=1)
        if deterministic
        else latent_distribution.sample()
    )
    return {
        "corrected_action": corrected,
        "residual_action": residual,
        "action_log_probability": log_probability,
        "next_latent_id": next_latent,
        "latent_log_probability": latent_distribution.log_prob(next_latent),
        "handoff": next_latent == actor_outputs["latent_logits"].shape[1] - 1,
    }


def conservative_crb_from_logits(
    critic_outputs: dict[str, Any], milestone_ids: Any
) -> Any:
    """Compute the differentiable conservative CRB for each batch item."""

    torch = _torch()
    logits = critic_outputs["reachability_logits"]
    if logits.ndim != 4 or logits.shape[2] != 3 or milestone_ids.shape != (logits.shape[1],):
        raise ValueError("RESOLVE critic-output shapes are invalid")
    probabilities = logits.sigmoid()
    recovery_lower = probabilities[:, :, 0].min(dim=0).values
    deletion_upper = probabilities[:, :, 1].max(dim=0).values
    baseline_upper = probabilities[:, :, 2].max(dim=0).values
    indices = milestone_ids[:, None]
    recovery = recovery_lower.gather(1, indices).squeeze(1)
    deletion = deletion_upper.gather(1, indices).squeeze(1)
    baseline = baseline_upper.gather(1, indices).squeeze(1)
    return recovery - torch.maximum(deletion, baseline)


def reachability_critic_loss(
    critic_outputs: dict[str, Any],
    reachability_targets: Any,
    cost_targets: Any,
    *,
    reachability_valid: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Fit physical R/D/B reachability and separately reported constraint costs."""

    torch = _torch()
    functional = torch.nn.functional
    logits = critic_outputs["reachability_logits"]
    costs = critic_outputs["cost_predictions"]
    if (
        reachability_targets.shape != logits.shape[1:]
        or cost_targets.shape != costs.shape[1:]
    ):
        raise ValueError("RESOLVE critic target shapes are invalid")
    expanded_targets = reachability_targets.to(logits.dtype).unsqueeze(0).expand_as(logits)
    losses = functional.binary_cross_entropy_with_logits(
        logits, expanded_targets, reduction="none"
    )
    if reachability_valid is not None:
        if reachability_valid.shape != reachability_targets.shape:
            raise ValueError("RESOLVE reachability-valid shape is invalid")
        mask = reachability_valid.to(losses.dtype).unsqueeze(0).expand_as(losses)
        reachability_loss = (losses * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        reachability_loss = losses.mean()
    cost_loss = functional.smooth_l1_loss(
        costs, cost_targets.to(costs.dtype).unsqueeze(0).expand_as(costs)
    )
    total = reachability_loss + cost_loss
    return total, {
        "loss": total.detach(),
        "reachability_loss": reachability_loss.detach(),
        "cost_loss": cost_loss.detach(),
    }


def count_trainable_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
