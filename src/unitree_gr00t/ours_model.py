"""Compact temporal recovery model used by Ours."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .ours import RECOVERY_OPTIONS, OursContractError


@dataclass(frozen=True)
class TemporalRecoveryModelConfig:
    encoder: str = "gru"
    history_length: int = 8
    context_width: int = 2048
    action_horizon: int = 16
    action_dim: int = 7
    selector_candidates: int = 17
    scalar_width: int = 6
    temporal_width: int = 256
    temporal_layers: int = 2
    temporal_heads: int = 8
    feedforward_width: int = 1024
    dropout: float = 0.0
    option_count: int = len(RECOVERY_OPTIONS)

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Ours temporal recovery model requires PyTorch") from exc
    return torch


def build_temporal_recovery_model(config: TemporalRecoveryModelConfig) -> Any:
    """Build one registered R0 architecture with a shared audited interface."""

    torch = _torch()
    nn = torch.nn
    if config.encoder not in {"linear", "mlp", "gru", "transformer"}:
        raise OursContractError(f"unsupported Ours temporal encoder: {config.encoder}")
    if (
        min(
            config.history_length,
            config.context_width,
            config.action_horizon,
            config.action_dim,
            config.scalar_width,
            config.temporal_width,
            config.temporal_layers,
            config.temporal_heads,
            config.feedforward_width,
            config.option_count,
        )
        < 1
        or config.temporal_width % config.temporal_heads
        or not 0.0 <= config.dropout < 1.0
    ):
        raise OursContractError("invalid Ours temporal model dimensions")

    class TemporalRecoveryController(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = config.temporal_width
            self.config = config
            self.context_projection = nn.Sequential(
                nn.LayerNorm(config.context_width), nn.Linear(config.context_width, width)
            )
            self.delta_projection = nn.Sequential(
                nn.LayerNorm(config.context_width), nn.Linear(config.context_width, width)
            )
            self.action_projection = nn.Sequential(
                nn.LayerNorm(config.action_horizon * config.action_dim),
                nn.Linear(config.action_horizon * config.action_dim, width),
            )
            self.selector_projection = nn.Sequential(
                nn.LayerNorm(config.selector_candidates * 2),
                nn.Linear(config.selector_candidates * 2, width),
            )
            self.scalar_projection = nn.Sequential(
                nn.LayerNorm(config.scalar_width), nn.Linear(config.scalar_width, width)
            )
            # Applying dropout after multimodal fusion regularizes every
            # encoder family, including the low-capacity linear and MLP heads.
            self.fusion = nn.Sequential(
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
            if config.encoder == "linear":
                self.temporal = nn.Identity()
            elif config.encoder == "mlp":
                self.temporal = nn.Sequential(
                    nn.LayerNorm(config.history_length * width),
                    nn.Linear(config.history_length * width, width),
                    nn.GELU(),
                    nn.Linear(width, width),
                )
            elif config.encoder == "gru":
                self.temporal = nn.GRU(
                    width,
                    width,
                    num_layers=config.temporal_layers,
                    batch_first=True,
                    dropout=config.dropout if config.temporal_layers > 1 else 0.0,
                )
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=width,
                    nhead=config.temporal_heads,
                    dim_feedforward=config.feedforward_width,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.temporal = nn.TransformerEncoder(layer, num_layers=config.temporal_layers)
            self.progress_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
            self.completion_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
            self.failure_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
            self.recovery_complete_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
            self.time_to_failure_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
            self.option_value_head = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, config.option_count)
            )

        def forward(
            self,
            contexts: Any,
            anchor_contexts: Any,
            action_chunks: Any,
            selector_features: Any,
            scalars: Any,
        ) -> dict[str, Any]:
            batch, history, context_width = contexts.shape
            expected = (batch, history, config.context_width)
            if contexts.shape != expected or anchor_contexts.shape != expected:
                raise ValueError("Ours context or anchor tensor has the wrong shape")
            if action_chunks.shape != (
                batch,
                history,
                config.action_horizon,
                config.action_dim,
            ):
                raise ValueError("Ours action history has the wrong shape")
            if selector_features.shape != (
                batch,
                history,
                config.selector_candidates * 2,
            ):
                raise ValueError("Ours selector feature history has the wrong shape")
            if scalars.shape != (batch, history, config.scalar_width):
                raise ValueError("Ours scalar history has the wrong shape")
            flat_actions = action_chunks.flatten(start_dim=2)
            tokens = self.fusion(
                self.context_projection(contexts)
                + self.delta_projection(contexts - anchor_contexts)
                + self.action_projection(flat_actions)
                + self.selector_projection(selector_features)
                + self.scalar_projection(scalars)
            )
            if config.encoder == "linear":
                state = tokens[:, -1]
            elif config.encoder == "mlp":
                state = self.temporal(tokens.flatten(start_dim=1))
            elif config.encoder == "gru":
                encoded, _ = self.temporal(tokens)
                state = encoded[:, -1]
            else:
                causal_mask = torch.triu(
                    torch.ones(history, history, device=tokens.device, dtype=torch.bool), diagonal=1
                )
                state = self.temporal(tokens, mask=causal_mask)[:, -1]
            return {
                "progress_logit": self.progress_head(state).squeeze(-1),
                "completion_logit": self.completion_head(state).squeeze(-1),
                "failure_logit": self.failure_head(state).squeeze(-1),
                "recovery_complete_logit": self.recovery_complete_head(state).squeeze(-1),
                "time_to_failure": self.time_to_failure_head(state).squeeze(-1),
                "option_values": self.option_value_head(state),
            }

    return TemporalRecoveryController()


def count_trainable_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def completion_progress_loss(
    outputs: dict[str, Any],
    target_complete: Any,
    target_progress: Any,
    target_progress_valid: Any | None = None,
    target_failure: Any | None = None,
    target_failure_valid: Any | None = None,
    target_option_values: Any | None = None,
    target_option_valid: Any | None = None,
    *,
    completion_weight: float = 1.0,
    progress_weight: float = 1.0,
    option_value_weight: float = 0.25,
    option_rank_weight: float = 0.25,
    option_classification_weight: float = 0.0,
    option_baseline_index: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """R0 loss for the false-STOP gate before failure branches are added."""

    torch = _torch()
    functional = torch.nn.functional
    completion = functional.binary_cross_entropy_with_logits(
        outputs["completion_logit"], target_complete.to(outputs["completion_logit"].dtype)
    )
    progress = functional.smooth_l1_loss(
        outputs["progress_logit"].sigmoid(),
        target_progress.to(outputs["progress_logit"].dtype),
        reduction="none",
    )
    if target_progress_valid is not None:
        progress_mask = target_progress_valid.to(progress.dtype)
        progress = (progress * progress_mask).sum() / progress_mask.sum().clamp_min(1.0)
    else:
        progress = progress.mean()
    failure = outputs["failure_logit"].sum() * 0.0
    if target_failure is not None and target_failure_valid is not None:
        failure_losses = functional.binary_cross_entropy_with_logits(
            outputs["failure_logit"],
            target_failure.to(outputs["failure_logit"].dtype),
            reduction="none",
        )
        failure_mask = target_failure_valid.to(failure_losses.dtype)
        failure = (failure_losses * failure_mask).sum() / failure_mask.sum().clamp_min(1.0)
    option_value = outputs["option_values"].sum() * 0.0
    option_rank = outputs["option_values"].sum() * 0.0
    option_classification = outputs["option_values"].sum() * 0.0
    if target_option_values is not None and target_option_valid is not None:
        normalized_target = target_option_values.to(outputs["option_values"].dtype) / 16.0
        if option_baseline_index is not None:
            if not 0 <= option_baseline_index < normalized_target.shape[1]:
                raise ValueError("option baseline index is outside the option head")
            labeled = target_option_valid.sum(dim=1) >= 2
            if not target_option_valid[labeled, option_baseline_index].all():
                raise ValueError("option baseline must be valid for every labeled row")
            normalized_target = (
                normalized_target
                - normalized_target[:, option_baseline_index : option_baseline_index + 1]
            )
        option_mask = target_option_valid.to(outputs["option_values"].dtype)
        option_losses = functional.smooth_l1_loss(
            outputs["option_values"],
            normalized_target,
            reduction="none",
        )
        option_value = (option_losses * option_mask).sum() / option_mask.sum().clamp_min(1.0)
        target_difference = normalized_target[:, :, None] - normalized_target[:, None, :]
        prediction_difference = (
            outputs["option_values"][:, :, None] - outputs["option_values"][:, None, :]
        )
        pair_mask = (
            target_option_valid[:, :, None]
            & target_option_valid[:, None, :]
            & (target_difference.abs() > 1e-4)
        )
        pair_losses = functional.relu(
            0.1 - prediction_difference * target_difference.sign()
        )
        pair_weight = pair_mask.to(pair_losses.dtype)
        option_rank = (pair_losses * pair_weight).sum() / pair_weight.sum().clamp_min(1.0)
        masked_target = normalized_target.masked_fill(~target_option_valid, -torch.inf)
        ordered_target = masked_target.sort(dim=1).values
        strict = (
            target_option_valid.sum(dim=1) >= 2
        ) & (ordered_target[:, -1] - ordered_target[:, -2] > 1e-4)
        if strict.any():
            masked_logits = outputs["option_values"].masked_fill(
                ~target_option_valid, -torch.inf
            )
            option_classification = functional.cross_entropy(
                masked_logits[strict], masked_target[strict].argmax(dim=1)
            )
    total = (
        completion_weight * completion
        + progress_weight * progress
        + 0.5 * failure
        + option_value_weight * option_value
        + option_rank_weight * option_rank
        + option_classification_weight * option_classification
    )
    return total, {
        "loss": total.detach(),
        "completion_loss": completion.detach(),
        "progress_loss": progress.detach(),
        "failure_loss": failure.detach(),
        "option_value_loss": option_value.detach(),
        "option_rank_loss": option_rank.detach(),
        "option_classification_loss": option_classification.detach(),
    }
