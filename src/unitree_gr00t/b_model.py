"""Torch implementation of B's SparkVLA-style unified selector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SelectorModelConfig:
    action_horizon: int = 16
    action_dim: int = 7
    context_width: int = 2048
    scoring_width: int = 1024
    scoring_layers: int = 2
    scoring_heads: int = 8
    feedforward_width: int = 4096
    dropout: float = 0.0

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("B selector requires PyTorch from the Isaac-GR00T environment") from exc
    return torch


def build_selector(config: SelectorModelConfig) -> Any:
    """Construct the model lazily so CPU contract tests do not require torch."""

    torch = _torch()
    nn = torch.nn

    class UnifiedStopPrefixSelector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if config.scoring_width % config.scoring_heads:
                raise ValueError("scoring width must be divisible by attention heads")
            self.config = config
            self.context_projection = nn.Sequential(
                nn.LayerNorm(config.context_width),
                nn.Linear(config.context_width, config.scoring_width),
            )
            self.anchor_projection = nn.Sequential(
                nn.LayerNorm(config.context_width),
                nn.Linear(config.context_width, config.scoring_width),
            )
            anchor_layer = nn.TransformerEncoderLayer(
                d_model=config.scoring_width,
                nhead=config.scoring_heads,
                dim_feedforward=config.feedforward_width,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.anchor_history = nn.TransformerEncoder(anchor_layer, num_layers=1)
            self.action_projection = nn.Sequential(
                nn.Linear(config.action_dim, config.scoring_width),
                nn.GELU(),
                nn.Linear(config.scoring_width, config.scoring_width),
            )
            self.length_embedding = nn.Embedding(config.action_horizon + 1, config.scoring_width)
            self.stop_embedding = nn.Parameter(torch.empty(config.scoring_width))
            scoring_layer = nn.TransformerEncoderLayer(
                d_model=config.scoring_width,
                nhead=config.scoring_heads,
                dim_feedforward=config.feedforward_width,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.scoring_transformer = nn.TransformerEncoder(
                scoring_layer, num_layers=config.scoring_layers
            )
            self.score = nn.Sequential(
                nn.LayerNorm(config.scoring_width),
                nn.Linear(config.scoring_width, config.scoring_width),
                nn.GELU(),
                nn.Linear(config.scoring_width, 1),
            )
            nn.init.normal_(self.stop_embedding, std=0.02)

        def forward(
            self,
            action_chunk: Any,
            current_context: Any,
            anchor_history: Any,
            anchor_valid: Any,
            candidate_valid: Any,
        ) -> Any:
            batch, horizon, action_dim = action_chunk.shape
            if horizon != config.action_horizon or action_dim != config.action_dim:
                raise ValueError("action chunk does not match selector model configuration")
            if current_context.shape != (batch, config.context_width):
                raise ValueError("current context does not match selector model configuration")
            if anchor_history.ndim != 3 or anchor_history.shape[0] != batch:
                raise ValueError("anchor history must have shape [B, K, context_width]")
            if anchor_history.shape[2] != config.context_width:
                raise ValueError("anchor history width does not match selector model configuration")
            if anchor_valid.shape != anchor_history.shape[:2]:
                raise ValueError("anchor validity shape mismatch")
            if candidate_valid.shape != (batch, horizon + 1):
                raise ValueError("candidate validity shape mismatch")
            if not torch.all(anchor_valid.any(dim=1)):
                raise ValueError("each selector sample requires at least one anchor")

            anchor_tokens = self.anchor_projection(anchor_history)
            fused_history = self.anchor_history(
                anchor_tokens, src_key_padding_mask=~anchor_valid.bool()
            )
            last_indices = anchor_valid.long().sum(dim=1) - 1
            fused_anchor = fused_history[
                torch.arange(batch, device=action_chunk.device), last_indices
            ]
            current_token = self.context_projection(current_context)
            length_ids = torch.arange(1, horizon + 1, device=action_chunk.device)
            action_tokens = self.action_projection(action_chunk) + self.length_embedding(length_ids)
            stop_token = self.stop_embedding + self.length_embedding.weight[0]
            stop_tokens = stop_token.view(1, 1, -1).expand(batch, 1, -1)
            tokens = torch.cat(
                (fused_anchor[:, None], current_token[:, None], stop_tokens, action_tokens), dim=1
            )
            contextual = self.scoring_transformer(tokens)
            scores = self.score(contextual[:, 2:]).squeeze(-1)
            return scores.masked_fill(~candidate_valid.bool(), torch.finfo(scores.dtype).min)

    return UnifiedStopPrefixSelector()


def selector_loss(
    scores: Any,
    priorities: Any,
    valid: Any,
    stop_labels: Any,
    rank_weights: Any,
    stop_weights: Any,
    *,
    stop_loss_weight: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    """Compute the paper's pairwise ranking and STOP log-sum-exp objectives."""

    torch = _torch()
    functional = torch.nn.functional
    if scores.shape != priorities.shape or scores.shape != valid.shape:
        raise ValueError("scores, priorities, and validity must share [B, H+1]")
    left = scores[:, :, None]
    right = scores[:, None, :]
    priority_left = priorities[:, :, None]
    priority_right = priorities[:, None, :]
    size = scores.shape[1]
    upper = torch.triu(torch.ones(size, size, device=scores.device, dtype=torch.bool), diagonal=1)
    pair_mask = (
        upper[None]
        & valid[:, :, None].bool()
        & valid[:, None, :].bool()
        & (priority_left != priority_right)
    )
    pair_labels = (priority_left > priority_right).to(scores.dtype)
    pair_losses = functional.binary_cross_entropy_with_logits(
        left - right, pair_labels, reduction="none"
    )
    weighted_pairs = pair_mask.to(scores.dtype) * rank_weights[:, None, None]
    rank_loss = (pair_losses * weighted_pairs).sum() / weighted_pairs.sum().clamp_min(1.0)

    continuation = scores[:, 1:].masked_fill(~valid[:, 1:].bool(), -torch.inf)
    stop_logit = scores[:, 0] - torch.logsumexp(continuation, dim=1)
    stop_losses = functional.binary_cross_entropy_with_logits(
        stop_logit, stop_labels.to(scores.dtype), reduction="none"
    )
    stop_loss = (stop_losses * stop_weights).sum() / stop_weights.sum().clamp_min(1.0)
    total = rank_loss + stop_loss_weight * stop_loss
    return total, {
        "loss": total.detach(),
        "rank_loss": rank_loss.detach(),
        "stop_loss": stop_loss.detach(),
        "pairs": pair_mask.sum().detach(),
    }
