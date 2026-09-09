"""Closed-loop RESOLVE actor wrapper around exact frozen B."""

from __future__ import annotations

import hashlib
from typing import Any

from .a0 import ACTION_KEYS
from .b_runtime import flat_action_chunk
from .ours import OursContractError
from .resolve_model import sample_recovery_action


def replace_action_chunk(action: dict[str, Any], corrected: Any, *, np: Any) -> dict[str, Any]:
    """Replace a GR00T action dictionary without altering its wire contract."""

    values = np.asarray(corrected, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != len(ACTION_KEYS):
        raise OursContractError("RESOLVE corrected action chunk has an invalid shape")
    result = dict(action)
    for index, key in enumerate(ACTION_KEYS):
        name = f"action.{key}"
        if (
            name not in action
            or np.asarray(action[name]).shape != values[..., index : index + 1].shape
        ):
            raise OursContractError("RESOLVE base action dictionary is incompatible")
        result[name] = values[..., index : index + 1]
    return result


def _sha256_tensor(value: Any, *, np: Any) -> str:
    array = np.asarray(value, dtype="<f4")
    return hashlib.sha256(array.tobytes()).hexdigest()


def build_resolve_sim_policy(
    selector_policy: Any,
    actor: Any,
    model_config: Any,
    *,
    actor_sha256: str,
) -> Any:
    """Wrap B with a causal adaptive recovery policy.

    The actor receives only contexts and B chunks observed up through the
    current decision.  HANDOFF preserves the exact action dictionary returned
    by B; a non-handoff replaces the whole H16 chunk with the actor's bounded
    action-logit correction.
    """

    import numpy as np
    import torch
    from gr00t.policy.policy import PolicyWrapper

    if not isinstance(actor_sha256, str) or len(actor_sha256) != 64:
        raise OursContractError("RESOLVE actor provenance is invalid")

    class ResolvePolicy(PolicyWrapper):
        def __init__(self, policy: Any) -> None:
            super().__init__(policy)
            self._history: list[tuple[Any, Any, Any, int, int]] = []
            self._latent = model_config.latent_codes
            self._device = next(actor.parameters()).device
            actor.eval()

        def check_observation(self, observation: dict[str, Any]) -> None:
            selector_policy.check_observation(observation)

        def check_action(self, action: dict[str, Any]) -> None:
            selector_policy.check_action(action)

        def _get_action(
            self, observation: dict[str, Any], options: dict[str, Any] | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            options = dict(options or {})
            request = options.get("resolve")
            if request is None:
                return selector_policy._get_action(observation, options)
            if not isinstance(request, dict):
                raise OursContractError("RESOLVE policy request is missing resolve options")
            active = request.get("active")
            scalars = np.asarray(request.get("scalars"), dtype=np.float32).copy()
            position = request.get("program_position")
            milestone = request.get("milestone_id")
            if (
                not isinstance(active, bool)
                or scalars.shape != (model_config.scalar_width,)
                or not np.isfinite(scalars).all()
                or not isinstance(position, int)
                or not 0 <= position < model_config.maximum_program_depth
                or not isinstance(milestone, int)
                or not 0 <= milestone < model_config.milestone_count
            ):
                raise OursContractError("RESOLVE policy request is invalid")
            selector_options = dict(options.get("b_selector") or {})
            selector_options["capture_training_context"] = True
            options["b_selector"] = selector_options
            action, info = selector_policy._get_action(observation, options)
            selector_info = info.get("b_selector")
            if not isinstance(selector_info, dict) or "training_context" not in selector_info:
                raise OursContractError("RESOLVE exact B omitted its causal context")
            context = np.asarray(selector_info["training_context"], dtype=np.float32)
            chunk = flat_action_chunk(action, np)[0]
            chunk[:, :6] = np.clip(chunk[:, :6], -1.0, 1.0)
            chunk[:, 6] = np.clip(chunk[:, 6], 0.0, 1.0)
            scalars[-1] = int(selector_info["candidate"]) / model_config.action_horizon
            if context.shape != (model_config.context_width,) or chunk.shape != (
                model_config.action_horizon,
                model_config.action_dim,
            ):
                raise OursContractError("RESOLVE runtime features are incompatible")
            self._history.append((context, chunk, scalars, self._latent, position))
            del self._history[: -model_config.history_length]
            rows = [self._history[0]] * (model_config.history_length - len(self._history))
            rows.extend(self._history)

            def tensor(value: Any) -> Any:
                return torch.as_tensor(value, dtype=torch.float32, device=self._device)

            contexts = tensor(np.stack([row[0] for row in rows]))[None]
            chunks = tensor(np.stack([row[1] for row in rows]))[None]
            state_scalars = tensor(np.stack([row[2] for row in rows]))[None]
            latents = torch.as_tensor(
                [[row[3] for row in rows]], dtype=torch.long, device=self._device
            )
            positions = torch.as_tensor(
                [[row[4] for row in rows]], dtype=torch.long, device=self._device
            )
            milestones = torch.as_tensor([milestone], dtype=torch.long, device=self._device)
            with torch.inference_mode():
                outputs = actor(
                    contexts,
                    chunks,
                    state_scalars,
                    latents,
                    positions,
                    milestones,
                )
                sample = sample_recovery_action(outputs, chunks[:, -1], deterministic=True)
            handoff = bool(sample["handoff"].item()) or not active
            corrected = sample["corrected_action"].float().cpu().numpy()
            residual = sample["residual_action"].float().cpu().numpy()
            next_latent = int(sample["next_latent_id"].item())
            if active and not handoff:
                action = replace_action_chunk(action, corrected, np=np)
                self._latent = next_latent
            metadata = {
                "active": active,
                "handoff": handoff,
                "program_position": position,
                "milestone_id": milestone,
                "latent_before": int(rows[-1][3]),
                "latent_after": self._latent,
                "actor_sha256": actor_sha256,
                "base_chunk_sha256": _sha256_tensor(chunk, np=np),
                "executed_chunk_sha256": _sha256_tensor(chunk if handoff else corrected[0], np=np),
                "residual_l1": float(np.abs(residual).sum()),
            }
            return action, dict(info) | {"resolve": metadata}

        def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
            self._history.clear()
            self._latent = model_config.latent_codes
            return selector_policy.reset(options)

    return ResolvePolicy(selector_policy)
