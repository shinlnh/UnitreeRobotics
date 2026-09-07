"""Shared GR00T backbone capture and B selector runtime integration."""

from __future__ import annotations

import hashlib
from typing import Any

from .a0 import ACTION_KEYS
from .b import BContractError, candidate_validity


class BackboneCapture:
    """Capture the exact backbone output used by the frozen A1 action head."""

    def __init__(self, backbone: Any):
        self._output: Any | None = None
        self._handle = backbone.register_forward_hook(self._capture)

    def _capture(self, _module: Any, _inputs: Any, output: Any) -> None:
        self._output = output

    def pop_context(self) -> Any:
        """Return the last valid GR00T language/context token as float32."""

        if self._output is None:
            raise RuntimeError("GR00T backbone did not produce a captured output")
        output = self._output
        self._output = None
        features = output["backbone_features"]
        valid = output["backbone_attention_mask"].bool()
        if features.ndim != 3 or valid.shape != features.shape[:2]:
            raise RuntimeError("unexpected GR00T backbone feature contract")
        if not valid.any(dim=1).all():
            raise RuntimeError("GR00T backbone output contains an empty sequence")
        torch = __import__("torch")
        indices = valid.long().sum(dim=1) - 1
        batch = torch.arange(features.shape[0], device=features.device)
        return features[batch, indices].detach().float()

    def close(self) -> None:
        self._handle.remove()


def flat_action_chunk(action: dict[str, Any], np: Any) -> Any:
    columns = []
    for key in ACTION_KEYS:
        value = np.asarray(action[f"action.{key}"], dtype=np.float32)
        if value.ndim != 3 or value.shape[2] != 1:
            raise BContractError(f"invalid action column for selector: action.{key}")
        columns.append(value)
    return np.concatenate(columns, axis=2)


def context_sha256(context: Any) -> str:
    np = __import__("numpy")
    value = np.asarray(context, dtype="<f4")
    return hashlib.sha256(value.tobytes()).hexdigest()


def build_selector_sim_policy(base_policy: Any, selector: Any, model_config: Any) -> Any:
    """Wrap the official sim policy and return action plus a unified B decision."""

    import numpy as np
    import torch
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

    class BSelectorSimPolicy(Gr00tSimPolicyWrapper):
        def __init__(self) -> None:
            super().__init__(base_policy)
            self.capture = BackboneCapture(base_policy.model.backbone)
            self.selector = selector
            self.selector.eval()

        def _get_action(
            self, observation: dict[str, Any], options: dict[str, Any] | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            options = options or {}
            selector_options = options.get("b_selector")
            if not isinstance(selector_options, dict):
                raise BContractError("B policy request is missing b_selector options")
            action, info = super()._get_action(observation, options)
            context = self.capture.pop_context()
            chunk = flat_action_chunk(action, np)
            batch = chunk.shape[0]
            if batch != 1:
                raise BContractError("B closed-loop server currently requires batch size one")
            anchors = np.asarray(selector_options.get("anchor_history", []), dtype=np.float32)
            if anchors.size == 0:
                anchors = np.empty((0, model_config.context_width), dtype=np.float32)
            elif anchors.ndim == 1:
                anchors = anchors[None]
            if anchors.ndim != 2 or anchors.shape[1] != model_config.context_width:
                raise BContractError("B anchor history has the wrong shape")
            subgoal_start = bool(selector_options.get("subgoal_start", False))
            if subgoal_start:
                anchors = np.concatenate((anchors, context.detach().cpu().numpy()), axis=0)
            if len(anchors) == 0:
                raise BContractError("B selector requires a subgoal-onset anchor")
            max_prefix = int(selector_options.get("max_prefix", model_config.action_horizon))
            remaining_steps = int(selector_options.get("remaining_steps", max_prefix))
            trajectory_steps = max(1, remaining_steps)
            valid = candidate_validity(
                decision_step=0,
                trajectory_steps=trajectory_steps,
                horizon=model_config.action_horizon,
                max_prefix=max_prefix,
            )
            with torch.inference_mode():
                scores = self.selector(
                    torch.as_tensor(chunk, device=context.device, dtype=torch.float32),
                    context,
                    torch.as_tensor(anchors[None], device=context.device, dtype=torch.float32),
                    torch.ones((1, len(anchors)), device=context.device, dtype=torch.bool),
                    torch.as_tensor([valid], device=context.device, dtype=torch.bool),
                )
            score_values = scores[0].float().cpu().numpy()
            candidate = int(np.argmax(score_values))
            selector_info = {
                "candidate": candidate,
                "scores": score_values,
                "valid": np.asarray(valid, dtype=np.bool_),
                "current_context_sha256": context_sha256(context.cpu().numpy()),
            }
            if subgoal_start:
                selector_info["raw_anchor"] = context[0].cpu().numpy()
            return action, dict(info) | {"b_selector": selector_info}

        def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
            return super().reset(options)

    return BSelectorSimPolicy()
