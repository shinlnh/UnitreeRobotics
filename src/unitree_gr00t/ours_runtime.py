"""Frozen B runtime plus Ours temporal recovery inference."""

from __future__ import annotations

from typing import Any

from .b import BContractError, candidate_validity, select_unified_candidate, selector_decision_seed
from .b_runtime import BackboneCapture, context_sha256, flat_action_chunk
from .ours import OursContractError, validate_runtime_payload
from .ours_train import normalized_selector_features, runtime_scalars


def build_recovery_sim_policy(
    base_policy: Any,
    selector: Any,
    selector_config: Any,
    recovery_model: Any,
    recovery_config: Any,
    *,
    completion_threshold: float,
    runtime_provenance: dict[str, Any],
) -> Any:
    """Wrap frozen GR00T/B and expose only outcome-blind learned beliefs."""

    import numpy as np
    import torch
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

    class OursRecoverySimPolicy(Gr00tSimPolicyWrapper):
        def __init__(self) -> None:
            super().__init__(base_policy)
            self.capture = BackboneCapture(base_policy.model.backbone)
            self.selector = selector
            self.selector.eval()
            self.recovery_model = recovery_model
            self.recovery_model.eval()
            self._memory: list[dict[str, Any]] = []
            self._device: Any | None = None

        def _reset_memory(self) -> None:
            self._memory.clear()

        def _append_feature(
            self,
            *,
            context: Any,
            anchor: Any,
            chunk: Any,
            scores: Any,
            valid: Any,
            candidate: int,
            elapsed_steps: int,
        ) -> None:
            selector_feature = normalized_selector_features(scores[None], valid[None], np)[0]
            scalar = runtime_scalars(
                elapsed_steps=np.asarray([elapsed_steps]),
                candidates=np.asarray([candidate]),
                chunks=chunk[None],
                scores=scores[None],
                valid=valid[None],
                np=np,
            )[0]
            self._memory.append(
                {
                    "context": context,
                    "anchor": anchor,
                    "action": chunk,
                    "selector": selector_feature,
                    "scalar": scalar,
                }
            )
            del self._memory[: -recovery_config.history_length]

        def _recovery_inputs(self) -> tuple[Any, Any, Any, Any, Any]:
            if not self._memory:
                raise OursContractError("Ours temporal memory is empty")
            if self._device is None:
                raise OursContractError("Ours temporal memory has no device")
            rows = [self._memory[0]] * (recovery_config.history_length - len(self._memory))
            rows.extend(self._memory)
            return tuple(
                torch.as_tensor(
                    np.stack([row[key] for row in rows]),
                    device=self._device,
                    dtype=torch.float32,
                )[None]
                for key in ("context", "anchor", "action", "selector", "scalar")
            )

        def _get_action(
            self, observation: dict[str, Any], options: dict[str, Any] | None = None
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            options = options or {}
            recovery_options = options.get("ours")
            if not isinstance(recovery_options, dict):
                raise OursContractError("Ours policy request is missing recovery options")
            validate_runtime_payload(recovery_options)
            episode_seed = recovery_options.get("episode_seed")
            decision_index = recovery_options.get("decision_index")
            elapsed_steps = recovery_options.get("subgoal_elapsed_steps")
            if not all(
                isinstance(value, int) for value in (episode_seed, decision_index, elapsed_steps)
            ):
                raise OursContractError("Ours request requires integer seed, decision, and elapsed")
            if min(episode_seed, decision_index, elapsed_steps) < 0:
                raise OursContractError("Ours request integer state cannot be negative")
            decision_seed = selector_decision_seed(episode_seed, decision_index)
            import random

            random.seed(decision_seed)
            np.random.seed(decision_seed)
            torch.manual_seed(decision_seed)
            torch.cuda.manual_seed_all(decision_seed)
            action, info = super()._get_action(observation, options)
            context = self.capture.pop_context().to(torch.float16).float()
            chunk_batch = flat_action_chunk(action, np).astype(np.float16).astype(np.float32)
            if len(chunk_batch) != 1:
                raise BContractError("Ours closed-loop server requires batch size one")
            chunk = chunk_batch[0]
            anchors = np.asarray(recovery_options.get("anchor_history", []), dtype=np.float32)
            if anchors.size == 0:
                anchors = np.empty((0, selector_config.context_width), dtype=np.float32)
            elif anchors.ndim == 1:
                anchors = anchors[None]
            if anchors.ndim != 2 or anchors.shape[1] != selector_config.context_width:
                raise BContractError("Ours anchor history has the wrong shape")
            subgoal_start = bool(recovery_options.get("subgoal_start", False))
            if subgoal_start:
                anchors = np.concatenate((anchors, context.detach().cpu().numpy()), axis=0)
                self._reset_memory()
            if len(anchors) == 0:
                raise BContractError("Ours requires a live subgoal anchor")
            max_prefix = int(recovery_options.get("max_prefix", selector_config.action_horizon))
            remaining_steps = int(recovery_options.get("remaining_steps", max_prefix))
            if remaining_steps < 1:
                raise BContractError("Ours requires at least one remaining control step")
            valid = np.asarray(
                candidate_validity(
                    decision_step=0,
                    trajectory_steps=remaining_steps,
                    horizon=selector_config.action_horizon,
                    max_prefix=max_prefix,
                ),
                dtype=np.bool_,
            )
            with torch.inference_mode():
                selector_scores = self.selector(
                    torch.as_tensor(chunk_batch, device=context.device, dtype=torch.float32),
                    context,
                    torch.as_tensor(anchors[None], device=context.device, dtype=torch.float32),
                    torch.ones((1, len(anchors)), device=context.device, dtype=torch.bool),
                    torch.as_tensor(valid[None], device=context.device, dtype=torch.bool),
                )
            scores = selector_scores[0].float().cpu().numpy()
            candidate = select_unified_candidate(scores.tolist(), valid.tolist())
            self._append_feature(
                context=context[0].cpu().numpy(),
                anchor=anchors[-1],
                chunk=chunk,
                scores=scores,
                valid=valid,
                candidate=candidate,
                elapsed_steps=elapsed_steps,
            )
            self._device = context.device
            recovery_inputs = self._recovery_inputs()
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=str(context.device).startswith("cuda"),
                ),
            ):
                recovery = self.recovery_model(*recovery_inputs)
            completion_probability = float(recovery["completion_logit"].sigmoid().float().cpu()[0])
            progress_probability = float(recovery["progress_logit"].sigmoid().float().cpu()[0])
            failure_probability = float(recovery["failure_logit"].sigmoid().float().cpu()[0])
            option_values = recovery["option_values"].float().cpu()[0].numpy()
            selector_info = {
                "candidate": candidate,
                "scores": scores,
                "valid": valid,
                "current_context_sha256": context_sha256(context.cpu().numpy()),
                "anchor_history_sha256": [context_sha256(anchor) for anchor in anchors],
                "episode_seed": episode_seed,
                "decision_index": decision_index,
                "decision_seed": decision_seed,
                "runtime_provenance": runtime_provenance["b_selector"],
            }
            if subgoal_start:
                selector_info["raw_anchor"] = context[0].cpu().numpy()
            recovery_info = {
                "completion_probability": completion_probability,
                "progress_probability": progress_probability,
                "failure_probability": failure_probability,
                "option_values": option_values,
                "completion_threshold": completion_threshold,
                "history_length": len(self._memory),
                "history_capacity": recovery_config.history_length,
                "runtime_provenance": runtime_provenance["ours"],
            }
            if bool(recovery_options.get("capture_training_context", False)):
                recovery_info["training_context"] = context[0].cpu().numpy()
            return action, dict(info) | {
                "b_selector": selector_info,
                "ours": recovery_info,
            }

        def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
            options = options or {}
            episode_seed = options.get("episode_seed")
            if not isinstance(episode_seed, int) or not 0 <= episode_seed < 2**31:
                raise OursContractError("Ours reset requires an integer episode seed")
            self._reset_memory()
            return dict(super().reset(None)) | {"episode_seed": episode_seed}

    return OursRecoverySimPolicy()
