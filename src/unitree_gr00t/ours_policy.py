"""Outcome-blind selective recovery decisions for Ours."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ours import RECOVERY_OPTIONS, OursContractError, RecoveryOption


@dataclass(frozen=True)
class RecoveryDirective:
    candidate: int
    action_chunk: Any
    suppress_stop_confirmation: bool
    option: str
    recovery_triggered: bool
    hypothesis_count: int
    completion_probability: float
    progress_probability: float
    apply_subgoal_transition: bool = False
    subgoal_delta: int = 0
    reanchor: bool = False


class SelectiveConsensusRecovery:
    """Recover only from low-completion, high-failure STOP proposals."""

    def __init__(
        self,
        *,
        gate_signal: str = "completion",
        gate_threshold: float | None = None,
        completion_threshold: float | None = None,
        consensus_hypotheses: int,
        max_recovery_attempts: int = 1,
        min_recovery_elapsed_steps: int = 0,
        use_option_values: bool = False,
        option_value_margin: float = 0.0,
        failure_threshold: float = 0.5,
        consensus_cooldown_decisions: int = 16,
        force_boundary_steps: int | None = None,
    ):
        # completion_threshold remains an explicit compatibility alias for the
        # first recorded smoke artifact.
        selected_threshold = gate_threshold if gate_threshold is not None else completion_threshold
        if gate_signal not in {"completion", "progress", "maximum"}:
            raise OursContractError("unsupported Ours gate signal")
        if selected_threshold is None or not 0.0 <= selected_threshold <= 1.0:
            raise OursContractError("gate threshold must be inside [0, 1]")
        if consensus_hypotheses < 1:
            raise OursContractError("consensus hypotheses must be positive")
        if max_recovery_attempts not in {1, 2}:
            raise OursContractError("max recovery attempts must be one or two")
        if min_recovery_elapsed_steps < 0:
            raise OursContractError("minimum recovery elapsed steps cannot be negative")
        if option_value_margin < 0.0:
            raise OursContractError("option value margin cannot be negative")
        if not 0.0 <= failure_threshold <= 1.0:
            raise OursContractError("failure threshold must be inside [0, 1]")
        if consensus_cooldown_decisions < 0:
            raise OursContractError("consensus cooldown cannot be negative")
        if force_boundary_steps is not None and force_boundary_steps < 1:
            raise OursContractError("forced collection boundary must be positive")
        self.gate_signal = gate_signal
        self.gate_threshold = selected_threshold
        self.consensus_hypotheses = consensus_hypotheses
        self.max_recovery_attempts = max_recovery_attempts
        self.min_recovery_elapsed_steps = min_recovery_elapsed_steps
        self.use_option_values = use_option_values
        self.option_value_margin = option_value_margin
        self.failure_threshold = failure_threshold
        self.consensus_cooldown_decisions = consensus_cooldown_decisions
        self.force_boundary_steps = force_boundary_steps
        self._proposals: list[tuple[Any, Any, Any]] = []
        self._cooldown_remaining = 0
        self._recovery_attempts = 0
        self._active_subgoal: int | None = None

    def reset(self) -> None:
        self._proposals.clear()
        self._cooldown_remaining = 0
        self._recovery_attempts = 0
        self._active_subgoal = None

    def _clear_proposals(self) -> None:
        self._proposals.clear()

    @staticmethod
    def _best_nonstop(scores: Any, valid: Any, np: Any) -> int:
        if len(scores) != len(valid) or len(scores) < 2:
            raise OursContractError("invalid selector proposal for recovery consensus")
        masked = np.where(valid[1:], scores[1:], -np.inf)
        if not np.isfinite(masked).any():
            raise OursContractError("recovery proposal has no valid non-STOP prefix")
        return int(masked.argmax()) + 1

    @staticmethod
    def _medoid(proposals: list[tuple[Any, Any, Any]], np: Any) -> int:
        chunks = np.stack([proposal[0] for proposal in proposals]).astype(np.float32)
        flattened = chunks.reshape(len(chunks), -1)
        scale = np.linalg.norm(flattened, axis=1, keepdims=True).clip(min=1e-6)
        normalized = flattened / scale
        distances = np.square(normalized[:, None] - normalized[None, :]).mean(axis=2)
        # NumPy argmin preserves the earliest registered hypothesis on ties.
        return int(distances.sum(axis=1).argmin())

    def decide(
        self,
        *,
        candidate: int,
        action_chunk: Any,
        scores: Any,
        valid: Any,
        completion_probability: float,
        progress_probability: float,
        failure_probability: float = 0.0,
        option_values: Any | None = None,
        subgoal_elapsed_steps: int = 0,
        subgoal_index: int = 0,
        stop_pending: bool = False,
        np: Any,
    ) -> RecoveryDirective:
        if subgoal_index < 0:
            raise OursContractError("subgoal index cannot be negative")
        if self._active_subgoal != subgoal_index:
            self._active_subgoal = subgoal_index
            self._recovery_attempts = 0
            self._cooldown_remaining = 0
            self._clear_proposals()
        cooldown_active = self._cooldown_remaining > 0
        if cooldown_active:
            self._cooldown_remaining -= 1
        if not all(
            0.0 <= probability <= 1.0
            for probability in (
                completion_probability,
                progress_probability,
                failure_probability,
            )
        ):
            raise OursContractError("Ours runtime probabilities must be inside [0, 1]")
        if candidate < 0 or candidate >= len(valid) or not bool(valid[candidate]):
            raise OursContractError("B proposed an invalid candidate to Ours")
        if subgoal_elapsed_steps < 0:
            raise OursContractError("subgoal elapsed steps cannot be negative")
        if not isinstance(stop_pending, bool):
            raise OursContractError("STOP pending state must be boolean")

        # Once selected, consensus consumes the registered number of fresh
        # hypotheses from the same live observation even when a later B sample
        # proposes a non-STOP prefix. No simulator step is consumed until the
        # medoid prefix is selected.
        if self._proposals:
            self._proposals.append((action_chunk.copy(), scores.copy(), valid.copy()))
            if len(self._proposals) < self.consensus_hypotheses:
                return RecoveryDirective(
                    candidate=0,
                    action_chunk=action_chunk,
                    suppress_stop_confirmation=True,
                    option=RecoveryOption.REOBSERVE.value,
                    recovery_triggered=True,
                    hypothesis_count=len(self._proposals),
                    completion_probability=completion_probability,
                    progress_probability=progress_probability,
                )
            chosen = self._medoid(self._proposals, np)
            chosen_chunk, chosen_scores, chosen_valid = self._proposals[chosen]
            chosen_candidate = self._best_nonstop(chosen_scores, chosen_valid, np)
            count = len(self._proposals)
            self._clear_proposals()
            self._cooldown_remaining = self.consensus_cooldown_decisions
            self._recovery_attempts += 1
            return RecoveryDirective(
                candidate=chosen_candidate,
                action_chunk=chosen_chunk,
                suppress_stop_confirmation=False,
                option=RecoveryOption.CONSENSUS_PREFIX.value,
                recovery_triggered=True,
                hypothesis_count=count,
                completion_probability=completion_probability,
                progress_probability=progress_probability,
            )

        if (
            self.force_boundary_steps is not None
            and subgoal_elapsed_steps >= self.force_boundary_steps
        ):
            self._clear_proposals()
            return RecoveryDirective(
                candidate=0,
                action_chunk=action_chunk,
                suppress_stop_confirmation=False,
                option=RecoveryOption.ADVANCE.value,
                recovery_triggered=False,
                hypothesis_count=1,
                completion_probability=completion_probability,
                progress_probability=progress_probability,
            )
        if candidate > 0:
            self._clear_proposals()
            return RecoveryDirective(
                candidate=candidate,
                action_chunk=action_chunk,
                suppress_stop_confirmation=False,
                option=RecoveryOption.ACCEPT_B.value,
                recovery_triggered=False,
                hypothesis_count=1,
                completion_probability=completion_probability,
                progress_probability=progress_probability,
            )
        gate_probability = {
            "completion": completion_probability,
            "progress": progress_probability,
            "maximum": max(completion_probability, progress_probability),
        }[self.gate_signal]
        if gate_probability >= self.gate_threshold:
            self._clear_proposals()
            return RecoveryDirective(
                candidate=0,
                action_chunk=action_chunk,
                suppress_stop_confirmation=False,
                option=RecoveryOption.ADVANCE.value,
                recovery_triggered=False,
                hypothesis_count=1,
                completion_probability=completion_probability,
                progress_probability=progress_probability,
            )

        if self._recovery_attempts >= self.max_recovery_attempts:
            self._clear_proposals()
            return RecoveryDirective(
                candidate=0,
                action_chunk=action_chunk,
                suppress_stop_confirmation=False,
                option=RecoveryOption.ADVANCE.value,
                recovery_triggered=False,
                hypothesis_count=1,
                completion_probability=completion_probability,
                progress_probability=progress_probability,
            )

        if (
            subgoal_elapsed_steps < self.min_recovery_elapsed_steps
            or failure_probability < self.failure_threshold
            or cooldown_active
        ):
            self._clear_proposals()
            return RecoveryDirective(
                candidate=0,
                action_chunk=action_chunk,
                suppress_stop_confirmation=False,
                option=RecoveryOption.ACCEPT_B.value,
                recovery_triggered=False,
                hypothesis_count=1,
                completion_probability=completion_probability,
                progress_probability=progress_probability,
            )

        selected_option = RecoveryOption.CONSENSUS_PREFIX
        if self.use_option_values:
            values = np.asarray(option_values, dtype=np.float32)
            if values.shape != (len(RECOVERY_OPTIONS),) or not np.isfinite(values).all():
                raise OursContractError("learned recovery option values are invalid")
            index = {option: RECOVERY_OPTIONS.index(option) for option in RECOVERY_OPTIONS}
            accept_options = [RecoveryOption.ACCEPT_B]
            if stop_pending:
                accept_options.append(RecoveryOption.ADVANCE)
            recover_options = [
                RecoveryOption.REOBSERVE,
                RecoveryOption.RETRY_CURRENT,
                RecoveryOption.CONSENSUS_PREFIX,
            ]
            if subgoal_index > 0:
                recover_options.append(RecoveryOption.BACKTRACK_ONE)
            best_accept = max(accept_options, key=lambda option: float(values[index[option]]))
            best_recover = max(recover_options, key=lambda option: float(values[index[option]]))
            accept_value = float(values[index[best_accept]])
            recover_value = float(values[index[best_recover]])
            if recover_value < accept_value + self.option_value_margin:
                self._clear_proposals()
                if best_accept is RecoveryOption.ADVANCE:
                    return RecoveryDirective(
                        candidate=0,
                        action_chunk=action_chunk,
                        suppress_stop_confirmation=True,
                        option=best_accept.value,
                        recovery_triggered=True,
                        hypothesis_count=1,
                        completion_probability=completion_probability,
                        progress_probability=progress_probability,
                        apply_subgoal_transition=True,
                        subgoal_delta=1,
                        reanchor=True,
                    )
                return RecoveryDirective(
                    candidate=0,
                    action_chunk=action_chunk,
                    suppress_stop_confirmation=False,
                    option=RecoveryOption.ACCEPT_B.value,
                    recovery_triggered=False,
                    hypothesis_count=1,
                    completion_probability=completion_probability,
                    progress_probability=progress_probability,
                )
            selected_option = best_recover

        if selected_option in {
            RecoveryOption.REOBSERVE,
            RecoveryOption.RETRY_CURRENT,
            RecoveryOption.BACKTRACK_ONE,
        }:
            self._recovery_attempts += 1
            self._cooldown_remaining = self.consensus_cooldown_decisions
            transition = selected_option in {
                RecoveryOption.RETRY_CURRENT,
                RecoveryOption.BACKTRACK_ONE,
            }
            return RecoveryDirective(
                candidate=0,
                action_chunk=action_chunk,
                suppress_stop_confirmation=True,
                option=selected_option.value,
                recovery_triggered=True,
                hypothesis_count=1,
                completion_probability=completion_probability,
                progress_probability=progress_probability,
                apply_subgoal_transition=transition,
                subgoal_delta=(-1 if selected_option is RecoveryOption.BACKTRACK_ONE else 0),
                reanchor=transition,
            )

        self._proposals.append((action_chunk.copy(), scores.copy(), valid.copy()))
        if len(self._proposals) < self.consensus_hypotheses:
            return RecoveryDirective(
                candidate=0,
                action_chunk=action_chunk,
                suppress_stop_confirmation=True,
                option=RecoveryOption.REOBSERVE.value,
                recovery_triggered=True,
                hypothesis_count=len(self._proposals),
                completion_probability=completion_probability,
                progress_probability=progress_probability,
            )
        chosen = self._medoid(self._proposals, np)
        chosen_chunk, chosen_scores, chosen_valid = self._proposals[chosen]
        chosen_candidate = self._best_nonstop(chosen_scores, chosen_valid, np)
        count = len(self._proposals)
        self._clear_proposals()
        self._cooldown_remaining = self.consensus_cooldown_decisions
        self._recovery_attempts += 1
        return RecoveryDirective(
            candidate=chosen_candidate,
            action_chunk=chosen_chunk,
            suppress_stop_confirmation=False,
            option=RecoveryOption.CONSENSUS_PREFIX.value,
            recovery_triggered=True,
            hypothesis_count=count,
            completion_probability=completion_probability,
            progress_probability=progress_probability,
        )
