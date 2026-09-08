"""Training-only simulator branching for counterfactual recovery supervision."""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any

from .ours import OursContractError, RecoveryOption, counterfactual_branch_seed


@dataclass(frozen=True)
class SimulatorSnapshot:
    flattened_state: Any
    state_progress: Any
    timestep: Any
    done: Any
    ctrl: Any
    mocap_pos: Any
    mocap_quat: Any


@dataclass(frozen=True)
class CounterfactualBranch:
    option: str
    branch_seed: int
    executed_steps: int
    completed_subtasks_before: int
    completed_subtasks_after: int
    predicate_count_before: int
    predicate_count_after: int
    final_success: bool
    return_value: float
    final_state_sha256: str


def _copy_optional(value: Any, name: str) -> Any:
    selected = getattr(value, name, None)
    return selected.copy() if selected is not None else None


def capture_simulator_snapshot(env: Any) -> SimulatorSnapshot:
    """Capture physical and evaluator bookkeeping state for training branches only."""

    state = env.sim.get_state()
    flattened = state.flatten() if hasattr(state, "flatten") else state
    return SimulatorSnapshot(
        flattened_state=flattened.copy(),
        state_progress=copy.deepcopy(getattr(env, "_state_progress", None)),
        timestep=copy.deepcopy(getattr(env, "timestep", None)),
        done=copy.deepcopy(getattr(env, "done", None)),
        ctrl=_copy_optional(env.sim.data, "ctrl"),
        mocap_pos=_copy_optional(env.sim.data, "mocap_pos"),
        mocap_quat=_copy_optional(env.sim.data, "mocap_quat"),
    )


def restore_simulator_snapshot(env: Any, snapshot: SimulatorSnapshot) -> Any:
    env.sim.set_state_from_flattened(snapshot.flattened_state)
    for name, value in (
        ("ctrl", snapshot.ctrl),
        ("mocap_pos", snapshot.mocap_pos),
        ("mocap_quat", snapshot.mocap_quat),
    ):
        selected = getattr(env.sim.data, name, None)
        if selected is not None and value is not None:
            selected[...] = value
    env.sim.forward()
    if snapshot.state_progress is not None:
        env._state_progress = copy.deepcopy(snapshot.state_progress)
    if snapshot.timestep is not None:
        env.timestep = copy.deepcopy(snapshot.timestep)
    if snapshot.done is not None:
        env.done = copy.deepcopy(snapshot.done)
    env._post_process()
    env._update_observables(force=True)
    return env._get_observations()


def snapshot_sha256(snapshot: SimulatorSnapshot, np: Any) -> str:
    digest = hashlib.sha256()
    for value in (
        snapshot.flattened_state,
        snapshot.ctrl,
        snapshot.mocap_pos,
        snapshot.mocap_quat,
    ):
        if value is not None:
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    digest.update(repr(snapshot.state_progress).encode())
    digest.update(repr(snapshot.timestep).encode())
    digest.update(repr(snapshot.done).encode())
    return digest.hexdigest()


def _predicate_count(env: Any, goal: dict[str, list[list[str]]]) -> int:
    return sum(
        bool(env._eval_predicate(predicate)) for states in goal.values() for predicate in states
    )


def evaluate_counterfactual_options(
    env: Any,
    *,
    goal: dict[str, list[list[str]]],
    option_actions: dict[RecoveryOption | str, Any],
    completed_subtasks_before: int,
    base_seed: int,
    state_index: int,
    np: Any,
) -> tuple[CounterfactualBranch, ...]:
    """Evaluate fixed candidate actions from one state and restore it exactly."""

    if base_seed not in {10007, 11007, 12007}:
        raise OursContractError("counterfactual branches are restricted to frozen train seeds")
    if state_index < 0 or completed_subtasks_before < 0 or not option_actions:
        raise OursContractError("counterfactual branch state is invalid")
    snapshot = capture_simulator_snapshot(env)
    source_sha256 = snapshot_sha256(snapshot, np)
    predicate_before = _predicate_count(env, goal)
    branches: list[CounterfactualBranch] = []
    try:
        for option_index, (raw_option, actions) in enumerate(option_actions.items()):
            option = RecoveryOption(raw_option)
            branch_seed = counterfactual_branch_seed(
                base_seed,
                state_index,
                option,
                option_index,
            )
            random.seed(branch_seed)
            np.random.seed(branch_seed)
            restore_simulator_snapshot(env, snapshot)
            executed_steps = 0
            for action in actions:
                env.step(np.asarray(action, dtype=np.float32))
                executed_steps += 1
            _, completed_after, final_success = env._check_success(goal)
            predicate_after = _predicate_count(env, goal)
            state_after = capture_simulator_snapshot(env)
            progress_gain = int(completed_after) - completed_subtasks_before
            predicate_gain = predicate_after - predicate_before
            return_value = (
                8.0 * progress_gain
                + 16.0 * int(bool(final_success))
                + float(predicate_gain)
                - 0.002 * executed_steps
            )
            branches.append(
                CounterfactualBranch(
                    option=option.value,
                    branch_seed=branch_seed,
                    executed_steps=executed_steps,
                    completed_subtasks_before=completed_subtasks_before,
                    completed_subtasks_after=int(completed_after),
                    predicate_count_before=predicate_before,
                    predicate_count_after=predicate_after,
                    final_success=bool(final_success),
                    return_value=return_value,
                    final_state_sha256=snapshot_sha256(state_after, np),
                )
            )
    finally:
        restore_simulator_snapshot(env, snapshot)
    if snapshot_sha256(capture_simulator_snapshot(env), np) != source_sha256:
        raise OursContractError("counterfactual simulator restore did not replay exactly")
    return tuple(branches)


def branch_payload(branch: CounterfactualBranch) -> dict[str, Any]:
    return asdict(branch)
