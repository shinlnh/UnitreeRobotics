"""Training-only simulator branching for counterfactual recovery supervision."""

from __future__ import annotations

import copy
import hashlib
import pickle
import random
from dataclasses import asdict, dataclass
from typing import Any

from .ours import OursContractError, RecoveryOption, counterfactual_branch_seed

EFFICIENCY_SHAPED_RETURN_TARGET = "efficiency-shaped-v1"
OUTCOME_FIRST_RETURN_TARGET = "outcome-first-physical-v1"
COUNTERFACTUAL_RETURN_TARGETS = (
    EFFICIENCY_SHAPED_RETURN_TARGET,
    OUTCOME_FIRST_RETURN_TARGET,
)


@dataclass(frozen=True)
class SimulatorSnapshot:
    flattened_state: Any
    state_progress: Any
    timestep: Any
    done: Any
    ctrl: Any
    mocap_pos: Any
    mocap_quat: Any
    evaluator_state: dict[str, Any]
    environment_state: dict[str, Any]
    simulator_aux_state: dict[str, Any]
    controller_states: tuple[dict[str, Any], ...]
    robot_buffer_states: tuple[dict[str, Any], ...]
    observation_cache: Any
    observable_states: tuple[tuple[str, dict[str, Any]], ...]


_EVALUATOR_STATE_NAMES = (
    "_state_progress",
    "_task_progress",
    "_last_regions",
    "_location_log",
)
_ENVIRONMENT_STATE_NAMES = ("cur_time", "timestep", "done")
_SIMULATOR_AUX_NAMES = (
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "userdata",
)
_ROBOT_BUFFER_NAMES = (
    "torques",
    "recent_ee_forcetorques",
    "recent_ee_pose",
    "recent_ee_vel",
    "recent_ee_vel_buffer",
    "recent_ee_acc",
    "recent_qpos",
    "recent_actions",
    "recent_torques",
)
_OBSERVABLE_STATE_NAMES = (
    "_time_since_last_sample",
    "_current_delay",
    "_current_observed_value",
    "_sampled",
)


@dataclass(frozen=True)
class CounterfactualBranch:
    option: str
    branch_seed: int
    executed_steps: int
    policy_calls: int
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


def _copy_present(value: Any, names: tuple[str, ...]) -> dict[str, Any]:
    return {name: copy.deepcopy(getattr(value, name)) for name in names if hasattr(value, name)}


def _controllers(env: Any) -> tuple[Any, ...]:
    controllers: list[Any] = []
    for robot in getattr(env, "robots", ()):
        selected = getattr(robot, "controller", None)
        if isinstance(selected, dict):
            controllers.extend(selected[key] for key in sorted(selected))
        elif selected is not None:
            controllers.append(selected)
    return tuple(controllers)


def _controller_state(controller: Any) -> dict[str, Any]:
    # The simulator reference must stay live. All remaining OSC fields are
    # numeric/configuration state and are safe to restore by value.
    return {
        name: copy.deepcopy(value) for name, value in controller.__dict__.items() if name != "sim"
    }


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
        evaluator_state=_copy_present(env, _EVALUATOR_STATE_NAMES),
        environment_state=_copy_present(env, _ENVIRONMENT_STATE_NAMES),
        simulator_aux_state=_copy_present(env.sim.data, _SIMULATOR_AUX_NAMES),
        controller_states=tuple(_controller_state(controller) for controller in _controllers(env)),
        robot_buffer_states=tuple(
            _copy_present(robot, _ROBOT_BUFFER_NAMES) for robot in getattr(env, "robots", ())
        ),
        observation_cache=copy.deepcopy(getattr(env, "_obs_cache", None)),
        observable_states=tuple(
            (name, _copy_present(observable, _OBSERVABLE_STATE_NAMES))
            for name, observable in getattr(env, "_observables", {}).items()
        ),
    )


def restore_simulator_snapshot(env: Any, snapshot: SimulatorSnapshot) -> Any:
    env.sim.set_state_from_flattened(snapshot.flattened_state)
    for name, value in snapshot.simulator_aux_state.items():
        getattr(env.sim.data, name)[...] = value
    for name, value in (
        ("ctrl", snapshot.ctrl),
        ("mocap_pos", snapshot.mocap_pos),
        ("mocap_quat", snapshot.mocap_quat),
    ):
        selected = getattr(env.sim.data, name, None)
        if selected is not None and value is not None:
            selected[...] = value
    env.sim.forward()
    for name, value in snapshot.evaluator_state.items():
        setattr(env, name, copy.deepcopy(value))
    for name, value in snapshot.environment_state.items():
        setattr(env, name, copy.deepcopy(value))
    controllers = _controllers(env)
    if len(controllers) != len(snapshot.controller_states):
        raise OursContractError("counterfactual controller inventory changed")
    for controller, state in zip(controllers, snapshot.controller_states, strict=True):
        for name, value in state.items():
            setattr(controller, name, copy.deepcopy(value))
    robots = tuple(getattr(env, "robots", ()))
    if len(robots) != len(snapshot.robot_buffer_states):
        raise OursContractError("counterfactual robot inventory changed")
    for robot, state in zip(robots, snapshot.robot_buffer_states, strict=True):
        for name, value in state.items():
            setattr(robot, name, copy.deepcopy(value))
    env._post_process()
    # mj_forward / post-processing may replace the constraint solver warm
    # start. The captured value must be the one consumed by the next step.
    for name, value in snapshot.simulator_aux_state.items():
        getattr(env.sim.data, name)[...] = value
    if snapshot.observation_cache is not None and hasattr(env, "_obs_cache"):
        env._obs_cache = copy.deepcopy(snapshot.observation_cache)
        observables = getattr(env, "_observables", {})
        if tuple(observables) != tuple(name for name, _ in snapshot.observable_states):
            raise OursContractError("counterfactual observable inventory changed")
        for name, state in snapshot.observable_states:
            for attribute, value in state.items():
                setattr(observables[name], attribute, copy.deepcopy(value))
    else:
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
    for value in (
        snapshot.evaluator_state,
        snapshot.environment_state,
        snapshot.simulator_aux_state,
        snapshot.controller_states,
        snapshot.robot_buffer_states,
        snapshot.observation_cache,
        snapshot.observable_states,
    ):
        digest.update(pickle.dumps(value, protocol=5))
    return digest.hexdigest()


def snapshot_mismatches(
    expected: SimulatorSnapshot,
    actual: SimulatorSnapshot,
    np: Any,
) -> tuple[str, ...]:
    mismatches: list[str] = []
    for name in ("flattened_state", "ctrl", "mocap_pos", "mocap_quat"):
        before = getattr(expected, name)
        after = getattr(actual, name)
        if (before is None) != (after is None) or (
            before is not None and not np.allclose(before, after, rtol=0.0, atol=1e-12)
        ):
            mismatches.append(name)
    for name in ("state_progress", "timestep", "done"):
        if getattr(expected, name) != getattr(actual, name):
            mismatches.append(name)
    for name in (
        "evaluator_state",
        "environment_state",
        "controller_states",
        "robot_buffer_states",
        "simulator_aux_state",
        "observation_cache",
        "observable_states",
    ):
        if pickle.dumps(getattr(expected, name), protocol=5) != pickle.dumps(
            getattr(actual, name), protocol=5
        ):
            mismatches.append(name)
    return tuple(mismatches)


def _predicate_count(env: Any, goal: dict[str, list[list[str]]]) -> int:
    return sum(
        bool(env._eval_predicate(predicate)) for states in goal.values() for predicate in states
    )


def counterfactual_return(
    *,
    progress_gain: int,
    final_success: bool,
    predicate_gain: int,
    executed_steps: int,
    policy_calls: int,
    return_target: str,
) -> float:
    """Score a branch with either legacy efficiency or outcome-first targets."""

    if return_target not in COUNTERFACTUAL_RETURN_TARGETS:
        raise OursContractError(f"unsupported counterfactual return target: {return_target}")
    if min(executed_steps, policy_calls) < 0:
        raise OursContractError("counterfactual return costs must be nonnegative")
    value = 8.0 * progress_gain + 16.0 * int(bool(final_success)) + float(predicate_gain)
    if return_target == EFFICIENCY_SHAPED_RETURN_TARGET:
        value -= 0.002 * executed_steps + 0.01 * policy_calls
    return value


def evaluate_counterfactual_options(
    env: Any,
    *,
    goal: dict[str, list[list[str]]],
    option_actions: dict[RecoveryOption | str, Any],
    completed_subtasks_before: int,
    base_seed: int,
    state_index: int,
    np: Any,
    return_target: str = EFFICIENCY_SHAPED_RETURN_TARGET,
) -> tuple[CounterfactualBranch, ...]:
    """Evaluate fixed candidate actions from one state and restore it exactly."""

    if base_seed not in {10007, 11007, 12007}:
        raise OursContractError("counterfactual branches are restricted to frozen train seeds")
    if state_index < 0 or completed_subtasks_before < 0 or not option_actions:
        raise OursContractError("counterfactual branch state is invalid")
    snapshot = capture_simulator_snapshot(env)
    predicate_before = _predicate_count(env, goal)
    branches: list[CounterfactualBranch] = []
    try:
        for raw_option, actions in option_actions.items():
            option = RecoveryOption(raw_option)
            branch_seed = counterfactual_branch_seed(base_seed, state_index)
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
            return_value = counterfactual_return(
                progress_gain=progress_gain,
                final_success=bool(final_success),
                predicate_gain=predicate_gain,
                executed_steps=executed_steps,
                policy_calls=0,
                return_target=return_target,
            )
            branches.append(
                CounterfactualBranch(
                    option=option.value,
                    branch_seed=branch_seed,
                    executed_steps=executed_steps,
                    policy_calls=0,
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
    restored = capture_simulator_snapshot(env)
    mismatches = snapshot_mismatches(snapshot, restored, np)
    if mismatches:
        raise OursContractError(
            "counterfactual simulator restore exceeded tolerance: " + ", ".join(mismatches)
        )
    return tuple(branches)


def evaluate_counterfactual_rollouts(
    env: Any,
    *,
    goal: dict[str, list[list[str]]],
    option_rollouts: dict[RecoveryOption | str, Any],
    completed_subtasks_before: int,
    base_seed: int,
    state_index: int,
    np: Any,
    return_target: str = EFFICIENCY_SHAPED_RETURN_TARGET,
) -> tuple[CounterfactualBranch, ...]:
    """Evaluate closed-loop option callbacks from one restored training state."""

    if base_seed not in {10007, 11007, 12007}:
        raise OursContractError("counterfactual branches are restricted to frozen train seeds")
    if state_index < 0 or completed_subtasks_before < 0 or not option_rollouts:
        raise OursContractError("counterfactual rollout state is invalid")
    snapshot = capture_simulator_snapshot(env)
    predicate_before = _predicate_count(env, goal)
    branches: list[CounterfactualBranch] = []
    try:
        for raw_option, rollout in option_rollouts.items():
            option = RecoveryOption(raw_option)
            branch_seed = counterfactual_branch_seed(base_seed, state_index)
            random.seed(branch_seed)
            np.random.seed(branch_seed)
            observation = restore_simulator_snapshot(env, snapshot)
            result = rollout(observation, branch_seed)
            if not isinstance(result, tuple) or len(result) != 2 or min(result) < 0:
                raise OursContractError("counterfactual rollout returned invalid cost counts")
            executed_steps, policy_calls = (int(value) for value in result)
            _, completed_after, final_success = env._check_success(goal)
            predicate_after = _predicate_count(env, goal)
            state_after = capture_simulator_snapshot(env)
            progress_gain = int(completed_after) - completed_subtasks_before
            predicate_gain = predicate_after - predicate_before
            return_value = counterfactual_return(
                progress_gain=progress_gain,
                final_success=bool(final_success),
                predicate_gain=predicate_gain,
                executed_steps=executed_steps,
                policy_calls=policy_calls,
                return_target=return_target,
            )
            branches.append(
                CounterfactualBranch(
                    option=option.value,
                    branch_seed=branch_seed,
                    executed_steps=executed_steps,
                    policy_calls=policy_calls,
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
    mismatches = snapshot_mismatches(snapshot, capture_simulator_snapshot(env), np)
    if mismatches:
        raise OursContractError(
            "counterfactual simulator restore exceeded tolerance: " + ", ".join(mismatches)
        )
    return tuple(branches)


def branch_payload(branch: CounterfactualBranch) -> dict[str, Any]:
    return asdict(branch)
