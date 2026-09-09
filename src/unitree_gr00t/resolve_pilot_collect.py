"""Collect paired R/D/B programs for the first RESOLVE physical pilot.

The collector is restricted to frozen training seeds and non-dynamic
RoboCerebra conditions. Every distinct arm is evaluated after resetting one
environment and exactly replaying the source action prefix, with the same
request-local GR00T diffusion seeds. Mathematically identical arms are executed
once and reused so simulator path noise cannot become a causal label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

from .a0 import (
    BenchmarkCase,
    RemotePolicyClient,
    build_policy_observation,
    discover_cases,
    load_goal,
    load_goal_steps,
    parse_task_description,
    to_libero_action,
    unpack_action_chunk,
)
from .a0_eval import (
    _apply_shifted_start,
    _configure_environment_imports,
    _init_state_path,
    _make_environment,
    _predicate_snapshot,
    _reset_environment,
)
from .a1 import sha256_file
from .b import select_unified_candidate, selector_decision_seed
from .ours import OursContractError
from .ours_counterfactual import (
    capture_simulator_snapshot,
    restore_simulator_snapshot,
    snapshot_mismatches,
    snapshot_sha256,
)
from .ours_counterfactual_prepare import _set_trace_state
from .ours_counterfactual_rollout_prepare import (
    RESIDUAL_SOURCE_SEEDS,
    _apply_residual_stop_confirmation,
    _validate_residual_source,
)
from .ours_data import OURS_CORPUS_MANIFEST, audit_recovery_corpus
from .ours_rollout_prepare import _episode_key, _read_jsonl, _write_json
from .resolve import RESOLVE_METHOD, RESOLVE_VARIANT, counterfactual_rescue_bottleneck

PILOT_SCHEMA_VERSION = 1
PILOT_MILESTONES = (
    "new_goal_predicate",
    "next_ordered_subtask",
    "final_success",
)
PILOT_TASK_TYPES = frozenset({"Ideal", "Memory_Execution", "Memory_Exploration"})


@dataclass(frozen=True)
class RecoveryProgram:
    program_id: str
    steering_codes: Any
    prefix_lengths: tuple[int, ...]
    subgoal_offsets: tuple[int, ...]


@dataclass(frozen=True)
class ControllerState:
    active_subgoal: int
    stop_streak: int
    retry_attempt_index: int
    subgoal_start: bool


@dataclass(frozen=True)
class ArmResult:
    reachability: Any
    completed_subtasks_after: int
    predicate_count_after: int
    reached_predicates: tuple[str, ...]
    final_success: bool
    executed_steps: int
    policy_calls: int
    final_state_sha256: str
    action_trace_sha256: str
    action_trace: Any
    recovery_contexts: Any
    base_chunks: Any
    recovery_chunks: Any
    selector_candidates: Any
    recovery_state_sha256s: tuple[str, ...]
    recovery_snapshots: tuple[Any, ...]
    all_context_sha256s: tuple[str, ...]
    all_base_chunk_sha256s: tuple[str, ...]
    all_state_sha256s: tuple[str, ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--source-corpus", type=Path, required=True)
    parser.add_argument("--robocerebra-source", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--sample-indices", type=int, nargs="+", required=True)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5559)
    parser.add_argument("--program-count", type=int, default=8)
    parser.add_argument("--program-depth", type=int, default=2)
    parser.add_argument("--code-dimension", type=int, default=4)
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--minimum-angle", type=float, default=0.06)
    parser.add_argument("--maximum-angle", type=float, default=0.24)
    parser.add_argument(
        "--proposal-center",
        type=float,
        nargs="+",
        help="Flattened depth-by-code-dimension center for local latent search",
    )
    parser.add_argument("--proposal-standard-deviation", type=float, default=0.03)
    parser.add_argument("--proposal-frozen-prefix", type=int, default=0)
    parser.add_argument(
        "--subgoal-offsets",
        type=int,
        nargs="+",
        help="One relative subgoal transition per recovery-program slot",
    )
    parser.add_argument("--rollout-steps", type=int, default=150)
    parser.add_argument("--max-policy-calls", type=int, default=48)
    parser.add_argument("--program-seed", type=int, default=41007)
    return parser


def generate_recovery_programs(
    *,
    count: int,
    depth: int,
    code_dimension: int,
    prefix_length: int,
    minimum_angle: float,
    maximum_angle: float,
    seed: int,
    np: Any,
    subgoal_offsets: tuple[int, ...] | None = None,
) -> tuple[RecoveryProgram, ...]:
    """Generate deterministic continuous programs without enumerating options."""

    if (
        min(count, depth, code_dimension, prefix_length) < 1
        or seed < 0
        or not 0.0 < minimum_angle <= maximum_angle <= math.pi / 2
    ):
        raise OursContractError("RESOLVE pilot program configuration is invalid")
    offsets = (0,) * depth if subgoal_offsets is None else subgoal_offsets
    if len(offsets) != depth:
        raise OursContractError("RESOLVE pilot subgoal offsets have invalid depth")
    rng = np.random.default_rng(seed)
    programs: list[RecoveryProgram] = []
    for index in range(count):
        directions = rng.normal(size=(depth, code_dimension)).astype(np.float32)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        if (norms <= 1e-12).any():
            raise OursContractError("RESOLVE pilot sampled a zero steering direction")
        magnitudes = rng.uniform(minimum_angle, maximum_angle, size=(depth, 1))
        codes = directions / norms * magnitudes.astype(np.float32)
        programs.append(
            RecoveryProgram(
                program_id=f"program-{seed}-{index:04d}",
                steering_codes=codes,
                prefix_lengths=(prefix_length,) * depth,
                subgoal_offsets=tuple(int(value) for value in offsets),
            )
        )
    return tuple(programs)


def generate_centered_recovery_programs(
    *,
    count: int,
    center_codes: Any,
    standard_deviation: float,
    prefix_length: int,
    frozen_prefix: int = 0,
    minimum_angle: float,
    maximum_angle: float,
    seed: int,
    np: Any,
    subgoal_offsets: tuple[int, ...] | None = None,
) -> tuple[RecoveryProgram, ...]:
    """Generate one exact center and Gaussian local proposals on bounded rays."""

    center = np.asarray(center_codes, dtype=np.float32)
    if (
        min(count, prefix_length) < 1
        or center.ndim != 2
        or min(center.shape) < 1
        or not 0 <= frozen_prefix <= center.shape[0]
        or not np.isfinite(center).all()
        or not math.isfinite(standard_deviation)
        or standard_deviation <= 0.0
        or not 0.0 < minimum_angle <= maximum_angle <= math.pi / 2
        or seed < 0
    ):
        raise OursContractError("RESOLVE centered proposal configuration is invalid")
    offsets = (0,) * center.shape[0] if subgoal_offsets is None else subgoal_offsets
    if len(offsets) != center.shape[0]:
        raise OursContractError("RESOLVE centered subgoal offsets have invalid depth")
    rng = np.random.default_rng(seed)
    programs: list[RecoveryProgram] = []
    for index in range(count):
        codes = center.copy()
        if index:
            codes[frozen_prefix:] += rng.normal(
                0.0,
                standard_deviation,
                size=codes[frozen_prefix:].shape,
            ).astype(np.float32)
        norms = np.linalg.norm(codes, axis=1, keepdims=True)
        if (norms <= 1e-12).any():
            raise OursContractError("RESOLVE centered proposal produced a zero code")
        bounded_norms = np.clip(norms, minimum_angle, maximum_angle)
        codes = codes / norms * bounded_norms
        programs.append(
            RecoveryProgram(
                program_id=f"centered-program-{seed}-{index:04d}",
                steering_codes=codes.astype(np.float32),
                prefix_lengths=(prefix_length,) * center.shape[0],
                subgoal_offsets=tuple(int(value) for value in offsets),
            )
        )
    return tuple(programs)


def _flatten_predicates(snapshot: dict[str, list[bool]]) -> tuple[bool, ...]:
    return tuple(value for states in snapshot.values() for value in states)


def newly_reached_predicates(
    before: dict[str, list[bool]], after: dict[str, list[bool]]
) -> tuple[str, ...]:
    if before.keys() != after.keys() or any(
        len(before[name]) != len(after[name]) for name in before
    ):
        raise OursContractError("RESOLVE predicate snapshots are incompatible")
    return tuple(
        f"{name}:{index}"
        for name in before
        for index, (left, right) in enumerate(zip(before[name], after[name], strict=True))
        if not left and right
    )


def physical_reachability(
    *,
    predicates_before: dict[str, list[bool]],
    predicates_after: dict[str, list[bool]],
    completed_before: int,
    completed_after: int,
    final_success: bool,
    np: Any,
) -> Any:
    """Return milestone indicators; costs never enter these labels."""

    before = _flatten_predicates(predicates_before)
    after = _flatten_predicates(predicates_after)
    if len(before) != len(after) or completed_before < 0 or completed_after < 0:
        raise OursContractError("RESOLVE pilot physical outcome is invalid")
    return np.asarray(
        [
            any(not left and right for left, right in zip(before, after, strict=True)),
            completed_after > completed_before,
            bool(final_success),
        ],
        dtype=np.float32,
    )


def program_crb_targets(
    recovery: Any, deletions: Any, baselines: Any, *, np: Any
) -> tuple[Any, Any]:
    """Compute per-slot and deletion-minimal whole-program CRB targets."""

    recovery = np.asarray(recovery, dtype=np.float32)
    deletions = np.asarray(deletions, dtype=np.float32)
    baselines = np.asarray(baselines, dtype=np.float32)
    if (
        recovery.shape != (len(PILOT_MILESTONES),)
        or deletions.ndim != 2
        or deletions.shape != baselines.shape
        or deletions.shape[1] != len(PILOT_MILESTONES)
        or len(deletions) < 1
    ):
        raise OursContractError("RESOLVE pilot R/D/B targets have invalid shapes")
    per_slot = np.stack(
        [
            counterfactual_rescue_bottleneck(recovery, deletion, baseline, np=np)
            for deletion, baseline in zip(deletions, baselines, strict=True)
        ]
    )
    return per_slot, per_slot.min(axis=0)


def canonical_baseline_sources(program_depth: int) -> tuple[tuple[str, int], ...]:
    """Return the unique execution source for every structural B_i estimand.

    B_0 is independent of the sampled recovery program and is shared across all
    programs at an anchor. At the final slot, B_last and D_last are the exact
    same policy, so B_last must reuse D_last rather than rerun it.
    """

    if program_depth < 1:
        raise OursContractError("RESOLVE pilot program depth must be positive")
    last = program_depth - 1
    return tuple(
        ("deletion", last)
        if index == last
        else (("shared_baseline", 0) if index == 0 else ("handoff", index))
        for index in range(program_depth)
    )


def _query_mosaic(
    client: RemotePolicyClient,
    observation: dict[str, Any],
    instruction: str,
    *,
    anchors: list[Any],
    subgoal_start: bool,
    episode_seed: int,
    decision_index: int,
    remaining_steps: int,
    program_id: str,
    rule_index: int,
    steering_code: Any,
    prefix_length: int,
    expected_runtime: dict[str, Any],
    np: Any,
) -> tuple[Any, Any, Any, int, Any, Any | None]:
    raw_action, info = client.get_action_with_info(
        build_policy_observation(observation, instruction, np),
        {
            "b_selector": {
                "anchor_history": anchors,
                "subgoal_start": subgoal_start,
                "max_prefix": 16,
                "episode_seed": episode_seed,
                "decision_index": decision_index,
                "remaining_steps": remaining_steps,
                "subgoal_elapsed_steps": 0,
                "capture_training_context": True,
            },
            "mosaic": {
                "steering_code": np.asarray(steering_code, dtype=np.float32),
                "prefix_length": prefix_length,
                "program_id": program_id,
                "rule_index": rule_index,
            },
        },
    )
    selector = info.get("b_selector")
    steering = info.get("mosaic")
    if (
        not isinstance(selector, dict)
        or not isinstance(steering, dict)
        or selector.get("runtime_provenance") != expected_runtime
        or selector.get("episode_seed") != episode_seed
        or selector.get("decision_index") != decision_index
        or selector.get("decision_seed") != selector_decision_seed(episode_seed, decision_index)
    ):
        raise OursContractError("RESOLVE pilot server provenance does not match")
    scores = np.asarray(selector["scores"], dtype=np.float32)
    valid = np.asarray(selector["valid"], dtype=np.bool_)
    candidate = int(selector["candidate"])
    if candidate != select_unified_candidate(scores.tolist(), valid.tolist()):
        raise OursContractError("RESOLVE pilot selector decision does not replay")
    context = np.asarray(selector.get("training_context"), dtype=np.float32)
    if context.shape != (2048,) or not np.isfinite(context).all():
        raise OursContractError("RESOLVE pilot server omitted its training context")
    raw_anchor = selector.get("raw_anchor")
    anchor = None if raw_anchor is None else np.asarray(raw_anchor, dtype=np.float32)
    if subgoal_start and (anchor is None or anchor.shape != (2048,)):
        raise OursContractError("RESOLVE pilot server returned an invalid anchor")
    chunk = unpack_action_chunk(raw_action, np)
    if chunk.shape != (16, 7) or not np.isfinite(chunk).all():
        raise OursContractError("RESOLVE pilot server returned an invalid action chunk")
    return chunk, scores, valid, candidate, context, anchor


def _zero_code(code_dimension: int, np: Any) -> Any:
    return np.zeros((code_dimension,), dtype=np.float32)


def _execute_actions(
    env: Any,
    observation: dict[str, Any],
    chunk: Any,
    count: int,
    *,
    remaining_steps: int,
    action_trace: list[Any],
    after_step: Any,
    np: Any,
) -> tuple[dict[str, Any], int]:
    selected = min(count, remaining_steps)
    for action in chunk[:selected]:
        libero_action = to_libero_action(action, np)
        action_trace.append(libero_action.copy())
        observation, _, _, _ = env.step(libero_action)
        after_step()
    return observation, selected


def _execute_recovery_macro(
    env: Any,
    observation: dict[str, Any],
    state: ControllerState,
    *,
    subgoals: tuple[str, ...],
    anchors: list[Any],
    episode_seed: int,
    decision_index: int,
    remaining_steps: int,
    action_trace: list[Any],
    after_step: Any,
    program: RecoveryProgram,
    rule_index: int,
    client: RemotePolicyClient,
    expected_runtime: dict[str, Any],
    np: Any,
) -> tuple[dict[str, Any], ControllerState, int, dict[str, Any]]:
    if state.active_subgoal >= len(subgoals):
        return observation, state, 0, {}
    code = np.asarray(program.steering_codes[rule_index], dtype=np.float32)
    prefix = int(program.prefix_lengths[rule_index])
    target_subgoal = max(
        0,
        min(
            len(subgoals) - 1,
            state.active_subgoal + int(program.subgoal_offsets[rule_index]),
        ),
    )
    starts_new_subgoal = state.subgoal_start or target_subgoal != state.active_subgoal
    common = {
        "anchors": anchors,
        "subgoal_start": starts_new_subgoal,
        "episode_seed": episode_seed,
        "decision_index": decision_index,
        "remaining_steps": remaining_steps,
        "program_id": program.program_id,
        "rule_index": rule_index,
        "prefix_length": prefix,
        "expected_runtime": expected_runtime,
        "np": np,
    }
    base = _query_mosaic(
        client,
        observation,
        subgoals[target_subgoal],
        steering_code=_zero_code(code.shape[0], np),
        **common,
    )
    recovery = _query_mosaic(
        client,
        observation,
        subgoals[target_subgoal],
        steering_code=code,
        **common,
    )
    base_chunk, _, _, base_candidate, base_context, base_anchor = base
    recovery_chunk, _, _, recovery_candidate, recovery_context, recovery_anchor = recovery
    if not np.array_equal(base_context, recovery_context):
        raise OursContractError("RESOLVE paired R/B contexts diverged before intervention")
    if starts_new_subgoal:
        if (
            base_anchor is None
            or recovery_anchor is None
            or not np.array_equal(base_anchor, recovery_anchor)
        ):
            raise OursContractError("RESOLVE paired R/B anchors diverged")
        anchors.append(recovery_anchor)
    observation, executed = _execute_actions(
        env,
        observation,
        recovery_chunk,
        prefix,
        remaining_steps=remaining_steps,
        action_trace=action_trace,
        after_step=after_step,
        np=np,
    )
    next_state = replace(
        state,
        active_subgoal=target_subgoal,
        stop_streak=0,
        retry_attempt_index=1,
        subgoal_start=False,
    )
    post_snapshot = capture_simulator_snapshot(env)
    record = {
        "context": recovery_context,
        "base_chunk": base_chunk,
        "recovery_chunk": recovery_chunk,
        "base_candidate": base_candidate,
        "recovery_candidate": recovery_candidate,
        "executed_prefix": executed,
        "post_state_sha256": snapshot_sha256(post_snapshot, np),
        "post_snapshot": post_snapshot,
    }
    return observation, next_state, executed, record


def _execute_baseline_macro(
    env: Any,
    observation: dict[str, Any],
    state: ControllerState,
    *,
    subgoals: tuple[str, ...],
    anchors: list[Any],
    episode_seed: int,
    decision_index: int,
    remaining_steps: int,
    action_trace: list[Any],
    after_step: Any,
    code_dimension: int,
    client: RemotePolicyClient,
    expected_runtime: dict[str, Any],
    np: Any,
) -> tuple[dict[str, Any], ControllerState, int, int, dict[str, Any]]:
    """Execute one physical B macro, including zero-time STOP transitions."""

    calls = 0
    last_record: dict[str, Any] = {}
    while state.active_subgoal < len(subgoals) and calls < 8:
        chunk, _, _, candidate, context, anchor = _query_mosaic(
            client,
            observation,
            subgoals[state.active_subgoal],
            anchors=anchors,
            subgoal_start=state.subgoal_start,
            episode_seed=episode_seed,
            decision_index=decision_index + calls,
            remaining_steps=remaining_steps,
            program_id="exact-B",
            rule_index=0,
            steering_code=_zero_code(code_dimension, np),
            prefix_length=1,
            expected_runtime=expected_runtime,
            np=np,
        )
        calls += 1
        if state.subgoal_start:
            assert anchor is not None
            anchors.append(anchor)
            state = replace(state, subgoal_start=False)
        last_record = {
            "context": context,
            "base_chunk": chunk,
            "recovery_chunk": chunk,
            "base_candidate": candidate,
            "recovery_candidate": candidate,
            "executed_prefix": 0,
            "post_state_sha256": None,
            "post_snapshot": None,
        }
        if candidate > 0:
            observation, executed = _execute_actions(
                env,
                observation,
                chunk,
                candidate,
                remaining_steps=remaining_steps,
                action_trace=action_trace,
                after_step=after_step,
                np=np,
            )
            state = replace(state, stop_streak=0)
            last_record["executed_prefix"] = executed
            post_snapshot = capture_simulator_snapshot(env)
            last_record["post_state_sha256"] = snapshot_sha256(post_snapshot, np)
            last_record["post_snapshot"] = post_snapshot
            return observation, state, executed, calls, last_record
        (
            next_subgoal,
            next_streak,
            next_attempt,
            retry_triggered,
            advanced,
        ) = _apply_residual_stop_confirmation(
            target_subgoal=state.active_subgoal,
            subgoal_count=len(subgoals),
            stop_streak=state.stop_streak,
            retry_attempt_index=state.retry_attempt_index,
        )
        state = ControllerState(
            active_subgoal=next_subgoal,
            stop_streak=next_streak,
            retry_attempt_index=next_attempt,
            subgoal_start=retry_triggered or advanced,
        )
    return observation, state, 0, calls, last_record


def _stack_records(
    records: list[dict[str, Any]], name: str, shape: tuple[int, ...], np: Any
) -> Any:
    values = [record[name] for record in records if record]
    if not values:
        return np.empty((0, *shape), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def _evaluate_arm(
    env: Any,
    observation: dict[str, Any],
    *,
    snapshot: Any,
    goal: dict[str, list[list[str]]],
    predicates_before: dict[str, list[bool]],
    completed_before: int,
    initial_state: ControllerState,
    source_anchors: list[Any],
    subgoals: tuple[str, ...],
    episode_seed: int,
    first_decision_index: int,
    program: RecoveryProgram,
    deletion_index: int | None,
    handoff_index: int | None,
    rollout_steps: int,
    max_policy_calls: int,
    client: RemotePolicyClient,
    expected_runtime: dict[str, Any],
    np: Any,
) -> ArmResult:
    """Evaluate R, one D_i, or one B_i from the identical source state."""

    if snapshot is not None:
        observation = restore_simulator_snapshot(env, snapshot)
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    client.reset({"episode_seed": episode_seed})
    state = initial_state
    anchors = [anchor.copy() for anchor in source_anchors]
    executed_steps = 0
    policy_calls = 0
    records: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    action_trace: list[Any] = []
    reachability = np.zeros(len(PILOT_MILESTONES), dtype=np.float32)
    maximum_completed = completed_before
    maximum_predicates = sum(_flatten_predicates(predicates_before))
    reached_final_success = False
    reached_predicates: set[str] = set()

    def observe_milestones() -> None:
        nonlocal maximum_completed, maximum_predicates, reached_final_success
        _, completed_now, success_now = env._check_success(goal)
        predicates_now = _predicate_snapshot(env, goal)
        reached_predicates.update(newly_reached_predicates(predicates_before, predicates_now))
        reachability[:] = np.maximum(
            reachability,
            physical_reachability(
                predicates_before=predicates_before,
                predicates_after=predicates_now,
                completed_before=completed_before,
                completed_after=int(completed_now),
                final_success=bool(success_now),
                np=np,
            ),
        )
        maximum_completed = max(maximum_completed, int(completed_now))
        maximum_predicates = max(maximum_predicates, sum(_flatten_predicates(predicates_now)))
        reached_final_success = reached_final_success or bool(success_now)

    depth = len(program.prefix_lengths)
    for rule_index in range(depth):
        if handoff_index == rule_index:
            break
        remaining = rollout_steps - executed_steps
        if remaining <= 0 or state.active_subgoal >= len(subgoals):
            break
        decision_index = first_decision_index + policy_calls
        if deletion_index == rule_index:
            observation, state, executed, calls, record = _execute_baseline_macro(
                env,
                observation,
                state,
                subgoals=subgoals,
                anchors=anchors,
                episode_seed=episode_seed,
                decision_index=decision_index,
                remaining_steps=remaining,
                action_trace=action_trace,
                after_step=observe_milestones,
                code_dimension=program.steering_codes.shape[1],
                client=client,
                expected_runtime=expected_runtime,
                np=np,
            )
            policy_calls += calls
        else:
            observation, state, executed, record = _execute_recovery_macro(
                env,
                observation,
                state,
                subgoals=subgoals,
                anchors=anchors,
                episode_seed=episode_seed,
                decision_index=decision_index,
                remaining_steps=remaining,
                action_trace=action_trace,
                after_step=observe_milestones,
                program=program,
                rule_index=rule_index,
                client=client,
                expected_runtime=expected_runtime,
                np=np,
            )
            policy_calls += 1
        executed_steps += executed
        records.append(record)
        if record:
            all_records.append(record)
        if policy_calls >= max_policy_calls:
            break

    while (
        executed_steps < rollout_steps
        and policy_calls < max_policy_calls
        and state.active_subgoal < len(subgoals)
    ):
        observation, state, executed, calls, record = _execute_baseline_macro(
            env,
            observation,
            state,
            subgoals=subgoals,
            anchors=anchors,
            episode_seed=episode_seed,
            decision_index=first_decision_index + policy_calls,
            remaining_steps=rollout_steps - executed_steps,
            action_trace=action_trace,
            after_step=observe_milestones,
            code_dimension=program.steering_codes.shape[1],
            client=client,
            expected_runtime=expected_runtime,
            np=np,
        )
        executed_steps += executed
        policy_calls += calls
        if record:
            all_records.append(record)
        if calls == 0:
            break

    observe_milestones()
    final_snapshot = capture_simulator_snapshot(env)
    actions = np.asarray(action_trace, dtype="<f4")
    return ArmResult(
        reachability=reachability,
        completed_subtasks_after=maximum_completed,
        predicate_count_after=maximum_predicates,
        reached_predicates=tuple(sorted(reached_predicates)),
        final_success=reached_final_success,
        executed_steps=executed_steps,
        policy_calls=policy_calls,
        final_state_sha256=snapshot_sha256(final_snapshot, np),
        action_trace_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
        action_trace=actions,
        recovery_contexts=_stack_records(records, "context", (2048,), np),
        base_chunks=_stack_records(records, "base_chunk", (16, 7), np),
        recovery_chunks=_stack_records(records, "recovery_chunk", (16, 7), np),
        selector_candidates=np.asarray(
            [record["base_candidate"] for record in records if record], dtype=np.int16
        ),
        recovery_state_sha256s=tuple(
            str(record["post_state_sha256"]) for record in records if record
        ),
        recovery_snapshots=tuple(
            record["post_snapshot"]
            for record in records
            if record and record["post_snapshot"] is not None
        ),
        all_context_sha256s=tuple(
            hashlib.sha256(np.asarray(record["context"], dtype="<f4").tobytes()).hexdigest()
            for record in all_records
        ),
        all_base_chunk_sha256s=tuple(
            hashlib.sha256(np.asarray(record["base_chunk"], dtype="<f4").tobytes()).hexdigest()
            for record in all_records
        ),
        all_state_sha256s=tuple(str(record["post_state_sha256"]) for record in all_records),
    )


def _arm_payload(result: ArmResult) -> dict[str, Any]:
    return {
        "reachability": result.reachability.tolist(),
        "completed_subtasks_after": result.completed_subtasks_after,
        "predicate_count_after": result.predicate_count_after,
        "reached_predicates": list(result.reached_predicates),
        "final_success": result.final_success,
        "executed_steps": result.executed_steps,
        "policy_calls": result.policy_calls,
        "final_state_sha256": result.final_state_sha256,
        "action_trace_sha256": result.action_trace_sha256,
    }


def _fresh_replay_environment(
    *,
    existing_env: Any | None = None,
    case: Any,
    task: Any,
    goal: dict[str, list[list[str]]],
    goal_steps: Any,
    prefix_rows: list[dict[str, Any]],
    benchmark_dir: Path,
    bddl_utils: Any,
    runtime: Any,
    control_frequency_hz: int,
    initial_wait_steps: int,
    reset_seed: int,
    np: Any,
) -> tuple[Any, dict[str, Any]]:
    """Rebuild a causal controller state by replaying, never teleporting."""

    env = existing_env
    if env is None:
        env = _make_environment(
            BenchmarkCase(case.task_type, case.case_name, case.path),
            bddl_utils,
            runtime,
            control_frequency_hz,
        )
    observation, _ = _reset_environment(env, _init_state_path(benchmark_dir, case), reset_seed, np)
    observation, _, start_event = _apply_shifted_start(env, case, task, goal, goal_steps)
    if start_event is None:
        hold = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        for _ in range(initial_wait_steps):
            observation, _, _, _ = env.step(hold)
    env._check_success(goal)
    for row in prefix_rows:
        for transition in row["transitions"]:
            action = np.asarray(transition["action"], dtype=np.float32)
            if action.shape != (7,):
                env.close()
                raise OursContractError("RESOLVE source replay action is invalid")
            observation, _, _, _ = env.step(action)
            env._check_success(goal)
    return env, observation


def _evaluate_fresh_arm(environment_factory: Any, **kwargs: Any) -> ArmResult:
    env, observation = environment_factory()
    try:
        return _evaluate_arm(env, observation, snapshot=None, **kwargs)
    finally:
        env.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        min(
            args.program_count,
            args.program_depth,
            args.code_dimension,
            args.prefix_length,
            args.rollout_steps,
            args.max_policy_calls,
        )
        < 1
        or args.prefix_length > 16
        or len(set(args.sample_indices)) != len(args.sample_indices)
    ):
        raise ValueError("RESOLVE pilot collection arguments are invalid")
    rollout = args.rollout.expanduser().resolve()
    source_corpus = args.source_corpus.expanduser().resolve()
    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite RESOLVE pilot: {destination}")
    audit_recovery_corpus(
        source_corpus,
        verify_hashes=True,
        require_development=False,
        require_both_completion_classes=False,
    )
    source_manifest_path = source_corpus / OURS_CORPUS_MANIFEST
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    run_manifest_path = rollout / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    summary_path = rollout / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    base_seed = int(run_manifest["base_seed"])
    if base_seed not in RESIDUAL_SOURCE_SEEDS or not bool(summary.get("complete")):
        raise OursContractError("RESOLVE pilot source is not a complete train rollout")
    _validate_residual_source(run_manifest, summary)

    np, bddl_utils, runtime = _configure_environment_imports(
        args.robocerebra_source.expanduser().resolve(), benchmark_dir
    )
    decisions: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in _read_jsonl(rollout / "decisions.jsonl"):
        if row.get("policy_invoked") is not False:
            decisions.setdefault(_episode_key(row), []).append(row)
    cases = {
        (case.task_type, case.case_name): case
        for case in discover_cases(benchmark_dir, run_manifest["task_types"], [])
    }
    expected_runtime = {
        "selector_weights_sha256": source_manifest["selector_weights_sha256"],
        "a1_checkpoint_weight_shards_sha256": source_manifest["a1_checkpoint_weight_shards_sha256"],
    }
    requested = set(args.sample_indices)
    found: set[int] = set()
    files_sha256: dict[str, str] = {}
    record_path = destination / "programs.jsonl"
    feature_dir = destination / "features"
    feature_dir.mkdir(parents=True)
    client = RemotePolicyClient(args.policy_host, args.policy_port)
    try:
        if not client.ping():
            raise RuntimeError("RESOLVE pilot policy server did not answer ping")
        for episode_index, key in enumerate(sorted(decisions)):
            rows = sorted(decisions[key], key=lambda row: int(row["policy_call"]))
            source_path = source_corpus / "features" / f"episode_{episode_index:06d}.npz"
            with np.load(source_path, allow_pickle=False) as source:
                sample_indices = source["sample_indices"].astype(np.int64)
            positions = {
                int(sample): position
                for position, sample in enumerate(sample_indices)
                if int(sample) in requested
            }
            if not positions:
                continue
            if key[0] not in PILOT_TASK_TYPES:
                raise OursContractError(
                    "RESOLVE R0 excludes dynamic conditions until evaluator RNG restore "
                    f"is wired: {key[0]}/{key[1]}"
                )
            case = cases[(key[0], key[1])]
            task = parse_task_description(case.path / "task_description.txt")
            goal = load_goal(case.path / "goal.json")
            goal_steps = load_goal_steps(case.path / "goal.json")
            env = _make_environment(
                BenchmarkCase(case.task_type, case.case_name, case.path),
                bddl_utils,
                runtime,
                int(run_manifest["control_frequency_hz"]),
            )
            try:
                observation, _ = _reset_environment(
                    env,
                    _init_state_path(benchmark_dir, case),
                    base_seed + key[2],
                    np,
                )
                observation, _, start_event = _apply_shifted_start(
                    env, case, task, goal, goal_steps
                )
                if start_event is None:
                    hold = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
                    for _ in range(int(run_manifest["initial_wait_steps"])):
                        observation, _, _, _ = env.step(hold)
                anchors: list[Any] = []
                for position, row in enumerate(rows):
                    env._check_success(goal)
                    raw_anchor = row.get("new_subgoal_anchor")
                    if raw_anchor is not None:
                        anchor = np.asarray(raw_anchor, dtype=np.float32)
                        if anchor.shape != (2048,):
                            raise OursContractError("RESOLVE source anchor is invalid")
                        anchors.append(anchor)
                    sample_index = int(sample_indices[position])
                    if sample_index in positions:
                        replay_match = (
                            _predicate_snapshot(env, goal) == row["success_predicates_before"]
                        )
                        if not replay_match:
                            raise OursContractError(
                                f"RESOLVE selected source state {sample_index} did not replay"
                            )
                        previous = rows[position - 1] if position else None
                        stop_streak = int(
                            previous["stop_confirmation_streak_after"]
                            if previous is not None
                            and int(previous["active_subgoal_index_after"])
                            == int(row["active_subgoal_index_before"])
                            else 0
                        )
                        if int(row["selector_candidate_before_recovery"]) != 0 or stop_streak < 1:
                            raise OursContractError(
                                "RESOLVE R0 anchors must be confirmed B-retry STOP states"
                            )
                        initial_state = ControllerState(
                            active_subgoal=int(row["active_subgoal_index_before"]),
                            stop_streak=stop_streak,
                            retry_attempt_index=int(row.get("retry_attempt_index_before", 0)),
                            subgoal_start=False,
                        )
                        environment_factory = partial(
                            _fresh_replay_environment,
                            case=case,
                            task=task,
                            goal=goal,
                            goal_steps=goal_steps,
                            prefix_rows=rows[:position],
                            benchmark_dir=benchmark_dir,
                            bddl_utils=bddl_utils,
                            runtime=runtime,
                            control_frequency_hz=int(run_manifest["control_frequency_hz"]),
                            initial_wait_steps=int(run_manifest["initial_wait_steps"]),
                            reset_seed=base_seed + key[2],
                            np=np,
                        )
                        reference_env, _ = environment_factory()
                        try:
                            predicates_before = _predicate_snapshot(reference_env, goal)
                            if predicates_before != row["success_predicates_before"]:
                                raise OursContractError(
                                    "RESOLVE fresh action replay did not recover source anchor"
                                )
                        finally:
                            reference_env.close()
                        completed_before = int(row["completed_subtasks_before"])
                        episode_seed = int(row["seed"])
                        first_decision_index = int(row["policy_call"]) - 1
                        source_max_steps = int(run_manifest["steps_per_subtask"]) * len(task.steps)
                        effective_rollout_steps = min(
                            args.rollout_steps,
                            source_max_steps - int(row["step_before"]),
                        )
                        if effective_rollout_steps < 1:
                            raise OursContractError(
                                "RESOLVE source anchor has no global budget remaining"
                            )
                        subgoal_offsets = (
                            (0,) * args.program_depth
                            if args.subgoal_offsets is None
                            else tuple(args.subgoal_offsets)
                        )
                        if len(subgoal_offsets) != args.program_depth:
                            raise OursContractError(
                                "RESOLVE subgoal offsets must match program depth"
                            )
                        if args.proposal_center is None:
                            programs = generate_recovery_programs(
                                count=args.program_count,
                                depth=args.program_depth,
                                code_dimension=args.code_dimension,
                                prefix_length=args.prefix_length,
                                minimum_angle=args.minimum_angle,
                                maximum_angle=args.maximum_angle,
                                seed=args.program_seed + sample_index,
                                np=np,
                                subgoal_offsets=subgoal_offsets,
                            )
                        else:
                            center = np.asarray(args.proposal_center, dtype=np.float32)
                            if center.size != (args.program_depth * args.code_dimension):
                                raise OursContractError("RESOLVE proposal center has invalid size")
                            programs = generate_centered_recovery_programs(
                                count=args.program_count,
                                center_codes=center.reshape(
                                    args.program_depth, args.code_dimension
                                ),
                                standard_deviation=(args.proposal_standard_deviation),
                                prefix_length=args.prefix_length,
                                minimum_angle=args.minimum_angle,
                                maximum_angle=args.maximum_angle,
                                seed=args.program_seed + sample_index,
                                np=np,
                                subgoal_offsets=subgoal_offsets,
                            )
                        shared_baseline_zero: ArmResult | None = None
                        for program_index, program in enumerate(programs):
                            branch_env, _ = environment_factory()
                            replay_same_environment = partial(
                                _fresh_replay_environment,
                                existing_env=branch_env,
                                **environment_factory.keywords,
                            )
                            common = {
                                "goal": goal,
                                "predicates_before": predicates_before,
                                "completed_before": completed_before,
                                "initial_state": initial_state,
                                "source_anchors": anchors,
                                "subgoals": task.steps,
                                "episode_seed": episode_seed,
                                "first_decision_index": first_decision_index,
                                "program": program,
                                "rollout_steps": effective_rollout_steps,
                                "max_policy_calls": args.max_policy_calls,
                                "client": client,
                                "expected_runtime": expected_runtime,
                                "np": np,
                            }

                            def evaluate_program_arm(
                                *,
                                deletion_index: int | None,
                                handoff_index: int | None,
                                _replay: Any = replay_same_environment,
                                _branch_env: Any = branch_env,
                                _common: dict[str, Any] = common,
                            ) -> ArmResult:
                                _, arm_observation = _replay()
                                return _evaluate_arm(
                                    _branch_env,
                                    arm_observation,
                                    snapshot=None,
                                    deletion_index=deletion_index,
                                    handoff_index=handoff_index,
                                    **_common,
                                )

                            recovery = evaluate_program_arm(
                                deletion_index=None,
                                handoff_index=None,
                            )
                            if args.program_depth == 1:
                                if shared_baseline_zero is None:
                                    shared_baseline_zero = evaluate_program_arm(
                                        deletion_index=None,
                                        handoff_index=0,
                                    )
                                deletions = [shared_baseline_zero]
                            else:
                                deletions = [
                                    evaluate_program_arm(
                                        deletion_index=index,
                                        handoff_index=None,
                                    )
                                    for index in range(args.program_depth)
                                ]
                            baselines: list[ArmResult] = []
                            for source, index in canonical_baseline_sources(args.program_depth):
                                if source == "deletion":
                                    result = deletions[index]
                                elif source == "shared_baseline":
                                    if shared_baseline_zero is None:
                                        shared_baseline_zero = evaluate_program_arm(
                                            deletion_index=None,
                                            handoff_index=0,
                                        )
                                    result = shared_baseline_zero
                                else:
                                    result = evaluate_program_arm(
                                        deletion_index=None,
                                        handoff_index=index,
                                    )
                                baselines.append(result)
                            # At the final structural slot, deleting recovery
                            # and handing off to B are exactly the same policy.
                            # Any difference is numerical/RNG contamination,
                            # not a causal recovery effect.
                            last_deletion = deletions[-1]
                            last_baseline = baselines[-1]
                            deterministic_fields = (
                                last_deletion is last_baseline,
                                last_deletion.action_trace_sha256
                                == last_baseline.action_trace_sha256,
                                last_deletion.final_state_sha256
                                == last_baseline.final_state_sha256,
                                last_deletion.executed_steps == last_baseline.executed_steps,
                                last_deletion.policy_calls == last_baseline.policy_calls,
                            )
                            if not all(deterministic_fields):
                                shared_steps = min(
                                    len(last_deletion.action_trace),
                                    len(last_baseline.action_trace),
                                )
                                action_delta = (
                                    np.abs(
                                        last_deletion.action_trace[:shared_steps]
                                        - last_baseline.action_trace[:shared_steps]
                                    )
                                    if shared_steps
                                    else np.empty((0,), dtype=np.float32)
                                )
                                different_rows = (
                                    np.flatnonzero((action_delta > 0).any(axis=1))
                                    if action_delta.ndim == 2
                                    else np.empty((0,), dtype=np.int64)
                                )
                                raise OursContractError(
                                    "RESOLVE D_last/B_last deterministic invariant failed: "
                                    + json.dumps(
                                        {
                                            "D_action": last_deletion.action_trace_sha256,
                                            "B_action": last_baseline.action_trace_sha256,
                                            "D_state": last_deletion.final_state_sha256,
                                            "B_state": last_baseline.final_state_sha256,
                                            "D_steps": last_deletion.executed_steps,
                                            "B_steps": last_baseline.executed_steps,
                                            "D_calls": last_deletion.policy_calls,
                                            "B_calls": last_baseline.policy_calls,
                                            "shared_steps": shared_steps,
                                            "maximum_action_delta": (
                                                float(action_delta.max())
                                                if action_delta.size
                                                else None
                                            ),
                                            "first_action_delta": (
                                                action_delta[0].tolist()
                                                if action_delta.size
                                                else None
                                            ),
                                            "first_different_action_index": (
                                                int(different_rows[0])
                                                if different_rows.size
                                                else None
                                            ),
                                            "first_different_action_delta": (
                                                action_delta[int(different_rows[0])].tolist()
                                                if different_rows.size
                                                else None
                                            ),
                                            "D_prefix_sha256": hashlib.sha256(
                                                last_deletion.recovery_chunks[:-1]
                                                .astype("<f4")
                                                .tobytes()
                                            ).hexdigest(),
                                            "B_prefix_sha256": hashlib.sha256(
                                                last_baseline.recovery_chunks.astype(
                                                    "<f4"
                                                ).tobytes()
                                            ).hexdigest(),
                                            "D_prefix_states": (
                                                last_deletion.recovery_state_sha256s[:-1]
                                            ),
                                            "B_prefix_states": (
                                                last_baseline.recovery_state_sha256s
                                            ),
                                            "D_contexts": (last_deletion.all_context_sha256s),
                                            "B_contexts": (last_baseline.all_context_sha256s),
                                            "D_chunks": (last_deletion.all_base_chunk_sha256s),
                                            "B_chunks": (last_baseline.all_base_chunk_sha256s),
                                            "D_states": last_deletion.all_state_sha256s,
                                            "B_states": last_baseline.all_state_sha256s,
                                            "prefix_snapshot_mismatches": (
                                                list(
                                                    snapshot_mismatches(
                                                        last_deletion.recovery_snapshots[0],
                                                        last_baseline.recovery_snapshots[0],
                                                        np,
                                                    )
                                                )
                                                if last_deletion.recovery_snapshots
                                                and last_baseline.recovery_snapshots
                                                else None
                                            ),
                                        },
                                        sort_keys=True,
                                    )
                                )
                            deletion_targets = np.stack(
                                [result.reachability for result in deletions]
                            )
                            baseline_targets = np.stack(
                                [result.reachability for result in baselines]
                            )
                            per_slot, program_target = program_crb_targets(
                                recovery.reachability,
                                deletion_targets,
                                baseline_targets,
                                np=np,
                            )
                            filename = f"state_{sample_index:06d}_program_{program_index:04d}.npz"
                            feature_path = feature_dir / filename
                            np.savez_compressed(
                                feature_path,
                                steering_codes=np.asarray(program.steering_codes, dtype=np.float32),
                                prefix_lengths=np.asarray(program.prefix_lengths, dtype=np.int16),
                                subgoal_offsets=np.asarray(program.subgoal_offsets, dtype=np.int16),
                                recovery_contexts=recovery.recovery_contexts.astype(np.float16),
                                base_chunks=recovery.base_chunks.astype(np.float16),
                                recovery_chunks=recovery.recovery_chunks.astype(np.float16),
                                selector_candidates=recovery.selector_candidates,
                                recovery_target=recovery.reachability,
                                deletion_targets=deletion_targets,
                                baseline_targets=baseline_targets,
                                per_slot_crb=per_slot,
                                program_crb=program_target,
                            )
                            files_sha256[filename] = sha256_file(feature_path)
                            with record_path.open("a", encoding="utf-8") as handle:
                                handle.write(
                                    json.dumps(
                                        {
                                            "schema_version": PILOT_SCHEMA_VERSION,
                                            "task_type": key[0],
                                            "case": key[1],
                                            "trial": key[2],
                                            "episode_index": episode_index,
                                            "sample_index": sample_index,
                                            "policy_call": int(row["policy_call"]),
                                            "source_step_before": int(row["step_before"]),
                                            "source_max_steps": source_max_steps,
                                            "effective_rollout_steps": (effective_rollout_steps),
                                            "program_id": program.program_id,
                                            "steering_codes": program.steering_codes.tolist(),
                                            "prefix_lengths": list(program.prefix_lengths),
                                            "subgoal_offsets": list(program.subgoal_offsets),
                                            "milestones": list(PILOT_MILESTONES),
                                            "recovery": _arm_payload(recovery),
                                            "deletions": [
                                                _arm_payload(result) for result in deletions
                                            ],
                                            "baselines": [
                                                _arm_payload(result) for result in baselines
                                            ],
                                            "per_slot_crb": per_slot.tolist(),
                                            "program_crb": program_target.tolist(),
                                            "features": filename,
                                        },
                                        separators=(",", ":"),
                                    )
                                    + "\n"
                                )
                            branch_env.close()
                        found.add(sample_index)
                    transitions = row["transitions"]
                    if transitions:
                        _set_trace_state(env, transitions[-1]["result_state"], np)
            finally:
                env.close()
    finally:
        client.close()
    if found != requested:
        raise OursContractError(
            f"RESOLVE pilot did not find requested samples: {sorted(requested - found)}"
        )
    manifest = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "experiment_id": "Ours",
        "variant": RESOLVE_VARIANT,
        "method": RESOLVE_METHOD,
        "stage": "R0-paired-physical-program-pilot",
        "source_rollout": str(rollout),
        "source_run_manifest_sha256": sha256_file(run_manifest_path),
        "source_summary_sha256": sha256_file(summary_path),
        "source_corpus_manifest_sha256": sha256_file(source_manifest_path),
        "base_seed": base_seed,
        "sample_indices": sorted(found),
        "program_count_per_state": args.program_count,
        "program_depth": args.program_depth,
        "program_seed": args.program_seed,
        "proposal": (
            "isotropic-random" if args.proposal_center is None else "bounded-local-Gaussian"
        ),
        "proposal_center": args.proposal_center,
        "proposal_standard_deviation": (
            None if args.proposal_center is None else args.proposal_standard_deviation
        ),
        "proposal_frozen_prefix": (
            None if args.proposal_center is None else args.proposal_frozen_prefix
        ),
        "code_dimension": args.code_dimension,
        "angle_interval": [args.minimum_angle, args.maximum_angle],
        "prefix_length": args.prefix_length,
        "subgoal_offsets": (
            [0] * args.program_depth if args.subgoal_offsets is None else args.subgoal_offsets
        ),
        "physical_horizon": args.rollout_steps,
        "maximum_physical_horizon": args.rollout_steps,
        "global_step_budget_contract": (
            "min-configured-horizon-and-source-global-steps-remaining-v1"
        ),
        "max_policy_calls": args.max_policy_calls,
        "milestones": list(PILOT_MILESTONES),
        "worlds": "one full R, every D_i, and every B_i per program",
        "dynamic_conditions_included": False,
        "pairing": (
            "same environment reset plus exact source-action replay for every "
            "distinct arm; same init state and request-local diffusion seed; "
            "mathematically identical arms evaluated once and reused"
        ),
        "canonical_identical_arm_reuse": [
            "B_0 shared across programs at the same anchor",
            "D_last reused as B_last",
        ],
        "supersedes_incomplete_snapshot_artifacts": True,
        "reward_shaping": False,
        "program_records_sha256": sha256_file(record_path),
        "files": len(files_sha256),
        "files_sha256": files_sha256,
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def main() -> int:
    print(json.dumps(run(_parser().parse_args()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
