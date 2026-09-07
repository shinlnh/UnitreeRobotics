"""Evaluate B's learned unified STOP/action-prefix execution policy."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
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
    _append_jsonl,
    _apply_shifted_start,
    _clean_partial_trace,
    _configure_environment_imports,
    _dynamic_injection_state,
    _final_predicates_hold,
    _init_state_path,
    _make_environment,
    _maybe_inject_dynamic,
    _percentile,
    _predicate_snapshot,
    _read_jsonl,
    _reset_environment,
    _result_state,
    _trace_frame,
    _write_json,
)
from .a1 import inspect_a1_checkpoint
from .a2 import (
    A2_PLAN_SOURCE,
    A2_PLANNER,
    audit_fixed_hierarchy,
    build_fixed_plan,
    hierarchy_audit_payload,
    plan_payload,
)
from .b import (
    B_ID,
    B_METHOD,
    B_PAPER,
    B_VARIANT,
    StopConfirmationState,
    confirm_stop,
    inspect_selector_checkpoint,
    select_unified_candidate,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robocerebra-source", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--experiment-id", default=B_ID)
    parser.add_argument("--variant", default=B_VARIANT)
    parser.add_argument("--method", default=B_METHOD)
    parser.add_argument("--paper", default=B_PAPER)
    parser.add_argument("--planner", default=A2_PLANNER)
    parser.add_argument("--plan-source", default=A2_PLAN_SOURCE)
    parser.add_argument("--task-types", nargs="+", required=True)
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--execution-horizon", type=int, choices=(8, 16), default=16)
    parser.add_argument("--stop-confirmation-window", type=int, default=2)
    parser.add_argument("--control-frequency-hz", type=int, default=20)
    parser.add_argument("--steps-per-subtask", type=int, default=150)
    parser.add_argument("--initial-wait-steps", type=int, default=15)
    parser.add_argument("--post-success-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5550)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-trace-images", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _run_episode(
    *,
    env: Any,
    case: BenchmarkCase,
    trial: int,
    client: RemotePolicyClient,
    benchmark_dir: Path,
    output_dir: Path,
    execution_horizon: int,
    stop_confirmation_window: int,
    control_frequency_hz: int,
    steps_per_subtask: int,
    initial_wait_steps: int,
    post_success_steps: int,
    seed: int,
    trace_images: bool,
    provenance: dict[str, str],
    np: Any,
) -> dict[str, Any]:
    task = parse_task_description(case.path / "task_description.txt")
    plan = build_fixed_plan(task, steps_per_subtask)
    goal = load_goal(case.path / "goal.json")
    goal_steps = load_goal_steps(case.path / "goal.json")
    episode_seed = seed + trial
    observation, base_reset_source = _reset_environment(
        env, _init_state_path(benchmark_dir, case), episode_seed, np
    )
    observation, excluded_subtasks, start_event = _apply_shifted_start(
        env, case, task, goal, goal_steps
    )
    initial_state_source = base_reset_source
    if start_event is not None:
        start_event["base_reset_source"] = base_reset_source
        start_event["planner_clock_reset"] = True
        initial_state_source = "annotated_demo_shift_state"
    dynamic_state = _dynamic_injection_state(env, case, task, steps_per_subtask, episode_seed)
    client.reset()

    hold_action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    if start_event is None:
        for _ in range(initial_wait_steps):
            observation, _, _, _ = env.step(hold_action)

    _, completed_before, reached_success = env._check_success(goal)
    current_success = bool(reached_success)
    max_steps = plan.max_steps
    first_success_step = 0 if reached_success else None
    stop_after = min(max_steps, post_success_steps) if reached_success else max_steps
    trace_path = output_dir / "decisions.jsonl"
    step = 0
    policy_calls = 0
    predicted_actions = 0
    selector_executed_actions = 0
    active_subgoal = 0
    visited_subgoals: set[int] = set()
    anchors: list[Any] = []
    subgoal_start = True
    confirmation_state = StopConfirmationState()
    selector_stop_proposals = 0
    selector_stop_commits = 0
    selector_prefix_histogram = [0] * (execution_horizon + 1)
    injection_count = 0
    zero_progress_decisions = 0
    policy_latencies: list[float] = []
    started = time.perf_counter()
    termination_reason = "step_budget"

    while step < min(max_steps, stop_after) and active_subgoal < len(plan.subgoals):
        if zero_progress_decisions > max_steps * stop_confirmation_window * 4:
            termination_reason = "selector_zero_progress_guard"
            break
        visited_subgoals.add(active_subgoal)
        instruction = plan.subgoals[active_subgoal]
        policy_observation = build_policy_observation(observation, instruction, np)
        options = {
            "b_selector": {
                "anchor_history": anchors,
                "subgoal_start": subgoal_start,
                "max_prefix": execution_horizon,
                # Use only the frozen global budget. The shorter evaluator-only
                # post-success window must not leak success into selector masks.
                "remaining_steps": max_steps - step,
            }
        }
        policy_started = time.perf_counter()
        raw_action, info = client.get_action_with_info(policy_observation, options)
        policy_latency = time.perf_counter() - policy_started
        policy_latencies.append(policy_latency)
        chunk = unpack_action_chunk(raw_action, np)
        if len(chunk) < 16:
            raise RuntimeError(f"B requires native H16, but checkpoint returned H{len(chunk)}")
        selector_info = info.get("b_selector")
        if not isinstance(selector_info, dict):
            raise RuntimeError("B server response is missing selector metadata")
        expected_runtime = {
            "selector_weights_sha256": provenance["selector_weights_sha256"],
            "a1_checkpoint_weight_shards_sha256": provenance["a1_checkpoint_weight_shards_sha256"],
        }
        if selector_info.get("runtime_provenance") != expected_runtime:
            raise RuntimeError("B server runtime provenance does not match evaluator artifacts")
        scores = np.asarray(selector_info["scores"], dtype=np.float32)
        valid = np.asarray(selector_info["valid"], dtype=np.bool_)
        candidate = int(selector_info["candidate"])
        replayed_candidate = select_unified_candidate(scores.tolist(), valid.tolist())
        if candidate != replayed_candidate:
            raise RuntimeError("B selector decision does not replay from scores and mask")
        if subgoal_start:
            raw_anchor = np.asarray(selector_info.get("raw_anchor"), dtype=np.float32)
            if raw_anchor.shape != (2048,):
                raise RuntimeError("B server returned an invalid subgoal anchor")
            anchors.append(raw_anchor)
            subgoal_start = False

        selection = confirm_stop(
            candidate,
            confirmation_state,
            confirmation_window=stop_confirmation_window,
        )
        confirmation_state = selection.state
        policy_calls += 1
        predicted_actions += len(chunk)
        selector_prefix_histogram[candidate] += 1
        if candidate == 0:
            selector_stop_proposals += 1
        frame_path = (
            _trace_frame(output_dir, case, trial, policy_calls, policy_observation, np)
            if trace_images
            else None
        )
        before_step = step
        completed_at_start = completed_before
        success_at_start = current_success
        predicates_at_start = _predicate_snapshot(env, goal)
        transitions: list[dict[str, Any]] = []
        injections: list[dict[str, Any]] = []

        if selection.stop_committed:
            selector_stop_commits += 1
            active_subgoal += 1
            subgoal_start = active_subgoal < len(plan.subgoals)
            zero_progress_decisions += 1
            if active_subgoal >= len(plan.subgoals):
                termination_reason = "selector_final_stop"
        elif selection.stop_pending:
            zero_progress_decisions += 1
        else:
            zero_progress_decisions = 0
            selected = min(selection.executed_prefix_length, min(max_steps, stop_after) - step)
            for action in chunk[:selected]:
                injection_segment = min(step // steps_per_subtask, len(task.steps) - 1)
                injection = _maybe_inject_dynamic(env, dynamic_state, step, injection_segment)
                if injection is not None:
                    injections.append(injection)
                    injection_count += 1
                libero_action = to_libero_action(action, np)
                hold_action[-1] = libero_action[-1]
                success_before_action = current_success
                completed_before_action = completed_before
                observation, _, _, _ = env.step(libero_action)
                step += 1
                selector_executed_actions += 1
                _, completed_before, now_success = env._check_success(goal)
                current_success = bool(now_success)
                transitions.append(
                    {
                        "step": step,
                        "action": [float(value) for value in libero_action],
                        "controller_hold": False,
                        "injection": injection,
                        "active_subgoal_index": active_subgoal,
                        "completed_subtasks_before": completed_before_action,
                        "completed_subtasks_after": completed_before,
                        "success_before": success_before_action,
                        "success_after": current_success,
                        "result_state": _result_state(env, observation),
                    }
                )
                if now_success and first_success_step is None:
                    first_success_step = step
                    stop_after = min(max_steps, step + post_success_steps)
                if step >= min(max_steps, stop_after):
                    break

        _append_jsonl(
            trace_path,
            {
                "experiment_id": B_ID,
                "variant": B_VARIANT,
                "method": B_METHOD,
                "paper": B_PAPER,
                "protocol": "continuous_no_restore",
                **provenance,
                "task_type": case.task_type,
                "case": case.case_name,
                "trial": trial,
                "seed": episode_seed,
                "initial_state_source": initial_state_source,
                "policy_call": policy_calls,
                "policy_latency_seconds": policy_latency,
                "step_before": before_step,
                "step_after": step,
                "full_task_instruction": task.instruction,
                "instruction": instruction,
                "planner": A2_PLANNER,
                "plan_source": plan.source,
                "plan_sha256": plan.sha256,
                "active_subgoal_index_before": active_subgoal - int(selection.stop_committed),
                "active_subgoal_index_after": active_subgoal,
                "frame_bundle": frame_path,
                "predicted_chunk": chunk.tolist(),
                "configured_execution_horizon": execution_horizon,
                "selector_scores": scores.tolist(),
                "selector_valid": valid.tolist(),
                "selector_candidate": candidate,
                "selector_context_sha256": selector_info["current_context_sha256"],
                "stop_pending": selection.stop_pending,
                "stop_committed": selection.stop_committed,
                "stop_confirmation_streak_after": selection.state.streak,
                "selected_prefix_length": len(transitions),
                "external_termination_truncated": (candidate > 0 and len(transitions) < candidate),
                "prefix_selection": "learned_unified_stop_prefix",
                "transitions": transitions,
                "completed_subtasks_before": completed_at_start,
                "completed_subtasks_after": completed_before,
                "success_before": success_at_start,
                "success_after": current_success,
                "success_predicates_before": predicates_at_start,
                "success_predicates_after": _predicate_snapshot(env, goal),
                "first_success_step": first_success_step,
                "condition_start": start_event if policy_calls == 1 else None,
                "injections": injections,
                "retry": False,
                "recovery": False,
            },
        )

    # Preserve the frozen 80-step reactivation observation window after a
    # successful final STOP. These are controller holds, not selector actions.
    monitoring_before = step
    monitoring_transitions: list[dict[str, Any]] = []
    monitoring_predicates_before = _predicate_snapshot(env, goal)
    while first_success_step is not None and step < min(max_steps, stop_after):
        injection_segment = min(step // steps_per_subtask, len(task.steps) - 1)
        injection = _maybe_inject_dynamic(env, dynamic_state, step, injection_segment)
        if injection is not None:
            injection_count += 1
        success_before_action = current_success
        completed_before_action = completed_before
        observation, _, _, _ = env.step(hold_action)
        step += 1
        _, completed_before, now_success = env._check_success(goal)
        current_success = bool(now_success)
        monitoring_transitions.append(
            {
                "step": step,
                "action": [float(value) for value in hold_action],
                "controller_hold": True,
                "injection": injection,
                "active_subgoal_index": max(0, active_subgoal - 1),
                "completed_subtasks_before": completed_before_action,
                "completed_subtasks_after": completed_before,
                "success_before": success_before_action,
                "success_after": current_success,
                "result_state": _result_state(env, observation),
            }
        )
    if monitoring_transitions:
        _append_jsonl(
            trace_path,
            {
                "experiment_id": B_ID,
                "variant": B_VARIANT,
                "method": B_METHOD,
                "paper": B_PAPER,
                "protocol": "continuous_no_restore",
                **provenance,
                "task_type": case.task_type,
                "case": case.case_name,
                "trial": trial,
                "seed": episode_seed,
                "policy_invoked": False,
                "policy_call": policy_calls,
                "policy_latency_seconds": 0.0,
                "step_before": monitoring_before,
                "step_after": step,
                "instruction": plan.subgoals[-1],
                "plan_sha256": plan.sha256,
                "configured_execution_horizon": execution_horizon,
                "selected_prefix_length": len(monitoring_transitions),
                "prefix_selection": "post_success_controller_hold",
                "transitions": monitoring_transitions,
                "success_predicates_before": monitoring_predicates_before,
                "success_predicates_after": _predicate_snapshot(env, goal),
                "first_success_step": first_success_step,
                "retry": False,
                "recovery": False,
            },
        )

    final_success = _final_predicates_hold(env, goal)
    possible_subtasks = sum(len(states) for states in goal.values()) - excluded_subtasks
    return {
        "experiment_id": B_ID,
        "variant": B_VARIANT,
        "method": B_METHOD,
        "paper": B_PAPER,
        "protocol": "continuous_no_restore",
        **provenance,
        "task_type": case.task_type,
        "case": case.case_name,
        "trial": trial,
        "seed": episode_seed,
        "initial_state_source": initial_state_source,
        "instruction": task.instruction,
        "planner": A2_PLANNER,
        "fixed_hierarchy": False,
        "plan": plan_payload(plan),
        "execution_horizon": execution_horizon,
        "control_frequency_hz": control_frequency_hz,
        "subtasks": len(task.steps),
        "planner_subgoals_visited": len(visited_subgoals),
        "selector_subgoals_completed": active_subgoal,
        "selector_stop_proposals": selector_stop_proposals,
        "selector_stop_commits": selector_stop_commits,
        "selector_prefix_histogram": selector_prefix_histogram,
        "termination_reason": termination_reason,
        "excluded_subtasks": excluded_subtasks,
        "possible_subtasks": possible_subtasks,
        "completed_subtasks": completed_before,
        "agent_completed_subtasks": max(0, completed_before - excluded_subtasks),
        "reached_success": first_success_step is not None,
        "first_success_step": first_success_step,
        "final_success": final_success,
        "post_success_reactivation": first_success_step is not None and not final_success,
        "max_steps": max_steps,
        "steps": step,
        "steps_after_first_success": (
            0 if first_success_step is None else max(0, step - first_success_step)
        ),
        "initial_wait_steps": 0 if start_event is not None else initial_wait_steps,
        "total_simulator_steps": step + (0 if start_event is not None else initial_wait_steps),
        "policy_calls": policy_calls,
        "predicted_actions": predicted_actions,
        "selector_executed_actions": selector_executed_actions,
        "controller_hold_steps": step - selector_executed_actions,
        "action_chunk_utilization": (
            selector_executed_actions / predicted_actions if predicted_actions else 0.0
        ),
        "policy_inference_seconds": sum(policy_latencies),
        "mean_policy_inference_ms": (
            1000.0 * sum(policy_latencies) / len(policy_latencies) if policy_latencies else 0.0
        ),
        "p95_policy_inference_ms": 1000.0 * _percentile(policy_latencies, 0.95),
        "injection_count": injection_count,
        "elapsed_seconds": time.perf_counter() - started,
        "retry": False,
        "recovery": False,
    }


def _run_manifest(
    args: argparse.Namespace,
    contract: Any,
    checkpoint_provenance: dict[str, Any],
    selector_audit: Any,
    selector_provenance: dict[str, Any],
    cases: list[BenchmarkCase],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": B_ID,
        "variant": B_VARIANT,
        "method": B_METHOD,
        "paper": B_PAPER,
        "protocol": "continuous_no_restore",
        "checkpoint": str(contract.checkpoint_dir),
        "checkpoint_source_experiment": checkpoint_provenance["experiment_id"],
        "checkpoint_training_revision": checkpoint_provenance["training_dataset_revision"],
        "selector_checkpoint": str(selector_audit.checkpoint_dir),
        "selector_weights_sha256": selector_audit.weights_sha256,
        "selector_provenance_sha256": selector_audit.provenance_sha256,
        "a1_checkpoint_weight_shards_sha256": selector_provenance[
            "a1_checkpoint_weight_shards_sha256"
        ],
        "selector_training": selector_provenance["training"],
        "benchmark_dir": str(args.benchmark_dir.expanduser().resolve()),
        "robocerebra_source": str(args.robocerebra_source.expanduser().resolve()),
        "benchmark_revision": args.benchmark_revision,
        "model_revision": args.model_revision,
        "dataset_revision": args.dataset_revision,
        "hierarchy": True,
        "planner": A2_PLANNER,
        "planner_mode": "learned_stop_outcome_blind",
        "plan_source": A2_PLAN_SOURCE,
        "planner_observes_task_outcomes": False,
        "planner_replans": False,
        "stop_or_adaptive_chunk": True,
        "retry": False,
        "recovery": False,
        "task_types": args.task_types,
        "cases": [f"{case.task_type}/{case.case_name}" for case in cases],
        "trials_per_case": args.trials,
        "expected_episodes": len(cases) * args.trials,
        "execution_horizon": args.execution_horizon,
        "native_action_horizon": contract.action_horizon,
        "stop_confirmation_window": args.stop_confirmation_window,
        "control_frequency_hz": args.control_frequency_hz,
        "steps_per_subtask": args.steps_per_subtask,
        "initial_wait_steps": args.initial_wait_steps,
        "post_success_steps": args.post_success_steps,
        "base_seed": args.seed,
        "trace_images": not args.no_trace_images,
        "output_dir": str(output_dir),
    }


def _build_summary(
    manifest: dict[str, Any], contract: Any, results: list[dict[str, Any]]
) -> dict[str, Any]:
    episodes = len(results)
    completed_subtasks = sum(result["agent_completed_subtasks"] for result in results)
    possible_subtasks = sum(result["possible_subtasks"] for result in results)
    return {
        **manifest,
        "fixed_hierarchy": False,
        "stop_or_adaptive_chunk": True,
        "retry": False,
        "recovery": False,
        "checkpoint_contract": asdict(contract) | {"checkpoint_dir": str(contract.checkpoint_dir)},
        "episodes": episodes,
        "complete": episodes == manifest["expected_episodes"],
        "completion_rate": episodes / manifest["expected_episodes"],
        "reached_successes": sum(int(result["reached_success"]) for result in results),
        "reached_success_rate": (
            sum(int(result["reached_success"]) for result in results) / episodes
            if episodes
            else 0.0
        ),
        "final_successes": sum(int(result["final_success"]) for result in results),
        "final_success_rate": (
            sum(int(result["final_success"]) for result in results) / episodes if episodes else 0.0
        ),
        "completed_subtasks": completed_subtasks,
        "possible_subtasks": possible_subtasks,
        "subtask_completion_rate": (
            completed_subtasks / possible_subtasks if possible_subtasks else 0.0
        ),
        "total_executed_steps": sum(result["steps"] for result in results),
        "total_policy_calls": sum(result["policy_calls"] for result in results),
        "total_selector_stop_proposals": sum(
            result["selector_stop_proposals"] for result in results
        ),
        "total_selector_stop_commits": sum(result["selector_stop_commits"] for result in results),
        "total_policy_inference_seconds": sum(
            result["policy_inference_seconds"] for result in results
        ),
        "total_elapsed_seconds": sum(result["elapsed_seconds"] for result in results),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.trials, args.control_frequency_hz, args.steps_per_subtask) < 1:
        raise ValueError("B evaluator counts must be positive")
    if (args.experiment_id, args.variant, args.method, args.paper) != (
        B_ID,
        B_VARIANT,
        B_METHOD,
        B_PAPER,
    ):
        raise ValueError("B evaluator identity is frozen")
    if args.planner != A2_PLANNER or args.plan_source != A2_PLAN_SOURCE:
        raise ValueError("B canonical planner source is frozen")
    contract, checkpoint_provenance = inspect_a1_checkpoint(
        args.checkpoint, expected_training_revision=args.model_revision
    )
    selector_audit, selector_provenance = inspect_selector_checkpoint(
        args.selector_checkpoint,
        expected_action_horizon=contract.action_horizon,
        expected_context_width=2048,
    )
    frozen_window = int(selector_provenance["selection"]["stop_confirmation_window"])
    if args.stop_confirmation_window != frozen_window:
        raise ValueError("B STOP confirmation window differs from the frozen checkpoint")

    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    cases = discover_cases(benchmark_dir, args.task_types, args.cases)
    hierarchy_audit = audit_fixed_hierarchy(cases, args.steps_per_subtask)
    if not hierarchy_audit.valid:
        raise ValueError(f"B hierarchy audit failed: {asdict(hierarchy_audit)}")
    output_dir = args.output.expanduser().resolve()
    artifact_names = ("run_manifest.json", "summary.json", "episodes.jsonl", "decisions.jsonl")
    existing = [name for name in artifact_names if (output_dir / name).exists()]
    if existing and not args.resume:
        raise FileExistsError(f"refusing to mix B runs in {output_dir}: {', '.join(existing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _run_manifest(
        args,
        contract,
        checkpoint_provenance,
        selector_audit,
        selector_provenance,
        cases,
        output_dir,
    )
    manifest["hierarchy_audit"] = hierarchy_audit_payload(hierarchy_audit)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("--resume configuration does not match run_manifest.json")
    elif existing:
        raise ValueError("cannot --resume B without run_manifest.json")
    else:
        _write_json(manifest_path, manifest)

    results = _read_jsonl(output_dir / "episodes.jsonl")
    completed_keys = {
        (str(result["task_type"]), str(result["case"]), int(result["trial"])) for result in results
    }
    if len(completed_keys) != len(results):
        raise ValueError("duplicate completed B episode")
    expected_keys = {
        (case.task_type, case.case_name, trial) for case in cases for trial in range(args.trials)
    }
    if completed_keys - expected_keys:
        raise ValueError("resume file contains unexpected B episodes")
    if args.resume:
        _clean_partial_trace(output_dir, completed_keys)

    np, bddl_utils, runtime = _configure_environment_imports(args.robocerebra_source, benchmark_dir)
    provenance = {
        "benchmark_revision": args.benchmark_revision,
        "model_revision": args.model_revision,
        "dataset_revision": args.dataset_revision,
        "selector_weights_sha256": selector_audit.weights_sha256,
        "a1_checkpoint_weight_shards_sha256": selector_provenance[
            "a1_checkpoint_weight_shards_sha256"
        ],
    }
    with RemotePolicyClient(args.policy_host, args.policy_port) as client:
        if not client.ping():
            raise RuntimeError("B policy server did not answer ping")
        for case in cases:
            pending = [
                trial
                for trial in range(args.trials)
                if (case.task_type, case.case_name, trial) not in completed_keys
            ]
            if not pending:
                continue
            env = _make_environment(case, bddl_utils, runtime, args.control_frequency_hz)
            try:
                for trial in pending:
                    result = _run_episode(
                        env=env,
                        case=case,
                        trial=trial,
                        client=client,
                        benchmark_dir=benchmark_dir,
                        output_dir=output_dir,
                        execution_horizon=args.execution_horizon,
                        stop_confirmation_window=args.stop_confirmation_window,
                        control_frequency_hz=args.control_frequency_hz,
                        steps_per_subtask=args.steps_per_subtask,
                        initial_wait_steps=args.initial_wait_steps,
                        post_success_steps=args.post_success_steps,
                        seed=args.seed,
                        trace_images=not args.no_trace_images,
                        provenance=provenance,
                        np=np,
                    )
                    results.append(result)
                    _append_jsonl(output_dir / "episodes.jsonl", result)
                    _write_json(
                        output_dir / "progress.json", _build_summary(manifest, contract, results)
                    )
            finally:
                env.close()

    summary = _build_summary(manifest, contract, results)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "progress.json", summary)
    return summary


def main() -> int:
    summary = run(_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
