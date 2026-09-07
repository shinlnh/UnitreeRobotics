"""A2 continuous evaluator with a frozen, outcome-blind subgoal hierarchy."""

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
    A2_ID,
    A2_PLAN_SOURCE,
    A2_PLANNER,
    A2_VARIANT,
    audit_fixed_hierarchy,
    build_fixed_plan,
    hierarchy_audit_payload,
    plan_payload,
    select_fixed_prefix_length,
    select_subgoal,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the fixed A2 hierarchy")
    parser.add_argument("--robocerebra-source", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--experiment-id", default=A2_ID)
    parser.add_argument("--variant", default=A2_VARIANT)
    parser.add_argument("--planner", default=A2_PLANNER)
    parser.add_argument("--plan-source", default=A2_PLAN_SOURCE)
    parser.add_argument("--task-types", nargs="+", required=True)
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--execution-horizon", type=int, default=16)
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
    control_frequency_hz: int,
    steps_per_subtask: int,
    initial_wait_steps: int,
    post_success_steps: int,
    seed: int,
    trace_images: bool,
    provenance: dict[str, str],
    experiment_id: str,
    variant: str,
    planner_name: str,
    np: Any,
) -> dict[str, Any]:
    task = parse_task_description(case.path / "task_description.txt")
    plan = build_fixed_plan(task, steps_per_subtask)
    goal = load_goal(case.path / "goal.json")
    goal_steps = load_goal_steps(case.path / "goal.json")
    episode_seed = seed + trial
    observation, base_reset_source = _reset_environment(
        env,
        _init_state_path(benchmark_dir, case),
        episode_seed,
        np,
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

    dummy_action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    if start_event is None:
        for _ in range(initial_wait_steps):
            observation, _, _, _ = env.step(dummy_action)

    _, completed_before, reached_success = env._check_success(goal)
    current_success = bool(reached_success)
    max_steps = plan.max_steps
    first_success_step = 0 if reached_success else None
    stop_after = min(max_steps, post_success_steps) if reached_success else max_steps
    trace_path = output_dir / "decisions.jsonl"
    step = 0
    policy_calls = 0
    predicted_actions = 0
    fixed_anchor_truncations = 0
    visited_subgoals: set[int] = set()
    policy_latencies: list[float] = []
    started = time.perf_counter()

    while step < min(max_steps, stop_after):
        planner_decision = select_subgoal(plan, step)
        visited_subgoals.add(planner_decision.subgoal_index)
        policy_observation = build_policy_observation(
            observation, planner_decision.subgoal_instruction, np
        )
        policy_started = time.perf_counter()
        raw_action = client.get_action(policy_observation)
        policy_latency = time.perf_counter() - policy_started
        policy_latencies.append(policy_latency)
        chunk = unpack_action_chunk(raw_action, np)
        if execution_horizon > len(chunk):
            raise RuntimeError(
                f"Requested H{execution_horizon}, but the checkpoint returned H{len(chunk)}"
            )
        policy_calls += 1
        predicted_actions += len(chunk)
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
        remaining_episode_steps = min(max_steps, stop_after) - step
        selected_prefix_length = select_fixed_prefix_length(
            planner_decision,
            execution_horizon,
            remaining_episode_steps,
        )
        anchor_truncated = selected_prefix_length < min(execution_horizon, remaining_episode_steps)
        if anchor_truncated:
            fixed_anchor_truncations += 1

        for action in chunk[:selected_prefix_length]:
            segment = planner_decision.subgoal_index
            injection = _maybe_inject_dynamic(env, dynamic_state, step, segment)
            if injection is not None:
                injections.append(injection)
            libero_action = to_libero_action(action, np)
            success_before_action = current_success
            completed_before_action = completed_before
            observation, _, _, _ = env.step(libero_action)
            step += 1
            _, completed_before, now_success = env._check_success(goal)
            current_success = bool(now_success)
            transitions.append(
                {
                    "step": step,
                    "action": [float(value) for value in libero_action],
                    "injection": injection,
                    "active_subgoal_index": planner_decision.subgoal_index,
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
                "experiment_id": experiment_id,
                "variant": variant,
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
                "instruction": planner_decision.subgoal_instruction,
                "planner": planner_name,
                "plan_source": plan.source,
                "plan_sha256": plan.sha256,
                "planner_decision": asdict(planner_decision),
                "frame_bundle": frame_path,
                "predicted_chunk": chunk.tolist(),
                "configured_execution_horizon": execution_horizon,
                "selected_prefix_length": len(transitions),
                "prefix_selection": (
                    "fixed_anchor_boundary" if anchor_truncated else "fixed_execution_horizon"
                ),
                "transitions": transitions,
                "completed_subtasks_before": completed_at_start,
                "completed_subtasks_after": completed_before,
                "success_before": success_at_start,
                "success_after": current_success,
                "success_predicates_before": predicates_at_start,
                "success_predicates_after": _predicate_snapshot(env, goal),
                "first_success_step": first_success_step,
                "condition_start": start_event if before_step == 0 else None,
                "injections": injections,
            },
        )

    final_success = _final_predicates_hold(env, goal)
    return {
        "experiment_id": experiment_id,
        "variant": variant,
        "protocol": "continuous_no_restore",
        **provenance,
        "task_type": case.task_type,
        "case": case.case_name,
        "trial": trial,
        "seed": episode_seed,
        "initial_state_source": initial_state_source,
        "instruction": task.instruction,
        "planner": planner_name,
        "fixed_hierarchy": True,
        "plan": plan_payload(plan),
        "execution_horizon": execution_horizon,
        "control_frequency_hz": control_frequency_hz,
        "subtasks": len(task.steps),
        "planner_subgoals_visited": len(visited_subgoals),
        "fixed_anchor_truncations": fixed_anchor_truncations,
        "excluded_subtasks": excluded_subtasks,
        "possible_subtasks": sum(len(states) for states in goal.values()) - excluded_subtasks,
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
        "action_chunk_utilization": step / predicted_actions if predicted_actions else 0.0,
        "policy_inference_seconds": sum(policy_latencies),
        "mean_policy_inference_ms": (
            1000.0 * sum(policy_latencies) / len(policy_latencies) if policy_latencies else 0.0
        ),
        "p95_policy_inference_ms": 1000.0 * _percentile(policy_latencies, 0.95),
        "injection_count": (
            0 if dynamic_state is None else len(dynamic_state["schedule"] & set(range(step + 1)))
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def _run_manifest(
    args: argparse.Namespace,
    contract: Any,
    checkpoint_provenance: dict[str, Any],
    cases: list[BenchmarkCase],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "variant": args.variant,
        "protocol": "continuous_no_restore",
        "checkpoint": str(contract.checkpoint_dir),
        "checkpoint_source_experiment": checkpoint_provenance["experiment_id"],
        "checkpoint_training_revision": checkpoint_provenance["training_dataset_revision"],
        "benchmark_dir": str(args.benchmark_dir.expanduser().resolve()),
        "robocerebra_source": str(args.robocerebra_source.expanduser().resolve()),
        "benchmark_revision": args.benchmark_revision,
        "model_revision": args.model_revision,
        "dataset_revision": args.dataset_revision,
        "hierarchy": True,
        "planner": args.planner,
        "planner_mode": "fixed_anchor_outcome_blind",
        "plan_source": args.plan_source,
        "planner_observes_images": False,
        "planner_observes_task_outcomes": False,
        "planner_replans": False,
        "stop_or_adaptive_chunk": False,
        "retry": False,
        "recovery": False,
        "task_types": args.task_types,
        "cases": [f"{case.task_type}/{case.case_name}" for case in cases],
        "trials_per_case": args.trials,
        "expected_episodes": len(cases) * args.trials,
        "execution_horizon": args.execution_horizon,
        "native_action_horizon": contract.action_horizon,
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
    final_successes = sum(int(result["final_success"]) for result in results)
    reached_successes = sum(int(result["reached_success"]) for result in results)
    completed_subtasks = sum(result["agent_completed_subtasks"] for result in results)
    possible_subtasks = sum(result["possible_subtasks"] for result in results)
    episodes = len(results)
    return {
        **manifest,
        "hierarchy": True,
        "fixed_hierarchy": True,
        "stop_or_adaptive_chunk": False,
        "retry": False,
        "recovery": False,
        "checkpoint_contract": asdict(contract) | {"checkpoint_dir": str(contract.checkpoint_dir)},
        "episodes": episodes,
        "complete": episodes == manifest["expected_episodes"],
        "completion_rate": episodes / manifest["expected_episodes"],
        "reached_successes": reached_successes,
        "reached_success_rate": reached_successes / episodes if episodes else 0.0,
        "final_successes": final_successes,
        "final_success_rate": final_successes / episodes if episodes else 0.0,
        "completed_subtasks": completed_subtasks,
        "possible_subtasks": possible_subtasks,
        "subtask_completion_rate": (
            completed_subtasks / possible_subtasks if possible_subtasks else 0.0
        ),
        "total_executed_steps": sum(result["steps"] for result in results),
        "total_policy_calls": sum(result["policy_calls"] for result in results),
        "total_planner_subgoals_visited": sum(
            result["planner_subgoals_visited"] for result in results
        ),
        "total_fixed_anchor_truncations": sum(
            result["fixed_anchor_truncations"] for result in results
        ),
        "total_policy_inference_seconds": sum(
            result["policy_inference_seconds"] for result in results
        ),
        "total_elapsed_seconds": sum(result["elapsed_seconds"] for result in results),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    if args.execution_horizon < 1:
        raise ValueError("--execution-horizon must be positive")
    if args.control_frequency_hz < 1:
        raise ValueError("--control-frequency-hz must be positive")
    if args.steps_per_subtask < 1:
        raise ValueError("--steps-per-subtask must be positive")
    if args.experiment_id != A2_ID or args.variant != A2_VARIANT:
        raise ValueError("A2 evaluator identity is frozen")
    if args.planner != A2_PLANNER or args.plan_source != A2_PLAN_SOURCE:
        raise ValueError("A2 planner contract is frozen")
    contract, checkpoint_provenance = inspect_a1_checkpoint(
        args.checkpoint, expected_training_revision=args.model_revision
    )
    if args.execution_horizon > contract.action_horizon:
        raise ValueError(f"--execution-horizon cannot exceed checkpoint H{contract.action_horizon}")

    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    cases = discover_cases(benchmark_dir, args.task_types, args.cases)
    hierarchy_audit = audit_fixed_hierarchy(cases, args.steps_per_subtask)
    if not hierarchy_audit.valid:
        raise ValueError(f"A2 hierarchy audit failed: {asdict(hierarchy_audit)}")
    output_dir = args.output.expanduser().resolve()
    existing = [
        name
        for name in ("run_manifest.json", "summary.json", "episodes.jsonl", "decisions.jsonl")
        if (output_dir / name).exists()
    ]
    if existing and not args.resume:
        raise FileExistsError(
            f"Refusing to mix A2 runs in {output_dir}; existing files: {', '.join(existing)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _run_manifest(args, contract, checkpoint_provenance, cases, output_dir)
    manifest["hierarchy_audit"] = hierarchy_audit_payload(hierarchy_audit)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual_manifest != manifest:
            raise ValueError("--resume configuration does not match run_manifest.json")
    elif existing:
        raise ValueError("Cannot --resume an A2 directory without run_manifest.json")
    else:
        _write_json(manifest_path, manifest)

    results = _read_jsonl(output_dir / "episodes.jsonl")
    completed_keys: set[tuple[str, str, int]] = set()
    for result in results:
        key = (str(result["task_type"]), str(result["case"]), int(result["trial"]))
        if key in completed_keys:
            raise ValueError(f"Duplicate completed episode in resume file: {key}")
        completed_keys.add(key)
    expected_keys = {
        (case.task_type, case.case_name, trial) for case in cases for trial in range(args.trials)
    }
    unexpected = completed_keys - expected_keys
    if unexpected:
        raise ValueError(f"Resume file contains unexpected episodes: {sorted(unexpected)!r}")
    if args.resume:
        _clean_partial_trace(output_dir, completed_keys)

    np, bddl_utils, runtime = _configure_environment_imports(args.robocerebra_source, benchmark_dir)
    provenance = {
        "benchmark_revision": args.benchmark_revision,
        "model_revision": args.model_revision,
        "dataset_revision": args.dataset_revision,
    }
    with RemotePolicyClient(args.policy_host, args.policy_port) as client:
        if not client.ping():
            raise RuntimeError("GR00T policy server did not answer ping")
        for case in cases:
            pending_trials = [
                trial
                for trial in range(args.trials)
                if (case.task_type, case.case_name, trial) not in completed_keys
            ]
            if not pending_trials:
                continue
            env = _make_environment(case, bddl_utils, runtime, args.control_frequency_hz)
            try:
                for trial in pending_trials:
                    result = _run_episode(
                        env=env,
                        case=case,
                        trial=trial,
                        client=client,
                        benchmark_dir=benchmark_dir,
                        output_dir=output_dir,
                        execution_horizon=args.execution_horizon,
                        control_frequency_hz=args.control_frequency_hz,
                        steps_per_subtask=args.steps_per_subtask,
                        initial_wait_steps=args.initial_wait_steps,
                        post_success_steps=args.post_success_steps,
                        seed=args.seed,
                        trace_images=not args.no_trace_images,
                        provenance=provenance,
                        experiment_id=args.experiment_id,
                        variant=args.variant,
                        planner_name=args.planner,
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
    args = _parser().parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
