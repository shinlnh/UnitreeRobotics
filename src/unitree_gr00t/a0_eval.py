"""Frozen no-hierarchy RoboCerebra rollout runner used by A0 and A1."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .a0 import (
    A0_ID,
    A0_VARIANT,
    BenchmarkCase,
    RemotePolicyClient,
    build_policy_observation,
    discover_cases,
    inspect_checkpoint,
    load_goal,
    load_goal_steps,
    parse_task_description,
    to_libero_action,
    unpack_action_chunk,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a no-hierarchy policy on RoboCerebra")
    parser.add_argument("--robocerebra-source", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--experiment-id", default=A0_ID)
    parser.add_argument("--variant", default=A0_VARIANT)
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


def _configure_environment_imports(
    robocerebra_source: Path, benchmark_dir: Path
) -> tuple[Any, Any, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    libero_root = robocerebra_source.expanduser().resolve() / "LIBERO"
    if not libero_root.is_dir():
        raise FileNotFoundError(f"RoboCerebra LIBERO source is missing: {libero_root}")
    libero_package_root = libero_root / "libero" / "libero"
    libero_config_dir = benchmark_dir.parent / "libero-config"
    libero_config_dir.mkdir(parents=True, exist_ok=True)
    config_path = libero_config_dir / "config.yaml"
    _write_json(
        config_path,
        {
            "benchmark_root": str(libero_package_root),
            "bddl_files": str(libero_package_root / "bddl_files"),
            "init_states": str(libero_package_root / "init_files"),
            "datasets": str(libero_root / "datasets"),
            "assets": str(libero_package_root / "assets"),
        },
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(libero_config_dir)
    sys.path.insert(0, str(libero_root))
    try:
        import libero.libero.envs.bddl_utils as bddl_utils
        import numpy as np
        from libero.libero.envs import TASK_MAPPING
        from robosuite import load_controller_config
    except ImportError as exc:
        raise RuntimeError(
            "A0 environment is not installed. Run scripts/setup_robocerebra_a0.sh first."
        ) from exc
    return np, bddl_utils, (TASK_MAPPING, load_controller_config)


def _make_environment(
    case: BenchmarkCase,
    bddl_utils: Any,
    runtime: Any,
    control_frequency_hz: int,
    bddl_file: Path | None = None,
) -> Any:
    task_mapping, load_controller_config = runtime
    if bddl_file is None:
        bddl_files = tuple(case.path.glob("*.bddl"))
        if len(bddl_files) != 1:
            raise RuntimeError(f"Expected one BDDL file in {case.path}, found {len(bddl_files)}")
        selected_bddl = bddl_files[0]
    else:
        selected_bddl = bddl_file
        if not selected_bddl.is_file() or selected_bddl.parent != case.path:
            raise RuntimeError(f"Selected BDDL does not belong to {case.path}: {selected_bddl}")
    problem = bddl_utils.get_problem_info(str(selected_bddl))
    problem_name = problem["problem_name"]
    if problem_name not in task_mapping:
        raise RuntimeError(f"LIBERO problem is not registered: {problem_name}")
    return task_mapping[problem_name](
        bddl_file_name=str(selected_bddl),
        robots=["Panda"],
        controller_configs=load_controller_config(default_controller="OSC_POSE"),
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        camera_heights=256,
        camera_widths=256,
        control_freq=control_frequency_hz,
    )


def _init_state_path(benchmark_dir: Path, case: BenchmarkCase) -> Path:
    task_type = (
        "Ideal"
        if case.task_type in {"Ideal", "Observation_Mismatching", "Random_Disturbance"}
        else case.task_type
    )
    return benchmark_dir / "init_files" / task_type / f"{case.case_name}.init"


def _reset_environment(env: Any, init_path: Path, seed: int, np: Any) -> tuple[dict[str, Any], str]:
    random.seed(seed)
    np.random.seed(seed)
    if hasattr(env, "seed"):
        env.seed(seed)
    env.reset()
    if not init_path.is_file():
        # Match RoboCerebra's public evaluator: a missing optional .init file
        # leaves the environment at its seeded reset state.
        return env._get_observations(), "seeded_environment_reset"
    with init_path.open("rb") as handle:
        initial_state = pickle.load(handle)  # noqa: S301 - pinned, local benchmark asset
    env.sim.set_state_from_flattened(initial_state)
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return env._get_observations(), "frozen_init_file"


def _load_demo_state(case: BenchmarkCase, frame_index: int) -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("A0 evaluator requires h5py for shifted benchmark states") from exc
    path = case.path / "demo.hdf5"
    if not path.is_file():
        raise FileNotFoundError(f"Demonstration state file is missing: {path}")
    with h5py.File(path, "r") as handle:
        states = handle["data"]["demo_1"]["states"]
        if frame_index < 0 or frame_index >= len(states):
            raise IndexError(
                f"Demo frame {frame_index} is outside [0, {len(states)}) for {case.path}"
            )
        return states[frame_index]


def _apply_shifted_start(
    env: Any,
    case: BenchmarkCase,
    task: Any,
    goal: dict[str, list[list[str]]],
    goal_steps: dict[str, list[int]],
) -> tuple[dict[str, Any], int, dict[str, Any] | None]:
    if case.task_type not in {"Observation_Mismatching", "Mix"}:
        return env._get_observations(), 0, None
    if len(task.start_indices) < 2:
        raise RuntimeError(f"{case.task_type}/{case.case_name} has no second step state")

    shifted_step = 1
    shifted_frame = task.start_indices[shifted_step]
    env.sim.set_state_from_flattened(_load_demo_state(case, shifted_frame))
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    env._check_success(goal)

    excluded = 0
    for object_name, steps in goal_steps.items():
        completed = sum(step < shifted_step for step in steps)
        env._state_progress[object_name] = completed
        excluded += completed
    event = {
        "type": "observation_mismatch_start",
        "task_step": shifted_step,
        "demo_frame": shifted_frame,
        "excluded_subtasks": excluded,
    }
    return env._get_observations(), excluded, event


def _find_object_y_address(env: Any, object_name: str) -> int | None:
    for joint_name in (
        f"{object_name}_1_joint0",
        f"{object_name}_joint0",
        f"{object_name}_joint",
    ):
        if joint_name in env.sim.model.joint_names:
            address = env.sim.model.get_joint_qpos_addr(joint_name)
            return int(address[0] if isinstance(address, (list, tuple)) else address) + 1
    return None


def _dynamic_injection_state(
    env: Any, case: BenchmarkCase, task: Any, steps_per_subtask: int, seed: int
) -> dict[str, Any] | None:
    if case.task_type not in {"Random_Disturbance", "Mix"}:
        return None
    description_path = case.path / "task_description.json"
    entries = json.loads(description_path.read_text(encoding="utf-8"))
    step_objects = [str(entry.get("object", "")) for entry in entries]
    related: list[tuple[int, float] | None] = []
    for object_name in step_objects:
        address = _find_object_y_address(env, object_name)
        related.append(
            (address, float(env.sim.data.qpos[address])) if address is not None else None
        )
    related_names = set(step_objects)
    unrelated: list[tuple[str, int, float]] = []
    for object_name in env.object_names:
        if object_name in related_names:
            continue
        address = _find_object_y_address(env, object_name)
        if address is not None:
            unrelated.append((object_name, address, float(env.sim.data.qpos[address])))
    if not any(related) and not unrelated:
        raise RuntimeError(f"No movable objects found for {case.task_type}/{case.case_name}")
    return {
        "related": related,
        "related_names": step_objects,
        "unrelated": unrelated,
        "rng": random.Random(seed),
        "toggle": -1,
        "schedule": {segment * steps_per_subtask + 10 for segment in range(1, len(task.steps))},
    }


def _maybe_inject_dynamic(
    env: Any, state: dict[str, Any] | None, step: int, segment: int
) -> dict[str, Any] | None:
    if state is None or step not in state["schedule"]:
        return None
    related = state["related"][segment] if segment < len(state["related"]) else None
    use_related = state["rng"].random() < 0.5 and related is not None
    if use_related:
        address, base_y = related
        object_name = state["related_names"][segment]
        relation = "related"
    else:
        if not state["unrelated"]:
            if related is None:
                raise RuntimeError("Dynamic injection has no eligible object")
            address, base_y = related
            object_name = state["related_names"][segment]
            relation = "related_fallback"
        else:
            object_name, address, base_y = state["rng"].choice(state["unrelated"])
            relation = "unrelated"
    offset = 0.15 * state["toggle"]
    env.sim.data.qpos[address] = base_y + offset
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    state["toggle"] *= -1
    return {
        "type": "object_y_displacement",
        "step": step,
        "segment": segment,
        "object": object_name,
        "relation": relation,
        "qpos_address": address,
        "base_y": base_y,
        "offset": offset,
    }


def _final_predicates_hold(env: Any, goal: dict[str, list[list[str]]]) -> bool:
    return bool(goal) and all(env._eval_predicate(states[-1]) for states in goal.values())


def _predicate_snapshot(env: Any, goal: dict[str, list[list[str]]]) -> dict[str, list[bool]]:
    return {
        object_name: [bool(env._eval_predicate(state)) for state in states]
        for object_name, states in goal.items()
    }


def _result_state(env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    """Capture the physical simulator state after one executed action."""

    return {
        "eef_position": observation["robot0_eef_pos"].tolist(),
        "eef_quaternion": observation["robot0_eef_quat"].tolist(),
        "gripper_qpos": observation["robot0_gripper_qpos"].tolist(),
        "simulator_qpos": env.sim.data.qpos.copy().tolist(),
        "simulator_qvel": env.sim.data.qvel.copy().tolist(),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _clean_partial_trace(output_dir: Path, completed_keys: set[tuple[str, str, int]]) -> None:
    """Discard only an interrupted episode's trace before a resume."""

    trace_path = output_dir / "decisions.jsonl"
    if not trace_path.is_file():
        return
    temporary = trace_path.with_suffix(".jsonl.tmp")
    referenced_frames: set[str] = set()
    with (
        trace_path.open(encoding="utf-8") as source,
        temporary.open("w", encoding="utf-8") as destination,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {trace_path}:{line_number}: {exc}") from exc
            key = (str(value["task_type"]), str(value["case"]), int(value["trial"]))
            if key not in completed_keys:
                continue
            destination.write(line)
            if value.get("frame_bundle"):
                referenced_frames.add(str(value["frame_bundle"]))
    temporary.replace(trace_path)

    frame_root = output_dir / "frames"
    if frame_root.is_dir():
        for frame in frame_root.rglob("*.npz"):
            if str(frame.relative_to(output_dir)) not in referenced_frames:
                frame.unlink()


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _trace_frame(
    output_dir: Path,
    case: BenchmarkCase,
    trial: int,
    policy_call: int,
    policy_observation: dict[str, Any],
    np: Any,
) -> str:
    path = (
        output_dir
        / "frames"
        / case.task_type
        / case.case_name
        / f"trial-{trial:02d}-call-{policy_call:05d}.npz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        model_agentview_image=policy_observation["video.image"],
        model_wrist_image=policy_observation["video.wrist_image"],
        state_x=policy_observation["state.x"],
        state_y=policy_observation["state.y"],
        state_z=policy_observation["state.z"],
        state_roll=policy_observation["state.roll"],
        state_pitch=policy_observation["state.pitch"],
        state_yaw=policy_observation["state.yaw"],
        state_gripper=policy_observation["state.gripper"],
    )
    return str(path.relative_to(output_dir))


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
    np: Any,
) -> dict[str, Any]:
    task = parse_task_description(case.path / "task_description.txt")
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
        initial_state_source = "annotated_demo_shift_state"
    dynamic_state = _dynamic_injection_state(env, case, task, steps_per_subtask, episode_seed)
    client.reset()

    dummy_action = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    if start_event is None:
        for _ in range(initial_wait_steps):
            observation, _, _, _ = env.step(dummy_action)

    _, completed_before, reached_success = env._check_success(goal)
    current_success = bool(reached_success)
    max_steps = steps_per_subtask * len(task.steps)
    first_success_step = 0 if reached_success else None
    stop_after = min(max_steps, post_success_steps) if reached_success else max_steps
    trace_path = output_dir / "decisions.jsonl"
    step = 0
    policy_calls = 0
    policy_latencies: list[float] = []
    started = time.perf_counter()

    while step < min(max_steps, stop_after):
        policy_observation = build_policy_observation(observation, task.instruction, np)
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

        for action in chunk[:execution_horizon]:
            segment = min(step // steps_per_subtask, len(task.steps) - 1)
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
                "instruction": task.instruction,
                "frame_bundle": frame_path,
                "predicted_chunk": chunk.tolist(),
                "configured_execution_horizon": execution_horizon,
                "selected_prefix_length": len(transitions),
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
        "execution_horizon": execution_horizon,
        "control_frequency_hz": control_frequency_hz,
        "subtasks": len(task.steps),
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
        "predicted_actions": policy_calls * 16,
        "action_chunk_utilization": step / (policy_calls * 16) if policy_calls else 0.0,
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
    cases: list[BenchmarkCase],
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "variant": args.variant,
        "protocol": "continuous_no_restore",
        "checkpoint": str(contract.checkpoint_dir),
        "benchmark_dir": str(args.benchmark_dir.expanduser().resolve()),
        "robocerebra_source": str(args.robocerebra_source.expanduser().resolve()),
        "benchmark_revision": args.benchmark_revision,
        "model_revision": args.model_revision,
        "dataset_revision": args.dataset_revision,
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
        "hierarchy": False,
        "stop_or_adaptive_chunk": False,
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
    contract = inspect_checkpoint(args.checkpoint)
    if args.execution_horizon > contract.action_horizon:
        raise ValueError(f"--execution-horizon cannot exceed checkpoint H{contract.action_horizon}")

    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    cases = discover_cases(benchmark_dir, args.task_types, args.cases)
    output_dir = args.output.expanduser().resolve()
    existing = [
        name
        for name in ("run_manifest.json", "summary.json", "episodes.jsonl", "decisions.jsonl")
        if (output_dir / name).exists()
    ]
    if existing and not args.resume:
        raise FileExistsError(
            f"Refusing to mix {args.experiment_id} runs in {output_dir}; "
            f"existing files: {', '.join(existing)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _run_manifest(args, contract, cases, output_dir)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.is_file():
        actual_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual_manifest != manifest:
            raise ValueError("--resume configuration does not match run_manifest.json")
    elif existing:
        raise ValueError(
            f"Cannot --resume an {args.experiment_id} directory without run_manifest.json"
        )
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
