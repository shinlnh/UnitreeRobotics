"""Reviewer-oriented metrics and artifact report for RoboCerebra baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

PAPER_URL = "https://arxiv.org/html/2506.06677v2"
AUDIT_URL = "https://arxiv.org/html/2606.04233"
EXPECTED_CONDITIONS = (
    "Ideal",
    "Memory_Execution",
    "Memory_Exploration",
    "Mix",
    "Observation_Mismatching",
    "Random_Disturbance",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the reviewer report for A0")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Completed run, for example H16=artifacts/A0/full-benchmark/H16",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--experiment-id", default="A0")
    parser.add_argument("--variant", default="GR00T-N1.7-LIBERO-original")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if trials < 1:
        return {"estimate": None, "lower_95": None, "upper_95": None, "successes": 0, "trials": 0}
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    half = (
        z
        * math.sqrt(probability * (1.0 - probability) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    return {
        "estimate": probability,
        "lower_95": max(0.0, center - half),
        "upper_95": min(1.0, center + half),
        "successes": successes,
        "trials": trials,
        "interval": "Wilson score",
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a quantile of an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cluster_bootstrap_mean(
    rows: list[Any],
    value: Callable[[Any], float],
    cluster: Callable[[Any], str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        return {"estimate": None, "lower_95": None, "upper_95": None}
    groups: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        groups[cluster(row)].append(row)
    names = sorted(groups)
    estimate = statistics.fmean(value(row) for row in rows)
    generator = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        sampled: list[Any] = []
        for name in generator.choices(names, k=len(names)):
            sampled.extend(groups[name])
        draws.append(statistics.fmean(value(row) for row in sampled))
    return {
        "estimate": estimate,
        "lower_95": _quantile(draws, 0.025),
        "upper_95": _quantile(draws, 0.975),
        "interval": f"task-cluster bootstrap ({samples} resamples)",
        "clusters": len(names),
    }


def cluster_bootstrap_ratio(
    rows: list[Any],
    numerator: Callable[[Any], float],
    denominator: Callable[[Any], float],
    cluster: Callable[[Any], str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap a pooled ratio while resampling complete task clusters."""
    if not rows:
        return {"estimate": None, "lower_95": None, "upper_95": None}
    groups: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        groups[cluster(row)].append(row)
    names = sorted(groups)

    def ratio(sampled: list[Any]) -> float:
        total_denominator = sum(denominator(row) for row in sampled)
        return (
            sum(numerator(row) for row in sampled) / total_denominator if total_denominator else 0.0
        )

    estimate = ratio(rows)
    generator = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        sampled: list[Any] = []
        for name in generator.choices(names, k=len(names)):
            sampled.extend(groups[name])
        draws.append(ratio(sampled))
    return {
        "estimate": estimate,
        "lower_95": _quantile(draws, 0.025),
        "upper_95": _quantile(draws, 0.975),
        "interval": f"task-cluster bootstrap ({samples} resamples)",
        "clusters": len(names),
        "aggregation": "pooled completed predicates / pooled possible predicates",
    }


def stratified_bootstrap_mean(
    rows: list[Any],
    value: Callable[[Any], float],
    stratum: Callable[[Any], str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Resample rollouts within every fixed benchmark task stratum."""
    if not rows:
        return {"estimate": None, "lower_95": None, "upper_95": None}
    groups: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)
    generator = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        sampled = [
            item
            for name in sorted(groups)
            for item in generator.choices(groups[name], k=len(groups[name]))
        ]
        draws.append(statistics.fmean(value(row) for row in sampled))
    return {
        "estimate": statistics.fmean(value(row) for row in rows),
        "lower_95": _quantile(draws, 0.025),
        "upper_95": _quantile(draws, 0.975),
        "interval": f"task-stratified rollout bootstrap ({samples} resamples)",
        "strata": len(groups),
    }


def stratified_bootstrap_ratio(
    rows: list[Any],
    numerator: Callable[[Any], float],
    denominator: Callable[[Any], float],
    stratum: Callable[[Any], str],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        return {"estimate": None, "lower_95": None, "upper_95": None}
    groups: dict[str, list[Any]] = defaultdict(list)
    for row in rows:
        groups[stratum(row)].append(row)

    def ratio(sampled: list[Any]) -> float:
        total_denominator = sum(denominator(row) for row in sampled)
        return sum(numerator(row) for row in sampled) / total_denominator

    generator = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        sampled = [
            item
            for name in sorted(groups)
            for item in generator.choices(groups[name], k=len(groups[name]))
        ]
        draws.append(ratio(sampled))
    return {
        "estimate": ratio(rows),
        "lower_95": _quantile(draws, 0.025),
        "upper_95": _quantile(draws, 0.975),
        "interval": f"task-stratified rollout bootstrap ({samples} resamples)",
        "strata": len(groups),
        "aggregation": "pooled completed predicates / pooled possible predicates",
    }


def _stratified_paired_ratio_delta(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        groups[_cluster_key(pair[0])].append(pair)
    names = sorted(groups)

    def delta(sampled: list[tuple[dict[str, Any], dict[str, Any]]]) -> float:
        left_possible = sum(float(pair[0]["possible_subtasks"]) for pair in sampled)
        right_possible = sum(float(pair[1]["possible_subtasks"]) for pair in sampled)
        left_rate = (
            sum(float(pair[0]["agent_completed_subtasks"]) for pair in sampled) / left_possible
        )
        right_rate = (
            sum(float(pair[1]["agent_completed_subtasks"]) for pair in sampled) / right_possible
        )
        return left_rate - right_rate

    generator = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        sampled = [
            pair for name in names for pair in generator.choices(groups[name], k=len(groups[name]))
        ]
        draws.append(delta(sampled))
    return {
        "estimate": delta(pairs),
        "lower_95": _quantile(draws, 0.025),
        "upper_95": _quantile(draws, 0.975),
        "interval": f"paired task-stratified rollout bootstrap ({samples} resamples)",
        "strata": len(names),
    }


def _episode_score(row: dict[str, Any]) -> float:
    possible = int(row["possible_subtasks"])
    return float(row["agent_completed_subtasks"]) / possible if possible else 0.0


def _cluster_key(row: dict[str, Any]) -> str:
    return f"{row['task_type']}/{row['case']}"


def _initial_state_source(row: dict[str, Any]) -> str:
    if row.get("initial_state_source"):
        return str(row["initial_state_source"])
    if row["task_type"] in {"Mix", "Observation_Mismatching"}:
        return "annotated_demo_shift_state"
    return "frozen_init_file"


def _descriptive(values: Iterable[float]) -> dict[str, float | int]:
    sequence = list(values)
    if not sequence:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(sequence),
        "mean": statistics.fmean(sequence),
        "median": statistics.median(sequence),
        "std": statistics.stdev(sequence) if len(sequence) > 1 else 0.0,
        "min": min(sequence),
        "max": max(sequence),
    }


def metric_block(rows: list[dict[str, Any]], *, samples: int, seed: int) -> dict[str, Any]:
    episodes = len(rows)
    completed = sum(int(row["agent_completed_subtasks"]) for row in rows)
    possible = sum(int(row["possible_subtasks"]) for row in rows)
    reached = sum(bool(row["reached_success"]) for row in rows)
    final = sum(bool(row["final_success"]) for row in rows)
    stable = sum(bool(row["reached_success"] and row["final_success"]) for row in rows)
    reactivated = sum(bool(row["post_success_reactivation"]) for row in rows)
    calls = sum(int(row["policy_calls"]) for row in rows)
    executed = sum(int(row["steps"]) for row in rows)
    predicted = sum(int(row["predicted_actions"]) for row in rows)
    policy_seconds = sum(float(row["policy_inference_seconds"]) for row in rows)
    elapsed_seconds = sum(float(row["elapsed_seconds"]) for row in rows)
    simulator_steps = sum(int(row["total_simulator_steps"]) for row in rows)
    control_frequency = int(rows[0].get("control_frequency_hz", 20)) if rows else 20
    declared_plan_lengths = [
        len(row["plan"]["subgoals"])
        for row in rows
        if isinstance(row.get("plan"), dict) and isinstance(row["plan"].get("subgoals"), list)
    ]
    visited_plan_lengths = [
        int(row["planner_subgoals_visited"])
        for row in rows
        if row.get("planner_subgoals_visited") is not None
    ]
    paper_sr = stratified_bootstrap_mean(
        rows,
        _episode_score,
        _cluster_key,
        samples=samples,
        seed=seed,
    )
    pooled_sr = stratified_bootstrap_ratio(
        rows,
        lambda row: float(row["agent_completed_subtasks"]),
        lambda row: float(row["possible_subtasks"]),
        _cluster_key,
        samples=samples,
        seed=seed + 10,
    )
    mean_steps = stratified_bootstrap_mean(
        rows,
        lambda row: float(row["steps"]),
        _cluster_key,
        samples=samples,
        seed=seed + 1,
    )
    return {
        "episodes": episodes,
        "task_instances": len({_cluster_key(row) for row in rows}),
        "paper_subtask_success_rate": paper_sr,
        "micro_subtask_completion_rate": completed / possible if possible else 0.0,
        "reference_pooled_subtask_success_rate": pooled_sr,
        "completed_subtasks": completed,
        "possible_subtasks": possible,
        "strict_full_task_success_rate": wilson_interval(final, episodes),
        "ever_reached_full_success_rate": wilson_interval(reached, episodes),
        "stable_success_given_reached": wilson_interval(stable, reached),
        "post_success_reactivation_rate": wilson_interval(reactivated, reached),
        "mean_executed_steps": mean_steps,
        "executed_steps": _descriptive(float(row["steps"]) for row in rows),
        "policy_calls": _descriptive(float(row["policy_calls"]) for row in rows),
        "total_executed_steps": executed,
        "total_policy_calls": calls,
        "total_predicted_actions": predicted,
        "action_chunk_utilization": executed / predicted if predicted else 0.0,
        "actions_per_completed_subtask": executed / completed if completed else None,
        "completed_subtasks_per_1000_actions": 1000.0 * completed / executed if executed else 0.0,
        "total_policy_inference_seconds": policy_seconds,
        "mean_policy_inference_ms": 1000.0 * policy_seconds / calls if calls else 0.0,
        "policy_latency_definition": (
            "Client-observed request/response wall time; includes server queueing, serialization, "
            "transport, preprocessing, model inference, and postprocessing."
        ),
        "episode_elapsed_seconds": _descriptive(float(row["elapsed_seconds"]) for row in rows),
        "total_episode_elapsed_seconds": elapsed_seconds,
        "simulated_seconds": simulator_steps / control_frequency,
        "simulation_realtime_factor": (
            simulator_steps / control_frequency / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "episodes_with_injection": sum(int(row["injection_count"] > 0) for row in rows),
        "total_injections": sum(int(row["injection_count"]) for row in rows),
        "initial_state_sources": dict(Counter(_initial_state_source(row) for row in rows)),
        "steps_after_first_success": _descriptive(
            float(row["steps_after_first_success"])
            for row in rows
            if row["first_success_step"] is not None
        ),
        "declared_plan_length": _descriptive(float(value) for value in declared_plan_lengths),
        "visited_plan_length": _descriptive(float(value) for value in visited_plan_lengths),
        "fixed_anchor_truncations": sum(
            int(row.get("fixed_anchor_truncations", 0)) for row in rows
        ),
    }


def _load_run(label: str, directory: Path) -> dict[str, Any]:
    root = directory.expanduser().resolve()
    manifest = _read_json(root / "run_manifest.json")
    summary = _read_json(root / "summary.json")
    episodes = _read_jsonl(root / "episodes.jsonl")
    if not summary.get("complete"):
        raise ValueError(f"Run {label} is not complete: {root}")
    expected = int(manifest["expected_episodes"])
    if expected != 600 or len(episodes) != expected:
        raise ValueError(f"Run {label} must contain exactly 600 episodes, found {len(episodes)}")
    counts = Counter(str(row["task_type"]) for row in episodes)
    if counts != Counter({name: 100 for name in EXPECTED_CONDITIONS}):
        raise ValueError(f"Run {label} has an invalid condition distribution: {dict(counts)}")
    keys = {(row["task_type"], row["case"], int(row["trial"])) for row in episodes}
    if len(keys) != len(episodes):
        raise ValueError(f"Run {label} contains duplicate episode keys")
    return {
        "label": label,
        "root": root,
        "manifest": manifest,
        "summary": summary,
        "episodes": episodes,
    }


def _scan_decisions(path: Path) -> dict[str, Any]:
    calls = 0
    transitions = 0
    latencies: list[float] = []
    prefix_lengths: Counter[int] = Counter()
    prefix_reasons: Counter[str] = Counter()
    active_subgoals: Counter[int] = Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            calls += 1
            transitions += len(row["transitions"])
            latencies.append(1000.0 * float(row["policy_latency_seconds"]))
            prefix_lengths[int(row["selected_prefix_length"])] += 1
            if row.get("prefix_selection"):
                prefix_reasons[str(row["prefix_selection"])] += 1
            planner_decision = row.get("planner_decision")
            if isinstance(planner_decision, dict) and "subgoal_index" in planner_decision:
                active_subgoals[int(planner_decision["subgoal_index"])] += 1
    return {
        "policy_calls": calls,
        "executed_transitions": transitions,
        "policy_latency_ms": _descriptive(latencies),
        "policy_latency_p95_ms": _quantile(latencies, 0.95),
        "policy_latency_p99_ms": _quantile(latencies, 0.99),
        "selected_prefix_histogram": dict(sorted(prefix_lengths.items())),
        "prefix_selection_reasons": dict(sorted(prefix_reasons.items())),
        "active_subgoal_call_histogram": dict(sorted(active_subgoals.items())),
    }


def _paired_delta(
    left: dict[str, Any], right: dict[str, Any], *, samples: int, seed: int
) -> dict[str, Any]:
    def keyed(run: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
        return {(row["task_type"], row["case"], int(row["trial"])): row for row in run["episodes"]}

    left_rows = keyed(left)
    right_rows = keyed(right)
    if left_rows.keys() != right_rows.keys():
        raise ValueError("Paired runs do not contain identical episode keys")
    pairs = [(left_rows[key], right_rows[key]) for key in sorted(left_rows)]

    def paired_metric(function: Callable[[dict[str, Any]], float], offset: int) -> dict[str, Any]:
        return stratified_bootstrap_mean(
            pairs,
            lambda pair: function(pair[0]) - function(pair[1]),
            lambda pair: _cluster_key(pair[0]),
            samples=samples,
            seed=seed + offset,
        )

    score_deltas = [_episode_score(a) - _episode_score(b) for a, b in pairs]
    return {
        "delta_definition": f"{left['label']} minus {right['label']}",
        "paired_episodes": len(pairs),
        "pairing_note": (
            "Environment task/trial keys and initial-state seeds are matched. Policy diffusion noise is "
            "not paired because concurrent clients share one stochastic policy-server RNG stream."
        ),
        "paper_subtask_success_rate_delta": paired_metric(_episode_score, 0),
        "reference_pooled_subtask_success_rate_delta": (
            _stratified_paired_ratio_delta(pairs, samples=samples, seed=seed + 10)
        ),
        "strict_full_task_success_rate_delta": paired_metric(
            lambda row: float(bool(row["final_success"])), 1
        ),
        "executed_steps_delta": paired_metric(lambda row: float(row["steps"]), 2),
        "policy_calls_delta": paired_metric(lambda row: float(row["policy_calls"]), 3),
        "policy_inference_seconds_delta": paired_metric(
            lambda row: float(row["policy_inference_seconds"]), 4
        ),
        "partial_completion_wins_ties_losses": {
            "wins": sum(delta > 0 for delta in score_deltas),
            "ties": sum(delta == 0 for delta in score_deltas),
            "losses": sum(delta < 0 for delta in score_deltas),
        },
    }


def _ideal_robustness(run: dict[str, Any], *, samples: int, seed: int) -> dict[str, Any]:
    rows = run["episodes"]
    ideal = {(row["case"], int(row["trial"])): row for row in rows if row["task_type"] == "Ideal"}
    result: dict[str, Any] = {}
    for offset, condition in enumerate(("Observation_Mismatching", "Random_Disturbance")):
        altered = {
            (row["case"], int(row["trial"])): row for row in rows if row["task_type"] == condition
        }
        pairs = [(ideal[key], altered[key]) for key in sorted(ideal.keys() & altered.keys())]
        result[condition] = {
            "drop_definition": "Ideal minus altered condition",
            "paired_episodes": len(pairs),
            "paper_subtask_success_rate_drop": stratified_bootstrap_mean(
                pairs,
                lambda pair: _episode_score(pair[0]) - _episode_score(pair[1]),
                lambda pair: str(pair[0]["case"]),
                samples=samples,
                seed=seed + offset,
            ),
            "reference_pooled_subtask_success_rate_drop": (
                _stratified_paired_ratio_delta(pairs, samples=samples, seed=seed + 20 + offset)
            ),
            "strict_full_task_success_rate_drop": stratified_bootstrap_mean(
                pairs,
                lambda pair: float(pair[0]["final_success"]) - float(pair[1]["final_success"]),
                lambda pair: str(pair[0]["case"]),
                samples=samples,
                seed=seed + 10 + offset,
            ),
        }
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(paths: list[Path], root: Path) -> str:
    """Hash names, sizes, and contents into one deterministic tree digest."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = str(path.relative_to(root)).encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _artifact_inventory(run: dict[str, Any]) -> dict[str, Any]:
    root = run["root"]
    frames = list((root / "frames").rglob("*.npz"))
    frame_mtimes = [path.stat().st_mtime for path in frames]
    core_names = ("run_manifest.json", "summary.json", "episodes.jsonl", "decisions.jsonl")
    return {
        "root": str(root),
        "frame_bundles": len(frames),
        "frame_bytes": sum(path.stat().st_size for path in frames),
        "frame_tree_sha256": _tree_sha256(frames, root),
        "frame_first_mtime_utc": (
            datetime.fromtimestamp(min(frame_mtimes), tz=timezone.utc).isoformat()  # noqa: UP017
            if frame_mtimes
            else None
        ),
        "frame_last_mtime_utc": (
            datetime.fromtimestamp(max(frame_mtimes), tz=timezone.utc).isoformat()  # noqa: UP017
            if frame_mtimes
            else None
        ),
        "trace_collection_span_seconds": (
            max(frame_mtimes) - min(frame_mtimes) if frame_mtimes else None
        ),
        "core_files": {
            name: {
                "bytes": (root / name).stat().st_size,
                "sha256": _sha256(root / name),
            }
            for name in core_names
        },
    }


def _command_output(argv: list[str]) -> str | None:
    try:
        return (
            subprocess.run(
                argv, check=False, capture_output=True, text=True, timeout=30
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _python_environment(python: Path) -> dict[str, Any] | None:
    if not python.is_file():
        return None
    program = """
import importlib.metadata as md
import json
import platform

names = ('numpy', 'mujoco', 'robosuite', 'pyzmq', 'msgpack', 'torch', 'transformers')
packages = {}
for name in names:
    try:
        packages[name] = md.version(name)
    except md.PackageNotFoundError:
        packages[name] = None
payload = {'python': platform.python_version(), 'packages': packages}
try:
    import torch
    payload['torch_cuda'] = torch.version.cuda
    payload['cudnn'] = torch.backends.cudnn.version()
    payload['cuda_available'] = torch.cuda.is_available()
except ImportError:
    pass
print(json.dumps(payload))
"""
    output = _command_output([str(python), "-c", program])
    return json.loads(output) if output else None


def _environment_metadata(experiment_id: str = "A0") -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    implementation_paths = [
        "configs/project.toml",
        "scripts/run_a0_full_benchmark.sh",
        "scripts/setup_robocerebra_a0.sh",
        "src/unitree_gr00t/a0.py",
        "src/unitree_gr00t/a0_eval.py",
        "src/unitree_gr00t/a0_merge.py",
        "src/unitree_gr00t/a0_report.py",
    ]
    if experiment_id == "A1":
        implementation_paths.extend(
            (
                "scripts/download_a1_training_data.sh",
                "scripts/run_a1_prepare_dataset.sh",
                "scripts/run_a1_full_benchmark.sh",
                "src/unitree_gr00t/a1.py",
                "src/unitree_gr00t/a1_data.py",
                "src/unitree_gr00t/a1_train.py",
            )
        )
    if experiment_id == "A2":
        implementation_paths.extend(
            (
                "scripts/run_a2_full_benchmark.sh",
                "scripts/run_a2_pipeline.sh",
                "src/unitree_gr00t/a2.py",
                "src/unitree_gr00t/a2_eval.py",
                "src/unitree_gr00t/a2_merge.py",
            )
        )
    packages = {}
    for name in ("numpy", "mujoco", "robosuite", "pyzmq", "msgpack", "torch"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu": _command_output(["lscpu"]),
        "gpu": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "packages_in_report_environment": packages,
        "evaluator_environment": _python_environment(project_root / ".venv-a0" / "bin/python"),
        "policy_server_environment": _python_environment(
            project_root / ".upstream" / "Isaac-GR00T-N1.7" / ".venv" / "bin/python"
        ),
        "git_head": _command_output(["git", "rev-parse", "HEAD"]),
        "git_branch": _command_output(["git", "branch", "--show-current"]),
        "git_worktree_status": _command_output(["git", "status", "--short"]),
        "implementation_sha256": {
            name: _sha256(project_root / name) for name in implementation_paths
        },
        "actual_upstream_git_heads": {
            "Isaac-GR00T-N1.7": _command_output(
                [
                    "git",
                    "-C",
                    str(project_root / ".upstream" / "Isaac-GR00T-N1.7"),
                    "rev-parse",
                    "HEAD",
                ]
            ),
            "RoboCerebra": _command_output(
                [
                    "git",
                    "-C",
                    str(project_root / ".upstream" / "RoboCerebra"),
                    "rev-parse",
                    "HEAD",
                ]
            ),
        },
    }


def _percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def _interval(metric: dict[str, Any]) -> str:
    if metric["estimate"] is None:
        return "N/A"
    return (
        f"{_percent(metric['estimate'])} "
        f"[{_percent(metric['lower_95'])}, {_percent(metric['upper_95'])}]"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_figure(path: Path, metrics: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ("Ideal", "Mem. Exec.", "Mem. Explor.", "Mix", "Obs. Mismatch", "Random Dist.")
    run_items = list(metrics["runs"].items())
    x = np.arange(len(EXPECTED_CONDITIONS))
    width = 0.8 / len(run_items)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    colors = ("#2563eb", "#f97316", "#16a34a", "#9333ea")

    for run_index, (label, run) in enumerate(run_items):
        offset = (run_index - (len(run_items) - 1) / 2) * width
        blocks = [run["by_condition"][condition] for condition in EXPECTED_CONDITIONS]
        paper = [block["paper_subtask_success_rate"] for block in blocks]
        strict = [block["strict_full_task_success_rate"] for block in blocks]
        paper_values = np.asarray([metric["estimate"] for metric in paper])
        strict_values = np.asarray([metric["estimate"] for metric in strict])
        axes[0, 0].bar(x + offset, paper_values, width, label=label, color=colors[run_index])
        axes[0, 0].errorbar(
            x + offset,
            paper_values,
            yerr=np.asarray(
                [
                    [max(0.0, metric["estimate"] - metric["lower_95"]) for metric in paper],
                    [max(0.0, metric["upper_95"] - metric["estimate"]) for metric in paper],
                ]
            ),
            fmt="none",
            ecolor="black",
            capsize=2,
            linewidth=0.8,
        )
        axes[0, 1].bar(x + offset, strict_values, width, label=label, color=colors[run_index])
        axes[0, 1].errorbar(
            x + offset,
            strict_values,
            yerr=np.asarray(
                [
                    [max(0.0, metric["estimate"] - metric["lower_95"]) for metric in strict],
                    [max(0.0, metric["upper_95"] - metric["estimate"]) for metric in strict],
                ]
            ),
            fmt="none",
            ecolor="black",
            capsize=2,
            linewidth=0.8,
        )
        axes[1, 0].bar(
            x + offset,
            [block["completed_subtasks_per_1000_actions"] for block in blocks],
            width,
            label=label,
            color=colors[run_index],
        )
        axes[1, 1].bar(
            x + offset,
            [block["mean_policy_inference_ms"] for block in blocks],
            width,
            label=label,
            color=colors[run_index],
        )

    axes[0, 0].set_title("Task-macro predicate SR (95% task-stratified CI)")
    axes[0, 1].set_title("Terminal goal-state SR (95% Wilson CI)")
    axes[1, 0].set_title("Completed predicates per 1,000 executed actions")
    axes[1, 1].set_title("Mean end-to-end policy RPC latency")
    paper_upper = max(
        block["paper_subtask_success_rate"]["upper_95"]
        for _, run in run_items
        for block in run["by_condition"].values()
    )
    terminal_upper = max(
        block["strict_full_task_success_rate"]["upper_95"]
        for _, run in run_items
        for block in run["by_condition"].values()
    )
    axes[0, 0].set_ylim(0, min(1.0, max(0.1, 1.2 * paper_upper)))
    axes[0, 1].set_ylim(0, min(1.0, max(0.1, 1.2 * terminal_upper)))
    axes[0, 0].set_ylabel("rate")
    axes[0, 1].set_ylabel("rate")
    axes[1, 0].set_ylabel("predicates / 1k actions")
    axes[1, 1].set_ylabel("milliseconds")
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.suptitle(
        f"{metrics['experiment_id']} — {metrics['variant']}, continuous no-restore",
        fontsize=14,
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _markdown(metrics: dict[str, Any]) -> str:
    experiment_id = str(metrics["experiment_id"])
    variant = str(metrics["variant"])
    if experiment_id == "A0":
        scope = (
            "A0 measures how far the unmodified `GR00T-N1.7-LIBERO/libero_10` "
            "low-level policy can execute a long-horizon full-task instruction"
        )
    elif experiment_id == "A1":
        scope = (
            f"{experiment_id} measures the shared post-trained `{variant}` low-level policy's "
            "ability to execute a long-horizon full-task instruction"
        )
    else:
        scope = (
            f"{experiment_id} measures `{variant}` with the frozen RoboCerebra fixed-anchor "
            "subgoal hierarchy"
        )
    hierarchy = experiment_id == "A2"
    isolation = (
        "with canonical step instructions switched at fixed 150-step anchors, but without "
        "outcome-aware switching, re-planning, stop/adaptive selection, retry, or recovery"
        if hierarchy
        else "without a hierarchy, stop/adaptive chunk selector, retry, or recovery"
    )
    lines = [
        f"# {experiment_id} full RoboCerebra benchmark — reviewer report",
        "",
        "## Scope and research question",
        "",
        f"{scope} {isolation}. The benchmark covers static, memory, partial-observation, disturbance, and mixed conditions on one continuous simulator timeline.",
        "",
        f"The official [RoboCerebra paper]({PAPER_URL}) defines 60 tasks and 10 rollouts per task, and reports predicate/subtask success as its SR. This report additionally retains terminal goal-state success, ordered-goal reach, confidence intervals, action efficiency, post-reach stability, latency, and per-episode artifacts. The statistical reporting follows the artifact-level caution recommended by the [2026 manipulation benchmark audit]({AUDIT_URL}).",
        "",
        "## Benchmark coverage",
        "",
        f"| Condition | Capability stressed | {experiment_id} continuous-track realization |",
        "| --- | --- | --- |",
        "| Ideal | Static, fully observable long-horizon execution | Official Ideal task and initial state |",
        "| Memory_Exploration | Active exploration to build an internal representation | Official exploration task, description, goals, scene, and initial state |",
        "| Memory_Execution | Retrieval of previously relevant state for goal completion | Official memory-execution task, description, goals, scene, and initial state |",
        "| Observation_Mismatching | Plan/perception misalignment | Official Ideal scene shifted to the second annotated demonstration state; already-forced predicates excluded |",
        "| Random_Disturbance | Unexpected environment changes | Seeded 0.15 m object-y displacements during one uninterrupted rollout |",
        "| Mix | Memory plus dynamic change and partial observation | Official Mix scene with shifted start and the same seeded displacement rule |",
        "",
        (
            f"The low-level policy sees the active canonical subgoal plus live agent/wrist RGB and proprioception. {experiment_id}'s planner is outcome-blind and supplies no symbolic memory, failure detector, retry, or state restoration."
            if hierarchy
            else f"Each policy sees the unchanged full-task language instruction and live agent/wrist RGB plus proprioception. {experiment_id} supplies no subgoal, symbolic memory, failure detector, retry, or state restoration."
        ),
        "",
        "## Headline results",
        "",
        "| Run | Episodes | Task-macro SR (paper Eq. 1) | Pooled predicate SR | Terminal goal-state SR | Mean steps | Calls | Mean policy RPC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, run in metrics["runs"].items():
        overall = run["overall"]
        lines.append(
            f"| {label} | {overall['episodes']} | {_interval(overall['paper_subtask_success_rate'])} | "
            f"{_interval(overall['reference_pooled_subtask_success_rate'])} | "
            f"{_interval(overall['strict_full_task_success_rate'])} | "
            f"{overall['executed_steps']['mean']:.2f} | {overall['policy_calls']['mean']:.2f} | "
            f"{overall['mean_policy_inference_ms']:.2f} ms |"
        )

    lines.extend(["", f"![{experiment_id} benchmark metric summary](summary_metrics.png)"])

    lines.extend(
        [
            "",
            "## Published RoboCerebra context",
            "",
            f"These Table 3 averages use the paper's benchmark-compatible resume/anchor protocol and are context only; they are not directly rank-comparable with this report's continuous no-restore {experiment_id} track.",
            "",
            "| Published system | Average SR |",
            "| --- | ---: |",
        ]
    )
    for system, value in metrics["published_context"]["table_3_average_sr"].items():
        lines.append(f"| {system} | {_percent(value)} |")

    lines.extend(["", "## Results by condition", ""])
    for label, run in metrics["runs"].items():
        lines.extend(
            [
                f"### {label}",
                "",
                "| Condition | Task-macro SR | Pooled SR | Terminal-state SR | Ordered-goal reached SR | Steps | Injections |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for condition in EXPECTED_CONDITIONS:
            block = run["by_condition"][condition]
            lines.append(
                f"| {condition} | {_interval(block['paper_subtask_success_rate'])} | "
                f"{_interval(block['reference_pooled_subtask_success_rate'])} | "
                f"{_interval(block['strict_full_task_success_rate'])} | "
                f"{_interval(block['ever_reached_full_success_rate'])} | "
                f"{block['executed_steps']['mean']:.2f} | {block['total_injections']} |"
            )
        lines.append("")

    paired = metrics.get("paired_comparison")
    if paired:
        lines.extend(
            [
                "## Fixed-horizon ablation",
                "",
                f"Delta convention: {paired['delta_definition']}.",
                "",
                f"- Paper SR delta: {_interval(paired['paper_subtask_success_rate_delta'])}",
                f"- Pooled predicate SR delta: {_interval(paired['reference_pooled_subtask_success_rate_delta'])}",
                f"- Terminal goal-state SR delta: {_interval(paired['strict_full_task_success_rate_delta'])}",
                f"- Mean executed-step delta: {paired['executed_steps_delta']['estimate']:.2f} [{paired['executed_steps_delta']['lower_95']:.2f}, {paired['executed_steps_delta']['upper_95']:.2f}]",
                f"- Partial-completion wins/ties/losses: {paired['partial_completion_wins_ties_losses']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Metric applicability",
            "",
            "- RoboCerebra task-macro SR: the primary paper-style value, computed from Eq. (1) per task/rollout and averaged with equal weight across the 60 task instances.",
            "- Reference-evaluator pooled SR: all completed state transitions divided by all possible transitions, matching the public evaluator's aggregate logger. It is reported separately because unequal task lengths make it differ from task-macro SR.",
            "- Terminal goal-state SR (machine-readable legacy key `strict_full_task_success_rate`): every object's terminal goal predicate must hold in the final frame. It can be true even when the ordered evaluator never observed the full transition sequence.",
            "- Ordered-goal reached SR: the public evaluator's sequential `_check_success` became true at least once. Stability/reactivation metrics are conditioned only on these reached episodes.",
            (
                "- Plan Match Accuracy: 100% by construction because A2 consumes the benchmark's canonical annotated plan; this is a structural contract check, not a learned-planner result. Plan Efficiency is reported per run as paper SR divided by mean declared plan length."
                if hierarchy
                else f"- Plan Match Accuracy and symbolic Plan Efficiency: N/A for {experiment_id} because it emits no symbolic high-level plan."
            ),
            f"- VideoQA Action Completion Accuracy: N/A because {experiment_id} has no reflection/VideoQA head.",
            f"- Failure-detection precision/recall/latency and recovery success: N/A because {experiment_id} intentionally has neither detector nor recovery policy. Injection exposure and conditional outcomes remain in the raw episodes/traces.",
            "",
            "## Reproducibility and limitations",
            "",
            "Every run fixes model, dataset, and evaluator revisions; seed, task, trial, initial state, action conversion, 20 Hz control, full model input tensors, predicted chunks, executed transitions, and simulator state are retained. Policy/evaluator semantics and H8/H16 were frozen before inspecting full-test outcomes; only resumability, hardware sharding, integrity checks, and reporting were changed during execution. Confidence intervals quantify rollout uncertainty but do not establish real-world transfer. Simulator workers share one seeded stochastic policy server, so task/trial initial states are matched across horizons but policy diffusion noise is not paired under concurrent request interleaving. This continuous no-restore protocol is intentionally stricter than RoboCerebra's anchor/resume mechanism, so the paper table is contextual rather than a direct leaderboard comparison.",
            "",
            f"See `metrics.json`, the condition/case CSV files, `environment.json`, and `artifact_inventory.json` for machine-readable evidence. The environment file hashes the exact {experiment_id} implementation sources; the inventory hashes every core JSON/JSONL file and the complete frame tree.",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(
    run_specs: list[str],
    output: Path,
    samples: int,
    *,
    experiment_id: str = "A0",
    variant: str = "GR00T-N1.7-LIBERO-original",
) -> dict[str, Any]:
    if samples < 1000:
        raise ValueError("--bootstrap-samples must be at least 1000")
    parsed: list[tuple[str, Path]] = []
    for spec in run_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --run value: {spec!r}")
        label, raw_path = spec.split("=", 1)
        if not label or not raw_path:
            raise ValueError(f"Invalid --run value: {spec!r}")
        parsed.append((label, Path(raw_path)))
    if len({label for label, _ in parsed}) != len(parsed):
        raise ValueError("Run labels must be unique")

    runs = [_load_run(label, path) for label, path in parsed]
    hierarchy = experiment_id == "A2"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "RoboCerebra",
        "experiment_id": experiment_id,
        "variant": variant,
        "protocol": "continuous_no_restore",
        "outcome_tuning_policy": (
            "Policy/evaluator semantics and fixed horizons were frozen before full-test outcome "
            "inspection; runtime sharding and reporting do not select actions from outcomes."
        ),
        "official_protocol": "60 tasks x 10 trials = 600 episodes per fixed horizon",
        "confidence_intervals": {
            "binary": "two-sided 95% Wilson score",
            "continuous": (f"two-sided 95% task-stratified rollout bootstrap, {samples} resamples"),
        },
        "published_context": {
            "source": PAPER_URL,
            "comparison_warning": (
                f"RoboCerebra Table 3 uses benchmark resume/anchor behavior; {experiment_id} uses continuous "
                "no-restore rollouts, so these values are contextual rather than a leaderboard."
            ),
            "table_3_average_sr": {
                "OpenVLA-Libero100": 0.0200,
                "OpenVLA*": 0.0457,
                "Planner + OpenVLA*": 0.1604,
                "Hierarchical Framework (HPE)": 0.1655,
            },
        },
        "paper_metric_applicability": {
            "task_macro_predicate_success_rate": "reported as the primary paper Eq. (1) metric",
            "reference_evaluator_pooled_subtask_rate": "reported separately",
            "strict_full_task_success_rate": (
                "legacy key: terminal goal-state success, reported as an additional reviewer metric"
            ),
            "average_plan_match_accuracy": 1.0 if hierarchy else None,
            "average_plan_match_accuracy_note": (
                "By construction: A2 consumes canonical benchmark annotations; not a learned-planner estimate."
                if hierarchy
                else None
            ),
            "plan_efficiency": {} if hierarchy else None,
            "videoqa_action_completion_accuracy": None,
            "reason": (
                "A2 has a frozen symbolic plan trace but no learned planner or VideoQA reflection head"
                if hierarchy
                else f"{experiment_id} has no high-level planner, symbolic action trace, or VideoQA reflection head"
            ),
        },
        "runs": {},
    }
    condition_csv: list[dict[str, Any]] = []
    case_csv: list[dict[str, Any]] = []
    for run_index, run in enumerate(runs):
        episodes = run["episodes"]
        by_condition = {
            condition: metric_block(
                [row for row in episodes if row["task_type"] == condition],
                samples=samples,
                seed=20260905 + run_index * 100 + index,
            )
            for index, condition in enumerate(EXPECTED_CONDITIONS)
        }
        by_case: dict[str, Any] = {}
        for index, key in enumerate(sorted({_cluster_key(row) for row in episodes})):
            block = metric_block(
                [row for row in episodes if _cluster_key(row) == key],
                samples=samples,
                seed=20261905 + run_index * 100 + index,
            )
            by_case[key] = block
            case_csv.append(_flat_metric_row(run["label"], key, block))
        overall = metric_block(episodes, samples=samples, seed=20260905 + run_index * 1000)
        decision_metrics = _scan_decisions(run["root"] / "decisions.jsonl")
        if decision_metrics["policy_calls"] != overall["total_policy_calls"]:
            raise ValueError(f"Decision count mismatch for {run['label']}")
        if decision_metrics["executed_transitions"] != overall["total_executed_steps"]:
            raise ValueError(f"Executed-transition count mismatch for {run['label']}")
        payload["runs"][run["label"]] = {
            "root": str(run["root"]),
            "manifest": run["manifest"],
            "overall": overall,
            "by_condition": by_condition,
            "by_case": by_case,
            "decision_metrics": decision_metrics,
            "robustness_vs_ideal": _ideal_robustness(
                run, samples=samples, seed=20262905 + run_index * 100
            ),
        }
        if hierarchy:
            if not run["manifest"].get("hierarchy") or run["manifest"].get("recovery"):
                raise ValueError(f"Run {run['label']} does not satisfy the frozen A2 contract")
            mean_plan_length = overall["declared_plan_length"]["mean"]
            paper_sr = overall["paper_subtask_success_rate"]["estimate"]
            payload["paper_metric_applicability"]["plan_efficiency"][run["label"]] = (
                paper_sr / mean_plan_length if mean_plan_length else None
            )
        for condition, block in by_condition.items():
            condition_csv.append(_flat_metric_row(run["label"], condition, block))

    if len(runs) == 2:
        payload["paired_comparison"] = _paired_delta(
            runs[0], runs[1], samples=samples, seed=20263905
        )

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "metrics.json", payload)
    _write_csv(output / "metrics_by_condition.csv", condition_csv)
    _write_csv(output / "metrics_by_case.csv", case_csv)
    _write_summary_figure(output / "summary_metrics.png", payload)
    _write_json(output / "environment.json", _environment_metadata(experiment_id))
    inventory = {run["label"]: _artifact_inventory(run) for run in runs}
    for run in runs:
        label = run["label"]
        if (
            run["manifest"].get("trace_images")
            and inventory[label]["frame_bundles"]
            != payload["runs"][label]["overall"]["total_policy_calls"]
        ):
            raise ValueError(f"Frame-bundle count mismatch for {label}")
    _write_json(output / "artifact_inventory.json", inventory)
    (output / "REVIEWER_REPORT.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def _flat_metric_row(label: str, group: str, block: dict[str, Any]) -> dict[str, Any]:
    paper = block["paper_subtask_success_rate"]
    pooled = block["reference_pooled_subtask_success_rate"]
    strict = block["strict_full_task_success_rate"]
    reached = block["ever_reached_full_success_rate"]
    return {
        "run": label,
        "group": group,
        "episodes": block["episodes"],
        "paper_sr": paper["estimate"],
        "paper_sr_ci95_low": paper["lower_95"],
        "paper_sr_ci95_high": paper["upper_95"],
        "pooled_subtask_sr": pooled["estimate"],
        "pooled_subtask_sr_ci95_low": pooled["lower_95"],
        "pooled_subtask_sr_ci95_high": pooled["upper_95"],
        "strict_full_task_sr": strict["estimate"],
        "strict_full_task_sr_ci95_low": strict["lower_95"],
        "strict_full_task_sr_ci95_high": strict["upper_95"],
        "ever_reached_sr": reached["estimate"],
        "mean_steps": block["executed_steps"]["mean"],
        "mean_policy_calls": block["policy_calls"]["mean"],
        "mean_policy_inference_ms": block["mean_policy_inference_ms"],
        "chunk_utilization": block["action_chunk_utilization"],
        "completed_subtasks_per_1000_actions": block["completed_subtasks_per_1000_actions"],
        "total_injections": block["total_injections"],
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = _parser().parse_args()
    payload = build_report(
        args.run,
        args.output,
        args.bootstrap_samples,
        experiment_id=args.experiment_id,
        variant=args.variant,
    )
    headline = {
        label: {
            "paper_sr": run["overall"]["paper_subtask_success_rate"],
            "strict_full_task_sr": run["overall"]["strict_full_task_success_rate"],
        }
        for label, run in payload["runs"].items()
    }
    print(json.dumps(headline, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
