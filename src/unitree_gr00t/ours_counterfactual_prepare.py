"""Replay train-only Ours states and generate same-state recovery option returns."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .a0 import (
    BenchmarkCase,
    discover_cases,
    load_goal,
    load_goal_steps,
    parse_task_description,
    to_libero_action,
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
from .ours import OURS_ID, RECOVERY_OPTIONS, OursContractError, RecoveryOption
from .ours_counterfactual import branch_payload, evaluate_counterfactual_options
from .ours_data import OURS_CORPUS_MANIFEST, audit_recovery_corpus
from .ours_rollout_prepare import _episode_key, _read_jsonl, _save_episode, _write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--source-corpus", type=Path, required=True)
    parser.add_argument("--robocerebra-source", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--stop-stride", type=int, default=32)
    parser.add_argument("--max-states-per-episode", type=int, default=8)
    return parser


def _set_trace_state(env: Any, state: dict[str, Any], np: Any) -> None:
    qpos = np.asarray(state["simulator_qpos"], dtype=np.float64)
    qvel = np.asarray(state["simulator_qvel"], dtype=np.float64)
    if qpos.shape != env.sim.data.qpos.shape or qvel.shape != env.sim.data.qvel.shape:
        raise OursContractError("trace simulator state has the wrong shape")
    env.sim.data.qpos[...] = qpos
    env.sim.data.qvel[...] = qvel
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)


def _selected_stop_positions(rows: list[dict[str, Any]], stride: int, maximum: int) -> set[int]:
    candidates = [
        index
        for index, row in enumerate(rows)
        if int(row["selector_candidate_before_recovery"]) == 0
    ]
    selected = candidates[::stride][:maximum]
    return set(selected)


def _branch_actions(row: dict[str, Any], np: Any) -> dict[RecoveryOption, list[Any]]:
    chunk = np.asarray(row["predicted_chunk"], dtype=np.float32)
    scores = np.asarray(row["selector_scores"], dtype=np.float32)
    valid = np.asarray(row["selector_valid"], dtype=np.bool_)
    if chunk.shape != (16, 7) or scores.shape != (17,) or valid.shape != (17,):
        raise OursContractError("counterfactual trace proposal has the wrong shape")
    masked = np.where(valid[1:], scores[1:], -np.inf)
    if not np.isfinite(masked).any():
        raise OursContractError("counterfactual proposal has no valid non-STOP prefix")
    prefix = int(masked.argmax()) + 1
    actions = [to_libero_action(action, np) for action in chunk[:prefix]]
    return {
        RecoveryOption.ACCEPT_B: [],
        RecoveryOption.ADVANCE: [],
        RecoveryOption.CONSENSUS_PREFIX: actions,
    }


def _semantic_return(branch: Any, *, complete: bool) -> Any:
    if branch.option in {RecoveryOption.ACCEPT_B.value, RecoveryOption.ADVANCE.value}:
        return replace(branch, return_value=branch.return_value + (8.0 if complete else -8.0))
    return branch


def run(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.stop_stride, args.max_states_per_episode) < 1:
        raise ValueError("counterfactual sampling counts must be positive")
    rollout = args.rollout.expanduser().resolve()
    source_corpus = args.source_corpus.expanduser().resolve()
    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite counterfactual corpus: {destination}")
    audit_recovery_corpus(
        source_corpus,
        verify_hashes=True,
        require_development=False,
        require_both_completion_classes=True,
    )
    source_manifest_path = source_corpus / OURS_CORPUS_MANIFEST
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    run_manifest_path = rollout / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    summary = json.loads((rollout / "summary.json").read_text(encoding="utf-8"))
    base_seed = int(run_manifest["base_seed"])
    if (
        run_manifest.get("experiment_id") != OURS_ID
        or base_seed not in {10007, 11007, 12007}
        or not bool(summary.get("complete"))
        or source_manifest.get("rollout_run_manifest_sha256") != sha256_file(run_manifest_path)
    ):
        raise OursContractError("counterfactual source is not a complete paired train rollout")

    np, bddl_utils, runtime = _configure_environment_imports(
        args.robocerebra_source.expanduser().resolve(), benchmark_dir
    )
    episodes = {_episode_key(row): row for row in _read_jsonl(rollout / "episodes.jsonl")}
    decisions: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in _read_jsonl(rollout / "decisions.jsonl"):
        if row.get("policy_invoked") is False:
            continue
        decisions.setdefault(_episode_key(row), []).append(row)
    cases = {
        (case.task_type, case.case_name): case
        for case in discover_cases(benchmark_dir, run_manifest["task_types"], [])
    }
    if set(decisions) != set(episodes):
        raise OursContractError("counterfactual rollout inventories differ")

    feature_dir = destination / "features"
    feature_dir.mkdir(parents=True)
    branch_path = destination / "counterfactual_branches.jsonl"
    files_sha256: dict[str, str] = {}
    branch_count = 0
    state_count = 0
    option_counts = {option.value: 0 for option in RECOVERY_OPTIONS}
    for episode_index, key in enumerate(sorted(decisions)):
        case = cases[(key[0], key[1])]
        rows = sorted(decisions[key], key=lambda row: int(row["policy_call"]))
        source_path = source_corpus / "features" / f"episode_{episode_index:06d}.npz"
        with np.load(source_path) as value:
            arrays = {name: value[name].copy() for name in value.files}
        if len(rows) != len(arrays["sample_indices"]):
            raise OursContractError("counterfactual trace and feature rows differ")
        option_values = np.zeros((len(rows), len(RECOVERY_OPTIONS)), dtype=np.float32)
        option_valid = np.zeros((len(rows), len(RECOVERY_OPTIONS)), dtype=np.bool_)
        selected_positions = _selected_stop_positions(
            rows,
            args.stop_stride,
            args.max_states_per_episode,
        )

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
                env,
                case,
                task,
                goal,
                goal_steps,
            )
            if start_event is None:
                hold = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
                for _ in range(int(run_manifest["initial_wait_steps"])):
                    observation, _, _, _ = env.step(hold)
            for position, row in enumerate(rows):
                env._check_success(goal)
                if _predicate_snapshot(env, goal) != row["success_predicates_before"]:
                    raise OursContractError(
                        f"counterfactual source state does not replay: {key} call {row['policy_call']}"
                    )
                if position in selected_positions:
                    completed_before = int(row["completed_subtasks_before"])
                    complete = completed_before > int(row["active_subgoal_index_before"]) + int(
                        episodes[key]["excluded_subtasks"]
                    )
                    branches = evaluate_counterfactual_options(
                        env,
                        goal=goal,
                        option_actions=_branch_actions(row, np),
                        completed_subtasks_before=completed_before,
                        base_seed=base_seed,
                        state_index=int(arrays["sample_indices"][position]),
                        np=np,
                    )
                    for branch in branches:
                        branch = _semantic_return(branch, complete=complete)
                        option_index = RECOVERY_OPTIONS.index(RecoveryOption(branch.option))
                        option_values[position, option_index] = branch.return_value
                        option_valid[position, option_index] = True
                        option_counts[branch.option] += 1
                        with branch_path.open("a", encoding="utf-8") as handle:
                            handle.write(
                                json.dumps(
                                    {
                                        "episode_index": episode_index,
                                        "sample_index": int(arrays["sample_indices"][position]),
                                        "task_type": key[0],
                                        "case": key[1],
                                        "trial": key[2],
                                        "policy_call": int(row["policy_call"]),
                                        "source_context_sha256": row["selector_context_sha256"],
                                        **branch_payload(branch),
                                    },
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                        branch_count += 1
                    state_count += 1
                transitions = row["transitions"]
                if transitions:
                    _set_trace_state(
                        env,
                        transitions[-1]["result_state"],
                        np,
                    )
        finally:
            env.close()
        arrays["target_option_values"] = option_values
        arrays["target_option_valid"] = option_valid
        filename = f"episode_{episode_index:06d}.npz"
        output_path = feature_dir / filename
        _save_episode(output_path, arrays, np)
        files_sha256[filename] = sha256_file(output_path)

    manifest = dict(source_manifest)
    manifest.update(
        {
            "source": "train-only same-state simulator branches over live Ours rollouts",
            "source_kind": "counterfactual_live_rollout",
            "contains_counterfactuals": True,
            "counterfactual_source_manifest_sha256": sha256_file(source_manifest_path),
            "counterfactual_sampling": {
                "candidate_states": "B STOP proposals",
                "stop_stride": args.stop_stride,
                "max_states_per_episode": args.max_states_per_episode,
                "state_count": state_count,
                "branch_count": branch_count,
                "options": option_counts,
                "training_only_restore": True,
                "runtime_restore": False,
            },
            "counterfactual_branches_sha256": sha256_file(branch_path),
            "files_sha256": files_sha256,
            "files": len(files_sha256),
        }
    )
    _write_json(destination / OURS_CORPUS_MANIFEST, manifest)
    return manifest


def main() -> int:
    print(json.dumps(run(_parser().parse_args()), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
