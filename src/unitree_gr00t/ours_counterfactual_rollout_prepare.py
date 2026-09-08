"""Generate train-only option values with closed-loop same-state roll-forwards."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
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
from .b import select_unified_candidate
from .ours import OURS_ID, RECOVERY_OPTIONS, OursContractError, RecoveryOption
from .ours_counterfactual import branch_payload, evaluate_counterfactual_rollouts
from .ours_counterfactual_prepare import _set_trace_state
from .ours_data import OURS_CORPUS_MANIFEST, audit_recovery_corpus
from .ours_rollout_prepare import _episode_key, _read_jsonl, _save_episode, _write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--source-corpus", type=Path, required=True)
    parser.add_argument("--robocerebra-source", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5551)
    parser.add_argument("--stop-stride", type=int, default=128)
    parser.add_argument("--max-states-per-episode", type=int, default=1)
    parser.add_argument("--min-source-elapsed-steps", type=int, default=75)
    parser.add_argument("--rollout-steps", type=int, default=75)
    parser.add_argument("--max-policy-calls", type=int, default=24)
    parser.add_argument("--consensus-hypotheses", type=int, choices=(4, 8), default=4)
    parser.add_argument("--max-replay-mismatch-rate", type=float, default=0.05)
    parser.add_argument(
        "--require-stop-pending",
        action="store_true",
        help="Label only second consecutive STOPs where B-retry would act",
    )
    parser.add_argument(
        "--residual-retry-baseline",
        action="store_true",
        help="Omit duplicate ACCEPT_B labels and compare real overrides with retry",
    )
    return parser


def _best_nonstop(scores: Any, valid: Any, np: Any) -> int:
    masked = np.where(valid[1:], scores[1:], -np.inf)
    if not np.isfinite(masked).any():
        raise OursContractError("counterfactual proposal has no valid non-STOP prefix")
    return int(masked.argmax()) + 1


def _medoid_index(chunks: list[Any], np: Any) -> int:
    if not chunks:
        raise OursContractError("counterfactual consensus requires hypotheses")
    flattened = np.stack(chunks).astype(np.float32).reshape(len(chunks), -1)
    normalized = flattened / np.linalg.norm(flattened, axis=1, keepdims=True).clip(min=1e-6)
    distances = np.square(normalized[:, None] - normalized[None, :]).mean(axis=2)
    return int(distances.sum(axis=1).argmin())


def _fresh_consensus_count(total_hypotheses: int) -> int:
    if total_hypotheses < 1:
        raise OursContractError("counterfactual consensus count must be positive")
    return total_hypotheses - 1


def _valid_options(
    active_subgoal: int,
    subgoal_count: int,
    *,
    stop_pending: bool = False,
    residual_retry_baseline: bool = False,
) -> tuple[RecoveryOption, ...]:
    if not 0 <= active_subgoal < subgoal_count:
        raise OursContractError("counterfactual active subgoal is invalid")
    options = [RecoveryOption.REOBSERVE, RecoveryOption.RETRY_CURRENT]
    if not residual_retry_baseline:
        options.insert(0, RecoveryOption.ACCEPT_B)
    if active_subgoal > 0:
        options.append(RecoveryOption.BACKTRACK_ONE)
    if stop_pending:
        options.append(RecoveryOption.ADVANCE)
    options.append(RecoveryOption.CONSENSUS_PREFIX)
    return tuple(options)


def _apply_stop_confirmation(
    *, target_subgoal: int, subgoal_count: int, stop_streak: int
) -> tuple[int, int, bool]:
    """Advance exactly once when a second consecutive STOP is observed."""

    if not 0 <= target_subgoal < subgoal_count or stop_streak < 0:
        raise OursContractError("counterfactual STOP confirmation state is invalid")
    next_streak = stop_streak + 1
    if next_streak < 2:
        return target_subgoal, next_streak, False
    return target_subgoal + 1, 0, True


def _apply_residual_stop_confirmation(
    *,
    target_subgoal: int,
    subgoal_count: int,
    stop_streak: int,
    retry_attempt_index: int,
) -> tuple[int, int, int, bool, bool]:
    """Apply B-retry, not B, after a residual counterfactual option."""

    if retry_attempt_index not in {0, 1}:
        raise OursContractError("counterfactual B-retry attempt is invalid")
    next_subgoal, next_streak, committed = _apply_stop_confirmation(
        target_subgoal=target_subgoal,
        subgoal_count=subgoal_count,
        stop_streak=stop_streak,
    )
    if not committed:
        return target_subgoal, next_streak, retry_attempt_index, False, False
    if retry_attempt_index == 0:
        return target_subgoal, 0, 1, True, False
    return next_subgoal, 0, 0, False, True


def _selected_rollout_positions(
    rows: list[dict[str, Any]],
    *,
    stride: int,
    maximum: int,
    min_elapsed_steps: int,
    require_stop_pending: bool = False,
) -> set[int]:
    if min(stride, maximum) < 1 or min_elapsed_steps < 0:
        raise OursContractError("counterfactual rollout sampling is invalid")
    segment_start: int | None = None
    candidates: list[int] = []
    for index, row in enumerate(rows):
        if row.get("new_subgoal_anchor") is not None or segment_start is None:
            segment_start = int(row["step_before"])
        elapsed = int(row["step_before"]) - segment_start
        previous = rows[index - 1] if index else None
        stop_pending_before = bool(
            require_stop_pending
            and previous is not None
            and int(previous["stop_confirmation_streak_after"]) > 0
            and int(previous["active_subgoal_index_after"])
            == int(row["active_subgoal_index_before"])
        )
        if (
            int(row["selector_candidate_before_recovery"]) == 0
            and elapsed >= min_elapsed_steps
            and (stop_pending_before or not require_stop_pending)
        ):
            candidates.append(index)
    return set(candidates[::stride][:maximum])


def _query_b(
    client: RemotePolicyClient,
    observation: dict[str, Any],
    instruction: str,
    *,
    anchors: list[Any],
    subgoal_start: bool,
    episode_seed: int,
    decision_index: int,
    remaining_steps: int,
    expected_runtime: dict[str, Any],
    np: Any,
) -> tuple[Any, Any, Any, int]:
    action, info = client.get_action_with_info(
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
            }
        },
    )
    selector = info.get("b_selector")
    if not isinstance(selector, dict) or selector.get("runtime_provenance") != expected_runtime:
        raise OursContractError("counterfactual B runtime provenance does not match source")
    scores = np.asarray(selector["scores"], dtype=np.float32)
    valid = np.asarray(selector["valid"], dtype=np.bool_)
    candidate = int(selector["candidate"])
    if candidate != select_unified_candidate(scores.tolist(), valid.tolist()):
        raise OursContractError("counterfactual B proposal does not replay")
    if subgoal_start:
        anchor = np.asarray(selector.get("raw_anchor"), dtype=np.float32)
        if anchor.shape != (2048,):
            raise OursContractError("counterfactual B returned an invalid anchor")
        anchors.append(anchor)
    return unpack_action_chunk(action, np), scores, valid, candidate


def _roll_option(
    env: Any,
    observation: dict[str, Any],
    branch_seed: int,
    *,
    option: RecoveryOption,
    stop_pending: bool,
    subgoals: tuple[str, ...],
    active_subgoal: int,
    source_anchors: list[Any],
    source_proposal: tuple[Any, Any, Any],
    client: RemotePolicyClient,
    rollout_steps: int,
    max_policy_calls: int,
    consensus_hypotheses: int,
    residual_retry_baseline: bool,
    expected_runtime: dict[str, Any],
    np: Any,
) -> tuple[int, int]:
    client.reset({"episode_seed": branch_seed})
    anchors = [anchor.copy() for anchor in source_anchors]
    target_subgoal = active_subgoal
    subgoal_start = False
    stop_streak = 0
    retry_attempt_index = 0
    if option is RecoveryOption.ACCEPT_B:
        if stop_pending:
            target_subgoal += 1
            subgoal_start = True
        else:
            stop_streak = 1
    elif option is RecoveryOption.RETRY_CURRENT:
        subgoal_start = True
        retry_attempt_index = int(residual_retry_baseline)
    elif option is RecoveryOption.BACKTRACK_ONE:
        target_subgoal -= 1
        subgoal_start = True
    elif option is RecoveryOption.ADVANCE:
        target_subgoal += 1
        subgoal_start = True
    if residual_retry_baseline and option in {
        RecoveryOption.REOBSERVE,
        RecoveryOption.CONSENSUS_PREFIX,
    }:
        # These alternatives replace B-retry's current-subtask retry.
        retry_attempt_index = 1
    if target_subgoal >= len(subgoals):
        return 0, 0

    executed_steps = 0
    policy_calls = 0
    first_proposal = True
    while executed_steps < rollout_steps and policy_calls < max_policy_calls:
        remaining = rollout_steps - executed_steps
        if option is RecoveryOption.CONSENSUS_PREFIX and first_proposal:
            source_chunk, source_scores, source_valid = source_proposal
            proposals = [
                (source_chunk.copy(), source_scores.copy(), source_valid.copy(), 0)
            ]
            # Runtime consensus counts the already-observed STOP proposal as
            # hypothesis one, so only the remaining hypotheses are new calls.
            for _ in range(_fresh_consensus_count(consensus_hypotheses)):
                if policy_calls >= max_policy_calls:
                    break
                proposal = _query_b(
                    client,
                    observation,
                    subgoals[target_subgoal],
                    anchors=anchors,
                    subgoal_start=False,
                    episode_seed=branch_seed,
                    decision_index=policy_calls,
                    remaining_steps=remaining,
                    expected_runtime=expected_runtime,
                    np=np,
                )
                proposals.append(proposal)
                policy_calls += 1
            chosen = proposals[_medoid_index([value[0] for value in proposals], np)]
            chunk, scores, valid, _ = chosen
            candidate = _best_nonstop(scores, valid, np)
        else:
            chunk, scores, valid, candidate = _query_b(
                client,
                observation,
                subgoals[target_subgoal],
                anchors=anchors,
                subgoal_start=subgoal_start,
                episode_seed=branch_seed,
                decision_index=policy_calls,
                remaining_steps=remaining,
                expected_runtime=expected_runtime,
                np=np,
            )
            policy_calls += 1
        first_proposal = False
        subgoal_start = False
        if candidate == 0:
            retry_triggered = False
            if residual_retry_baseline:
                (
                    target_subgoal,
                    stop_streak,
                    retry_attempt_index,
                    retry_triggered,
                    advanced,
                ) = _apply_residual_stop_confirmation(
                    target_subgoal=target_subgoal,
                    subgoal_count=len(subgoals),
                    stop_streak=stop_streak,
                    retry_attempt_index=retry_attempt_index,
                )
            else:
                target_subgoal, stop_streak, advanced = _apply_stop_confirmation(
                    target_subgoal=target_subgoal,
                    subgoal_count=len(subgoals),
                    stop_streak=stop_streak,
                )
            if advanced and target_subgoal >= len(subgoals):
                break
            subgoal_start = retry_triggered or advanced
            continue
        stop_streak = 0
        selected = min(candidate, remaining)
        for action in chunk[:selected]:
            observation, _, _, _ = env.step(to_libero_action(action, np))
            executed_steps += 1
    return executed_steps, policy_calls


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        min(
            args.stop_stride,
            args.max_states_per_episode,
            args.rollout_steps,
            args.max_policy_calls,
        )
        < 1
        or args.min_source_elapsed_steps < 0
        or not 0.0 <= args.max_replay_mismatch_rate <= 0.05
    ):
        raise ValueError("counterfactual roll-forward settings are invalid")
    if args.residual_retry_baseline and not args.require_stop_pending:
        raise ValueError("residual B-retry labels require confirmed STOP states")
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
        # The primary demonstration corpus supplies both completion classes.
        # A train-only rollout seed may legitimately contain no successful
        # subgoal while still providing valid STOP states for option branches.
        require_both_completion_classes=False,
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

    expected_runtime = {
        "selector_weights_sha256": source_manifest["selector_weights_sha256"],
        "a1_checkpoint_weight_shards_sha256": source_manifest[
            "a1_checkpoint_weight_shards_sha256"
        ],
    }
    feature_dir = destination / "features"
    feature_dir.mkdir(parents=True)
    branch_path = destination / "counterfactual_branches.jsonl"
    files_sha256: dict[str, str] = {}
    branch_count = 0
    state_count = 0
    replay_rows = 0
    replay_mismatches = 0
    option_counts = {option.value: 0 for option in RECOVERY_OPTIONS}
    client = RemotePolicyClient(args.policy_host, args.policy_port)
    try:
        if not client.ping():
            raise RuntimeError("counterfactual B policy server did not answer ping")
        for episode_index, key in enumerate(sorted(decisions)):
            case = cases[(key[0], key[1])]
            rows = sorted(decisions[key], key=lambda row: int(row["policy_call"]))
            source_path = source_corpus / "features" / f"episode_{episode_index:06d}.npz"
            with np.load(source_path, allow_pickle=False) as value:
                arrays = {name: value[name].copy() for name in value.files}
            if len(rows) != len(arrays["sample_indices"]):
                raise OursContractError("counterfactual trace and feature rows differ")
            option_values = np.zeros((len(rows), len(RECOVERY_OPTIONS)), dtype=np.float32)
            option_valid = np.zeros((len(rows), len(RECOVERY_OPTIONS)), dtype=np.bool_)
            selected_positions = _selected_rollout_positions(
                rows,
                stride=args.stop_stride,
                maximum=args.max_states_per_episode,
                min_elapsed_steps=args.min_source_elapsed_steps,
                require_stop_pending=args.require_stop_pending,
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
                    hold = np.asarray(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32
                    )
                    for _ in range(int(run_manifest["initial_wait_steps"])):
                        observation, _, _, _ = env.step(hold)
                anchors: list[Any] = []
                for position, row in enumerate(rows):
                    env._check_success(goal)
                    new_anchor = row.get("new_subgoal_anchor")
                    if new_anchor is not None:
                        anchor = np.asarray(new_anchor, dtype=np.float32)
                        if anchor.shape != (2048,):
                            raise OursContractError("source rollout contains an invalid anchor")
                        anchors.append(anchor)
                    replay_match = _predicate_snapshot(env, goal) == row[
                        "success_predicates_before"
                    ]
                    replay_rows += 1
                    replay_mismatches += int(not replay_match)
                    if position in selected_positions and replay_match:
                        completed_before = int(row["completed_subtasks_before"])
                        active_subgoal = int(row["active_subgoal_index_before"])
                        complete = completed_before > active_subgoal + int(
                            episodes[key]["excluded_subtasks"]
                        )
                        previous = rows[position - 1] if position else None
                        stop_pending_before = bool(
                            previous is not None
                            and int(previous["stop_confirmation_streak_after"]) > 0
                            and int(previous["active_subgoal_index_after"]) == active_subgoal
                        )
                        rollouts = {}
                        for option in _valid_options(
                            active_subgoal,
                            len(task.steps),
                            stop_pending=stop_pending_before,
                            residual_retry_baseline=args.residual_retry_baseline,
                        ):
                            rollouts[option] = partial(
                                _roll_option,
                                env,
                                option=option,
                                stop_pending=stop_pending_before,
                                subgoals=task.steps,
                                active_subgoal=active_subgoal,
                                source_anchors=[anchor.copy() for anchor in anchors],
                                source_proposal=(
                                    np.asarray(row["predicted_chunk"], dtype=np.float32),
                                    np.asarray(row["selector_scores"], dtype=np.float32),
                                    np.asarray(row["selector_valid"], dtype=np.bool_),
                                ),
                                client=client,
                                rollout_steps=args.rollout_steps,
                                max_policy_calls=args.max_policy_calls,
                                consensus_hypotheses=args.consensus_hypotheses,
                                residual_retry_baseline=args.residual_retry_baseline,
                                expected_runtime=expected_runtime,
                                np=np,
                            )
                        branches = evaluate_counterfactual_rollouts(
                            env,
                            goal=goal,
                            option_rollouts=rollouts,
                            completed_subtasks_before=completed_before,
                            base_seed=base_seed,
                            state_index=int(arrays["sample_indices"][position]),
                            np=np,
                        )
                        for branch in branches:
                            if branch.option == RecoveryOption.ADVANCE.value:
                                branch = replace(
                                    branch,
                                    return_value=branch.return_value
                                    + (8.0 if complete else -8.0),
                                )
                            option_index = RECOVERY_OPTIONS.index(RecoveryOption(branch.option))
                            option_values[position, option_index] = branch.return_value
                            option_valid[position, option_index] = True
                            option_counts[branch.option] += 1
                            with branch_path.open("a", encoding="utf-8") as handle:
                                handle.write(
                                    json.dumps(
                                        {
                                            "episode_index": episode_index,
                                            "sample_index": int(
                                                arrays["sample_indices"][position]
                                            ),
                                            "task_type": key[0],
                                            "case": key[1],
                                            "trial": key[2],
                                            "policy_call": int(row["policy_call"]),
                                            "source_stop_pending": stop_pending_before,
                                            "source_context_sha256": row[
                                                "selector_context_sha256"
                                            ],
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
                        _set_trace_state(env, transitions[-1]["result_state"], np)
            finally:
                env.close()
            arrays["target_option_values"] = option_values
            arrays["target_option_valid"] = option_valid
            filename = f"episode_{episode_index:06d}.npz"
            output_path = feature_dir / filename
            _save_episode(output_path, arrays, np)
            files_sha256[filename] = sha256_file(output_path)
    finally:
        client.close()

    replay_mismatch_rate = replay_mismatches / replay_rows if replay_rows else 1.0
    if replay_mismatch_rate > args.max_replay_mismatch_rate:
        raise OursContractError(
            f"counterfactual replay mismatch rate {replay_mismatch_rate:.6f} exceeds cap"
        )
    manifest = dict(source_manifest)
    manifest.update(
        {
            "source": "train-only closed-loop same-state option roll-forwards",
            "source_kind": "counterfactual_live_rollout",
            "contains_counterfactuals": True,
            "counterfactual_source_manifest_sha256": sha256_file(source_manifest_path),
            "counterfactual_sampling": {
                "candidate_states": (
                    "confirmed B STOP proposals"
                    if args.require_stop_pending
                    else "B STOP proposals"
                ),
                "require_stop_pending": args.require_stop_pending,
                "residual_retry_baseline": args.residual_retry_baseline,
                "continuation_policy": (
                    "B-retry-confirmed-stop-one-retry-per-subtask"
                    if args.residual_retry_baseline
                    else "B-confirmed-stop-advance"
                ),
                "stop_stride": args.stop_stride,
                "max_states_per_episode": args.max_states_per_episode,
                "min_source_elapsed_steps": args.min_source_elapsed_steps,
                "state_count": state_count,
                "branch_count": branch_count,
                "options": option_counts,
                "rollout_steps": args.rollout_steps,
                "max_policy_calls": args.max_policy_calls,
                "consensus_hypotheses": args.consensus_hypotheses,
                "consensus_source_proposal_included": True,
                "future_injections": False,
                "replay_rows": replay_rows,
                "replay_mismatches": replay_mismatches,
                "replay_mismatch_rate": replay_mismatch_rate,
                "max_replay_mismatch_rate": args.max_replay_mismatch_rate,
                "training_only_restore": True,
                "runtime_restore": False,
                "restore_absolute_tolerance": 1e-12,
                "advance_requires_pending_stop": True,
                "confirmed_stop_continues_remaining_horizon": True,
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
