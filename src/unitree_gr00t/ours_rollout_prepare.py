"""Convert train-only live Ours traces into an audited temporal corpus."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .a1 import sha256_file
from .b_runtime import context_sha256
from .ours import OURS_ID, OURS_METHOD, OURS_PARENT, OURS_VARIANT, OursContractError
from .ours_data import OURS_CORPUS_MANIFEST, OURS_CORPUS_SCHEMA


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--failure-onset-steps", type=int, default=75)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _save_episode(path: Path, arrays: dict[str, Any], np: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _episode_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return str(record["task_type"]), str(record["case"]), int(record["trial"])


def build_failure_targets(
    subgoals: Sequence[Any],
    elapsed: Sequence[int],
    complete: Sequence[bool],
    failure_after_injection: Sequence[bool],
    *,
    failure_onset_steps: int,
    np: Any,
) -> Any:
    """Label failed temporal segments with future outcome only during data generation."""

    if (
        failure_onset_steps < 1
        or len({len(subgoals), len(elapsed), len(complete), len(failure_after_injection)}) != 1
    ):
        raise ValueError("failure target inputs are inconsistent")
    successful_subgoals = {
        subgoal for subgoal, is_complete in zip(subgoals, complete, strict=True) if is_complete
    }
    return np.asarray(
        [
            (injected or (subgoal not in successful_subgoals and age >= failure_onset_steps))
            and not is_complete
            for injected, subgoal, age, is_complete in zip(
                failure_after_injection,
                subgoals,
                elapsed,
                complete,
                strict=True,
            )
        ],
        dtype=np.bool_,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.failure_onset_steps < 1:
        raise ValueError("failure onset must be positive")
    rollout = args.rollout.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite live Ours corpus: {destination}")
    run_manifest_path = rollout / "run_manifest.json"
    summary_path = rollout / "summary.json"
    decisions_path = rollout / "decisions.jsonl"
    episodes_path = rollout / "episodes.jsonl"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    identity = (
        run_manifest.get("experiment_id"),
        run_manifest.get("variant"),
        run_manifest.get("method"),
    )
    if identity != (OURS_ID, OURS_VARIANT, OURS_METHOD):
        raise OursContractError("live corpus source is not an Ours rollout")
    base_seed = int(run_manifest["base_seed"])
    if base_seed not in {10007, 11007, 12007}:
        raise OursContractError("live corpus source is outside frozen train seeds")
    if (
        not bool(run_manifest.get("capture_training_context"))
        or not bool(summary.get("complete"))
        or int(summary.get("episodes", -1)) != int(run_manifest["expected_episodes"])
    ):
        raise OursContractError("live corpus source is incomplete or lacks context export")

    import numpy as np

    episode_results = {_episode_key(row): row for row in _read_jsonl(episodes_path)}
    decisions: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(decisions_path):
        if row.get("policy_invoked") is False:
            continue
        decisions[_episode_key(row)].append(row)
    if set(decisions) != set(episode_results):
        raise OursContractError("live corpus decision and episode inventories differ")

    feature_dir = destination / "features"
    feature_dir.mkdir(parents=True)
    files_sha256: dict[str, str] = {}
    sample_index = 0
    positives = 0
    negatives = 0
    failure_positives = 0
    episode_ids: list[int] = []
    for episode_index, key in enumerate(sorted(decisions)):
        rows = sorted(decisions[key], key=lambda row: int(row["policy_call"]))
        excluded = int(episode_results[key]["excluded_subtasks"])
        contexts: list[Any] = []
        anchors: list[Any] = []
        chunks: list[Any] = []
        scores: list[Any] = []
        valid: list[Any] = []
        candidates: list[int] = []
        subgoals: list[int] = []
        frames: list[int] = []
        elapsed: list[int] = []
        complete: list[bool] = []
        failure_after_injection: list[bool] = []
        anchor_positions: list[int] = []
        current_anchor: Any | None = None
        current_anchor_position: int | None = None
        active_subgoal: int | None = None
        subgoal_start_step = 0
        injection_seen = False
        for row in rows:
            context = np.asarray(row.get("recovery_training_context"), dtype=np.float32)
            if (
                context.shape != (2048,)
                or context_sha256(context) != row["selector_context_sha256"]
            ):
                raise OursContractError("live context is missing or differs from its trace hash")
            subgoal = int(row["active_subgoal_index_before"])
            if active_subgoal != subgoal:
                active_subgoal = subgoal
                subgoal_start_step = int(row["step_before"])
            new_anchor = row.get("new_subgoal_anchor")
            if new_anchor is not None:
                current_anchor = np.asarray(new_anchor, dtype=np.float32)
                current_anchor_position = len(contexts)
            if (
                current_anchor is None
                or current_anchor.shape != (2048,)
                or current_anchor_position is None
            ):
                raise OursContractError("live rollout is missing its causal subgoal anchor")
            completed_before = max(0, int(row["completed_subtasks_before"]) - excluded)
            is_complete = completed_before > subgoal
            contexts.append(context.astype(np.float16))
            anchors.append(current_anchor.astype(np.float16))
            anchor_positions.append(current_anchor_position)
            chunks.append(np.asarray(row["predicted_chunk"], dtype=np.float16))
            scores.append(np.asarray(row["selector_scores"], dtype=np.float32))
            valid.append(np.asarray(row["selector_valid"], dtype=np.bool_))
            candidates.append(int(row["selector_candidate_before_recovery"]))
            subgoals.append(subgoal)
            frames.append(int(row["step_before"]))
            elapsed.append(int(row["step_before"]) - subgoal_start_step)
            complete.append(is_complete)
            failure_after_injection.append(injection_seen and not is_complete)
            positives += int(is_complete)
            negatives += int(not is_complete)
            injection_seen = injection_seen or any(
                transition.get("injection") is not None for transition in row["transitions"]
            )
        count = len(rows)
        failure = build_failure_targets(
            subgoals,
            elapsed,
            complete,
            failure_after_injection,
            failure_onset_steps=args.failure_onset_steps,
            np=np,
        )
        failure_positives += int(failure.sum())
        ids = np.arange(sample_index, sample_index + count, dtype=np.int64)
        sample_index += count
        filename = f"episode_{episode_index:06d}.npz"
        path = feature_dir / filename
        _save_episode(
            path,
            {
                "sample_indices": ids,
                "contexts": np.stack(contexts),
                "anchor_contexts": np.stack(anchors),
                "action_chunks": np.stack(chunks),
                "selector_scores": np.stack(scores),
                "selector_valid": np.stack(valid),
                "selector_candidates": np.asarray(candidates, dtype=np.int8),
                "anchor_positions": np.asarray(anchor_positions, dtype=np.int32),
                "subgoal_indices": np.asarray(subgoals, dtype=np.int16),
                "frame_indices": np.asarray(frames, dtype=np.int32),
                "elapsed_steps": np.asarray(elapsed, dtype=np.int32),
                "target_progress": np.zeros(count, dtype=np.float16),
                "target_progress_valid": np.zeros(count, dtype=np.bool_),
                "target_complete": np.asarray(complete, dtype=np.bool_),
                "target_failure": failure,
            },
            np,
        )
        files_sha256[filename] = sha256_file(path)
        episode_ids.append(episode_index)

    manifest = {
        "schema_version": OURS_CORPUS_SCHEMA,
        "experiment_id": OURS_ID,
        "variant": OURS_VARIANT,
        "method": OURS_METHOD,
        "parent_experiment": OURS_PARENT,
        "source": "train-only continuous live rollouts with offline privileged labels",
        "source_kind": "live_rollout",
        "rollout": str(rollout),
        "rollout_run_manifest_sha256": sha256_file(run_manifest_path),
        "rollout_summary_sha256": sha256_file(summary_path),
        "rollout_decisions_sha256": sha256_file(decisions_path),
        "rollout_episodes_sha256": sha256_file(episodes_path),
        "selector_weights_sha256": run_manifest["selector_weights_sha256"],
        "a1_checkpoint_weight_shards_sha256": run_manifest["a1_checkpoint_weight_shards_sha256"],
        "context_width": 2048,
        "action_horizon": 16,
        "action_dim": 7,
        "consumed_rollout_base_seeds": [base_seed],
        "contains_counterfactuals": False,
        "failure_label": (
            "privileged post-injection or unsuccessful-subgoal temporal onset; "
            "never a runtime input"
        ),
        "failure_onset_steps": args.failure_onset_steps,
        "limited": False,
        "files": len(files_sha256),
        "samples": sample_index,
        "files_sha256": files_sha256,
        "split_episodes": {"train": episode_ids, "development": []},
        "label_counts": {
            "completion_negative": negatives,
            "completion_positive": positives,
            "failure_positive": failure_positives,
        },
    }
    _write_json(destination / OURS_CORPUS_MANIFEST, manifest)
    return manifest


def main() -> int:
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
