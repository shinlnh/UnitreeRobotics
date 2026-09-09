"""Evaluate a learned one-macro RESOLVE actor against paired frozen B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .a0 import (
    RemotePolicyClient,
    build_policy_observation,
    to_libero_action,
    unpack_action_chunk,
)
from .a1 import sha256_file
from .b import selector_decision_seed
from .ours import OursContractError
from .ours_counterfactual import capture_simulator_snapshot, snapshot_sha256
from .ours_counterfactual_rollout_prepare import _apply_residual_stop_confirmation
from .resolve_frontier_prepare import _configure_environment, _make_environment
from .resolve_frontier_rollout import (
    _baseline_arm,
    _load_demonstration,
    _read_jsonl,
    _reset_to_anchor,
    _runtime_provenance,
    flatten_frontier_anchors,
)
from .resolve_pilot_collect import _query_mosaic, _zero_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontiers", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--robocerebra-source", type=Path, required=True)
    parser.add_argument("--exclude-pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actor-sha256", required=True)
    parser.add_argument("--anchor-offset", type=int, default=32)
    parser.add_argument("--selection-seed", type=int, default=86007)
    parser.add_argument("--limit-anchors", type=int, default=32)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5559)
    parser.add_argument("--base-seed", type=int, default=86007)
    parser.add_argument("--control-frequency-hz", type=int, default=20)
    parser.add_argument("--max-policy-calls", type=int, default=16)
    parser.add_argument("--code-dimension", type=int, default=4)
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def excluded_source_groups(pairs: Path) -> set[int]:
    root = pairs.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = root / "anchors.jsonl"
    if sha256_file(records) != manifest.get("anchors_sha256"):
        raise OursContractError("RESOLVE exclusion-pair hash mismatch")
    return {int(row["source_manifest_index"]) for row in _read_jsonl(records)}


def _query_resolve(
    client: RemotePolicyClient,
    observation: dict[str, Any],
    instruction: str,
    *,
    anchors: list[Any],
    subgoal_start: bool,
    episode_seed: int,
    decision_index: int,
    remaining_steps: int,
    scalars: Any,
    actor_sha256: str,
    code_dimension: int,
    expected_runtime: dict[str, Any],
    np: Any,
) -> tuple[Any, int, Any | None, dict[str, Any]]:
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
            },
            "mosaic": {
                "steering_code": _zero_code(code_dimension, np),
                "prefix_length": 1,
                "program_id": "resolve-frontier-eval",
                "rule_index": 0,
            },
            "resolve": {
                "active": True,
                "scalars": np.asarray(scalars, dtype=np.float32),
                "program_position": 0,
                "milestone_id": 1,
            },
        },
    )
    selector = info.get("b_selector")
    resolve = info.get("resolve")
    if (
        not isinstance(selector, dict)
        or not isinstance(resolve, dict)
        or selector.get("runtime_provenance") != expected_runtime
        or selector.get("decision_seed") != selector_decision_seed(episode_seed, decision_index)
        or resolve.get("actor_sha256") != actor_sha256
        or resolve.get("active") is not True
    ):
        raise OursContractError("RESOLVE actor server provenance is invalid")
    chunk = unpack_action_chunk(raw_action, np)
    handoff = bool(resolve["handoff"])
    candidate = int(selector["candidate"]) if handoff else 16
    raw_anchor = selector.get("raw_anchor")
    anchor = None if raw_anchor is None else np.asarray(raw_anchor, dtype=np.float32)
    if subgoal_start and (anchor is None or anchor.shape != (2048,)):
        raise OursContractError("RESOLVE actor omitted its subgoal anchor")
    executable = chunk.copy()
    executable[:, :6] = np.clip(executable[:, :6], -1.0, 1.0)
    executable[:, 6] = np.clip(executable[:, 6], 0.0, 1.0)
    if (
        hashlib.sha256(executable.astype("<f4").tobytes()).hexdigest()
        != resolve["executed_chunk_sha256"]
    ):
        raise OursContractError("RESOLVE actor executed-chunk hash mismatch")
    return chunk, candidate, anchor, resolve


def _recovery_arm(
    env: Any,
    observation: dict[str, Any],
    *,
    instruction: str,
    predicate: list[str],
    scalars: Any,
    actor_sha256: str,
    episode_seed: int,
    steps: int,
    max_policy_calls: int,
    code_dimension: int,
    client: RemotePolicyClient,
    expected_runtime: dict[str, Any],
    np: Any,
) -> dict[str, Any]:
    active_subgoal = 0
    stop_streak = 0
    retry_attempt = 0
    starts_subgoal = True
    anchors: list[Any] = []
    reached = bool(env._eval_predicate(predicate))
    executed = 0
    calls = 0
    actor_metadata = None
    trace: list[Any] = []
    while executed < steps and calls < max_policy_calls and active_subgoal == 0:
        if calls == 0:
            chunk, candidate, anchor, actor_metadata = _query_resolve(
                client,
                observation,
                instruction,
                anchors=anchors,
                subgoal_start=starts_subgoal,
                episode_seed=episode_seed,
                decision_index=calls,
                remaining_steps=steps - executed,
                scalars=scalars,
                actor_sha256=actor_sha256,
                code_dimension=code_dimension,
                expected_runtime=expected_runtime,
                np=np,
            )
        else:
            chunk, _, _, candidate, _, anchor = _query_mosaic(
                client,
                observation,
                instruction,
                anchors=anchors,
                subgoal_start=starts_subgoal,
                episode_seed=episode_seed,
                decision_index=calls,
                remaining_steps=steps - executed,
                program_id="resolve-frontier-B-continuation",
                rule_index=0,
                steering_code=_zero_code(code_dimension, np),
                prefix_length=1,
                expected_runtime=expected_runtime,
                np=np,
            )
        calls += 1
        if starts_subgoal:
            if anchor is None:
                raise OursContractError("RESOLVE actor omitted its subgoal anchor")
            anchors.append(anchor)
            starts_subgoal = False
        if candidate == 0:
            (
                active_subgoal,
                stop_streak,
                retry_attempt,
                starts_subgoal,
                _,
            ) = _apply_residual_stop_confirmation(
                target_subgoal=active_subgoal,
                subgoal_count=1,
                stop_streak=stop_streak,
                retry_attempt_index=retry_attempt,
            )
            continue
        stop_streak = 0
        for action in chunk[: min(candidate, steps - executed)]:
            libero = to_libero_action(action, np)
            trace.append(libero.copy())
            observation, _, _, _ = env.step(libero)
            executed += 1
            reached = reached or bool(env._eval_predicate(predicate))
    if actor_metadata is None:
        raise OursContractError("RESOLVE actor arm made no policy request")
    action_trace = np.asarray(trace, dtype="<f4")
    return {
        "reached": reached,
        "executed_steps": executed,
        "policy_calls": calls,
        "actor": actor_metadata,
        "action_trace_sha256": hashlib.sha256(action_trace.tobytes()).hexdigest(),
        "final_state_sha256": snapshot_sha256(capture_simulator_snapshot(env), np),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        len(args.actor_sha256) != 64
        or min(
            args.anchor_offset,
            args.limit_anchors,
            args.control_frequency_hz,
            args.max_policy_calls,
            args.code_dimension,
        )
        < 1
    ):
        raise OursContractError("RESOLVE actor-evaluation arguments are invalid")
    frontier_root = args.frontiers.expanduser().resolve()
    frontier_manifest = json.loads((frontier_root / "manifest.json").read_text(encoding="utf-8"))
    rows = _read_jsonl(frontier_root / "frontiers.jsonl")
    if sha256_file(frontier_root / "frontiers.jsonl") != frontier_manifest.get("frontiers_sha256"):
        raise OursContractError("RESOLVE actor frontier hash mismatch")
    excluded = excluded_source_groups(args.exclude_pairs)
    rows = [row for row in rows if int(row["source_manifest_index"]) not in excluded]
    selected = flatten_frontier_anchors(
        rows,
        split="train",
        anchor_offset=args.anchor_offset,
        selection_seed=args.selection_seed,
        start=0,
        limit=args.limit_anchors,
    )
    destination = args.output.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite RESOLVE actor evaluation: {destination}")
    destination.mkdir(parents=True)
    np, bddl_utils, runtime = _configure_environment(args.robocerebra_source)
    expected_runtime = _runtime_provenance(args.runtime_manifest.expanduser().resolve())
    client = RemotePolicyClient(args.policy_host, args.policy_port)
    output_rows: list[dict[str, Any]] = []
    try:
        if not client.ping():
            raise RuntimeError("RESOLVE actor policy server did not answer ping")
        for index, row in enumerate(selected):
            runtime_xml, states, _ = _load_demonstration(row, runtime)
            anchor = int(row["anchor_frame"])
            steps = int(row["first_true_frame"]) - anchor
            predicate = list(row["predicate"])
            env = _make_environment(
                Path(row["bddl"]),
                bddl_utils=bddl_utils,
                runtime=runtime,
                control_frequency_hz=args.control_frequency_hz,
                camera_observations=True,
            )
            episode_seed = args.base_seed + index
            try:
                observation = _reset_to_anchor(env, runtime_xml, states[anchor])
                client.reset({"episode_seed": episode_seed})
                baseline, _ = _baseline_arm(
                    env,
                    observation,
                    instruction=str(row["subgoal_instruction"]),
                    predicate=predicate,
                    episode_seed=episode_seed,
                    steps=steps,
                    max_policy_calls=args.max_policy_calls,
                    code_dimension=args.code_dimension,
                    client=client,
                    expected_runtime=expected_runtime,
                    np=np,
                )
                observation = _reset_to_anchor(env, runtime_xml, states[anchor])
                client.reset({"episode_seed": episode_seed})
                scalars = np.asarray(
                    (
                        anchor / max(1, int(row["first_true_frame"])),
                        min(1.0, steps / 64.0),
                        min(1.0, int(row["subgoal_index"]) / 15.0),
                        0.0,
                        0.0,
                        0.0,
                    ),
                    dtype=np.float32,
                )
                recovery = _recovery_arm(
                    env,
                    observation,
                    instruction=str(row["subgoal_instruction"]),
                    predicate=predicate,
                    scalars=scalars,
                    actor_sha256=args.actor_sha256,
                    episode_seed=episode_seed,
                    steps=steps,
                    max_policy_calls=args.max_policy_calls,
                    code_dimension=args.code_dimension,
                    client=client,
                    expected_runtime=expected_runtime,
                    np=np,
                )
            finally:
                env.close()
            output_rows.append(
                {
                    "anchor_index": index,
                    "source_manifest_index": int(row["source_manifest_index"]),
                    "scene": row["scene"],
                    "case": row["case"],
                    "predicate": predicate,
                    "anchor_frame": anchor,
                    "paired_physical_steps": steps,
                    "episode_seed": episode_seed,
                    "baseline": baseline,
                    "recovery": recovery,
                    "paired_effect": int(recovery["reached"]) - int(baseline["reached"]),
                }
            )
    finally:
        client.close()
    records = destination / "anchors.jsonl"
    _write_jsonl(records, output_rows)
    manifest = {
        "schema_version": 1,
        "experiment_id": "Ours",
        "stage": "RESOLVE one-macro actor paired physical evaluation",
        "novelty_claim": False,
        "actor_sha256": args.actor_sha256,
        "frontier_manifest_sha256": sha256_file(frontier_root / "manifest.json"),
        "excluded_pairs_manifest_sha256": sha256_file(
            args.exclude_pairs.expanduser().resolve() / "manifest.json"
        ),
        "excluded_source_groups": len(excluded),
        "anchor_offset": args.anchor_offset,
        "selection_seed": args.selection_seed,
        "anchors": len(output_rows),
        "baseline_reached": sum(int(row["baseline"]["reached"]) for row in output_rows),
        "recovery_reached": sum(int(row["recovery"]["reached"]) for row in output_rows),
        "positive_effects": sum(row["paired_effect"] > 0 for row in output_rows),
        "negative_effects": sum(row["paired_effect"] < 0 for row in output_rows),
        "actor_interventions": sum(not row["recovery"]["actor"]["handoff"] for row in output_rows),
        "records_sha256": sha256_file(records),
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def main() -> int:
    try:
        result = run(_parser().parse_args())
    except (FileNotFoundError, OursContractError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
