"""Run exact expert and frozen-B arms from verified train-only goal frontiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .a0 import RemotePolicyClient, to_libero_action
from .a1 import sha256_file
from .a1_data import model_action, mujoco_autolimits_xml
from .ours import OursContractError
from .ours_counterfactual import capture_simulator_snapshot, snapshot_sha256
from .ours_counterfactual_rollout_prepare import _apply_residual_stop_confirmation
from .resolve_frontier_prepare import (
    _configure_environment,
    _make_environment,
    portable_model_xml,
)
from .resolve_pilot_collect import _query_mosaic, _zero_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontiers", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--robocerebra-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "development"), default="train")
    parser.add_argument(
        "--anchor-offset",
        type=int,
        help="retain only anchors this many physical steps before the frontier",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        help="SHA-256-rank anchors before applying the limit",
    )
    parser.add_argument("--anchor-start", type=int, default=0)
    parser.add_argument("--limit-anchors", type=int)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5559)
    parser.add_argument("--control-frequency-hz", type=int, default=20)
    parser.add_argument("--max-policy-calls", type=int, default=16)
    parser.add_argument("--code-dimension", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=81007)
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten_frontier_anchors(
    rows: list[dict[str, Any]],
    *,
    split: str,
    anchor_offset: int | None,
    selection_seed: int | None,
    start: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Expand immutable frontier events into deterministically ordered anchors."""

    if (
        split not in {"train", "development"}
        or (anchor_offset is not None and anchor_offset < 1)
        or (selection_seed is not None and selection_seed < 0)
        or start < 0
        or (limit is not None and limit < 1)
    ):
        raise OursContractError("RESOLVE frontier rollout selection is invalid")
    anchors: list[dict[str, Any]] = []
    for frontier_index, row in enumerate(rows):
        if row["split"] != split:
            continue
        for anchor in row["anchor_frames"]:
            offset = int(row["first_true_frame"]) - int(anchor)
            if anchor_offset is not None and offset != anchor_offset:
                continue
            anchors.append(
                row
                | {
                    "frontier_index": frontier_index,
                    "anchor_frame": int(anchor),
                    "anchor_offset": offset,
                }
            )
    if selection_seed is None:
        anchors.sort(
            key=lambda row: (
                int(row["source_manifest_index"]),
                int(row["subgoal_index"]),
                -int(row["anchor_offset"]),
            )
        )
    else:
        anchors.sort(
            key=lambda row: hashlib.sha256(
                json.dumps(
                    (
                        "resolve-frontier-selection-v1",
                        selection_seed,
                        int(row["source_manifest_index"]),
                        int(row["subgoal_index"]),
                        row["predicate"],
                        int(row["anchor_frame"]),
                    ),
                    separators=(",", ":"),
                ).encode()
            ).digest()
        )
    return anchors[start:] if limit is None else anchors[start : start + limit]


def expert_model_chunk(actions: Any, *, frame: int, horizon: int, np: Any) -> Any:
    """Convert raw LIBERO actions to a padded GR00T-domain expert chunk."""

    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != 7 or horizon < 1 or not np.isfinite(values).all():
        raise OursContractError("RESOLVE frontier expert actions are invalid")
    if not 0 <= frame < len(values):
        raise OursContractError("RESOLVE frontier expert frame is out of range")
    selected = [
        model_action(values[min(frame + offset, len(values) - 1)], np) for offset in range(horizon)
    ]
    return np.asarray(selected, dtype=np.float32)


def _runtime_provenance(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "selector_weights_sha256": manifest.get("selector_weights_sha256"),
        "a1_checkpoint_weight_shards_sha256": manifest.get("a1_checkpoint_weight_shards_sha256"),
    }
    if not isinstance(expected["selector_weights_sha256"], str) or not isinstance(
        expected["a1_checkpoint_weight_shards_sha256"], dict
    ):
        raise OursContractError("RESOLVE frontier runtime manifest is invalid")
    return expected


def _load_demonstration(row: dict[str, Any], runtime: Any) -> tuple[str, Any, Any]:
    import h5py
    import numpy as np

    path = Path(row["demonstration"])
    if sha256_file(path) != row["demonstration_sha256"]:
        raise OursContractError("RESOLVE frontier demonstration hash mismatch")
    if sha256_file(Path(row["bddl"])) != row["bddl_sha256"]:
        raise OursContractError("RESOLVE frontier BDDL hash mismatch")
    with h5py.File(path, "r") as handle:
        group = handle["data"]["demo_1"]
        states = group["states"][:]
        actions = group["actions"][:]
        source_xml = str(group.attrs["model_file"])
    if hashlib.sha256(source_xml.encode()).hexdigest() != row["source_model_xml_sha256"]:
        raise OursContractError("RESOLVE frontier source XML hash mismatch")
    runtime_xml = mujoco_autolimits_xml(
        portable_model_xml(
            source_xml,
            libero_package_root=runtime[2],
            robosuite_package_root=runtime[3],
        )
    )
    if hashlib.sha256(runtime_xml.encode()).hexdigest() != row["runtime_model_xml_sha256"]:
        raise OursContractError("RESOLVE frontier runtime XML hash mismatch")
    if states.ndim != 2 or actions.ndim != 2 or len(states) != len(actions):
        raise OursContractError("RESOLVE frontier HDF5 arrays are invalid")
    return runtime_xml, np.asarray(states), np.asarray(actions)


def _reset_to_anchor(env: Any, runtime_xml: str, state: Any) -> dict[str, Any]:
    env.reset_from_xml_string(runtime_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)
    return env._get_observations()


def _expert_arm(
    env: Any,
    observation: dict[str, Any],
    *,
    actions: Any,
    anchor: int,
    steps: int,
    predicate: list[str],
    np: Any,
) -> tuple[bool, int, str]:
    reached = bool(env._eval_predicate(predicate))
    executed = 0
    for action in actions[anchor : anchor + steps]:
        observation, _, _, _ = env.step(np.asarray(action, dtype=np.float32))
        del observation
        executed += 1
        reached = reached or bool(env._eval_predicate(predicate))
    final = snapshot_sha256(capture_simulator_snapshot(env), np)
    return reached, executed, final


def _baseline_arm(
    env: Any,
    observation: dict[str, Any],
    *,
    instruction: str,
    predicate: list[str],
    episode_seed: int,
    steps: int,
    max_policy_calls: int,
    code_dimension: int,
    client: RemotePolicyClient,
    expected_runtime: dict[str, Any],
    np: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    active_subgoal = 0
    stop_streak = 0
    retry_attempt = 0
    starts_subgoal = True
    anchors: list[Any] = []
    reached = bool(env._eval_predicate(predicate))
    executed = 0
    calls = 0
    first_context = None
    first_chunk = None
    first_candidate = None
    action_trace: list[Any] = []
    while executed < steps and calls < max_policy_calls and active_subgoal == 0:
        chunk, _, _, candidate, context, anchor = _query_mosaic(
            client,
            observation,
            instruction,
            anchors=anchors,
            subgoal_start=starts_subgoal,
            episode_seed=episode_seed,
            decision_index=calls,
            remaining_steps=steps - executed,
            program_id="exact-B-frontier",
            rule_index=0,
            steering_code=_zero_code(code_dimension, np),
            prefix_length=1,
            expected_runtime=expected_runtime,
            np=np,
        )
        if first_context is None:
            first_context = context.copy()
            first_chunk = chunk.copy()
            first_candidate = candidate
        calls += 1
        if starts_subgoal:
            if anchor is None:
                raise OursContractError("RESOLVE frontier B omitted its subgoal anchor")
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
        count = min(candidate, steps - executed)
        for action in chunk[:count]:
            libero = to_libero_action(action, np)
            action_trace.append(libero.copy())
            observation, _, _, _ = env.step(libero)
            executed += 1
            reached = reached or bool(env._eval_predicate(predicate))
    if first_context is None or first_chunk is None or first_candidate is None:
        raise OursContractError("RESOLVE frontier B made no policy query")
    trace = np.asarray(action_trace, dtype="<f4")
    return {
        "reached": reached,
        "executed_steps": executed,
        "policy_calls": calls,
        "initial_candidate": first_candidate,
        "action_trace_sha256": hashlib.sha256(trace.tobytes()).hexdigest(),
        "final_state_sha256": snapshot_sha256(capture_simulator_snapshot(env), np),
    }, {"context": first_context, "base_chunk": first_chunk}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        min(args.control_frequency_hz, args.max_policy_calls, args.code_dimension) < 1
        or args.base_seed < 0
    ):
        raise ValueError("RESOLVE frontier rollout arguments are invalid")
    root = args.frontiers.expanduser().resolve()
    source_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("physical_label") != (
        "false-to-true BDDL predicate under recorded simulator state"
    ):
        raise OursContractError("RESOLVE frontier source is not physically verified")
    rows = _read_jsonl(root / "frontiers.jsonl")
    if sha256_file(root / "frontiers.jsonl") != source_manifest["frontiers_sha256"]:
        raise OursContractError("RESOLVE frontier index hash mismatch")
    selected = flatten_frontier_anchors(
        rows,
        split=args.split,
        anchor_offset=args.anchor_offset,
        selection_seed=args.selection_seed,
        start=args.anchor_start,
        limit=args.limit_anchors,
    )
    if not selected:
        raise OursContractError("RESOLVE frontier rollout selected no anchors")
    destination = args.output.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite RESOLVE frontier rollout: {destination}")
    feature_dir = destination / "features"
    feature_dir.mkdir(parents=True)
    np, bddl_utils, runtime = _configure_environment(args.robocerebra_source)
    expected_runtime = _runtime_provenance(args.runtime_manifest.expanduser().resolve())
    output_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    files_sha256: dict[str, str] = {}
    client = RemotePolicyClient(args.policy_host, args.policy_port)
    try:
        if not client.ping():
            raise RuntimeError("RESOLVE frontier policy server did not answer ping")
        for anchor_index, row in enumerate(selected):
            selection_index = args.anchor_start + anchor_index
            runtime_xml, states, actions = _load_demonstration(row, runtime)
            anchor = int(row["anchor_frame"])
            steps = int(row["first_true_frame"]) - anchor
            if steps < 1:
                raise OursContractError("RESOLVE frontier has no intervention horizon")
            env = _make_environment(
                Path(row["bddl"]),
                bddl_utils=bddl_utils,
                runtime=runtime,
                control_frequency_hz=args.control_frequency_hz,
                camera_observations=True,
            )
            try:
                predicate = list(row["predicate"])
                observation = _reset_to_anchor(env, runtime_xml, states[anchor])
                if env._eval_predicate(predicate):
                    exclusions.append(
                        {
                            "selection_index": selection_index,
                            "source_manifest_index": int(row["source_manifest_index"]),
                            "predicate": predicate,
                            "reason": "predicate true after executable reset post-processing",
                        }
                    )
                    continue
                expert_reached, expert_steps, expert_final = _expert_arm(
                    env,
                    observation,
                    actions=actions,
                    anchor=anchor,
                    steps=steps,
                    predicate=predicate,
                    np=np,
                )
                observation = _reset_to_anchor(env, runtime_xml, states[anchor])
                episode_seed = args.base_seed + selection_index
                client.reset({"episode_seed": episode_seed})
                baseline, feature = _baseline_arm(
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
            finally:
                env.close()
            expert_chunk = expert_model_chunk(actions, frame=anchor, horizon=16, np=np)
            filename = f"anchor_{anchor_index:06d}.npz"
            feature_path = feature_dir / filename
            np.savez_compressed(
                feature_path,
                context=feature["context"].astype(np.float32),
                base_chunk=feature["base_chunk"].astype(np.float32),
                expert_chunk=expert_chunk,
            )
            files_sha256[filename] = sha256_file(feature_path)
            output_rows.append(
                {
                    "anchor_index": anchor_index,
                    "selection_index": selection_index,
                    "source_manifest_index": int(row["source_manifest_index"]),
                    "scene": row["scene"],
                    "case": row["case"],
                    "split": row["split"],
                    "subgoal_index": int(row["subgoal_index"]),
                    "subgoal_instruction": row["subgoal_instruction"],
                    "predicate": predicate,
                    "anchor_frame": anchor,
                    "first_true_frame": int(row["first_true_frame"]),
                    "paired_physical_steps": steps,
                    "episode_seed": episode_seed,
                    "expert_reached": expert_reached,
                    "expert_executed_steps": expert_steps,
                    "expert_final_state_sha256": expert_final,
                    "baseline": baseline,
                    "expert_superiority": int(expert_reached) - int(baseline["reached"]),
                    "feature_file": filename,
                }
            )
    finally:
        client.close()
    records_path = destination / "anchors.jsonl"
    _write_jsonl(records_path, output_rows)
    manifest = {
        "schema_version": 1,
        "experiment_id": "Ours",
        "stage": "RESOLVE train-only expert/B frontier arms",
        "novelty_claim": False,
        "frontier_source": str(root),
        "frontier_manifest_sha256": sha256_file(root / "manifest.json"),
        "runtime_manifest": str(args.runtime_manifest.expanduser().resolve()),
        "runtime_manifest_sha256": sha256_file(args.runtime_manifest.expanduser().resolve()),
        "runtime_provenance": expected_runtime,
        "split": args.split,
        "anchor_offset": args.anchor_offset,
        "selection_seed": args.selection_seed,
        "anchor_start": args.anchor_start,
        "limited": args.limit_anchors is not None,
        "selected_anchors": len(selected),
        "anchors": len(output_rows),
        "exclusions": exclusions,
        "expert_reached": sum(int(row["expert_reached"]) for row in output_rows),
        "baseline_reached": sum(int(row["baseline"]["reached"]) for row in output_rows),
        "positive_gaps": sum(row["expert_superiority"] > 0 for row in output_rows),
        "negative_gaps": sum(row["expert_superiority"] < 0 for row in output_rows),
        "anchors_sha256": sha256_file(records_path),
        "files_sha256": files_sha256,
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def main() -> int:
    result = run(_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
