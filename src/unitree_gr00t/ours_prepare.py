"""Build Ours' compact temporal corpus from frozen B demonstration features."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .a1 import sha256_file
from .b import candidate_validity, inspect_selector_checkpoint, select_unified_candidate
from .b_model import SelectorModelConfig, build_selector
from .ours import OURS_ID, OURS_METHOD, OURS_PARENT, OURS_VARIANT
from .ours_data import OURS_CORPUS_MANIFEST, OURS_CORPUS_SCHEMA, build_demonstration_targets


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-dataset", type=Path, required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def ordered_episode_rows(samples: list[dict[str, Any]], episode_index: int) -> list[dict[str, Any]]:
    rows = [row for row in samples if int(row["episode_index"]) == episode_index]
    rows.sort(key=lambda row: int(row["sample_index"]))
    if not rows:
        raise ValueError(f"episode {episode_index} has no B samples")
    last_frame_by_subgoal: dict[int, int] = {}
    for row in rows:
        subgoal = int(row["subgoal_index"])
        frame = int(row["frame_index"])
        if frame < last_frame_by_subgoal.get(subgoal, frame):
            raise ValueError(f"episode {episode_index} B samples are not causal")
        last_frame_by_subgoal[subgoal] = frame
    return rows


def _model_config(provenance: dict[str, Any]) -> SelectorModelConfig:
    return SelectorModelConfig(
        action_horizon=int(provenance["action_horizon"]),
        action_dim=int(provenance["action_dim"]),
        context_width=int(provenance["context_width"]),
        scoring_width=int(provenance["scoring_width"]),
        scoring_layers=int(provenance["scoring_layers"]),
        scoring_heads=int(provenance["scoring_heads"]),
        feedforward_width=int(provenance["feedforward_width"]),
        dropout=float(provenance["dropout"]),
    )


def _score_episode(
    *,
    selector: Any,
    model_config: SelectorModelConfig,
    rows: list[dict[str, Any]],
    contexts: Any,
    chunks: Any,
    batch_size: int,
    device: str,
    np: Any,
    torch: Any,
) -> tuple[Any, Any, Any, Any]:
    first_by_subgoal: dict[int, int] = {}
    for position, row in enumerate(rows):
        first_by_subgoal.setdefault(int(row["subgoal_index"]), position)
    # Invalid B candidates use a finite float32 sentinel near -3e38. Keeping
    # scores in float32 avoids silently turning that audited mask value into
    # negative infinity during corpus export.
    scores = np.empty((len(rows), model_config.action_horizon + 1), dtype=np.float32)
    valid = np.empty_like(scores, dtype=np.bool_)
    candidates = np.empty(len(rows), dtype=np.int8)
    anchor_positions = np.empty(len(rows), dtype=np.int32)
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        max_anchors = max(int(row["subgoal_index"]) + 1 for row in batch_rows)
        anchor_history = np.zeros(
            (len(batch_rows), max_anchors, model_config.context_width), dtype=np.float32
        )
        anchor_valid = np.zeros((len(batch_rows), max_anchors), dtype=np.bool_)
        candidate_valid = np.zeros(
            (len(batch_rows), model_config.action_horizon + 1), dtype=np.bool_
        )
        for offset, row in enumerate(batch_rows):
            subgoal_index = int(row["subgoal_index"])
            for anchor_index in range(subgoal_index + 1):
                if anchor_index not in first_by_subgoal:
                    raise ValueError("B episode is missing a preceding subgoal anchor")
                anchor_history[offset, anchor_index] = contexts[first_by_subgoal[anchor_index]]
                anchor_valid[offset, anchor_index] = True
            remaining = max(1, int(row["trajectory_steps"]) - int(row["frame_index"]))
            candidate_valid[offset] = candidate_validity(
                decision_step=0,
                trajectory_steps=remaining,
                horizon=model_config.action_horizon,
                max_prefix=model_config.action_horizon,
            )
            anchor_positions[start + offset] = first_by_subgoal[subgoal_index]
        with torch.inference_mode():
            batch_scores = (
                selector(
                    torch.as_tensor(
                        chunks[start : start + len(batch_rows)],
                        device=device,
                        dtype=torch.float32,
                    ),
                    torch.as_tensor(
                        contexts[start : start + len(batch_rows)],
                        device=device,
                        dtype=torch.float32,
                    ),
                    torch.as_tensor(anchor_history, device=device, dtype=torch.float32),
                    torch.as_tensor(anchor_valid, device=device, dtype=torch.bool),
                    torch.as_tensor(candidate_valid, device=device, dtype=torch.bool),
                )
                .float()
                .cpu()
                .numpy()
            )
        scores[start : start + len(batch_rows)] = batch_scores
        valid[start : start + len(batch_rows)] = candidate_valid
        for offset, (row_scores, row_valid) in enumerate(
            zip(batch_scores, candidate_valid, strict=True)
        ):
            candidates[start + offset] = select_unified_candidate(
                row_scores.tolist(), row_valid.tolist()
            )
    return scores, valid, candidates, anchor_positions


def _save_episode(path: Path, arrays: dict[str, Any], np: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1 or (args.limit_episodes is not None and args.limit_episodes < 1):
        raise ValueError("Ours preparation counts must be positive")
    dataset = args.b_dataset.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    manifest_path = destination / OURS_CORPUS_MANIFEST
    if manifest_path.exists():
        raise FileExistsError(f"Ours compact corpus is already complete: {destination}")
    if destination.exists() and any(destination.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite Ours corpus: {destination}")

    b_manifest_path = dataset / "manifest.json"
    b_feature_manifest_path = dataset / "feature_manifest.json"
    b_manifest = json.loads(b_manifest_path.read_text(encoding="utf-8"))
    feature_manifest = json.loads(b_feature_manifest_path.read_text(encoding="utf-8"))
    if b_manifest.get("experiment_id") != "B" or feature_manifest.get("experiment_id") != "B":
        raise ValueError("Ours preparation requires frozen B inputs")
    if not feature_manifest.get("complete") or feature_manifest.get("limited"):
        raise ValueError("Ours preparation requires the complete B feature corpus")
    audit, provenance = inspect_selector_checkpoint(
        args.selector_checkpoint,
        expected_action_horizon=int(feature_manifest["action_horizon"]),
        expected_context_width=int(feature_manifest["context_width"]),
    )
    if provenance.get("a1_checkpoint_weight_shards_sha256") != feature_manifest.get(
        "checkpoint_weight_shards_sha256"
    ):
        raise ValueError("B selector and feature corpus bind different A1 weights")

    import numpy as np
    import torch
    from safetensors.torch import load_file

    model_config = _model_config(provenance)
    selector = build_selector(model_config).to(args.device)
    selector.load_state_dict(load_file(str(audit.checkpoint_dir / "model.safetensors")))
    selector.eval()

    episodes = _read_jsonl(dataset / "episodes.jsonl")
    samples = _read_jsonl(dataset / "samples.jsonl")
    rows_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        rows_by_episode[int(row["episode_index"])].append(row)
    selected_episodes = sorted(episodes, key=lambda row: int(row["episode_index"]))
    if args.limit_episodes is not None:
        selected_episodes = selected_episodes[: args.limit_episodes]
    feature_dir = destination / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    output_hashes: dict[str, str] = {}
    split_episodes: dict[str, list[int]] = {"train": [], "development": []}
    label_counts = {"completion_negative": 0, "completion_positive": 0}
    sample_count = 0
    started = time.perf_counter()
    input_hashes = feature_manifest["feature_files_sha256"]
    for completed, episode in enumerate(selected_episodes, start=1):
        episode_index = int(episode["episode_index"])
        split = "development" if episode["split"] == "dev" else str(episode["split"])
        if split not in split_episodes:
            raise ValueError(f"unsupported B split for Ours: {split}")
        filename = f"episode_{episode_index:06d}.npz"
        source_path = dataset / "features" / filename
        if input_hashes.get(filename) != sha256_file(source_path):
            raise ValueError(f"frozen B feature hash mismatch: {filename}")
        destination_path = feature_dir / filename
        if args.resume and destination_path.is_file():
            output_hashes[filename] = sha256_file(destination_path)
            with np.load(destination_path) as cached:
                complete = cached["target_complete"].astype(np.bool_, copy=False)
                sample_count += len(complete)
                label_counts["completion_positive"] += int(complete.sum())
                label_counts["completion_negative"] += int((~complete).sum())
            split_episodes[split].append(episode_index)
            continue

        rows = ordered_episode_rows(rows_by_episode[episode_index], episode_index)
        targets = build_demonstration_targets(rows, episode["subgoals"])
        with np.load(source_path) as source:
            source_ids = source["sample_indices"].astype(np.int64, copy=False)
            contexts = source["contexts"].astype(np.float16, copy=False)
            chunks = source["action_chunks"].astype(np.float16, copy=False)
        expected_ids = np.asarray([target.sample_index for target in targets], dtype=np.int64)
        order = np.argsort(source_ids)
        if not np.array_equal(source_ids[order], expected_ids):
            raise ValueError(f"B feature order differs from samples index: {filename}")
        source_ids = source_ids[order]
        contexts = contexts[order]
        chunks = chunks[order]
        scores, valid, candidates, anchor_positions = _score_episode(
            selector=selector,
            model_config=model_config,
            rows=rows,
            contexts=contexts,
            chunks=chunks,
            batch_size=args.batch_size,
            device=args.device,
            np=np,
            torch=torch,
        )
        complete = np.asarray([target.complete for target in targets], dtype=np.bool_)
        arrays = {
            "sample_indices": source_ids,
            "contexts": contexts,
            "action_chunks": chunks,
            "selector_scores": scores,
            "selector_valid": valid,
            "selector_candidates": candidates,
            "anchor_positions": anchor_positions,
            "subgoal_indices": np.asarray(
                [target.subgoal_index for target in targets], dtype=np.int16
            ),
            "frame_indices": np.asarray([target.frame_index for target in targets], dtype=np.int32),
            "elapsed_steps": np.asarray(
                [target.elapsed_steps for target in targets], dtype=np.int32
            ),
            "target_progress": np.asarray(
                [target.progress for target in targets], dtype=np.float16
            ),
            "target_complete": complete,
        }
        _save_episode(destination_path, arrays, np)
        output_hashes[filename] = sha256_file(destination_path)
        split_episodes[split].append(episode_index)
        sample_count += len(targets)
        label_counts["completion_positive"] += int(complete.sum())
        label_counts["completion_negative"] += int((~complete).sum())
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "episodes": completed,
                    "total": len(selected_episodes),
                    "samples": sample_count,
                    "elapsed_seconds": elapsed,
                }
            ),
            flush=True,
        )

    limited = args.limit_episodes is not None
    manifest = {
        "schema_version": OURS_CORPUS_SCHEMA,
        "experiment_id": OURS_ID,
        "variant": OURS_VARIANT,
        "method": OURS_METHOD,
        "parent_experiment": OURS_PARENT,
        "source": "frozen B successful-demonstration contexts and proposals",
        "b_dataset": str(dataset),
        "b_manifest_sha256": sha256_file(b_manifest_path),
        "b_feature_manifest_sha256": sha256_file(b_feature_manifest_path),
        "selector_checkpoint": str(audit.checkpoint_dir),
        "selector_weights_sha256": audit.weights_sha256,
        "a1_checkpoint_weight_shards_sha256": provenance["a1_checkpoint_weight_shards_sha256"],
        "context_width": model_config.context_width,
        "action_horizon": model_config.action_horizon,
        "action_dim": model_config.action_dim,
        "consumed_rollout_base_seeds": [],
        "contains_counterfactuals": False,
        "limited": limited,
        "files": len(output_hashes),
        "samples": sample_count,
        "files_sha256": dict(sorted(output_hashes.items())),
        "split_episodes": split_episodes,
        "label_counts": label_counts,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
