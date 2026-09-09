"""Warm-start the RESOLVE recovery actor on successful expert continuations.

This stage is deliberately not a paper contribution.  It uses the frozen-GR00T
contexts already extracted for B and teaches the recovery actor that a
physically valid correction exists before sparse paired R/D/B RL begins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .a1 import sha256_file
from .ours import OursContractError
from .resolve_model import (
    ResolveModelConfig,
    build_recovery_actor,
    count_trainable_parameters,
    sample_recovery_action,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=71007)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--model-width", type=int, default=512)
    parser.add_argument("--transformer-layers", type=int, default=8)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--feedforward-width", type=int, default=2048)
    parser.add_argument("--history-length", type=int, default=8)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def bounded_model_actions(actions: Any, *, gripper_index: int | None, np: Any) -> Any:
    """Project logged model actions into the simulator's executable box."""

    values = np.asarray(actions, dtype=np.float32).copy()
    if values.ndim < 1 or values.shape[-1] < 1 or not np.isfinite(values).all():
        raise OursContractError("RESOLVE expert actions are invalid")
    values[...] = np.clip(values, -1.0, 1.0)
    if gripper_index is not None:
        if not 0 <= gripper_index < values.shape[-1]:
            raise OursContractError("RESOLVE gripper index is invalid")
        values[..., gripper_index] = np.clip(
            np.asarray(actions, dtype=np.float32)[..., gripper_index], 0.0, 1.0
        )
    return values


def expert_action_chunks(
    actions: Any, frame_indices: Any, *, horizon: int, gripper_index: int | None, np: Any
) -> Any:
    """Gather fixed-horizon expert chunks, padding only past episode end."""

    actions = bounded_model_actions(actions, gripper_index=gripper_index, np=np)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if actions.ndim != 2 or frames.ndim != 1 or horizon < 1 or len(actions) < 1:
        raise OursContractError("RESOLVE expert chunk source is invalid")
    if (frames < 0).any() or (frames >= len(actions)).any():
        raise OursContractError("RESOLVE expert frame index is out of range")
    offsets = np.arange(horizon, dtype=np.int64)[None, :]
    indices = np.minimum(frames[:, None] + offsets, len(actions) - 1)
    return actions[indices]


def causal_history_indices(
    episode_indices: Any,
    subgoal_indices: Any,
    frame_indices: Any,
    *,
    history_length: int,
    np: Any,
) -> Any:
    """Build left-padded histories without crossing episode/subgoal boundaries."""

    episodes = np.asarray(episode_indices, dtype=np.int64)
    subgoals = np.asarray(subgoal_indices, dtype=np.int64)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if (
        episodes.ndim != 1
        or episodes.shape != subgoals.shape
        or episodes.shape != frames.shape
        or history_length < 1
        or len(episodes) < 1
    ):
        raise OursContractError("RESOLVE history metadata are invalid")
    histories = np.empty((len(episodes), history_length), dtype=np.int64)
    queues: dict[tuple[int, int], deque[int]] = defaultdict(lambda: deque(maxlen=history_length))
    last_frame: dict[tuple[int, int], int] = {}
    for index, (episode, subgoal, frame) in enumerate(zip(episodes, subgoals, frames, strict=True)):
        key = (int(episode), int(subgoal))
        if key in last_frame and int(frame) < last_frame[key]:
            raise OursContractError("RESOLVE samples are not causal within a subgoal")
        last_frame[key] = int(frame)
        queue = queues[key]
        queue.append(index)
        available = list(queue)
        histories[index] = [available[0]] * (history_length - len(available)) + available
    return histories


def _dataset_fingerprint(manifest: dict[str, Any]) -> str:
    selected = {
        key: manifest[key]
        for key in (
            "selector_index_manifest_sha256",
            "checkpoint_provenance_sha256",
            "checkpoint_weight_shards_sha256",
            "feature_run_contract_sha256",
            "samples",
            "feature_files_sha256",
        )
    }
    payload = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_expert_warmstart_dataset(
    root: Path,
    *,
    history_length: int,
    action_horizon: int = 16,
    action_dim: int = 7,
    gripper_index: int | None = 6,
    limit_episodes: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load causal histories, B proposals, and expert action continuations."""

    import numpy as np
    import pandas as pd

    root = root.expanduser().resolve()
    manifest_path = root / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment_id") != "B"
        or not manifest.get("complete")
        or manifest.get("limited")
        or int(manifest.get("context_width", -1)) != 2048
        or int(manifest.get("action_horizon", -1)) != action_horizon
    ):
        raise OursContractError("RESOLVE warm-start feature manifest is incompatible")
    for name in ("episodes.jsonl", "samples.jsonl"):
        expected = manifest.get(f"{name.removesuffix('.jsonl')}_sha256")
        if expected is not None and sha256_file(root / name) != expected:
            raise OursContractError(f"RESOLVE warm-start hash mismatch: {name}")

    episodes = _read_jsonl(root / "episodes.jsonl")
    if limit_episodes is not None:
        if limit_episodes < 1:
            raise OursContractError("RESOLVE episode limit must be positive")
        episodes = episodes[:limit_episodes]
    selected = {int(row["episode_index"]): row for row in episodes}
    samples = {
        int(row["sample_index"]): row
        for row in _read_jsonl(root / "samples.jsonl")
        if int(row["episode_index"]) in selected
    }
    if not samples:
        raise OursContractError("RESOLVE warm-start dataset contains no samples")

    contexts: list[Any] = []
    base_chunks: list[Any] = []
    expert_chunks: list[Any] = []
    scalars: list[Any] = []
    episode_ids: list[int] = []
    subgoal_ids: list[int] = []
    frame_ids: list[int] = []
    split_train: list[bool] = []
    handoff: list[float] = []
    sample_ids: list[int] = []
    feature_hashes = manifest["feature_files_sha256"]

    for episode_id, episode in selected.items():
        filename = f"episode_{episode_id:06d}.npz"
        path = root / "features" / filename
        if filename not in feature_hashes or sha256_file(path) != feature_hashes[filename]:
            raise OursContractError(f"RESOLVE warm-start feature hash mismatch: {filename}")
        with np.load(path, allow_pickle=False) as value:
            ids = value["sample_indices"].astype(np.int64)
            episode_contexts = value["contexts"].astype(np.float16)
            episode_base = value["action_chunks"].astype(np.float16)
        rows = [samples[int(index)] for index in ids]
        if (
            episode_contexts.shape != (len(rows), 2048)
            or episode_base.shape != (len(rows), action_horizon, action_dim)
            or any(int(row["episode_index"]) != episode_id for row in rows)
        ):
            raise OursContractError("RESOLVE warm-start episode features are invalid")
        frame = pd.read_parquet(episode["parquet"], columns=["action"])
        episode_actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
        row_frames = np.asarray([int(row["frame_index"]) for row in rows])
        episode_expert = expert_action_chunks(
            episode_actions,
            row_frames,
            horizon=action_horizon,
            gripper_index=gripper_index,
            np=np,
        ).astype(np.float16)
        episode_base = bounded_model_actions(
            episode_base, gripper_index=gripper_index, np=np
        ).astype(np.float16)
        subgoal_count = max(1, len(episode["subgoals"]))
        for row in rows:
            frame_index = int(row["frame_index"])
            completion = int(row["completion_step"])
            trajectory = max(1, int(row["trajectory_steps"]))
            subgoal = int(row["subgoal_index"])
            scalars.append(
                np.asarray(
                    (
                        frame_index / trajectory,
                        max(0, trajectory - frame_index) / trajectory,
                        subgoal / max(1, subgoal_count - 1),
                        subgoal / subgoal_count,
                        max(0, completion - frame_index) / max(1, completion),
                        0.0,
                    ),
                    dtype=np.float16,
                )
            )
            episode_ids.append(episode_id)
            subgoal_ids.append(subgoal)
            frame_ids.append(frame_index)
            split_train.append(str(row["split"]) == "train")
            handoff.append(float(frame_index + action_horizon >= completion))
            sample_ids.append(int(row["sample_index"]))
        contexts.append(episode_contexts)
        base_chunks.append(episode_base)
        expert_chunks.append(episode_expert)

    packed = {
        "contexts": np.concatenate(contexts),
        "base_chunks": np.concatenate(base_chunks),
        "expert_chunks": np.concatenate(expert_chunks),
        "scalars": np.stack(scalars),
        "episode_indices": np.asarray(episode_ids, dtype=np.int32),
        "subgoal_indices": np.asarray(subgoal_ids, dtype=np.int16),
        "frame_indices": np.asarray(frame_ids, dtype=np.int32),
        "train": np.asarray(split_train, dtype=np.bool_),
        "handoff": np.asarray(handoff, dtype=np.float32),
        "sample_indices": np.asarray(sample_ids, dtype=np.int64),
    }
    packed["history_indices"] = causal_history_indices(
        packed["episode_indices"],
        packed["subgoal_indices"],
        packed["frame_indices"],
        history_length=history_length,
        np=np,
    )
    expected_samples = sum(len(value) for value in contexts)
    if any(len(value) != expected_samples for value in packed.values()):
        raise OursContractError("RESOLVE warm-start packed arrays are misaligned")
    metadata = {
        "feature_manifest": str(manifest_path),
        "feature_manifest_sha256": sha256_file(manifest_path),
        "dataset_fingerprint": _dataset_fingerprint(manifest),
        "episodes": len(episodes),
        "samples": expected_samples,
        "train_samples": int(packed["train"].sum()),
        "development_samples": int((~packed["train"]).sum()),
        "limited": limit_episodes is not None,
    }
    return packed, metadata


def _batch(
    data: dict[str, Any], indices: Any, config: ResolveModelConfig, device: Any, torch: Any
) -> tuple[tuple[Any, ...], Any, Any, Any]:
    history = data["history_indices"][indices]

    def tensor(value: Any) -> Any:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    contexts = tensor(data["contexts"][history])
    base = tensor(data["base_chunks"][history])
    scalars = tensor(data["scalars"][history])
    batch_size = len(indices)
    latent = torch.full(
        (batch_size, config.history_length),
        config.latent_codes,
        dtype=torch.long,
        device=device,
    )
    positions = torch.arange(config.history_length, device=device)[None].expand(batch_size, -1)
    milestones = torch.ones(batch_size, dtype=torch.long, device=device)
    expert = tensor(data["expert_chunks"][indices])
    handoff = tensor(data["handoff"][indices])
    current_base = base[:, -1]
    return (contexts, base, scalars, latent, positions, milestones), current_base, expert, handoff


def _loss(
    actor: Any,
    batch: tuple[tuple[Any, ...], Any, Any, Any],
    *,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    inputs, base, expert, handoff_target = batch
    outputs = actor(*inputs)
    sampled = sample_recovery_action(outputs, base, deterministic=True)
    corrected = sampled["corrected_action"]
    action_loss = torch.nn.functional.smooth_l1_loss(corrected, expert)

    # A direct bounded-action loss has almost no useful gradient when expert
    # demonstrations use saturated Cartesian commands.  Supervise the exact
    # logit displacement executed by the actor, while retaining an action-space
    # term for calibration near the interior of the box.
    epsilon = 1e-4
    base_unit = base.clamp(-1.0 + epsilon, 1.0 - epsilon).clone()
    expert_unit = expert.clamp(-1.0 + epsilon, 1.0 - epsilon).clone()
    gripper_index = outputs.get("gripper_index")
    if gripper_index is not None:
        base_unit[..., gripper_index] = (2.0 * base[..., gripper_index] - 1.0).clamp(
            -1.0 + epsilon, 1.0 - epsilon
        )
        expert_unit[..., gripper_index] = (2.0 * expert[..., gripper_index] - 1.0).clamp(
            -1.0 + epsilon, 1.0 - epsilon
        )
    target_logit_shift = torch.atanh(expert_unit) - torch.atanh(base_unit)
    predicted_logit_shift = outputs["residual_trust"] * outputs["residual_mean"]
    logit_loss = torch.nn.functional.smooth_l1_loss(predicted_logit_shift, target_logit_shift)
    continuation_logit = torch.logsumexp(outputs["latent_logits"][:, :-1], dim=1)
    handoff_logit = outputs["latent_logits"][:, -1] - continuation_logit
    handoff_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        handoff_logit, handoff_target
    )
    deviation = sampled["residual_action"].square().mean()
    total = logit_loss + 0.25 * action_loss + 0.1 * handoff_loss + 1e-4 * deviation
    return total, {
        "loss": total,
        "action_loss": action_loss,
        "logit_loss": logit_loss,
        "handoff_loss": handoff_loss,
        "deviation": deviation,
        "corrected": corrected,
        "base": base,
        "expert": expert,
        "handoff_logit": handoff_logit,
        "handoff_target": handoff_target,
    }


def _evaluate(
    actor: Any,
    data: dict[str, Any],
    indices: Any,
    *,
    batch_size: int,
    config: ResolveModelConfig,
    device: Any,
    torch: Any,
) -> dict[str, float]:
    actor.eval()
    action_absolute = 0.0
    action_squared = 0.0
    base_absolute = 0.0
    base_squared = 0.0
    sign_correct = 0.0
    base_sign_correct = 0.0
    handoff_correct = 0.0
    elements = 0
    samples = 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            batch = _batch(data, selected, config, device, torch)
            _, metrics = _loss(actor, batch, torch=torch)
            corrected = metrics["corrected"]
            base = metrics["base"]
            expert = metrics["expert"]
            delta = corrected - expert
            base_delta = base - expert
            action_absolute += float(delta.abs().sum())
            action_squared += float(delta.square().sum())
            base_absolute += float(base_delta.abs().sum())
            base_squared += float(base_delta.square().sum())
            sign_correct += float(
                (torch.sign(corrected[..., :6]) == torch.sign(expert[..., :6])).sum()
            )
            base_sign_correct += float(
                (torch.sign(base[..., :6]) == torch.sign(expert[..., :6])).sum()
            )
            handoff_correct += float(
                ((metrics["handoff_logit"] >= 0) == (metrics["handoff_target"] >= 0.5)).sum()
            )
            elements += delta.numel()
            samples += len(selected)
    if samples < 1 or elements < 1:
        raise OursContractError("RESOLVE warm-start evaluation split is empty")
    return {
        "samples": float(samples),
        "actor_mae": action_absolute / elements,
        "actor_rmse": math.sqrt(action_squared / elements),
        "base_mae": base_absolute / elements,
        "base_rmse": math.sqrt(base_squared / elements),
        "actor_to_base_mae_ratio": action_absolute / max(base_absolute, 1e-12),
        "direction_agreement": sign_correct / (samples * config.action_horizon * 6),
        "base_direction_agreement": base_sign_correct / (samples * config.action_horizon * 6),
        "handoff_accuracy": handoff_correct / samples,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        min(args.steps, args.batch_size, args.eval_interval, args.eval_batch_size) < 1
        or args.learning_rate <= 0.0
        or args.weight_decay < 0.0
    ):
        raise OursContractError("RESOLVE warm-start optimization settings are invalid")
    import numpy as np
    import torch

    config = ResolveModelConfig(
        history_length=args.history_length,
        model_width=args.model_width,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        feedforward_width=args.feedforward_width,
        maximum_residual_scale=4.0,
        gripper_index=6,
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    data, source = load_expert_warmstart_dataset(
        args.features,
        history_length=config.history_length,
        action_horizon=config.action_horizon,
        action_dim=config.action_dim,
        gripper_index=config.gripper_index,
        limit_episodes=args.limit_episodes,
    )
    train = np.flatnonzero(data["train"])
    development = np.flatnonzero(~data["train"])
    if len(train) < args.batch_size or len(development) < 1:
        raise OursContractError("RESOLVE warm-start needs nonempty train/development splits")
    actor = build_recovery_actor(config).to(device)
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = np.random.default_rng(args.seed)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "experiment_id": "Ours",
        "stage": "RESOLVE-expert-recovery-actor-warm-start",
        "novelty_claim": False,
        "purpose": "initialize a reachable recovery action class before paired sparse RL",
        "source": source,
        "model": asdict(config),
        "trainable_parameters": count_trainable_parameters(actor),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
    }
    _write_json(output / "run_manifest.json", manifest)
    log_path = output / "training.jsonl"
    baseline = _evaluate(
        actor,
        data,
        development,
        batch_size=args.eval_batch_size,
        config=config,
        device=device,
        torch=torch,
    )
    best_mae = baseline["actor_mae"]
    best_step = 0
    torch.save(
        {
            "schema_version": 1,
            "stage": manifest["stage"],
            "model_config": asdict(config),
            "model_state_dict": actor.state_dict(),
            "step": 0,
            "development": baseline,
            "dataset_fingerprint": source["dataset_fingerprint"],
        },
        output / "actor.pt",
    )
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            actor.train()
            selected = rng.choice(train, size=args.batch_size, replace=False)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _loss(
                actor,
                _batch(data, selected, config, device, torch),
                torch=torch,
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            if step % args.eval_interval == 0 or step == args.steps:
                evaluation = _evaluate(
                    actor,
                    data,
                    development,
                    batch_size=args.eval_batch_size,
                    config=config,
                    device=device,
                    torch=torch,
                )
                record = {
                    "step": step,
                    "train_loss": float(metrics["loss"].detach()),
                    "train_action_loss": float(metrics["action_loss"].detach()),
                    "train_logit_loss": float(metrics["logit_loss"].detach()),
                    "train_handoff_loss": float(metrics["handoff_loss"].detach()),
                    "gradient_norm": float(gradient_norm),
                    **evaluation,
                }
                log.write(json.dumps(record, separators=(",", ":")) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                if evaluation["actor_mae"] < best_mae:
                    best_mae = evaluation["actor_mae"]
                    best_step = step
                    torch.save(
                        {
                            "schema_version": 1,
                            "stage": manifest["stage"],
                            "model_config": asdict(config),
                            "model_state_dict": actor.state_dict(),
                            "step": step,
                            "development": evaluation,
                            "dataset_fingerprint": source["dataset_fingerprint"],
                        },
                        output / "actor.pt",
                    )
    checkpoint = torch.load(output / "actor.pt", map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint["model_state_dict"])
    final = _evaluate(
        actor,
        data,
        development,
        batch_size=args.eval_batch_size,
        config=config,
        device=device,
        torch=torch,
    )
    summary = {
        "complete": True,
        "best_step": best_step,
        "warmstart_improved": best_step > 0,
        "untrained_development": baseline,
        "selected_development": final,
        "checkpoint": str(output / "actor.pt"),
        "checkpoint_sha256": sha256_file(output / "actor.pt"),
        "training_log": str(log_path),
    }
    _write_json(output / "summary.json", summary)
    return summary


def main() -> int:
    try:
        result = run(_parser().parse_args())
    except (FileNotFoundError, OursContractError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
