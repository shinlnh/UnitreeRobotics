"""Train and freeze B's unified STOP/action-prefix selector."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .a1 import sha256_file
from .b import B_CHECKPOINT_PROVENANCE, B_ID, B_METHOD, B_PAPER, B_VARIANT, build_ordinal_targets
from .b_model import SelectorModelConfig, build_selector, selector_loss


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--context-width", type=int, default=2048)
    parser.add_argument("--scoring-width", type=int, default=1024)
    parser.add_argument("--scoring-layers", type=int, default=2)
    parser.add_argument("--scoring-heads", type=int, default=8)
    parser.add_argument("--feedforward-width", type=int, default=4096)
    parser.add_argument("--boundary-jitter-steps", type=int, default=2)
    parser.add_argument("--near-boundary-steps", type=int, default=80)
    parser.add_argument("--stop-loss-weight", type=float, default=1.0)
    parser.add_argument("--stop-positive-weight", type=float, default=3.0)
    parser.add_argument("--near-boundary-stop-weight", type=float, default=3.0)
    parser.add_argument("--unsuccessful-rank-weight", type=float, default=0.1)
    parser.add_argument("--unsuccessful-stop-weight", type=float, default=1.5)
    parser.add_argument("--stop-confirmation-window", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--validate-steps", type=int, default=1_000)
    parser.add_argument("--validation-samples", type=int, default=4_096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def anchor_sample_map(samples: list[dict[str, Any]]) -> dict[tuple[int, int], int]:
    """Map every demonstrated subgoal to its earliest cached context."""

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    for sample in samples:
        key = (int(sample["episode_index"]), int(sample["subgoal_index"]))
        candidate = (int(sample["frame_index"]), int(sample["sample_index"]))
        if key not in anchors or candidate < anchors[key]:
            anchors[key] = candidate
    return {key: value[1] for key, value in anchors.items()}


def _load_features(
    dataset: Path,
    samples: list[dict[str, Any]],
    expected_hashes: dict[str, str],
    np: Any,
) -> tuple[Any, Any]:
    sample_count = len(samples)
    contexts: Any | None = None
    chunks: Any | None = None
    seen = np.zeros(sample_count, dtype=np.bool_)
    for path in sorted((dataset / "features").glob("episode_*.npz")):
        if expected_hashes.get(path.name) != sha256_file(path):
            raise ValueError(f"feature cache hash mismatch: {path}")
        with np.load(path) as value:
            ids = value["sample_indices"].astype(np.int64, copy=False)
            episode_contexts = value["contexts"]
            episode_chunks = value["action_chunks"]
            if not (len(ids) == len(episode_contexts) == len(episode_chunks)):
                raise ValueError(f"feature row mismatch: {path}")
            if contexts is None:
                contexts = np.empty((sample_count, episode_contexts.shape[1]), dtype=np.float16)
                chunks = np.empty((sample_count, *episode_chunks.shape[1:]), dtype=np.float16)
            if np.any(ids < 0) or np.any(ids >= sample_count) or np.any(seen[ids]):
                raise ValueError(f"invalid or duplicate feature sample IDs: {path}")
            contexts[ids] = episode_contexts
            chunks[ids] = episode_chunks
            seen[ids] = True
    if contexts is None or chunks is None or not seen.all():
        missing = int((~seen).sum())
        raise ValueError(f"feature cache is incomplete ({missing} missing samples)")
    return contexts, chunks


def _batch(
    indices: list[int],
    samples: list[dict[str, Any]],
    contexts: Any,
    chunks: Any,
    anchors: dict[tuple[int, int], int],
    *,
    model_config: SelectorModelConfig,
    jitter_steps: int,
    near_boundary_steps: int,
    stop_positive_weight: float,
    near_boundary_stop_weight: float,
    unsuccessful_rank_weight: float,
    unsuccessful_stop_weight: float,
    rng: random.Random,
    np: Any,
) -> dict[str, Any]:
    max_anchors = max(int(samples[index]["subgoal_index"]) + 1 for index in indices)
    anchor_history = np.zeros(
        (len(indices), max_anchors, model_config.context_width), dtype=np.float32
    )
    anchor_valid = np.zeros((len(indices), max_anchors), dtype=np.bool_)
    priorities = []
    valid = []
    stop_labels = []
    rank_weights = []
    stop_weights = []
    jitters = []
    for row_index, index in enumerate(indices):
        sample = samples[index]
        episode_index = int(sample["episode_index"])
        subgoal_index = int(sample["subgoal_index"])
        for anchor_index in range(subgoal_index + 1):
            anchor_id = anchors[(episode_index, anchor_index)]
            anchor_history[row_index, anchor_index] = contexts[anchor_id]
            anchor_valid[row_index, anchor_index] = True
        completion = int(sample["completion_step"])
        trajectory_steps = int(sample["trajectory_steps"])
        low = max(-jitter_steps, -completion)
        high = min(jitter_steps, trajectory_steps - completion)
        jitter = rng.randint(low, high)
        targets = build_ordinal_targets(
            decision_step=int(sample["frame_index"]),
            trajectory_steps=trajectory_steps,
            horizon=model_config.action_horizon,
            successful=True,
            completion_step=completion,
            boundary_jitter=jitter,
            near_boundary_steps=near_boundary_steps,
            unsuccessful_rank_weight=unsuccessful_rank_weight,
            stop_positive_weight=stop_positive_weight,
            near_boundary_stop_weight=near_boundary_stop_weight,
            unsuccessful_stop_weight=unsuccessful_stop_weight,
        )
        priorities.append(targets.priorities)
        valid.append(targets.valid)
        stop_labels.append(targets.stop_label)
        rank_weights.append(targets.rank_weight)
        stop_weights.append(targets.stop_weight)
        jitters.append(jitter)
    return {
        "action_chunks": chunks[indices].astype(np.float32),
        "contexts": contexts[indices].astype(np.float32),
        "anchor_history": anchor_history,
        "anchor_valid": anchor_valid,
        "priorities": np.asarray(priorities, dtype=np.int64),
        "valid": np.asarray(valid, dtype=np.bool_),
        "stop_labels": np.asarray(stop_labels, dtype=np.float32),
        "rank_weights": np.asarray(rank_weights, dtype=np.float32),
        "stop_weights": np.asarray(stop_weights, dtype=np.float32),
        "jitters": jitters,
    }


def _to_device(batch: dict[str, Any], device: str, torch: Any) -> dict[str, Any]:
    return {
        key: torch.as_tensor(value, device=device)
        for key, value in batch.items()
        if key != "jitters"
    }


def _validate(
    model: Any,
    indices: list[int],
    make_batch: Any,
    *,
    batch_size: int,
    stop_loss_weight: float,
    device: str,
    torch: Any,
) -> dict[str, float]:
    totals = defaultdict(float)
    count = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            chosen = indices[start : start + batch_size]
            batch = _to_device(make_batch(chosen), device, torch)
            scores = model(
                batch["action_chunks"],
                batch["contexts"],
                batch["anchor_history"],
                batch["anchor_valid"],
                batch["valid"],
            )
            loss, metrics = selector_loss(
                scores,
                batch["priorities"],
                batch["valid"],
                batch["stop_labels"],
                batch["rank_weights"],
                batch["stop_weights"],
                stop_loss_weight=stop_loss_weight,
            )
            size = len(chosen)
            totals["loss"] += float(loss) * size
            totals["rank_loss"] += float(metrics["rank_loss"]) * size
            totals["stop_loss"] += float(metrics["stop_loss"]) * size
            predicted = scores.argmax(dim=1)
            target = batch["priorities"].argmax(dim=1)
            totals["candidate_accuracy"] += float((predicted == target).sum())
            totals["stop_accuracy"] += float(((predicted == 0) == (target == 0)).sum())
            count += size
    return {key: value / count for key, value in totals.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.steps, args.batch_size, args.validate_steps, args.validation_samples) < 1:
        raise ValueError("B training counts must be positive")
    dataset = args.dataset.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    resume = bool(getattr(args, "resume", False))
    final_weights_path = destination / "model.safetensors"
    final_provenance_path = destination / B_CHECKPOINT_PROVENANCE
    final_parts = (final_weights_path.is_file(), final_provenance_path.is_file())
    if all(final_parts):
        raise FileExistsError(f"selector checkpoint is already complete: {destination}")
    if final_parts[1] or (final_parts[0] and not resume):
        raise ValueError(f"selector checkpoint has an incomplete final export: {destination}")
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise FileExistsError(f"refusing to overwrite selector checkpoint: {destination}")
    manifest_path = dataset / "manifest.json"
    feature_manifest_path = dataset / "feature_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if not feature_manifest.get("complete") or feature_manifest.get("limited"):
        raise ValueError("B feature extraction must be complete and unlimited")
    if int(feature_manifest["samples"]) != int(manifest["samples"]):
        raise ValueError("B feature/sample counts do not match")
    parent_weight_hashes = feature_manifest.get("checkpoint_weight_shards_sha256")
    if not isinstance(parent_weight_hashes, dict) or not parent_weight_hashes:
        raise ValueError("B feature manifest is missing frozen A1 weight hashes")

    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file

    model_config = SelectorModelConfig(
        action_horizon=args.action_horizon,
        context_width=args.context_width,
        scoring_width=args.scoring_width,
        scoring_layers=args.scoring_layers,
        scoring_heads=args.scoring_heads,
        feedforward_width=args.feedforward_width,
    )
    if args.action_horizon != int(manifest["action_horizon"]):
        raise ValueError("selector action horizon does not match index")
    if args.context_width != int(feature_manifest["context_width"]):
        raise ValueError("selector context width does not match feature cache")
    samples = _read_jsonl(dataset / "samples.jsonl")
    if [int(sample["sample_index"]) for sample in samples] != list(range(len(samples))):
        raise ValueError("selector sample IDs must be contiguous and ordered")
    feature_hashes = feature_manifest.get("feature_files_sha256")
    if not isinstance(feature_hashes, dict) or len(feature_hashes) != int(
        feature_manifest["feature_files"]
    ):
        raise ValueError("B feature manifest is missing per-file hashes")
    contexts, chunks = _load_features(dataset, samples, feature_hashes, np)
    anchors = anchor_sample_map(samples)
    train_indices = [index for index, row in enumerate(samples) if row["split"] == "train"]
    development_indices = [
        index for index, row in enumerate(samples) if row["split"] == "development"
    ]
    if not train_indices or not development_indices:
        raise ValueError("selector requires non-empty train and development splits")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_rng = random.Random(args.seed)
    development_rng = random.Random(args.seed + 1)

    def make_batch(indices: list[int], *, rng: random.Random = train_rng) -> dict[str, Any]:
        return _batch(
            indices,
            samples,
            contexts,
            chunks,
            anchors,
            model_config=model_config,
            jitter_steps=args.boundary_jitter_steps,
            near_boundary_steps=args.near_boundary_steps,
            stop_positive_weight=args.stop_positive_weight,
            near_boundary_stop_weight=args.near_boundary_stop_weight,
            unsuccessful_rank_weight=args.unsuccessful_rank_weight,
            unsuccessful_stop_weight=args.unsuccessful_stop_weight,
            rng=rng,
            np=np,
        )

    model = build_selector(model_config).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    destination.mkdir(parents=True, exist_ok=True)
    resume_slots = tuple(
        (
            destination / f"resume_model_{slot}.safetensors",
            destination / f"training_state_{slot}.pt",
        )
        for slot in range(2)
    )
    resume_contract = {
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "a1_checkpoint_weight_shards_sha256": parent_weight_hashes,
        "model_config": model_config.payload(),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "gradient_clip_norm": args.gradient_clip_norm,
        "steps": args.steps,
        "validate_steps": args.validate_steps,
        "validation_samples": args.validation_samples,
        "boundary_jitter_steps": args.boundary_jitter_steps,
        "near_boundary_steps": args.near_boundary_steps,
        "stop_loss_weight": args.stop_loss_weight,
        "stop_positive_weight": args.stop_positive_weight,
        "near_boundary_stop_weight": args.near_boundary_stop_weight,
        "unsuccessful_rank_weight": args.unsuccessful_rank_weight,
        "unsuccessful_stop_weight": args.unsuccessful_stop_weight,
    }
    history: list[dict[str, Any]] = []
    start_step = 0
    if resume:
        resume_artifacts = [path for pair in resume_slots for path in pair if path.exists()]
        allowed_artifacts = set(resume_artifacts)
        if final_weights_path.exists():
            allowed_artifacts.add(final_weights_path)
        temporary_patterns = (
            "resume_model_[01].tmp-*.safetensors",
            "training_state_[01].tmp-*.pt",
            "model.tmp-*.safetensors",
        )
        temporary_artifacts = {
            path
            for pattern in temporary_patterns
            for path in destination.glob(pattern)
            if path.is_file()
        }
        allowed_artifacts.update(temporary_artifacts)
        unrelated = [path for path in destination.iterdir() if path not in allowed_artifacts]
        if unrelated:
            raise ValueError("B checkpoint directory is non-empty but has no resumable state")
        valid_slots: list[tuple[int, Path, dict[str, Any]]] = []
        for resume_model_path, resume_state_path in resume_slots:
            if not (resume_model_path.is_file() and resume_state_path.is_file()):
                continue
            try:
                state = torch.load(resume_state_path, map_location="cpu", weights_only=False)
                required_state = {
                    "step",
                    "optimizer",
                    "train_rng_state",
                    "torch_rng_state",
                    "cuda_rng_state_all",
                    "history",
                    "contract",
                    "resume_weights_sha256",
                }
                if not required_state.issubset(state):
                    continue
                if state["contract"] != resume_contract:
                    continue
                if state["resume_weights_sha256"] != sha256_file(resume_model_path):
                    continue
                valid_slots.append((int(state["step"]), resume_model_path, state))
            except (EOFError, KeyError, OSError, RuntimeError, ValueError, pickle.UnpicklingError):
                continue
        if resume_artifacts and not valid_slots:
            raise ValueError("B resume checkpoint has no complete hash-matched slot")
        if valid_slots:
            start_step, resume_model_path, state = max(valid_slots, key=lambda value: value[0])
            start_step = int(state["step"])
            if start_step >= args.steps:
                raise ValueError("B resume step must be smaller than requested training steps")
            model.load_state_dict(load_file(str(resume_model_path)))
            optimizer.load_state_dict(state["optimizer"])
            train_rng.setstate(state["train_rng_state"])
            torch.set_rng_state(state["torch_rng_state"])
            if torch.cuda.is_available():
                cuda_rng_state_all = state["cuda_rng_state_all"]
                if len(cuda_rng_state_all) != torch.cuda.device_count():
                    raise ValueError("B resume checkpoint CUDA device count does not match")
                torch.cuda.set_rng_state_all(cuda_rng_state_all)
            history = list(state["history"])
        for path in temporary_artifacts:
            path.unlink(missing_ok=True)

    def learning_rate(step: int) -> float:
        if step <= args.warmup_steps:
            return args.learning_rate * step / max(1, args.warmup_steps)
        remaining = (args.steps - step) / max(1, args.steps - args.warmup_steps)
        return args.learning_rate * max(0.0, remaining)

    fixed_development = development_indices[:]
    development_rng.shuffle(fixed_development)
    fixed_development = fixed_development[: args.validation_samples]
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        selected = [train_rng.choice(train_indices) for _ in range(args.batch_size)]
        batch = _to_device(make_batch(selected), args.device, torch)
        optimizer.zero_grad(set_to_none=True)
        scores = model(
            batch["action_chunks"],
            batch["contexts"],
            batch["anchor_history"],
            batch["anchor_valid"],
            batch["valid"],
        )
        loss, _ = selector_loss(
            scores,
            batch["priorities"],
            batch["valid"],
            batch["stop_labels"],
            batch["rank_weights"],
            batch["stop_weights"],
            stop_loss_weight=args.stop_loss_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
        rate = learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = rate
        optimizer.step()
        if step == 1 or step % args.validate_steps == 0 or step == args.steps:
            validation_rng = random.Random(args.seed + 2)
            metrics = _validate(
                model,
                fixed_development,
                lambda indices, rng=validation_rng: make_batch(indices, rng=rng),
                batch_size=args.batch_size,
                stop_loss_weight=args.stop_loss_weight,
                device=args.device,
                torch=torch,
            )
            record = {"step": step, "learning_rate": rate, **metrics}
            history.append(record)
            print(json.dumps(record), flush=True)
            resume_weights = {
                key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()
            }
            slot = (len(history) - 1) % len(resume_slots)
            resume_model_path, resume_state_path = resume_slots[slot]
            temporary_model = resume_model_path.with_suffix(f".tmp-{os.getpid()}.safetensors")
            save_file(resume_weights, temporary_model)
            temporary_model.replace(resume_model_path)
            temporary_state = resume_state_path.with_suffix(f".tmp-{os.getpid()}.pt")
            torch.save(
                {
                    "step": step,
                    "optimizer": optimizer.state_dict(),
                    "train_rng_state": train_rng.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                    ),
                    "history": history,
                    "contract": resume_contract,
                    "resume_weights_sha256": sha256_file(resume_model_path),
                },
                temporary_state,
            )
            temporary_state.replace(resume_state_path)

    weights_path = final_weights_path
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    temporary_weights = weights_path.with_suffix(f".tmp-{os.getpid()}.safetensors")
    save_file(state, temporary_weights)
    temporary_weights.replace(weights_path)
    provenance = {
        "schema_version": 1,
        "experiment_id": B_ID,
        "variant": B_VARIANT,
        "method": B_METHOD,
        "paper": B_PAPER,
        **model_config.payload(),
        "weights_sha256": sha256_file(weights_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "a1_checkpoint_weight_shards_sha256": parent_weight_hashes,
        "samples_sha256": manifest["samples_sha256"],
        "training": {
            "successful_demonstrations_only": True,
            "policy_rollouts_included": False,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "gradient_clip_norm": args.gradient_clip_norm,
            "boundary_jitter_steps": args.boundary_jitter_steps,
            "boundary_jitter_sampling": "uniform over [-k,k], clipped to valid trajectory boundary",
            "near_boundary_steps": args.near_boundary_steps,
            "stop_loss_weight": args.stop_loss_weight,
            "stop_positive_weight": args.stop_positive_weight,
            "near_boundary_stop_weight": args.near_boundary_stop_weight,
            "unsuccessful_rank_weight": args.unsuccessful_rank_weight,
            "unsuccessful_stop_weight": args.unsuccessful_stop_weight,
            "seed": args.seed,
        },
        "selection": {
            "stop_confirmation_window": args.stop_confirmation_window,
            "confirmation_window_source": "frozen smallest non-trivial window before held-out evaluation",
        },
        "development": {
            "samples": len(fixed_development),
            "metrics": history,
            "final": history[-1],
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    _write_json(final_provenance_path, provenance)
    for resume_model_path, resume_state_path in resume_slots:
        resume_model_path.unlink(missing_ok=True)
        resume_state_path.unlink(missing_ok=True)
    for pattern in ("resume_model_[01].tmp-*.safetensors", "training_state_[01].tmp-*.pt"):
        for path in destination.glob(pattern):
            path.unlink(missing_ok=True)
    return provenance


def main() -> int:
    summary = run(_parser().parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
