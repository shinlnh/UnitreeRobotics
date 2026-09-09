"""Warm-start RESOLVE only from physically decisive expert/B frontier pairs.

This is deliberately an initialization stage rather than the claimed RL
algorithm.  A recovery target is admitted only when executable expert actions
reach the registered predicate and frozen B does not.  Exact B is the target
when B reaches it.  Pairs where neither arm reaches are unresolved and cannot
silently become negative recovery labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
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
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=84007)
    parser.add_argument("--development-fraction", type=float, default=0.2)
    parser.add_argument("--model-width", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--feedforward-width", type=int, default=512)
    parser.add_argument("--history-length", type=int, default=4)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _development_groups(rows: list[dict[str, Any]], *, fraction: float, seed: int) -> set[int]:
    """Select whole demonstrations, stratified by whether they contain rescue."""

    groups: dict[int, bool] = {}
    for row in rows:
        group = int(row["source_manifest_index"])
        groups[group] = groups.get(group, False) or bool(row["positive"])
    selected: set[int] = set()
    for positive in (False, True):
        members = [group for group, label in groups.items() if label == positive]
        members.sort(
            key=lambda group: hashlib.sha256(
                f"resolve-frontier-split-v1:{seed}:{group}".encode()
            ).digest()
        )
        if len(members) < 2:
            continue
        count = min(len(members) - 1, max(1, round(fraction * len(members))))
        selected.update(members[:count])
    return selected


def _executable_chunks(values: Any, *, np: Any) -> Any:
    """Apply the exact action box used by the LIBERO policy adapter."""

    chunks = np.asarray(values, dtype=np.float32).copy()
    if chunks.shape != (16, 7) or not np.isfinite(chunks).all():
        raise OursContractError("RESOLVE frontier action chunk is invalid")
    chunks[..., :6] = np.clip(chunks[..., :6], -1.0, 1.0)
    chunks[..., 6] = np.clip(chunks[..., 6], 0.0, 1.0)
    return chunks


def load_decisive_frontier_pairs(
    root: Path, *, development_fraction: float, seed: int, np: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load hash-verified pairs without turning unresolved failures into labels."""

    root = root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment_id") != "Ours"
        or manifest.get("stage") != "RESOLVE train-only expert/B frontier arms"
        or not 0.0 < development_fraction < 0.5
    ):
        raise OursContractError("RESOLVE frontier-pair manifest is incompatible")
    records_path = root / "anchors.jsonl"
    if sha256_file(records_path) != manifest.get("anchors_sha256"):
        raise OursContractError("RESOLVE frontier-pair record hash mismatch")
    hashes = manifest.get("files_sha256")
    if not isinstance(hashes, dict):
        raise OursContractError("RESOLVE frontier-pair feature registry is invalid")

    decisive: list[dict[str, Any]] = []
    unresolved = 0
    for row in _read_jsonl(records_path):
        expert = bool(row["expert_reached"])
        baseline = bool(row["baseline"]["reached"])
        if not expert and not baseline:
            unresolved += 1
            continue
        decisive.append(row | {"positive": expert and not baseline})
    if not decisive or not any(row["positive"] for row in decisive):
        raise OursContractError("RESOLVE frontier pairs contain no decisive rescue")
    development_groups = _development_groups(decisive, fraction=development_fraction, seed=seed)

    contexts: list[Any] = []
    base_chunks: list[Any] = []
    target_chunks: list[Any] = []
    scalars: list[Any] = []
    positive: list[bool] = []
    development: list[bool] = []
    source_groups: list[int] = []
    for row in decisive:
        filename = str(row["feature_file"])
        path = root / "features" / filename
        if filename not in hashes or sha256_file(path) != hashes[filename]:
            raise OursContractError(f"RESOLVE frontier-pair hash mismatch: {filename}")
        with np.load(path, allow_pickle=False) as value:
            context = value["context"].astype(np.float32)
            base = _executable_chunks(value["base_chunk"], np=np)
            expert = _executable_chunks(value["expert_chunk"], np=np)
        if context.shape != (2048,):
            raise OursContractError("RESOLVE frontier-pair feature shapes are invalid")
        is_positive = bool(row["positive"])
        frontier = max(1, int(row["first_true_frame"]))
        context_candidate = int(row["baseline"]["initial_candidate"])
        contexts.append(context)
        base_chunks.append(base)
        target_chunks.append(expert if is_positive else base)
        scalars.append(
            np.asarray(
                (
                    int(row["anchor_frame"]) / frontier,
                    min(1.0, int(row["paired_physical_steps"]) / 64.0),
                    min(1.0, int(row["subgoal_index"]) / 15.0),
                    0.0,
                    0.0,
                    context_candidate / 16.0,
                ),
                dtype=np.float32,
            )
        )
        group = int(row["source_manifest_index"])
        positive.append(is_positive)
        development.append(group in development_groups)
        source_groups.append(group)

    packed = {
        "contexts": np.stack(contexts),
        "base_chunks": np.stack(base_chunks),
        "target_chunks": np.stack(target_chunks),
        "scalars": np.stack(scalars),
        "positive": np.asarray(positive, dtype=np.bool_),
        "development": np.asarray(development, dtype=np.bool_),
        "source_groups": np.asarray(source_groups, dtype=np.int32),
    }
    train = ~packed["development"]
    validation = packed["development"]
    for split in (train, validation):
        if not split.any() or not (packed["positive"] & split).any():
            raise OursContractError("RESOLVE frontier split lacks a positive rescue")
    metadata = {
        "pairs_manifest": str(manifest_path),
        "pairs_manifest_sha256": sha256_file(manifest_path),
        "pairs": int(manifest["anchors"]),
        "decisive": len(decisive),
        "unresolved_excluded": unresolved,
        "positive": int(packed["positive"].sum()),
        "baseline_safe": int((~packed["positive"]).sum()),
        "train": int(train.sum()),
        "development": int(validation.sum()),
        "train_groups": len(set(packed["source_groups"][train].tolist())),
        "development_groups": len(set(packed["source_groups"][validation].tolist())),
        "group_disjoint": not bool(
            set(packed["source_groups"][train].tolist())
            & set(packed["source_groups"][validation].tolist())
        ),
    }
    return packed, metadata


def _inputs(
    data: dict[str, Any], indices: Any, config: ResolveModelConfig, torch: Any, device: Any
):
    def tensor(value: Any) -> Any:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    batch = len(indices)
    context = tensor(data["contexts"][indices])[:, None].expand(-1, config.history_length, -1)
    base_current = tensor(data["base_chunks"][indices])
    base = base_current[:, None].expand(-1, config.history_length, -1, -1)
    scalar_current = tensor(data["scalars"][indices])
    scalars = scalar_current[:, None].expand(-1, config.history_length, -1)
    latent = torch.full(
        (batch, config.history_length),
        config.latent_codes,
        dtype=torch.long,
        device=device,
    )
    positions = torch.arange(config.history_length, device=device)[None].expand(batch, -1)
    milestones = torch.ones(batch, dtype=torch.long, device=device)
    target = tensor(data["target_chunks"][indices])
    positive = torch.as_tensor(data["positive"][indices], dtype=torch.bool, device=device)
    return (context, base, scalars, latent, positions, milestones), base_current, target, positive


def _loss(actor: Any, batch: Any, *, torch: Any) -> tuple[Any, dict[str, Any]]:
    inputs, base, target, positive = batch
    outputs = actor(*inputs)
    sample = sample_recovery_action(outputs, base, deterministic=True)
    corrected = sample["corrected_action"]
    positive_weight = positive.to(corrected.dtype)[:, None, None]
    positive_count = positive_weight.sum().clamp_min(1.0)
    action = (
        torch.nn.functional.smooth_l1_loss(corrected, target, reduction="none") * positive_weight
    ).sum() / (positive_count * corrected.shape[1] * corrected.shape[2])

    epsilon = 1e-4
    base_unit = base.clamp(-1.0 + epsilon, 1.0 - epsilon).clone()
    target_unit = target.clamp(-1.0 + epsilon, 1.0 - epsilon).clone()
    gripper = outputs.get("gripper_index")
    if gripper is not None:
        base_unit[..., gripper] = (2.0 * base[..., gripper] - 1.0).clamp(
            -1.0 + epsilon, 1.0 - epsilon
        )
        target_unit[..., gripper] = (2.0 * target[..., gripper] - 1.0).clamp(
            -1.0 + epsilon, 1.0 - epsilon
        )
    target_shift = torch.atanh(target_unit) - torch.atanh(base_unit)
    predicted_shift = outputs["residual_trust"] * outputs["residual_mean"]
    logit = (
        torch.nn.functional.smooth_l1_loss(predicted_shift, target_shift, reduction="none")
        * positive_weight
    ).sum() / (positive_count * corrected.shape[1] * corrected.shape[2])

    continuation = torch.logsumexp(outputs["latent_logits"][:, :-1], dim=1)
    handoff_logit = outputs["latent_logits"][:, -1] - continuation
    handoff_target = (~positive).to(handoff_logit.dtype)
    handoff = torch.nn.functional.binary_cross_entropy_with_logits(handoff_logit, handoff_target)
    negative = ~positive
    abstention = (
        sample["residual_action"][negative].square().mean()
        if negative.any()
        else corrected.sum() * 0.0
    )
    total = logit + 0.25 * action + 0.25 * handoff + 0.1 * abstention
    return total, {
        "total": total,
        "action": action,
        "logit": logit,
        "handoff": handoff,
        "abstention": abstention,
        "corrected": corrected,
        "base": base,
        "target": target,
        "positive": positive,
        "handoff_logit": handoff_logit,
    }


def _evaluate(
    actor: Any,
    data: dict[str, Any],
    indices: Any,
    config: ResolveModelConfig,
    torch: Any,
    device: Any,
) -> dict[str, float]:
    actor.eval()
    with torch.inference_mode():
        loss, metrics = _loss(actor, _inputs(data, indices, config, torch, device), torch=torch)
    positive = metrics["positive"]
    negative = ~positive
    corrected = metrics["corrected"]
    base = metrics["base"]
    target = metrics["target"]
    action_elements = config.action_horizon * config.action_dim
    predicted_positive = metrics["handoff_logit"] < 0.0
    executed = torch.where(predicted_positive[:, None, None], corrected, base)
    positive_absolute = (executed[positive] - target[positive]).abs().sum()
    proposal_absolute = (corrected[positive] - target[positive]).abs().sum()
    base_absolute = (base[positive] - target[positive]).abs().sum()
    return {
        "loss": float(loss),
        "samples": float(len(indices)),
        "positive_samples": float(positive.sum()),
        "positive_actor_mae": float(positive_absolute / (positive.sum() * action_elements)),
        "positive_proposal_mae": float(proposal_absolute / (positive.sum() * action_elements)),
        "positive_base_mae": float(base_absolute / (positive.sum() * action_elements)),
        "positive_mae_ratio": float(positive_absolute / base_absolute.clamp_min(1e-12)),
        "positive_intervention_rate": float(predicted_positive[positive].float().mean()),
        "false_intervention_rate": float(predicted_positive[negative].float().mean()),
        "baseline_safe_residual_mae": float(executed[negative].sub(base[negative]).abs().mean()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        min(args.steps, args.batch_size, args.eval_interval) < 1
        or args.learning_rate <= 0.0
        or args.weight_decay < 0.0
    ):
        raise OursContractError("RESOLVE frontier training arguments are invalid")
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("high")
    config = ResolveModelConfig(
        history_length=args.history_length,
        model_width=args.model_width,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        feedforward_width=args.feedforward_width,
    )
    device = torch.device(args.device)
    data, source = load_decisive_frontier_pairs(
        args.pairs,
        development_fraction=args.development_fraction,
        seed=args.seed,
        np=np,
    )
    train = np.flatnonzero(~data["development"])
    validation = np.flatnonzero(data["development"])
    train_positive = train[data["positive"][train]]
    train_safe = train[~data["positive"][train]]
    if not len(train_positive) or not len(train_safe):
        raise OursContractError("RESOLVE frontier training needs both decisive classes")
    actor = build_recovery_actor(config).to(device)
    optimizer = torch.optim.AdamW(
        actor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite RESOLVE frontier model: {output}")
    output.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "experiment_id": "Ours",
        "stage": "RESOLVE-decisive-frontier-warm-start",
        "novelty_claim": False,
        "unresolved_failures_are_negative_labels": False,
        "source": source,
        "model": asdict(config),
        "trainable_parameters": count_trainable_parameters(actor),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
    }
    _write_json(output / "run_manifest.json", manifest)
    baseline = _evaluate(actor, data, validation, config, torch, device)
    best_loss = math.inf
    best_step = 0
    torch.save(
        {
            "schema_version": 1,
            "stage": manifest["stage"],
            "model_config": asdict(config),
            "model_state_dict": actor.state_dict(),
            "step": 0,
            "development": baseline,
            "pairs_manifest_sha256": source["pairs_manifest_sha256"],
        },
        output / "actor.pt",
    )
    with (output / "training.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, args.steps + 1):
            actor.train()
            positive_count = min(len(train_positive), max(1, args.batch_size // 2))
            safe_count = args.batch_size - positive_count
            positive_indices = rng.choice(
                train_positive, size=positive_count, replace=len(train_positive) < positive_count
            )
            safe_indices = rng.choice(
                train_safe, size=safe_count, replace=len(train_safe) < safe_count
            )
            indices = np.concatenate((positive_indices, safe_indices))
            rng.shuffle(indices)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _loss(actor, _inputs(data, indices, config, torch, device), torch=torch)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            if step % args.eval_interval == 0 or step == args.steps:
                evaluation = _evaluate(actor, data, validation, config, torch, device)
                row = {
                    "step": step,
                    "train_loss": float(metrics["total"].detach()),
                    "gradient_norm": float(gradient_norm),
                    **evaluation,
                }
                log.write(json.dumps(row, separators=(",", ":")) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
                eligible = (
                    evaluation["positive_intervention_rate"] >= 0.5
                    and evaluation["false_intervention_rate"] <= 0.1
                    and evaluation["positive_mae_ratio"] < 1.0
                )
                if eligible and evaluation["loss"] < best_loss:
                    best_loss = evaluation["loss"]
                    best_step = step
                    torch.save(
                        {
                            "schema_version": 1,
                            "stage": manifest["stage"],
                            "model_config": asdict(config),
                            "model_state_dict": actor.state_dict(),
                            "step": step,
                            "development": evaluation,
                            "pairs_manifest_sha256": source["pairs_manifest_sha256"],
                        },
                        output / "actor.pt",
                    )
    checkpoint = torch.load(output / "actor.pt", map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint["model_state_dict"])
    selected = _evaluate(actor, data, validation, config, torch, device)
    summary = {
        "complete": True,
        "best_step": best_step,
        "accepted_warmstart": best_step > 0,
        "selection_gate": {
            "minimum_positive_intervention_rate": 0.5,
            "maximum_false_intervention_rate": 0.1,
            "maximum_positive_mae_ratio": 1.0,
        },
        "untrained_development": baseline,
        "selected_development": selected,
        "checkpoint": str(output / "actor.pt"),
        "checkpoint_sha256": sha256_file(output / "actor.pt"),
    }
    _write_json(output / "summary.json", summary)
    return summary


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
