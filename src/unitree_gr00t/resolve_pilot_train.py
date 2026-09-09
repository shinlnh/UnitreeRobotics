"""Train compute-matched ordinary and CRB program actor-critics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .a1 import sha256_file
from .ours import OursContractError
from .ours_rollout_prepare import _write_json


@dataclass(frozen=True)
class PilotModelConfig:
    program_depth: int = 2
    context_width: int = 2048
    action_horizon: int = 16
    action_dim: int = 7
    code_dimension: int = 4
    model_width: int = 192
    transformer_layers: int = 3
    transformer_heads: int = 6
    feedforward_width: int = 768
    dropout: float = 0.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--milestone", default="next_ordered_subtask")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--actor-temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=61007)
    parser.add_argument("--validation-modulus", type=int, default=5)
    parser.add_argument("--validation-remainder", type=int, default=4)
    return parser


def build_program_scorer(config: PilotModelConfig) -> Any:
    """One shared potential-outcome network; worlds differ only by intervention."""

    import torch

    nn = torch.nn
    if (
        min(
            config.program_depth,
            config.context_width,
            config.action_horizon,
            config.action_dim,
            config.code_dimension,
            config.model_width,
            config.transformer_layers,
            config.transformer_heads,
            config.feedforward_width,
        )
        < 1
        or config.model_width % config.transformer_heads
    ):
        raise OursContractError("RESOLVE pilot model configuration is invalid")

    class ProgramScorer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = config.model_width
            action_width = config.action_horizon * config.action_dim
            self.context = nn.Sequential(
                nn.LayerNorm(config.context_width), nn.Linear(config.context_width, width)
            )
            self.base = nn.Sequential(nn.LayerNorm(action_width), nn.Linear(action_width, width))
            self.residual = nn.Sequential(
                nn.LayerNorm(action_width), nn.Linear(action_width, width)
            )
            self.code = nn.Linear(config.code_dimension, width)
            self.prefix = nn.Linear(1, width)
            self.subgoal_offset = nn.Linear(1, width)
            self.position = nn.Embedding(config.program_depth, width)
            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=config.transformer_heads,
                dim_feedforward=config.feedforward_width,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.program = nn.TransformerEncoder(layer, config.transformer_layers)
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

        def forward(
            self,
            contexts: Any,
            base_chunks: Any,
            residual_chunks: Any,
            codes: Any,
            prefix_lengths: Any,
            subgoal_offsets: Any,
        ) -> Any:
            batch, depth, context_width = contexts.shape
            expected_action = (
                batch,
                depth,
                config.action_horizon,
                config.action_dim,
            )
            if (
                (depth, context_width) != (config.program_depth, config.context_width)
                or base_chunks.shape != expected_action
                or residual_chunks.shape != expected_action
                or codes.shape != (batch, depth, config.code_dimension)
                or prefix_lengths.shape != (batch, depth)
                or subgoal_offsets.shape != (batch, depth)
            ):
                raise ValueError("RESOLVE pilot scorer input shapes are invalid")
            positions = torch.arange(depth, device=contexts.device)
            tokens = (
                self.context(contexts)
                + self.base(base_chunks.flatten(start_dim=2))
                + self.residual(residual_chunks.flatten(start_dim=2))
                + self.code(codes)
                + self.prefix(prefix_lengths[..., None])
                + self.subgoal_offset(subgoal_offsets[..., None])
                + self.position(positions)[None]
            )
            return self.head(self.program(tokens).mean(dim=1)).squeeze(1)

    return ProgramScorer()


def intervention_worlds(residuals: Any, codes: Any, *, torch: Any) -> tuple[Any, Any]:
    """Return shared-network inputs ordered R, D_0..D_d, B_0..B_d."""

    if residuals.ndim != 4 or codes.ndim != 3 or residuals.shape[:2] != codes.shape[:2]:
        raise ValueError("RESOLVE pilot intervention tensors are invalid")
    batch, depth = residuals.shape[:2]
    residual_worlds = [residuals]
    code_worlds = [codes]
    for index in range(depth):
        deleted_residual = residuals.clone()
        deleted_code = codes.clone()
        deleted_residual[:, index] = 0
        deleted_code[:, index] = 0
        residual_worlds.append(deleted_residual)
        code_worlds.append(deleted_code)
    for index in range(depth):
        baseline_residual = residuals.clone()
        baseline_code = codes.clone()
        baseline_residual[:, index:] = 0
        baseline_code[:, index:] = 0
        residual_worlds.append(baseline_residual)
        code_worlds.append(baseline_code)
    return torch.stack(residual_worlds, dim=1), torch.stack(code_worlds, dim=1)


def intervention_sequence_worlds(values: Any, *, torch: Any) -> Any:
    """Encode R, one-slot deletion, and suffix handoff for a sequence tensor."""

    if values.ndim < 2:
        raise ValueError("RESOLVE intervention sequence is invalid")
    depth = values.shape[1]
    worlds = [values]
    for index in range(depth):
        deleted = values.clone()
        deleted[:, index] = 0
        worlds.append(deleted)
    for index in range(depth):
        baseline = values.clone()
        baseline[:, index:] = 0
        worlds.append(baseline)
    return torch.stack(worlds, dim=1)


def causal_program_view(
    contexts: Any, base_chunks: Any, residual_chunks: Any, *, torch: Any
) -> tuple[Any, Any, Any]:
    """Remove post-treatment observations from an anchor-time program input.

    The complete latent code sequence is a proposal available at the anchor.
    Future observations and VLA chunks are not. Only the current context, B
    proposal, and first residual are retained; the anchor features are repeated
    over structural tokens so the transformer can combine them with each code.
    """

    if (
        contexts.ndim != 3
        or base_chunks.ndim != 4
        or residual_chunks.ndim != 4
        or contexts.shape[:2] != base_chunks.shape[:2]
        or base_chunks.shape != residual_chunks.shape
    ):
        raise ValueError("RESOLVE causal program tensors are invalid")
    depth = contexts.shape[1]
    anchor_context = contexts[:, :1].expand(-1, depth, -1)
    anchor_base = base_chunks[:, :1].expand(-1, depth, -1, -1)
    current_residual = torch.zeros_like(residual_chunks)
    current_residual[:, 0] = residual_chunks[:, 0]
    return anchor_context, anchor_base, current_residual


def crb_from_world_probabilities(probabilities: Any) -> Any:
    """Map R,D_i,B_i probabilities to the program's worst-slot CRB."""

    if probabilities.ndim != 2 or probabilities.shape[1] < 3 or probabilities.shape[1] % 2 != 1:
        raise ValueError("RESOLVE pilot world probabilities are invalid")
    depth = (probabilities.shape[1] - 1) // 2
    recovery = probabilities[:, 0]
    deletion = probabilities[:, 1 : 1 + depth]
    baseline = probabilities[:, 1 + depth :]
    return recovery - __import__("torch").maximum(deletion, baseline).max(dim=1).values


def _split_state(key: str, modulus: int, remainder: int) -> bool:
    digest = hashlib.sha256(f"RESOLVE-pilot-split-v1:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulus == remainder


def load_pilot_corpora(
    roots: list[Path], *, milestone: str, validation_modulus: int, validation_remainder: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    if validation_modulus < 2 or not 0 <= validation_remainder < validation_modulus:
        raise ValueError("RESOLVE pilot split is invalid")
    arrays: dict[str, list[Any]] = {
        name: []
        for name in (
            "contexts",
            "base_chunks",
            "residual_chunks",
            "codes",
            "prefix_lengths",
            "subgoal_offsets",
            "targets",
            "state_keys",
            "program_ids",
        )
    }
    manifests: list[dict[str, Any]] = []
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests.append(manifest)
        milestones = tuple(manifest["milestones"])
        if milestone not in milestones:
            raise OursContractError(f"unknown RESOLVE pilot milestone: {milestone}")
        milestone_index = milestones.index(milestone)
        base_seed = int(manifest["base_seed"])
        for line in (root / "programs.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            feature_path = root / "features" / row["features"]
            if sha256_file(feature_path) != manifest["files_sha256"][row["features"]]:
                raise OursContractError("RESOLVE pilot feature hash mismatch")
            with np.load(feature_path, allow_pickle=False) as value:
                contexts = value["recovery_contexts"].astype(np.float32)
                base = value["base_chunks"].astype(np.float32)
                recovery = value["recovery_chunks"].astype(np.float32)
                codes = value["steering_codes"].astype(np.float32)
                prefixes = value["prefix_lengths"].astype(np.float32) / 16.0
                offsets = (
                    value["subgoal_offsets"].astype(np.float32)
                    if "subgoal_offsets" in value
                    else np.zeros(len(prefixes), dtype=np.float32)
                )
                offsets = np.clip(offsets / 16.0, -1.0, 1.0)
                r = float(value["recovery_target"][milestone_index])
                d = value["deletion_targets"][:, milestone_index].astype(np.float32)
                b = value["baseline_targets"][:, milestone_index].astype(np.float32)
            depth = int(manifest["program_depth"])
            if (
                contexts.shape != (depth, 2048)
                or base.shape != (depth, 16, 7)
                or recovery.shape != (depth, 16, 7)
                or codes.shape != (depth, int(manifest["code_dimension"]))
                or d.shape != (depth,)
                or b.shape != (depth,)
            ):
                raise OursContractError("RESOLVE pilot feature shape mismatch")
            arrays["contexts"].append(contexts)
            arrays["base_chunks"].append(base)
            arrays["residual_chunks"].append(recovery - base)
            arrays["codes"].append(codes)
            arrays["prefix_lengths"].append(prefixes)
            arrays["subgoal_offsets"].append(offsets)
            arrays["targets"].append(np.concatenate(([r], d, b)))
            arrays["state_keys"].append(f"{base_seed}:{int(row['sample_index'])}")
            arrays["program_ids"].append(str(row["program_id"]))
    if not arrays["targets"]:
        raise OursContractError("RESOLVE pilot corpora contain no programs")
    packed = {
        name: np.stack(values) if name not in {"state_keys", "program_ids"} else values
        for name, values in arrays.items()
    }
    validation = np.asarray(
        [
            _split_state(key, validation_modulus, validation_remainder)
            for key in packed["state_keys"]
        ],
        dtype=np.bool_,
    )
    if not validation.any() or validation.all():
        unique = sorted(set(packed["state_keys"]))
        if len(unique) < 2:
            raise OursContractError("RESOLVE pilot needs at least two physical anchors")
        held_out = unique[-1]
        validation = np.asarray([key == held_out for key in packed["state_keys"]], dtype=np.bool_)
    packed["validation"] = validation
    return packed, {"manifests": manifests, "milestone": milestone}


def _tensor_batch(data: dict[str, Any], indices: Any, device: Any, torch: Any) -> tuple[Any, ...]:
    return tuple(
        torch.as_tensor(data[name][indices], dtype=torch.float32, device=device)
        for name in (
            "contexts",
            "base_chunks",
            "residual_chunks",
            "codes",
            "prefix_lengths",
            "subgoal_offsets",
            "targets",
        )
    )


def _world_logits(
    model: Any,
    contexts: Any,
    base_chunks: Any,
    residual_chunks: Any,
    codes: Any,
    prefixes: Any,
    subgoal_offsets: Any,
    *,
    torch: Any,
) -> Any:
    contexts, base_chunks, residual_chunks = causal_program_view(
        contexts, base_chunks, residual_chunks, torch=torch
    )
    residual_worlds, code_worlds = intervention_worlds(residual_chunks, codes, torch=torch)
    offset_worlds = intervention_sequence_worlds(subgoal_offsets, torch=torch)
    batch, worlds, depth = residual_worlds.shape[:3]
    repeated_contexts = contexts[:, None].expand(-1, worlds, -1, -1)
    repeated_base = base_chunks[:, None].expand(-1, worlds, -1, -1, -1)
    repeated_prefixes = prefixes[:, None].expand(-1, worlds, -1)
    recovery_atoms = code_worlds.abs().sum(dim=-1) > 0
    repeated_prefixes = repeated_prefixes * recovery_atoms
    logits = model(
        repeated_contexts.reshape(batch * worlds, depth, -1),
        repeated_base.reshape(batch * worlds, depth, *base_chunks.shape[-2:]),
        residual_worlds.reshape(batch * worlds, depth, *residual_chunks.shape[-2:]),
        code_worlds.reshape(batch * worlds, depth, codes.shape[-1]),
        repeated_prefixes.reshape(batch * worlds, depth),
        offset_worlds.reshape(batch * worlds, depth),
    )
    return logits.reshape(batch, worlds)


def _actor_logits(model: Any, batch: tuple[Any, ...], *, torch: Any) -> Any:
    contexts, base, residuals, codes, prefixes, subgoal_offsets, _ = batch
    contexts, base, residuals = causal_program_view(contexts, base, residuals, torch=torch)
    return model(contexts, base, residuals, codes, prefixes, subgoal_offsets)


def _group_indices(state_keys: list[str], allowed: Any) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index in allowed.tolist():
        groups.setdefault(state_keys[index], []).append(index)
    return groups


def _evaluate_policy(
    actor: Any,
    data: dict[str, Any],
    indices: Any,
    *,
    device: Any,
    torch: Any,
) -> dict[str, Any]:
    import numpy as np

    actor.eval()
    groups = _group_indices(data["state_keys"], indices)
    selected: list[dict[str, Any]] = []
    with torch.inference_mode():
        for state_key, candidates in sorted(groups.items()):
            candidate_indices = np.asarray(candidates, dtype=np.int64)
            batch = _tensor_batch(data, candidate_indices, device, torch)
            logits = _actor_logits(actor, batch, torch=torch)
            local = int(logits.argmax().item())
            chosen = candidates[local]
            target = data["targets"][chosen]
            recovery = float(target[0])
            crb = float(target[0] - target[1:].max())
            selected.append(
                {
                    "state_key": state_key,
                    "program_id": data["program_ids"][chosen],
                    "recovery": recovery,
                    "crb": crb,
                    "certified": crb > 0.0,
                }
            )
    if not selected:
        raise OursContractError("RESOLVE pilot evaluation split is empty")
    return {
        "anchors": len(selected),
        "mean_recovery": float(np.mean([item["recovery"] for item in selected])),
        "mean_crb": float(np.mean([item["crb"] for item in selected])),
        "certified_rate": float(np.mean([item["certified"] for item in selected])),
        "selected": selected,
    }


def _critic_validation(
    critic: Any,
    data: dict[str, Any],
    indices: Any,
    *,
    ordinary: bool,
    device: Any,
    torch: Any,
) -> dict[str, float]:
    functional = torch.nn.functional
    critic.eval()
    with torch.inference_mode():
        batch = _tensor_batch(data, indices, device, torch)
        logits = _world_logits(critic, *batch[:-1], torch=torch)
        targets = batch[-1]
        if ordinary:
            logits = logits[:, :1]
            targets = targets[:, :1]
        loss = functional.binary_cross_entropy_with_logits(logits, targets)
        probabilities = logits.sigmoid()
        mae = (probabilities - targets).abs().mean()
    return {"bce": float(loss.cpu()), "mae": float(mae.cpu())}


def train(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    if (
        min(args.steps, args.batch_size, args.validation_modulus) < 1
        or args.learning_rate <= 0
        or args.weight_decay < 0
        or args.actor_temperature <= 0
        or args.seed < 0
    ):
        raise ValueError("RESOLVE pilot training arguments are invalid")
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite RESOLVE pilot training: {output}")
    data, provenance = load_pilot_corpora(
        args.corpus,
        milestone=args.milestone,
        validation_modulus=args.validation_modulus,
        validation_remainder=args.validation_remainder,
    )
    validation_mask = data["validation"]
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    train_groups = _group_indices(data["state_keys"], train_indices)
    if len(train_groups) < 1:
        raise OursContractError("RESOLVE pilot training split is empty")

    first_manifest = provenance["manifests"][0]
    config = PilotModelConfig(
        program_depth=int(first_manifest["program_depth"]),
        code_dimension=int(first_manifest["code_dimension"]),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    critic_initial = build_program_scorer(config).to(device)
    actor_initial = build_program_scorer(config).to(device)
    models = {
        "ordinary_critic": copy.deepcopy(critic_initial),
        "ordinary_actor": copy.deepcopy(actor_initial),
        "resolve_critic": copy.deepcopy(critic_initial),
        "resolve_actor": copy.deepcopy(actor_initial),
    }
    optimizers = {
        name: torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        for name, model in models.items()
    }
    generator = np.random.default_rng(args.seed)
    functional = torch.nn.functional
    group_values = list(train_groups.values())
    history: list[dict[str, float]] = []
    for step in range(1, args.steps + 1):
        sampled = generator.choice(
            train_indices, size=min(args.batch_size, len(train_indices)), replace=True
        )
        batch = _tensor_batch(data, sampled, device, torch)
        targets = batch[-1]
        losses: dict[str, Any] = {}
        for method in ("ordinary", "resolve"):
            critic = models[f"{method}_critic"]
            critic.train()
            logits = _world_logits(critic, *batch[:-1], torch=torch)
            critic_loss = (
                functional.binary_cross_entropy_with_logits(logits[:, 0], targets[:, 0])
                if method == "ordinary"
                else functional.binary_cross_entropy_with_logits(logits, targets)
            )
            optimizers[f"{method}_critic"].zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizers[f"{method}_critic"].step()
            losses[f"{method}_critic"] = critic_loss.detach()

            actor = models[f"{method}_actor"]
            actor.train()
            actor_losses = []
            chosen_groups = generator.choice(
                len(group_values), size=min(8, len(group_values)), replace=True
            )
            for group_index in chosen_groups:
                candidates = np.asarray(group_values[int(group_index)], dtype=np.int64)
                candidate_batch = _tensor_batch(data, candidates, device, torch)
                with torch.no_grad():
                    critic_logits = _world_logits(critic, *candidate_batch[:-1], torch=torch)
                    score = (
                        critic_logits[:, 0].sigmoid()
                        if method == "ordinary"
                        else crb_from_world_probabilities(critic_logits.sigmoid())
                    )
                    target_policy = torch.softmax(score / args.actor_temperature, dim=0)
                actor_score = _actor_logits(actor, candidate_batch, torch=torch)
                actor_losses.append(-(target_policy * torch.log_softmax(actor_score, dim=0)).sum())
            actor_loss = torch.stack(actor_losses).mean()
            optimizers[f"{method}_actor"].zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizers[f"{method}_actor"].step()
            losses[f"{method}_actor"] = actor_loss.detach()
        if step == 1 or step % 100 == 0 or step == args.steps:
            history.append(
                {"step": step} | {name: float(value.cpu()) for name, value in losses.items()}
            )

    evaluation = {
        method: {
            "critic": _critic_validation(
                models[f"{method}_critic"],
                data,
                validation_indices,
                ordinary=method == "ordinary",
                device=device,
                torch=torch,
            ),
            "policy": _evaluate_policy(
                models[f"{method}_actor"],
                data,
                validation_indices,
                device=device,
                torch=torch,
            ),
        }
        for method in ("ordinary", "resolve")
    }
    output.mkdir(parents=True)
    from safetensors.torch import save_file

    weights = {
        f"{name}.{key}": value.detach().cpu().contiguous()
        for name, model in models.items()
        for key, value in model.state_dict().items()
    }
    weights_path = output / "model.safetensors"
    save_file(weights, str(weights_path))
    report = {
        "schema_version": 1,
        "stage": "R1a-compute-matched-contextual-program-actor-critic-pilot",
        "claim_status": "diagnostic-only",
        "learning_scope": (
            "anchor-time Monte Carlo reachability; the full closed-loop TD "
            "learner is not claimed by this pilot"
        ),
        "post_treatment_features_visible": False,
        "milestone": args.milestone,
        "model_config": asdict(config),
        "seed": args.seed,
        "steps_per_method": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "actor_temperature": args.actor_temperature,
        "programs": len(data["targets"]),
        "train_programs": len(train_indices),
        "validation_programs": len(validation_indices),
        "train_anchors": len(set(data["state_keys"][i] for i in train_indices)),
        "validation_anchors": len(set(data["state_keys"][i] for i in validation_indices)),
        "comparison_contract": (
            "same programs, physical rollouts, network architecture, optimizer, "
            "updates, and initialization seed; ordinary fits R only while RESOLVE "
            "fits paired R/D_i/B_i and improves on CRB"
        ),
        "evaluation": evaluation,
        "history": history,
        "weights_sha256": sha256_file(weights_path),
        "source_manifests": provenance["manifests"],
    }
    _write_json(output / "report.json", report)
    return report


def main() -> int:
    print(json.dumps(train(_parser().parse_args()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
