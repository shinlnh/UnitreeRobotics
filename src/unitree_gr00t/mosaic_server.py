"""Serve exact B with registered MOSAIC continuous action-noise interventions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .a1 import inspect_a1_checkpoint, sha256_file
from .b import inspect_selector_checkpoint, verify_a1_weight_hashes
from .b_model import SelectorModelConfig, build_selector
from .b_runtime import build_selector_sim_policy
from .mosaic_runtime import build_mosaic_steering_policy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--embodiment", default="LIBERO_PANDA")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5550)
    parser.add_argument("--seed", type=int, default=30007)
    parser.add_argument("--code-dimension", type=int, default=4)
    parser.add_argument("--basis-seed", type=int, default=10007)
    parser.add_argument("--maximum-angle", type=float, default=0.2)
    parser.add_argument("--resolve-actor-checkpoint", type=Path)
    return parser


def run(args: argparse.Namespace) -> None:
    contract, _ = inspect_a1_checkpoint(
        args.checkpoint, expected_training_revision=args.model_revision
    )
    audit, provenance = inspect_selector_checkpoint(
        args.selector_checkpoint,
        expected_action_horizon=contract.action_horizon,
        expected_context_width=2048,
    )
    parent_weight_hashes = verify_a1_weight_hashes(
        contract, provenance.get("a1_checkpoint_weight_shards_sha256")
    )

    import numpy as np
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer
    from safetensors.torch import load_file

    # Counterfactual arms are invalid if identical request seeds can produce
    # different action chunks. Fail on a nondeterministic CUDA kernel instead
    # of silently turning numerical jitter into a causal effect.
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_config = SelectorModelConfig(
        action_horizon=int(provenance["action_horizon"]),
        action_dim=int(provenance["action_dim"]),
        context_width=int(provenance["context_width"]),
        scoring_width=int(provenance["scoring_width"]),
        scoring_layers=int(provenance["scoring_layers"]),
        scoring_heads=int(provenance["scoring_heads"]),
        feedforward_width=int(provenance["feedforward_width"]),
        dropout=float(provenance["dropout"]),
    )
    embodiment = EmbodimentTag.resolve(args.embodiment)
    base = Gr00tPolicy(
        embodiment,
        str(contract.checkpoint_dir),
        device=args.device,
        strict=True,
    )
    selector = build_selector(model_config).to(args.device)
    selector.load_state_dict(load_file(str(audit.checkpoint_dir / "model.safetensors")))
    selector.eval()
    exact_b = build_selector_sim_policy(
        base,
        selector,
        model_config,
        runtime_provenance={
            "selector_weights_sha256": audit.weights_sha256,
            "a1_checkpoint_weight_shards_sha256": parent_weight_hashes,
        },
    )
    policy = build_mosaic_steering_policy(
        exact_b,
        code_dimension=args.code_dimension,
        basis_seed=args.basis_seed,
        maximum_angle=args.maximum_angle,
    )
    resolve_actor_sha256 = None
    if args.resolve_actor_checkpoint is not None:
        from .resolve_model import ResolveModelConfig, build_recovery_actor
        from .resolve_runtime import build_resolve_sim_policy

        actor_checkpoint = args.resolve_actor_checkpoint.expanduser().resolve()
        payload = torch.load(actor_checkpoint, map_location=args.device, weights_only=True)
        if payload.get("stage") != "RESOLVE-decisive-frontier-warm-start":
            raise RuntimeError("RESOLVE server actor checkpoint has an incompatible stage")
        actor_config = ResolveModelConfig(**payload["model_config"])
        actor = build_recovery_actor(actor_config).to(args.device)
        actor.load_state_dict(payload["model_state_dict"], strict=True)
        actor.eval()
        resolve_actor_sha256 = sha256_file(actor_checkpoint)
        policy = build_resolve_sim_policy(
            policy,
            actor,
            actor_config,
            actor_sha256=resolve_actor_sha256,
        )
    print(
        json.dumps(
            {
                "experiment_id": "Ours",
                "variant": "GR00T-RC-MOSAIC-VLA",
                "checkpoint": str(contract.checkpoint_dir),
                "selector_checkpoint": str(audit.checkpoint_dir),
                "selector_weights_sha256": audit.weights_sha256,
                "a1_checkpoint_weight_shards_sha256": parent_weight_hashes,
                "code_dimension": args.code_dimension,
                "basis_seed": args.basis_seed,
                "maximum_angle": args.maximum_angle,
                "resolve_actor_checkpoint": (
                    None
                    if args.resolve_actor_checkpoint is None
                    else str(args.resolve_actor_checkpoint.expanduser().resolve())
                ),
                "resolve_actor_sha256": resolve_actor_sha256,
                "host": args.host,
                "port": args.port,
                "seed": args.seed,
            },
            indent=2,
        ),
        flush=True,
    )
    PolicyServer(policy=policy, host=args.host, port=args.port).run()


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
