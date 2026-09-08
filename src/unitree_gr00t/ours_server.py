"""Serve frozen GR00T-RC/B with Ours temporal recovery beliefs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .a1 import inspect_a1_checkpoint
from .b import inspect_selector_checkpoint, verify_a1_weight_hashes
from .b_model import SelectorModelConfig, build_selector
from .ours_model import TemporalRecoveryModelConfig, build_temporal_recovery_model
from .ours_runtime import build_recovery_sim_policy
from .ours_train import inspect_recovery_checkpoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--recovery-checkpoint", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--embodiment", default="LIBERO_PANDA")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5550)
    parser.add_argument("--seed", type=int, default=30007)
    return parser


def run(args: argparse.Namespace) -> None:
    contract, _ = inspect_a1_checkpoint(
        args.checkpoint, expected_training_revision=args.model_revision
    )
    selector_audit, selector_provenance = inspect_selector_checkpoint(
        args.selector_checkpoint,
        expected_action_horizon=contract.action_horizon,
        expected_context_width=2048,
    )
    parent_weight_hashes = verify_a1_weight_hashes(
        contract, selector_provenance.get("a1_checkpoint_weight_shards_sha256")
    )
    recovery_audit, recovery_provenance = inspect_recovery_checkpoint(
        args.recovery_checkpoint,
        expected_selector_sha256=selector_audit.weights_sha256,
    )
    if recovery_provenance.get("a1_checkpoint_weight_shards_sha256") != parent_weight_hashes:
        raise ValueError("Ours checkpoint binds different frozen A1 weights")

    import numpy as np
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from gr00t.policy.server_client import PolicyServer
    from safetensors.torch import load_file

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    selector_config = SelectorModelConfig(
        action_horizon=int(selector_provenance["action_horizon"]),
        action_dim=int(selector_provenance["action_dim"]),
        context_width=int(selector_provenance["context_width"]),
        scoring_width=int(selector_provenance["scoring_width"]),
        scoring_layers=int(selector_provenance["scoring_layers"]),
        scoring_heads=int(selector_provenance["scoring_heads"]),
        feedforward_width=int(selector_provenance["feedforward_width"]),
        dropout=float(selector_provenance["dropout"]),
    )
    recovery_config = TemporalRecoveryModelConfig(**recovery_provenance["model"])
    embodiment = EmbodimentTag.resolve(args.embodiment)
    base = Gr00tPolicy(embodiment, str(contract.checkpoint_dir), device=args.device, strict=True)
    selector = build_selector(selector_config).to(args.device)
    selector.load_state_dict(load_file(str(selector_audit.checkpoint_dir / "model.safetensors")))
    selector.eval()
    recovery = build_temporal_recovery_model(recovery_config).to(args.device)
    recovery.load_state_dict(load_file(str(recovery_audit.checkpoint_dir / "model.safetensors")))
    recovery.eval()
    runtime_provenance = {
        "b_selector": {
            "selector_weights_sha256": selector_audit.weights_sha256,
            "a1_checkpoint_weight_shards_sha256": parent_weight_hashes,
        },
        "ours": {
            "recovery_weights_sha256": recovery_audit.weights_sha256,
            "recovery_provenance_sha256": recovery_audit.provenance_sha256,
            "selector_weights_sha256": selector_audit.weights_sha256,
        },
    }
    policy = build_recovery_sim_policy(
        base,
        selector,
        selector_config,
        recovery,
        recovery_config,
        completion_threshold=recovery_audit.completion_threshold,
        runtime_provenance=runtime_provenance,
    )
    print(
        json.dumps(
            {
                "experiment_id": "Ours",
                "checkpoint": str(contract.checkpoint_dir),
                "selector_checkpoint": str(selector_audit.checkpoint_dir),
                "selector_weights_sha256": selector_audit.weights_sha256,
                "recovery_checkpoint": str(recovery_audit.checkpoint_dir),
                "recovery_weights_sha256": recovery_audit.weights_sha256,
                "completion_threshold": recovery_audit.completion_threshold,
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
