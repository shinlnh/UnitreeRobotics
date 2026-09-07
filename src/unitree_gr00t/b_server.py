"""Serve frozen GR00T-RC with B's learned unified selector."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .a1 import inspect_a1_checkpoint
from .b import inspect_selector_checkpoint
from .b_model import SelectorModelConfig, build_selector
from .b_runtime import build_selector_sim_policy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selector-checkpoint", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--embodiment", default="LIBERO_PANDA")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5550)
    parser.add_argument("--seed", type=int, default=7)
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
    if model_config.context_width != 2048:
        raise ValueError("B server only supports the frozen GR00T context width")
    selector = build_selector(model_config).to(args.device)
    selector.load_state_dict(load_file(str(audit.checkpoint_dir / "model.safetensors")))
    selector.eval()
    embodiment = EmbodimentTag.resolve(args.embodiment)
    base = Gr00tPolicy(
        embodiment,
        str(contract.checkpoint_dir),
        device=args.device,
        strict=True,
    )
    policy = build_selector_sim_policy(base, selector, model_config)
    print(
        json.dumps(
            {
                "experiment_id": provenance["experiment_id"],
                "checkpoint": str(contract.checkpoint_dir),
                "selector_checkpoint": str(audit.checkpoint_dir),
                "selector_weights_sha256": audit.weights_sha256,
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
