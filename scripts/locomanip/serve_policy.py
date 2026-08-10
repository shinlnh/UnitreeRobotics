#!/usr/bin/env python3
"""Serve NVIDIA's G1 locomanipulation checkpoint from an isolated N1.5 environment."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from gr00t.eval.robot import RobotInferenceServer  # noqa: E402
from gr00t.model.policy import Gr00tPolicy  # noqa: E402

from configs.groot.g1_locomanipulation_data_config import G1LocomanipulationSDGDataConfig  # noqa: E402

DEFAULT_MODEL = (
    PROJECT_ROOT
    / "checkpoints/g1_locomanip_finetune_hf/g1_locomanip_finetune_20260129_231610/checkpoint-20000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(f"locomanipulation checkpoint is missing: {args.model}")
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")

    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(args.seed)

    data_config = G1LocomanipulationSDGDataConfig()
    print(f"[INFO] Loading G1 locomanipulation policy: {args.model} (seed={args.seed})", flush=True)
    policy = Gr00tPolicy(
        model_path=str(args.model.resolve()),
        embodiment_tag="new_embodiment",
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        denoising_steps=args.denoising_steps,
        device=args.device,
    )
    print(f"[READY] GR00T N1.5 locomanipulation server on {args.host}:{args.port}", flush=True)
    server = RobotInferenceServer(policy, host=args.host, port=args.port)

    def reset_policy_seed(payload: dict) -> dict[str, int]:
        seed = int(payload["seed"])
        set_seed(seed)
        print(f"[SEED] Policy diffusion RNG reset to {seed}", flush=True)
        return {"seed": seed}

    server.register_endpoint("set_seed", reset_policy_seed)
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
