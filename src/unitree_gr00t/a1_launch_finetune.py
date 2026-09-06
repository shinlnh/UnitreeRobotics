"""Memory-bounded GR00T N1.7 post-training launcher for one 16 GiB GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--embodiment-tag", required=True)
    parser.add_argument("--num-gpus", type=int, choices=(1,), default=1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-steps", type=int, required=True)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--dataloader-num-workers", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--state-dropout-prob", type=float, required=True)
    parser.add_argument("--episode-sampling-rate", type=float, default=1.0)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--num-shards-per-epoch", type=int, default=100_000)
    return parser


def run(args: argparse.Namespace) -> None:
    import torch
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.experiment import run as run_experiment
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline

    embodiment = EmbodimentTag.resolve(args.embodiment_tag).value
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [str(Path(args.dataset_path).resolve())],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment,
                    }
                ],
            }
        }
    )
    config.load_config_path = None
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.state_dropout_prob = args.state_dropout_prob
    config.model.color_jitter_params = {
        "brightness": 0.3,
        "contrast": 0.4,
        "saturation": 0.5,
        "hue": 0.08,
    }
    config.model.extra_augmentation_config = None
    config.model.load_bf16 = True
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = False
    config.model.use_relative_action = True

    config.training.start_from_checkpoint = str(Path(args.base_model_path).resolve())
    config.training.optim = "adafactor"
    config.training.global_batch_size = args.global_batch_size
    config.training.dataloader_num_workers = args.dataloader_num_workers
    config.training.learning_rate = args.learning_rate
    config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.training.gradient_checkpointing = True
    config.training.output_dir = str(Path(args.output_dir).resolve())
    config.training.save_steps = args.save_steps
    config.training.save_total_limit = args.save_total_limit
    config.training.num_gpus = 1
    config.training.use_wandb = False
    config.training.max_steps = args.max_steps
    config.training.weight_decay = 1e-5
    config.training.warmup_ratio = 0.05

    config.data.shard_size = args.shard_size
    config.data.episode_sampling_rate = args.episode_sampling_rate
    config.data.num_shards_per_epoch = args.num_shards_per_epoch

    torch.set_float32_matmul_precision("high")
    original_create_model = Gr00tN1d7Pipeline._create_model

    def create_bf16_trainable_model(pipeline: Gr00tN1d7Pipeline):
        model = original_create_model(pipeline)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(torch.bfloat16)
        return model

    Gr00tN1d7Pipeline._create_model = create_bf16_trainable_model
    os.environ.setdefault("LOGURU_LEVEL", "INFO")
    print(
        json.dumps(
            {
                "a1_memory_profile": {
                    "backbone_dtype": "bfloat16",
                    "trainable_head_dtype": "bfloat16",
                    "optimizer": "adafactor",
                    "gradient_checkpointing": True,
                    "tuned_modules": ["projector", "diffusion_model"],
                }
            },
            indent=2,
        ),
        flush=True,
    )
    run_experiment(config)


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
