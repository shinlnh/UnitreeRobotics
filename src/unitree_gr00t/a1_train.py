"""Run and provenance-lock the shared A1 GR00T-RC post-training job."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .a1 import validate_converted_dataset, write_checkpoint_provenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gr00t-source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--training-dataset", required=True)
    parser.add_argument("--training-dataset-revision", required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--dataloader-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--state-dropout-probability", type=float, default=0.2)
    parser.add_argument("--save-steps", type=int, default=1000)
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _output(argv: list[str]) -> str | None:
    try:
        completed = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() or None


def _hardware() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": _output(["lscpu"]),
        "memory": _output(["free", "-h"]),
        "gpu": _output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        min(
            args.max_steps,
            args.micro_batch_size,
            args.gradient_accumulation_steps,
            args.dataloader_workers,
        )
        < 1
    ):
        raise ValueError("A1 training counts must be positive")
    dataset = validate_converted_dataset(
        args.dataset,
        expected_episodes=args.expected_episodes,
        expected_revision=args.training_dataset_revision,
    )
    if not dataset.valid:
        raise ValueError(f"A1 converted dataset is invalid: {dataset.issues}")
    gr00t_source = args.gr00t_source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    artifact = args.artifact.expanduser().resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "unitree_gr00t.a1_launch_finetune",
        "--base-model-path",
        str(args.base_checkpoint.expanduser().resolve()),
        "--dataset-path",
        str(dataset.root),
        "--embodiment-tag",
        "LIBERO_PANDA",
        "--num-gpus",
        "1",
        "--output-dir",
        str(output),
        "--save-steps",
        str(args.save_steps),
        "--save-total-limit",
        "2",
        "--max-steps",
        str(args.max_steps),
        "--global-batch-size",
        str(args.micro_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--dataloader-num-workers",
        str(args.dataloader_workers),
        "--learning-rate",
        str(args.learning_rate),
        "--state-dropout-prob",
        str(args.state_dropout_probability),
        "--episode-sampling-rate",
        "1.0",
        "--shard-size",
        "1024",
        "--num-shards-per-epoch",
        "100000",
    ]
    manifest = {
        "schema_version": 1,
        "experiment_id": "A1",
        "variant": "GR00T-RC",
        "training_dataset": args.training_dataset,
        "training_dataset_revision": args.training_dataset_revision,
        "dataset_audit": {
            **dataset.__dict__,
            "root": str(dataset.root),
        },
        "base_checkpoint": str(args.base_checkpoint.expanduser().resolve()),
        "output": str(output),
        "max_steps": args.max_steps,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.micro_batch_size * args.gradient_accumulation_steps,
        "dataloader_workers": args.dataloader_workers,
        "learning_rate": args.learning_rate,
        "state_dropout_probability": args.state_dropout_probability,
        "tune_llm": False,
        "tune_visual": False,
        "tune_projector": True,
        "tune_diffusion_model": True,
        "memory_profile": {
            "backbone_dtype": "bfloat16",
            "trainable_head_dtype": "bfloat16",
            "optimizer": "adafactor",
            "gradient_checkpointing": True,
        },
        "command": command,
    }
    _write_json(artifact / "run_manifest.json", manifest)
    _write_json(artifact / "hardware.json", _hardware())
    log_path = artifact / "training.log"
    environment = os.environ.copy()
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    environment.setdefault("OMP_NUM_THREADS", "8")
    environment.setdefault("MKL_NUM_THREADS", "8")
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=gr00t_source,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("A1 trainer stdout pipe was not created")
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        failure = {"complete": False, "return_code": return_code, "log": str(log_path)}
        _write_json(artifact / "summary.json", failure)
        raise RuntimeError(f"A1 post-training failed with exit code {return_code}")
    processor_dir = output / "processor"
    for name in ("embodiment_id.json", "processor_config.json", "statistics.json"):
        source = processor_dir / name
        if not source.is_file():
            raise RuntimeError(f"A1 trainer did not emit {source}")
        shutil.copy2(source, output / name)
    provenance = write_checkpoint_provenance(
        output,
        training_dataset=args.training_dataset,
        training_dataset_revision=args.training_dataset_revision,
        converted_dataset=dataset,
        base_checkpoint=args.base_checkpoint,
        max_steps=args.max_steps,
        effective_batch_size=args.micro_batch_size * args.gradient_accumulation_steps,
    )
    summary = {
        "complete": True,
        "return_code": 0,
        "checkpoint": str(output),
        "checkpoint_provenance": str(provenance),
        "training_log": str(log_path),
    }
    _write_json(artifact / "summary.json", summary)
    return summary


def main() -> int:
    try:
        summary = run(_parser().parse_args())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
