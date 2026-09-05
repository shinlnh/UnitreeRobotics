"""Build and optionally execute reproducible upstream GR00T/SONIC commands."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import ProjectConfig


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    cwd: Path
    description: str

    def display(self) -> str:
        return f"cd {shlex.quote(str(self.cwd))} && {shlex.join(self.argv)}"


def _controller_flags(config: ProjectConfig) -> list[str]:
    variant = config.sonic.variant
    if variant == "release":
        return []
    if variant not in {"low_latency", "sonic_v1_1"}:
        raise ValueError(f"Unsupported SONIC variant: {variant}")
    return [
        "--deploy-checkpoint",
        f"policy/{variant}/model",
        "--deploy-obs-config",
        f"policy/{variant}/observation_config.yaml",
    ]


def build_train_command(
    config: ProjectConfig,
    dataset_path: str | Path,
    output_dir: str | Path,
    num_gpus: int | None = None,
    max_steps: int | None = None,
) -> CommandSpec:
    dataset = Path(dataset_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    argv = (
        "uv",
        "run",
        "python",
        "gr00t/experiment/launch_finetune.py",
        "--base-model-path",
        config.model.base_model,
        "--dataset-path",
        str(dataset),
        "--embodiment-tag",
        config.model.embodiment,
        "--modality-config-path",
        "gr00t/configs/data/embodiment_configs.py",
        "--num-gpus",
        str(num_gpus or config.training.num_gpus),
        "--output-dir",
        str(output),
        "--save-total-limit",
        "5",
        "--save-steps",
        str(config.training.save_steps),
        "--max-steps",
        str(max_steps or config.training.max_steps),
        "--global-batch-size",
        str(config.training.global_batch_size),
        "--color-jitter-params",
        "brightness",
        "0.3",
        "contrast",
        "0.4",
        "saturation",
        "0.5",
        "hue",
        "0.08",
        "--dataloader-num-workers",
        "4",
    )
    return CommandSpec(argv, config.isaac_gr00t_dir, "Fine-tune GR00T N1.7 for G1 SONIC")


def validate_sonic_checkpoint(config: ProjectConfig, checkpoint: str | None) -> str:
    selected = (checkpoint or config.model.checkpoint).strip()
    if not selected:
        raise ValueError(
            "A fine-tuned UNITREE_G1_SONIC checkpoint is required. "
            "The base GR00T checkpoint cannot serve this post-training embodiment."
        )
    if selected == config.model.base_model:
        raise ValueError(
            f"{config.model.base_model} is the base model; UNITREE_G1_SONIC requires a fine-tuned checkpoint"
        )
    return selected


def build_server_command(config: ProjectConfig, checkpoint: str | None = None) -> CommandSpec:
    selected = validate_sonic_checkpoint(config, checkpoint)
    argv = (
        "uv",
        "run",
        "python",
        "gr00t/eval/run_gr00t_server.py",
        "--model-path",
        selected,
        "--embodiment-tag",
        config.model.embodiment,
        "--device",
        config.model.device,
        "--host",
        config.model.bind_host,
        "--port",
        str(config.model.port),
    )
    return CommandSpec(argv, config.isaac_gr00t_dir, "Start the GR00T policy server")


def build_deploy_command(
    config: ProjectConfig,
    mode: str,
    prompt: str,
    record: bool | None = None,
) -> CommandSpec:
    if mode not in {"sim", "real"}:
        raise ValueError("mode must be sim or real")
    should_record = config.sonic.record_during_inference if record is None else record
    argv = [
        "python",
        "gear_sonic/scripts/launch_inference.py",
        "--policy-host",
        config.model.host,
        "--policy-port",
        str(config.model.port),
        "--camera-host",
        config.sonic.camera_host,
        "--camera-port",
        str(config.sonic.camera_port),
        "--prompt",
        prompt,
        "--action-publish-rate",
        str(config.sonic.action_publish_rate),
        "--action-horizon",
        str(config.sonic.action_horizon),
        *_controller_flags(config),
    ]
    if mode == "sim":
        argv.append("--sim")
    if not should_record:
        argv.append("--no-data-exporter")
    return CommandSpec(tuple(argv), config.sonic_dir, f"Launch SONIC inference on {mode}")


def build_collect_command(
    config: ProjectConfig,
    mode: str,
    prompt: str,
    dataset_name: str | None = None,
) -> CommandSpec:
    if mode not in {"sim", "real"}:
        raise ValueError("mode must be sim or real")
    argv = [
        "python",
        "gear_sonic/scripts/launch_data_collection.py",
        "--task-prompt",
        prompt,
        "--camera-host",
        config.sonic.camera_host,
        "--camera-port",
        str(config.sonic.camera_port),
        *_controller_flags(config),
    ]
    if mode == "sim":
        argv.append("--sim")
    if dataset_name:
        argv.extend(["--dataset-name", dataset_name])
    return CommandSpec(tuple(argv), config.sonic_dir, f"Collect G1 demonstrations on {mode}")


def build_process_dataset_command(
    config: ProjectConfig,
    dataset_path: str | Path,
    output_path: str | Path,
) -> CommandSpec:
    dataset = Path(dataset_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    argv = (
        "./.venv_data_collection/bin/python",
        "gear_sonic/scripts/process_dataset.py",
        "--dataset-path",
        str(dataset),
        "--output-path",
        str(output),
    )
    return CommandSpec(argv, config.sonic_dir, "Clean and filter a collected SONIC dataset")


def build_open_loop_command(
    config: ProjectConfig,
    dataset_path: str | Path,
    trajectory_ids: list[int],
    execution_horizon: int = 8,
) -> CommandSpec:
    if not trajectory_ids:
        raise ValueError("At least one trajectory id is required")
    dataset = Path(dataset_path).expanduser().resolve()
    argv = [
        "uv",
        "run",
        "python",
        "gr00t/eval/open_loop_eval.py",
        "--dataset-path",
        str(dataset),
        "--embodiment-tag",
        config.model.embodiment,
        "--host",
        config.model.host,
        "--port",
        str(config.model.port),
        "--traj-ids",
        *(str(value) for value in trajectory_ids),
        "--execution-horizon",
        str(execution_horizon),
    ]
    return CommandSpec(tuple(argv), config.isaac_gr00t_dir, "Run open-loop checkpoint evaluation")


def run_or_preview(spec: CommandSpec, execute: bool) -> int:
    print(f"{spec.description}:\n{spec.display()}")
    if not execute:
        print("\nPreview only. Add --execute to run it.")
        return 0
    if not spec.cwd.is_dir():
        raise FileNotFoundError(f"Upstream checkout not found: {spec.cwd}")
    completed = subprocess.run(spec.argv, cwd=spec.cwd, check=False)
    return completed.returncode
