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


def build_a0_server_command(config: ProjectConfig, seed: int = 7) -> CommandSpec:
    """Start the unmodified NVIDIA LIBERO checkpoint behind its official API."""

    argv = (
        "uv",
        "run",
        "--no-sync",
        "python",
        "gr00t/eval/run_gr00t_server.py",
        "--model-path",
        str(config.robocerebra.checkpoint_dir),
        "--embodiment-tag",
        config.robocerebra.embodiment,
        "--device",
        config.model.device,
        "--host",
        config.model.bind_host,
        "--port",
        str(config.model.port),
        "--seed",
        str(seed),
        "--use-sim-policy-wrapper",
    )
    return CommandSpec(argv, config.isaac_gr00t_dir, "Start the frozen A0 policy server")


def build_a0_eval_command(
    config: ProjectConfig,
    *,
    task_types: list[str],
    case_names: list[str],
    trials: int,
    execution_horizon: int,
    seed: int,
    output_dir: str | Path,
    trace_images: bool = True,
    resume: bool = False,
) -> CommandSpec:
    """Build the separate RoboCerebra client command for experiment A0."""

    if trials < 1:
        raise ValueError("trials must be at least 1")
    if execution_horizon not in config.robocerebra.fixed_execution_horizons:
        allowed = ", ".join(f"H{value}" for value in config.robocerebra.fixed_execution_horizons)
        raise ValueError(f"A0 execution horizon must be one of: {allowed}")
    unknown_types = sorted(set(task_types) - set(config.robocerebra.task_types))
    if unknown_types:
        raise ValueError(f"Unknown RoboCerebra task types: {', '.join(unknown_types)}")

    evaluator_python = config.root / ".venv-a0" / "bin" / "python"
    output = Path(output_dir).expanduser().resolve()
    argv = [
        "/usr/bin/env",
        f"PYTHONPATH={config.root / 'src'}",
        str(evaluator_python),
        "-m",
        "unitree_gr00t.a0_eval",
        "--robocerebra-source",
        str(config.robocerebra_dir),
        "--benchmark-dir",
        str(config.robocerebra.benchmark_dir),
        "--checkpoint",
        str(config.robocerebra.checkpoint_dir),
        "--benchmark-revision",
        config.upstream.robocerebra_revision,
        "--model-revision",
        config.robocerebra.model_revision,
        "--dataset-revision",
        config.robocerebra.dataset_revision,
        "--task-types",
        *task_types,
    ]
    if case_names:
        argv.extend(("--cases", *case_names))
    argv.extend(
        (
            "--trials",
            str(trials),
            "--execution-horizon",
            str(execution_horizon),
            "--control-frequency-hz",
            str(config.robocerebra.control_frequency_hz),
            "--steps-per-subtask",
            str(config.robocerebra.steps_per_subtask),
            "--initial-wait-steps",
            str(config.robocerebra.initial_wait_steps),
            "--post-success-steps",
            str(config.robocerebra.post_success_observation_steps),
            "--seed",
            str(seed),
            "--policy-host",
            config.model.host,
            "--policy-port",
            str(config.model.port),
            "--output",
            str(output),
        )
    )
    if not trace_images:
        argv.append("--no-trace-images")
    if resume:
        argv.append("--resume")
    return CommandSpec(tuple(argv), config.root, "Run A0 on RoboCerebra")


def build_a1_prepare_command(
    config: ProjectConfig,
    *,
    workers: int | None = None,
    limit: int | None = None,
    resume: bool = True,
    output_dir: str | Path | None = None,
) -> CommandSpec:
    """Build the raw-state replay and GR00T LeRobot conversion command."""

    a1 = config.robocerebra_posttrain
    selected_workers = workers or a1.conversion_workers
    if selected_workers < 1:
        raise ValueError("A1 conversion workers must be positive")
    destination = Path(output_dir).expanduser().resolve() if output_dir else a1.lerobot_training_dir
    argv = [
        "/usr/bin/env",
        f"PYTHONPATH={config.root / 'src'}",
        str(config.root / ".venv-a0" / "bin" / "python"),
        "-m",
        "unitree_gr00t.a1_data",
        "--robocerebra-source",
        str(config.robocerebra_dir),
        "--benchmark-dir",
        str(config.robocerebra.benchmark_dir),
        "--source",
        str(a1.raw_training_dir),
        "--manifest",
        str(a1.training_manifest),
        "--expected-manifest-sha256",
        a1.training_manifest_sha256,
        "--destination",
        str(destination),
        "--source-revision",
        a1.training_dataset_revision,
        "--expected-manifest-rows",
        str(a1.expected_training_manifest_rows),
        "--expected-episodes",
        str(a1.expected_training_episodes),
        "--workers",
        str(selected_workers),
        "--fps",
        str(a1.dataset_fps),
    ]
    if limit is not None:
        argv.extend(("--limit", str(limit)))
    if resume:
        argv.append("--resume")
    return CommandSpec(tuple(argv), config.root, "Prepare the A1 RoboCerebra training dataset")


def build_a1_train_command(
    config: ProjectConfig,
    *,
    max_steps: int | None = None,
    dataset_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    artifact_dir: str | Path | None = None,
) -> CommandSpec:
    """Build the single-GPU, memory-bounded A1 post-training command."""

    a1 = config.robocerebra_posttrain
    steps = max_steps or a1.max_steps
    if steps < 1:
        raise ValueError("A1 max steps must be positive")
    dataset = Path(dataset_dir).expanduser().resolve() if dataset_dir else a1.lerobot_training_dir
    output = Path(checkpoint_dir).expanduser().resolve() if checkpoint_dir else a1.checkpoint_dir
    artifact = (
        Path(artifact_dir).expanduser().resolve()
        if artifact_dir
        else config.artifact_dir / "A1" / "training"
    )
    argv = (
        "/usr/bin/env",
        f"PYTHONPATH={config.root / 'src'}",
        str(config.isaac_gr00t_dir / ".venv" / "bin" / "python"),
        "-m",
        "unitree_gr00t.a1_train",
        "--gr00t-source",
        str(config.isaac_gr00t_dir),
        "--dataset",
        str(dataset),
        "--base-checkpoint",
        str(a1.base_checkpoint_dir),
        "--output",
        str(output),
        "--artifact",
        str(artifact),
        "--training-dataset",
        a1.training_dataset,
        "--training-dataset-revision",
        a1.training_dataset_revision,
        "--expected-episodes",
        str(a1.expected_training_episodes),
        "--max-steps",
        str(steps),
        "--micro-batch-size",
        str(a1.micro_batch_size),
        "--gradient-accumulation-steps",
        str(a1.gradient_accumulation_steps),
        "--dataloader-workers",
        str(a1.dataloader_workers),
        "--learning-rate",
        str(a1.learning_rate),
        "--state-dropout-probability",
        str(a1.state_dropout_probability),
        "--save-steps",
        str(min(a1.save_steps, steps)),
    )
    return CommandSpec(argv, config.root, "Post-train the shared A1 GR00T-RC checkpoint")


def build_a1_server_command(config: ProjectConfig, seed: int = 7) -> CommandSpec:
    """Start the frozen post-trained A1 checkpoint behind the official API."""

    argv = (
        "uv",
        "run",
        "--no-sync",
        "python",
        "gr00t/eval/run_gr00t_server.py",
        "--model-path",
        str(config.robocerebra_posttrain.checkpoint_dir),
        "--embodiment-tag",
        config.robocerebra.embodiment,
        "--device",
        config.model.device,
        "--host",
        config.model.bind_host,
        "--port",
        str(config.model.port),
        "--seed",
        str(seed),
        "--use-sim-policy-wrapper",
    )
    return CommandSpec(argv, config.isaac_gr00t_dir, "Start the frozen A1 policy server")


def build_a1_eval_command(
    config: ProjectConfig,
    *,
    task_types: list[str],
    case_names: list[str],
    trials: int,
    execution_horizon: int,
    seed: int,
    output_dir: str | Path,
    trace_images: bool = True,
    resume: bool = False,
) -> CommandSpec:
    """Reuse the frozen continuous evaluator with A1 identity/provenance."""

    spec = build_a0_eval_command(
        config,
        task_types=task_types,
        case_names=case_names,
        trials=trials,
        execution_horizon=execution_horizon,
        seed=seed,
        output_dir=output_dir,
        trace_images=trace_images,
        resume=resume,
    )
    argv = list(spec.argv)
    checkpoint_index = argv.index("--checkpoint") + 1
    model_revision_index = argv.index("--model-revision") + 1
    argv[checkpoint_index] = str(config.robocerebra_posttrain.checkpoint_dir)
    argv[model_revision_index] = config.robocerebra_posttrain.training_dataset_revision
    argv.extend(
        (
            "--experiment-id",
            config.robocerebra_posttrain.experiment_id,
            "--variant",
            config.robocerebra_posttrain.variant,
        )
    )
    return CommandSpec(tuple(argv), config.root, "Run A1 on RoboCerebra")


def build_a2_server_command(config: ProjectConfig, seed: int = 7) -> CommandSpec:
    """A2 uses the exact frozen A1 checkpoint and policy-server contract."""

    spec = build_a1_server_command(config, seed)
    return CommandSpec(spec.argv, spec.cwd, "Start the frozen A2 low-level policy server")


def build_a2_eval_command(
    config: ProjectConfig,
    *,
    task_types: list[str],
    case_names: list[str],
    trials: int,
    execution_horizon: int,
    seed: int,
    output_dir: str | Path,
    trace_images: bool = True,
    resume: bool = False,
) -> CommandSpec:
    """Build the fixed-anchor hierarchical A2 evaluation command."""

    a2 = config.robocerebra_hierarchy
    spec = build_a1_eval_command(
        config,
        task_types=task_types,
        case_names=case_names,
        trials=trials,
        execution_horizon=execution_horizon,
        seed=seed,
        output_dir=output_dir,
        trace_images=trace_images,
        resume=resume,
    )
    argv = list(spec.argv)
    module_index = argv.index("unitree_gr00t.a0_eval")
    argv[module_index] = "unitree_gr00t.a2_eval"
    argv[argv.index("--experiment-id") + 1] = a2.experiment_id
    argv[argv.index("--variant") + 1] = a2.variant
    argv[argv.index("--steps-per-subtask") + 1] = str(a2.subgoal_horizon_steps)
    argv.extend(("--planner", a2.planner, "--plan-source", a2.plan_source))
    return CommandSpec(tuple(argv), config.root, "Run A2 fixed hierarchy on RoboCerebra")


def build_b_prepare_command(
    config: ProjectConfig, *, limit: int | None = None, output_dir: str | Path | None = None
) -> CommandSpec:
    """Build B's audited demonstration-boundary index."""

    a1 = config.robocerebra_posttrain
    b = config.robocerebra_selector
    destination = Path(output_dir).expanduser().resolve() if output_dir else b.dataset_dir
    argv = [
        "/usr/bin/env",
        f"PYTHONPATH={config.root / 'src'}",
        str(config.root / ".venv-a0" / "bin" / "python"),
        "-m",
        "unitree_gr00t.b_data",
        "--source",
        str(a1.raw_training_dir),
        "--manifest",
        str(a1.training_manifest),
        "--converted-dataset",
        str(a1.lerobot_training_dir),
        "--benchmark-dir",
        str(config.robocerebra.benchmark_dir),
        "--source-revision",
        a1.training_dataset_revision,
        "--expected-manifest-sha256",
        a1.training_manifest_sha256,
        "--expected-manifest-rows",
        str(a1.expected_training_manifest_rows),
        "--expected-episodes",
        str(a1.expected_training_episodes),
        "--destination",
        str(destination),
        "--action-horizon",
        str(b.action_horizon),
        "--sample-stride",
        str(b.sample_stride),
        "--post-boundary-steps",
        str(b.near_boundary_steps),
        "--development-fraction",
        str(b.development_fraction),
        "--seed",
        str(b.training_seed),
    ]
    if limit is not None:
        argv.extend(("--limit", str(limit)))
    return CommandSpec(tuple(argv), config.root, "Prepare B selector boundary index")


def build_b_features_command(
    config: ProjectConfig,
    *,
    batch_size: int = 1,
    limit_episodes: int | None = None,
    resume: bool = True,
) -> CommandSpec:
    """Extract frozen A1 contexts and H16 proposals for B."""

    if batch_size < 1:
        raise ValueError("B feature batch size must be positive")
    a1 = config.robocerebra_posttrain
    b = config.robocerebra_selector
    python = config.isaac_gr00t_dir / ".venv" / "bin" / "python"
    argv = [
        "/usr/bin/env",
        f"PYTHONPATH={config.root / 'src'}:{config.isaac_gr00t_dir}",
        str(python),
        "-m",
        "unitree_gr00t.b_features",
        "--index",
        str(b.dataset_dir),
        "--checkpoint",
        str(a1.checkpoint_dir),
        "--model-revision",
        a1.training_dataset_revision,
        "--destination",
        str(b.dataset_dir),
        "--device",
        config.model.device,
        "--batch-size",
        str(batch_size),
        "--seed",
        str(b.training_seed),
    ]
    if limit_episodes is not None:
        argv.extend(("--limit-episodes", str(limit_episodes)))
    if resume:
        argv.append("--resume")
    return CommandSpec(tuple(argv), config.root, "Extract frozen A1 features for B")


def build_b_train_command(
    config: ProjectConfig, *, steps: int | None = None, batch_size: int | None = None
) -> CommandSpec:
    """Train the B selector under its frozen paper-derived contract."""

    b = config.robocerebra_selector
    selected_steps = steps or b.max_steps
    selected_batch = batch_size or b.batch_size
    python = config.isaac_gr00t_dir / ".venv" / "bin" / "python"
    argv = (
        "/usr/bin/env",
        f"PYTHONPATH={config.root / 'src'}:{config.isaac_gr00t_dir}",
        str(python),
        "-m",
        "unitree_gr00t.b_train",
        "--dataset",
        str(b.dataset_dir),
        "--destination",
        str(b.checkpoint_dir),
        "--device",
        config.model.device,
        "--action-horizon",
        str(b.action_horizon),
        "--context-width",
        str(b.context_width),
        "--scoring-width",
        str(b.scoring_width),
        "--scoring-layers",
        str(b.scoring_layers),
        "--scoring-heads",
        str(b.scoring_heads),
        "--feedforward-width",
        str(b.feedforward_width),
        "--boundary-jitter-steps",
        str(b.boundary_jitter_steps),
        "--near-boundary-steps",
        str(b.near_boundary_steps),
        "--stop-loss-weight",
        str(b.stop_loss_weight),
        "--stop-positive-weight",
        str(b.stop_positive_weight),
        "--near-boundary-stop-weight",
        str(b.near_boundary_stop_weight),
        "--unsuccessful-rank-weight",
        str(b.unsuccessful_rank_weight),
        "--unsuccessful-stop-weight",
        str(b.unsuccessful_stop_weight),
        "--stop-confirmation-window",
        str(b.stop_confirmation_window),
        "--steps",
        str(selected_steps),
        "--batch-size",
        str(selected_batch),
        "--learning-rate",
        str(b.learning_rate),
        "--weight-decay",
        str(b.weight_decay),
        "--warmup-steps",
        str(min(b.warmup_steps, selected_steps)),
        "--gradient-clip-norm",
        str(b.gradient_clip_norm),
        "--validate-steps",
        str(min(b.save_steps, selected_steps)),
        "--seed",
        str(b.training_seed),
        "--resume",
    )
    return CommandSpec(argv, config.root, "Train and freeze B unified selector")


def build_b_server_command(config: ProjectConfig, seed: int = 7) -> CommandSpec:
    """Serve frozen A1 with the frozen B selector."""

    a1 = config.robocerebra_posttrain
    b = config.robocerebra_selector
    return CommandSpec(
        (
            "/usr/bin/env",
            f"PYTHONPATH={config.root / 'src'}:{config.isaac_gr00t_dir}",
            str(config.isaac_gr00t_dir / ".venv" / "bin" / "python"),
            "-m",
            "unitree_gr00t.b_server",
            "--checkpoint",
            str(a1.checkpoint_dir),
            "--selector-checkpoint",
            str(b.checkpoint_dir),
            "--model-revision",
            a1.training_dataset_revision,
            "--embodiment",
            config.robocerebra.embodiment,
            "--device",
            config.model.device,
            "--host",
            config.model.bind_host,
            "--port",
            str(config.model.port),
            "--seed",
            str(seed),
        ),
        config.root,
        "Start the frozen B GR00T-RC plus selector server",
    )


def build_b_eval_command(
    config: ProjectConfig,
    *,
    task_types: list[str],
    case_names: list[str],
    trials: int,
    execution_horizon: int,
    seed: int,
    output_dir: str | Path,
    trace_images: bool = True,
    resume: bool = False,
) -> CommandSpec:
    """Build B's continuous adaptive selector evaluation command."""

    spec = build_a2_eval_command(
        config,
        task_types=task_types,
        case_names=case_names,
        trials=trials,
        execution_horizon=execution_horizon,
        seed=seed,
        output_dir=output_dir,
        trace_images=trace_images,
        resume=resume,
    )
    argv = list(spec.argv)
    argv[argv.index("unitree_gr00t.a2_eval")] = "unitree_gr00t.b_eval"
    argv[argv.index("--experiment-id") + 1] = config.robocerebra_selector.experiment_id
    argv[argv.index("--variant") + 1] = config.robocerebra_selector.variant
    argv.extend(
        (
            "--selector-checkpoint",
            str(config.robocerebra_selector.checkpoint_dir),
            "--method",
            config.robocerebra_selector.method,
            "--paper",
            config.robocerebra_selector.paper,
            "--stop-confirmation-window",
            str(config.robocerebra_selector.stop_confirmation_window),
        )
    )
    return CommandSpec(tuple(argv), config.root, "Run adaptive B on RoboCerebra")


def run_or_preview(spec: CommandSpec, execute: bool) -> int:
    print(f"{spec.description}:\n{spec.display()}")
    if not execute:
        print("\nPreview only. Add --execute to run it.")
        return 0
    if not spec.cwd.is_dir():
        raise FileNotFoundError(f"Upstream checkout not found: {spec.cwd}")
    completed = subprocess.run(spec.argv, cwd=spec.cwd, check=False)
    return completed.returncode
