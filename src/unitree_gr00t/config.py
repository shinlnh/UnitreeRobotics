"""Typed project configuration loaded from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UpstreamConfig:
    root: Path
    isaac_gr00t_repo: str
    isaac_gr00t_revision: str
    isaac_gr00t_dirname: str
    sonic_repo: str
    sonic_revision: str
    sonic_dirname: str
    robocerebra_repo: str
    robocerebra_revision: str
    robocerebra_dirname: str
    sparkvla_repo: str
    sparkvla_revision: str
    sparkvla_dirname: str


@dataclass(frozen=True)
class RoboCerebraConfig:
    dataset: str
    dataset_revision: str
    model: str
    model_revision: str
    checkpoint_subdir: str
    checkpoint_dir: Path
    benchmark_dir: Path
    embodiment: str
    task_types: tuple[str, ...]
    trials_per_task: int
    control_frequency_hz: int
    action_horizon: int
    fixed_execution_horizons: tuple[int, ...]
    steps_per_subtask: int
    initial_wait_steps: int
    post_success_observation_steps: int


@dataclass(frozen=True)
class RoboCerebraPosttrainConfig:
    experiment_id: str
    variant: str
    training_dataset: str
    training_dataset_revision: str
    training_manifest: Path
    training_manifest_sha256: str
    raw_training_dir: Path
    lerobot_training_dir: Path
    base_checkpoint_dir: Path
    checkpoint_dir: Path
    expected_training_manifest_rows: int
    expected_training_episodes: int
    download_workers: int
    conversion_workers: int
    dataset_fps: int
    max_steps: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    dataloader_workers: int
    learning_rate: float
    state_dropout_probability: float
    save_steps: int


@dataclass(frozen=True)
class RoboCerebraHierarchyConfig:
    experiment_id: str
    variant: str
    planner: str
    plan_source: str
    subgoal_horizon_steps: int


@dataclass(frozen=True)
class ModelConfig:
    base_model: str
    embodiment: str
    checkpoint: str
    device: str
    host: str
    bind_host: str
    port: int


@dataclass(frozen=True)
class SonicConfig:
    variant: str
    camera_host: str
    camera_port: int
    action_publish_rate: int
    action_horizon: int
    record_during_inference: bool


@dataclass(frozen=True)
class TrainingConfig:
    num_gpus: int
    max_steps: int
    global_batch_size: int
    save_steps: int


@dataclass(frozen=True)
class SafetyConfig:
    max_cartesian_delta: float
    workspace_x: tuple[float, float]
    workspace_y: tuple[float, float]
    workspace_z: tuple[float, float]
    max_episode_steps: int


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    name: str
    artifact_dir: Path
    upstream: UpstreamConfig
    robocerebra: RoboCerebraConfig
    robocerebra_posttrain: RoboCerebraPosttrainConfig
    robocerebra_hierarchy: RoboCerebraHierarchyConfig
    model: ModelConfig
    sonic: SonicConfig
    training: TrainingConfig
    safety: SafetyConfig

    @property
    def isaac_gr00t_dir(self) -> Path:
        return self.upstream.root / self.upstream.isaac_gr00t_dirname

    @property
    def sonic_dir(self) -> Path:
        return self.upstream.root / self.upstream.sonic_dirname

    @property
    def robocerebra_dir(self) -> Path:
        return self.upstream.root / self.upstream.robocerebra_dirname

    @property
    def sparkvla_dir(self) -> Path:
        return self.upstream.root / self.upstream.sparkvla_dirname


def _pair(value: Any, key: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{key} must be a two-item TOML array")
    low, high = float(value[0]), float(value[1])
    if low >= high:
        raise ValueError(f"{key} lower bound must be smaller than upper bound")
    return (low, high)


def load_config(path: str | Path = "configs/project.toml") -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Project config not found: {config_path}")

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    root = config_path.parent.parent
    project = raw["project"]
    upstream = raw["upstream"]
    robocerebra = raw["robocerebra"]
    robocerebra_posttrain = raw["robocerebra_posttrain"]
    robocerebra_hierarchy = raw["robocerebra_hierarchy"]
    model = raw["model"]
    sonic = raw["sonic"]
    training = raw["training"]
    safety = raw["safety"]

    upstream_root = Path(upstream["root"])
    if not upstream_root.is_absolute():
        upstream_root = root / upstream_root

    artifact_dir = Path(project["artifact_dir"])
    if not artifact_dir.is_absolute():
        artifact_dir = root / artifact_dir

    checkpoint_dir = Path(robocerebra["checkpoint_dir"])
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = root / checkpoint_dir

    benchmark_dir = Path(robocerebra["benchmark_dir"])
    if not benchmark_dir.is_absolute():
        benchmark_dir = root / benchmark_dir

    def project_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    return ProjectConfig(
        root=root,
        name=str(project["name"]),
        artifact_dir=artifact_dir,
        upstream=UpstreamConfig(
            root=upstream_root,
            isaac_gr00t_repo=str(upstream["isaac_gr00t_repo"]),
            isaac_gr00t_revision=str(upstream["isaac_gr00t_revision"]),
            isaac_gr00t_dirname=str(upstream.get("isaac_gr00t_dirname", "Isaac-GR00T")),
            sonic_repo=str(upstream["sonic_repo"]),
            sonic_revision=str(upstream["sonic_revision"]),
            sonic_dirname=str(upstream.get("sonic_dirname", "GR00T-WholeBodyControl")),
            robocerebra_repo=str(upstream["robocerebra_repo"]),
            robocerebra_revision=str(upstream["robocerebra_revision"]),
            robocerebra_dirname=str(upstream.get("robocerebra_dirname", "RoboCerebra")),
            sparkvla_repo=str(upstream["sparkvla_repo"]),
            sparkvla_revision=str(upstream["sparkvla_revision"]),
            sparkvla_dirname=str(upstream.get("sparkvla_dirname", "SparkVLA")),
        ),
        robocerebra=RoboCerebraConfig(
            dataset=str(robocerebra["dataset"]),
            dataset_revision=str(robocerebra["dataset_revision"]),
            model=str(robocerebra["model"]),
            model_revision=str(robocerebra["model_revision"]),
            checkpoint_subdir=str(robocerebra["checkpoint_subdir"]),
            checkpoint_dir=checkpoint_dir,
            benchmark_dir=benchmark_dir,
            embodiment=str(robocerebra["embodiment"]),
            task_types=tuple(str(value) for value in robocerebra["task_types"]),
            trials_per_task=int(robocerebra["trials_per_task"]),
            control_frequency_hz=int(robocerebra["control_frequency_hz"]),
            action_horizon=int(robocerebra["action_horizon"]),
            fixed_execution_horizons=tuple(
                int(value) for value in robocerebra["fixed_execution_horizons"]
            ),
            steps_per_subtask=int(robocerebra["steps_per_subtask"]),
            initial_wait_steps=int(robocerebra["initial_wait_steps"]),
            post_success_observation_steps=int(robocerebra["post_success_observation_steps"]),
        ),
        robocerebra_posttrain=RoboCerebraPosttrainConfig(
            experiment_id=str(robocerebra_posttrain["experiment_id"]),
            variant=str(robocerebra_posttrain["variant"]),
            training_dataset=str(robocerebra_posttrain["training_dataset"]),
            training_dataset_revision=str(robocerebra_posttrain["training_dataset_revision"]),
            training_manifest=project_path(str(robocerebra_posttrain["training_manifest"])),
            training_manifest_sha256=str(robocerebra_posttrain["training_manifest_sha256"]),
            raw_training_dir=project_path(str(robocerebra_posttrain["raw_training_dir"])),
            lerobot_training_dir=project_path(str(robocerebra_posttrain["lerobot_training_dir"])),
            base_checkpoint_dir=project_path(str(robocerebra_posttrain["base_checkpoint_dir"])),
            checkpoint_dir=project_path(str(robocerebra_posttrain["checkpoint_dir"])),
            expected_training_manifest_rows=int(
                robocerebra_posttrain["expected_training_manifest_rows"]
            ),
            expected_training_episodes=int(robocerebra_posttrain["expected_training_episodes"]),
            download_workers=int(robocerebra_posttrain["download_workers"]),
            conversion_workers=int(robocerebra_posttrain["conversion_workers"]),
            dataset_fps=int(robocerebra_posttrain["dataset_fps"]),
            max_steps=int(robocerebra_posttrain["max_steps"]),
            micro_batch_size=int(robocerebra_posttrain["micro_batch_size"]),
            gradient_accumulation_steps=int(robocerebra_posttrain["gradient_accumulation_steps"]),
            dataloader_workers=int(robocerebra_posttrain["dataloader_workers"]),
            learning_rate=float(robocerebra_posttrain["learning_rate"]),
            state_dropout_probability=float(robocerebra_posttrain["state_dropout_probability"]),
            save_steps=int(robocerebra_posttrain["save_steps"]),
        ),
        robocerebra_hierarchy=RoboCerebraHierarchyConfig(
            experiment_id=str(robocerebra_hierarchy["experiment_id"]),
            variant=str(robocerebra_hierarchy["variant"]),
            planner=str(robocerebra_hierarchy["planner"]),
            plan_source=str(robocerebra_hierarchy["plan_source"]),
            subgoal_horizon_steps=int(robocerebra_hierarchy["subgoal_horizon_steps"]),
        ),
        model=ModelConfig(
            base_model=str(model["base_model"]),
            embodiment=str(model["embodiment"]),
            checkpoint=str(model.get("checkpoint", "")),
            device=str(model["device"]),
            host=str(model["host"]),
            bind_host=str(model["bind_host"]),
            port=int(model["port"]),
        ),
        sonic=SonicConfig(
            variant=str(sonic["variant"]),
            camera_host=str(sonic["camera_host"]),
            camera_port=int(sonic["camera_port"]),
            action_publish_rate=int(sonic["action_publish_rate"]),
            action_horizon=int(sonic["action_horizon"]),
            record_during_inference=bool(sonic["record_during_inference"]),
        ),
        training=TrainingConfig(
            num_gpus=int(training["num_gpus"]),
            max_steps=int(training["max_steps"]),
            global_batch_size=int(training["global_batch_size"]),
            save_steps=int(training["save_steps"]),
        ),
        safety=SafetyConfig(
            max_cartesian_delta=float(safety["max_cartesian_delta"]),
            workspace_x=_pair(safety["workspace_x"], "safety.workspace_x"),
            workspace_y=_pair(safety["workspace_y"], "safety.workspace_y"),
            workspace_z=_pair(safety["workspace_z"], "safety.workspace_z"),
            max_episode_steps=int(safety["max_episode_steps"]),
        ),
    )
