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
    sonic_repo: str
    sonic_revision: str


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
    model: ModelConfig
    sonic: SonicConfig
    training: TrainingConfig
    safety: SafetyConfig

    @property
    def isaac_gr00t_dir(self) -> Path:
        return self.upstream.root / "Isaac-GR00T"

    @property
    def sonic_dir(self) -> Path:
        return self.upstream.root / "GR00T-WholeBodyControl"


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

    return ProjectConfig(
        root=root,
        name=str(project["name"]),
        artifact_dir=artifact_dir,
        upstream=UpstreamConfig(
            root=upstream_root,
            isaac_gr00t_repo=str(upstream["isaac_gr00t_repo"]),
            isaac_gr00t_revision=str(upstream["isaac_gr00t_revision"]),
            sonic_repo=str(upstream["sonic_repo"]),
            sonic_revision=str(upstream["sonic_revision"]),
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
