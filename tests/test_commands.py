from pathlib import Path

import pytest

from unitree_gr00t.commands import (
    build_deploy_command,
    build_server_command,
    build_train_command,
)
from unitree_gr00t.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_train_command_uses_n17_sonic_contract(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "project.toml")
    spec = build_train_command(config, tmp_path / "dataset", tmp_path / "checkpoint")
    command = list(spec.argv)
    assert "nvidia/GR00T-N1.7-3B" in command
    assert "UNITREE_G1_SONIC" in command
    assert "gr00t/configs/data/embodiment_configs.py" in command


def test_server_rejects_base_checkpoint() -> None:
    config = load_config(ROOT / "configs" / "project.toml")
    with pytest.raises(ValueError, match="base model"):
        build_server_command(config, config.model.base_model)


def test_deploy_uses_matching_sonic_variant() -> None:
    config = load_config(ROOT / "configs" / "project.toml")
    spec = build_deploy_command(config, "sim", "pick up the apple", record=False)
    rendered = spec.display()
    assert "--sim" in spec.argv
    assert "policy/sonic_v1_1/model" in spec.argv
    assert "policy/sonic_v1_1/observation_config.yaml" in spec.argv
    assert "--no-data-exporter" in spec.argv
    assert "pick up the apple" in rendered
