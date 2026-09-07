from pathlib import Path

import pytest

from unitree_gr00t.commands import (
    build_a0_eval_command,
    build_a0_server_command,
    build_a1_eval_command,
    build_a1_prepare_command,
    build_a1_train_command,
    build_a2_eval_command,
    build_a2_server_command,
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


def test_a0_server_uses_original_libero_checkpoint_and_wrapper() -> None:
    config = load_config(ROOT / "configs" / "project.toml")
    spec = build_a0_server_command(config, seed=13)
    assert str(config.robocerebra.checkpoint_dir) in spec.argv
    assert "LIBERO_PANDA" in spec.argv
    assert "--use-sim-policy-wrapper" in spec.argv
    assert spec.cwd == ROOT / ".upstream" / "Isaac-GR00T-N1.7"


def test_a0_eval_is_frozen_to_declared_execution_horizons(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "project.toml")
    spec = build_a0_eval_command(
        config,
        task_types=["Ideal"],
        case_names=["case1"],
        trials=1,
        execution_horizon=16,
        seed=7,
        output_dir=tmp_path / "a0",
        resume=True,
    )
    assert "unitree_gr00t.a0_eval" in spec.argv
    assert "--model-revision" in spec.argv
    assert "--dataset-revision" in spec.argv
    assert "--resume" in spec.argv
    assert "16" in spec.argv

    with pytest.raises(ValueError, match="H8, H16"):
        build_a0_eval_command(
            config,
            task_types=["Ideal"],
            case_names=[],
            trials=1,
            execution_horizon=4,
            seed=7,
            output_dir=tmp_path / "invalid",
        )


def test_a1_commands_pin_shared_posttraining_contract(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "project.toml")
    prepare = build_a1_prepare_command(config, workers=8)
    assert "995" in prepare.argv
    assert "1000" in prepare.argv
    assert "--resume" in prepare.argv

    train = build_a1_train_command(config, max_steps=20_000)
    assert str(config.robocerebra_posttrain.base_checkpoint_dir) in train.argv
    assert "--micro-batch-size" in train.argv
    assert "--gradient-accumulation-steps" in train.argv
    assert "LIBERO_PANDA" not in train.argv  # enforced inside the provenance-locked runner

    evaluate = build_a1_eval_command(
        config,
        task_types=["Ideal"],
        case_names=["case1"],
        trials=1,
        execution_horizon=16,
        seed=7,
        output_dir=tmp_path / "a1",
    )
    assert str(config.robocerebra_posttrain.checkpoint_dir) in evaluate.argv
    assert "A1" in evaluate.argv
    assert "GR00T-RC" in evaluate.argv


def test_a2_commands_reuse_a1_checkpoint_with_separate_hierarchy(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs" / "project.toml")
    server = build_a2_server_command(config, seed=13)
    evaluate = build_a2_eval_command(
        config,
        task_types=["Ideal"],
        case_names=["case1"],
        trials=1,
        execution_horizon=16,
        seed=7,
        output_dir=tmp_path / "a2",
        resume=True,
    )

    assert str(config.robocerebra_posttrain.checkpoint_dir) in server.argv
    assert "unitree_gr00t.a2_eval" in evaluate.argv
    assert "unitree_gr00t.a0_eval" not in evaluate.argv
    assert "A2" in evaluate.argv
    assert "GR00T-RC-fixed-hierarchy" in evaluate.argv
    assert "RoboCerebra-HPE-fixed-anchor-reimplementation" in evaluate.argv
    assert "--resume" in evaluate.argv
