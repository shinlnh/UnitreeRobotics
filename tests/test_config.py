from pathlib import Path

from unitree_gr00t.config import load_config
from unitree_gr00t.tasks import load_tasks, select_tasks

ROOT = Path(__file__).resolve().parents[1]


def test_project_config_resolves_paths() -> None:
    config = load_config(ROOT / "configs" / "project.toml")
    assert config.root == ROOT
    assert config.model.embodiment == "UNITREE_G1_SONIC"
    assert config.sonic.action_horizon == 40
    assert config.isaac_gr00t_dir == ROOT / ".upstream" / "Isaac-GR00T-N1.7"
    assert config.robocerebra_dir == ROOT / ".upstream" / "RoboCerebra"
    assert config.robocerebra.action_horizon == 16
    assert config.robocerebra.fixed_execution_horizons == (8, 16)
    assert config.robocerebra_posttrain.experiment_id == "A1"
    assert config.robocerebra_posttrain.expected_training_manifest_rows == 1000
    assert len(config.robocerebra_posttrain.training_manifest_sha256) == 64
    assert config.robocerebra_posttrain.expected_training_episodes == 995
    assert config.robocerebra_posttrain.micro_batch_size == 2
    assert config.robocerebra_posttrain.gradient_accumulation_steps == 16
    assert config.robocerebra_hierarchy.experiment_id == "A2"
    assert config.robocerebra_hierarchy.variant == "GR00T-RC-fixed-hierarchy"
    assert config.robocerebra_hierarchy.subgoal_horizon_steps == 150
    assert config.robocerebra_selector.experiment_id == "B"
    assert config.robocerebra_selector.action_horizon == 16
    assert config.robocerebra_selector.context_width == 2048
    assert config.robocerebra_selector.stop_confirmation_window == 2
    assert config.robocerebra_selector.checkpoint_dir == (
        ROOT / "checkpoints" / "robocerebra" / "GR00T-RC-SparkVLA-selector"
    )
    assert config.robocerebra_retry.experiment_id == "B-retry"
    assert config.robocerebra_retry.max_retries_per_subtask == 1
    assert config.robocerebra_retry.preserve_global_step_budget
    assert not config.robocerebra_retry.failure_detector
    assert not config.robocerebra_retry.recovery_memory
    assert not config.robocerebra_retry.recovery_policy


def test_task_catalog_and_selection() -> None:
    tasks = load_tasks(ROOT / "configs" / "tasks.toml")
    assert {task.kind for task in tasks} == {"pick", "place", "stack"}
    selected = select_tasks(tasks, "pick_blue_cube,stack_red_on_blue")
    assert [task.id for task in selected] == ["pick_blue_cube", "stack_red_on_blue"]
