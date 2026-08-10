from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _versions() -> dict[str, str]:
    return dict(
        line.split("=", maxsplit=1)
        for line in (ROOT / "configs/setup/versions.env").read_text().splitlines()
        if line and not line.startswith("#")
    )


def test_reproducibility_lock_is_complete() -> None:
    lock = _versions()
    expected = {
        "ISAAC_SIM_VERSION",
        "ISAAC_LAB_VERSION",
        "ISAAC_LAB_REF",
        "RSL_RL_VERSION",
        "GR00T_RELEASE",
        "GR00T_MODEL",
        "GR00T_MODEL_REVISION",
        "GR00T_REF",
        "GR00T_FETCH_DATASET",
        "GR00T_FETCH_DATASET_REVISION",
        "GR00T_APPLEPNP_MODEL",
        "GR00T_APPLEPNP_MODEL_REVISION",
        "GR00T_G1_MULTITASK_DATASET",
        "GR00T_G1_MULTITASK_DATASET_REVISION",
        "LEAPP_REF",
        "SONIC_REF",
        "ROBOSUITE_G1_REF",
    }
    assert expected <= set(lock)
    assert len(lock["ISAAC_LAB_REF"]) == 40
    assert len(lock["GR00T_REF"]) == 40
    assert len(lock["GR00T_MODEL_REVISION"]) == 40
    assert len(lock["SONIC_REF"]) == 40
    assert len(lock["GR00T_FETCH_DATASET_REVISION"]) == 40
    assert len(lock["GR00T_APPLEPNP_MODEL_REVISION"]) == 40
    assert len(lock["GR00T_G1_MULTITASK_DATASET_REVISION"]) == 40
    assert len(lock["LEAPP_REF"]) == 40
    assert len(lock["ROBOSUITE_G1_REF"]) == 40


def test_all_project_tasks_are_registered_in_source() -> None:
    registrations = (
        ROOT / "source/unitree_rl_groot/unitree_rl_groot/tasks/locomotion/g1/__init__.py"
    ).read_text()
    assert "Unitree-G1-Velocity-Flat-Robust" in registrations
    assert "Unitree-G1-Velocity-Rough-Robust" in registrations
    assert "Unitree-G1-GR00T-Navigation" in registrations


def test_two_environment_boundary_is_preserved() -> None:
    lab_setup = (ROOT / "scripts/setup/bootstrap_isaaclab.sh").read_text()
    groot_setup = (ROOT / "scripts/setup/bootstrap_groot.sh").read_text()
    assert ".deps/IsaacLab" in lab_setup
    assert ".deps/Isaac-GR00T" in groot_setup
    assert "uv sync" in lab_setup and "uv sync" in groot_setup


def test_fetch_pipeline_uses_official_sonic_action_contract() -> None:
    finetune = (ROOT / "scripts/sonic/finetune_fetch.sh").read_text()
    audit = (ROOT / "scripts/sonic/audit_fetch_dataset.py").read_text()
    assert "UNITREE_G1_SONIC" in finetune
    assert '"total": 78' in audit
    assert "action.motion_token" in audit


def test_applepnp_pipeline_keeps_onnx_and_sonic_contracts_separate() -> None:
    policy = (ROOT / "scripts/applepnp/applepnp_policy.py").read_text()
    demo = (ROOT / "scripts/applepnp/run_wbc_demo.py").read_text()
    assert '"navigate_command"' in policy
    assert '"base_height_command"' in policy
    assert "HAND_STATE_TO_MODEL" in policy
    assert "WholeBodyControlWrapper" in demo
    assert "action.motion_token" not in policy
    assert "LMPnPAppleToPlateDC_G1_gear_wbc" in demo


def test_multitask_pipeline_trains_one_balanced_43d_policy() -> None:
    train = (ROOT / "scripts/multitask/finetune_g1_fruits.sh").read_text()
    modality = (ROOT / "configs/groot/g1_fruits_multitask_config.py").read_text()
    server = (ROOT / "scripts/multitask/serve_g1_fruits.sh").read_text()
    assert all(f"g1-pick-{name}" in train for name in ("apple", "pear", "grapes", "starfruit"))
    assert "--ds-weights-alpha 0.0" in train
    assert "--embodiment-tag NEW_EMBODIMENT" in train
    assert "UNITREE_G1_SONIC" not in train
    assert "list(range(40))" in modality
    assert '"left_leg"' in modality and '"right_hand"' in modality
    assert "--use-sim-policy-wrapper" in server
    assert 'embodiment_tag="REAL_G1"' in server
    assert 'embodiment_tag="NEW_EMBODIMENT"' in server


def test_multitask_default_demo_is_real_g1_inside_isaac() -> None:
    makefile = (ROOT / "Makefile").read_text()
    runner = (ROOT / "scripts/multitask/run_isaac_g1_fruits.py").read_text()
    bridge = (ROOT / "source/unitree_rl_groot/unitree_rl_groot/groot/isaac_multitask.py").read_text()
    assert "demo_isaac_g1_fruits.sh" in makefile
    assert "Pink IK + AGILE locomotion -> Isaac Sim" in runner
    assert '"video.ego_view"' in bridge
    assert '"navigate_command"' in bridge
    assert "POLICY_HANDS_TO_PINK" in bridge
    assert "IsaacActionSafetyFilter" in runner
    assert "torch.tensor(relative_action[None]" in runner
    assert "torch.as_tensor(relative_action[None]" not in runner
    assert "[SAFETY STOP]" in runner
    assert "simulation_app.close(exit_code=exit_code)" in runner


def test_multitask_launcher_propagates_failed_physical_rollout() -> None:
    launcher = (ROOT / "scripts/multitask/demo_isaac_g1_fruits.sh").read_text()
    assert "runner_status=${PIPESTATUS[0]}" in launcher
    assert 'exit "${runner_status}"' in launcher


def test_rough_ppo_uses_positive_gaussian_scale_parameterization() -> None:
    config = (
        ROOT / "source/unitree_rl_groot/unitree_rl_groot/tasks/locomotion/g1/agents/rsl_rl_ppo_cfg.py"
    ).read_text()
    assert 'std_type="log"' in config
    assert "BoundedGaussianDistribution" in config

    environment = (
        ROOT / "source/unitree_rl_groot/unitree_rl_groot/tasks/locomotion/g1/env_cfg.py"
    ).read_text()
    assert "gpu_found_lost_pairs_capacity = 2**26" in environment
    assert "gpu_total_aggregate_pairs_capacity = 2**22" in environment
    assert "safe_base_height_l2" in environment
    assert "safe_joint_acc_l2" in environment
