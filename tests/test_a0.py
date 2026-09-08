import json
from pathlib import Path

import numpy as np
import pytest

from unitree_gr00t.a0 import (
    A0ContractError,
    RemotePolicyClient,
    build_policy_observation,
    discover_cases,
    inspect_checkpoint,
    load_goal,
    load_goal_steps,
    parse_task_description,
    to_libero_action,
    unpack_action_chunk,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _checkpoint(tmp_path: Path, horizon: int = 16) -> Path:
    root = tmp_path / "checkpoint"
    root.mkdir()
    _write_json(root / "config.json", {"action_horizon": 40})
    _write_json(root / "embodiment_id.json", {"libero_sim": 2})
    _write_json(root / "statistics.json", {})
    _write_json(
        root / "model.safetensors.index.json",
        {"weight_map": {"layer.weight": "model-00001-of-00001.safetensors"}},
    )
    (root / "model-00001-of-00001.safetensors").write_bytes(b"fixture")
    _write_json(
        root / "processor_config.json",
        {
            "processor_kwargs": {
                "modality_configs": {
                    "libero_sim": {
                        "video": {"modality_keys": ["image", "wrist_image"]},
                        "state": {
                            "modality_keys": [
                                "x",
                                "y",
                                "z",
                                "roll",
                                "pitch",
                                "yaw",
                                "gripper",
                            ]
                        },
                        "action": {
                            "modality_keys": [
                                "x",
                                "y",
                                "z",
                                "roll",
                                "pitch",
                                "yaw",
                                "gripper",
                            ],
                            "delta_indices": list(range(horizon)),
                        },
                        "language": {"modality_keys": ["annotation.human.action.task_description"]},
                    }
                }
            }
        },
    )
    return root


def test_checkpoint_contract_uses_embodiment_horizon_not_model_max(tmp_path: Path) -> None:
    contract = inspect_checkpoint(_checkpoint(tmp_path))
    assert contract.embodiment == "libero_sim"
    assert contract.embodiment_id == 2
    assert contract.action_horizon == 16
    assert contract.action_keys[-1] == "gripper"


def test_checkpoint_contract_rejects_missing_weight_shard(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    (checkpoint / "model-00001-of-00001.safetensors").unlink()
    with pytest.raises(A0ContractError, match="missing weight shards"):
        inspect_checkpoint(checkpoint)


def test_case_discovery_and_full_task_prompt(tmp_path: Path) -> None:
    case = tmp_path / "Ideal" / "case2"
    case.mkdir(parents=True)
    (case / "scene.bddl").write_text("fixture", encoding="utf-8")
    (case / "task_description.txt").write_text(
        "Task: Put both objects away.\n"
        "Step: Pick up object\n"
        "[0, 10]\n"
        "Step: Put object in box\n"
        "[10, 20]\n",
        encoding="utf-8",
    )
    _write_json(
        case / "goal.json",
        {
            "object_1": [
                {"state_pair": ["Pick", "object_1"], "task_step": 0},
                {"state_pair": ["In", "object_1", "box_1"], "task_step": 1},
            ]
        },
    )

    cases = discover_cases(tmp_path, ["Ideal"], ["case2"])
    description = parse_task_description(case / "task_description.txt")
    goal = load_goal(case / "goal.json")
    goal_steps = load_goal_steps(case / "goal.json")
    assert [(value.task_type, value.case_name) for value in cases] == [("Ideal", "case2")]
    assert description.instruction == "Put both objects away."
    assert description.steps == ("Pick up object", "Put object in box")
    assert description.start_indices == (0, 10)
    assert goal["object_1"][0][0] == "pick"
    assert goal_steps == {"object_1": [0, 1]}

    disturbed = discover_cases(tmp_path, ["Random_Disturbance"], ["case2"])
    assert disturbed[0].task_type == "Random_Disturbance"
    assert disturbed[0].path == case


def test_policy_observation_and_action_conversion_match_libero_contract() -> None:
    raw = {
        "agentview_image": np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3),
        "robot0_eye_in_hand_image": np.zeros((3, 4, 3), dtype=np.uint8),
        "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.4, -0.4], dtype=np.float32),
    }
    observation = build_policy_observation(raw, "perform the whole task", np)
    assert observation["video.image"].shape == (1, 1, 3, 4, 3)
    assert (
        observation["video.image"][0, 0, 0, 0].tolist() == raw["agentview_image"][-1, -1].tolist()
    )
    assert observation["state.gripper"].shape == (1, 1, 2)
    assert observation["annotation.human.action.task_description"] == ["perform the whole task"]

    response = {
        f"action.{key}": np.full((1, 16, 1), index / 10, dtype=np.float32)
        for index, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw", "gripper"))
    }
    chunk = unpack_action_chunk(response, np)
    assert chunk.shape == (16, 7)
    assert chunk[0].tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert to_libero_action(chunk[0], np)[-1] == -1.0


def test_remote_policy_client_preserves_adaptive_options_and_info() -> None:
    client = object.__new__(RemotePolicyClient)
    captured = {}

    def call(endpoint: str, data: object) -> object:
        captured["endpoint"] = endpoint
        captured["data"] = data
        return [{"action.x": "chunk"}, {"b_selector": {"candidate": 3}}]

    client.call = call  # type: ignore[method-assign]
    action, info = client.get_action_with_info(
        {"video.image": "frame"}, {"b_selector": {"max_prefix": 8}}
    )
    assert action == {"action.x": "chunk"}
    assert info["b_selector"]["candidate"] == 3
    assert captured["endpoint"] == "get_action"
    assert captured["data"]["options"]["b_selector"]["max_prefix"] == 8
    client.reset({"episode_seed": 9})
    assert captured["endpoint"] == "reset"
    assert captured["data"] == {"options": {"episode_seed": 9}}
