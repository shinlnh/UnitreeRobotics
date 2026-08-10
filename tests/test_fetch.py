from __future__ import annotations

import json

import numpy as np
import pytest
from unitree_rl_groot.groot.fetch import FetchEvent, FetchExecutive, FetchMission, FetchPhase

from scripts.sonic.audit_fetch_dataset import audit
from scripts.sonic.policy_client import MessageSerializer


def test_fetch_executive_completes_all_observable_phases() -> None:
    executive = FetchExecutive(FetchMission("red can", "table", "delivery bin"))
    events = (
        FetchEvent.OBJECT_FOUND,
        FetchEvent.PICKUP_POSE_REACHED,
        FetchEvent.PREGRASP_ALIGNED,
        FetchEvent.GRASP_CLOSED,
        FetchEvent.OBJECT_LIFTED,
        FetchEvent.DESTINATION_REACHED,
        FetchEvent.OBJECT_RELEASED,
        FetchEvent.DELIVERY_CONFIRMED,
    )
    for event in events:
        executive.advance(event)
    assert executive.phase is FetchPhase.COMPLETE
    assert executive.done
    assert "complete" in executive.prompt


def test_fetch_executive_recovery_is_bounded() -> None:
    executive = FetchExecutive(FetchMission("bottle", "shelf", "tray"), max_retries=1)
    assert executive.advance(FetchEvent.FAILURE) is FetchPhase.RECOVER
    assert executive.advance(FetchEvent.RETRY_READY) is FetchPhase.SEARCH
    executive.advance(FetchEvent.FAILURE)
    assert executive.failed
    with pytest.raises(RuntimeError, match="terminated"):
        executive.advance(FetchEvent.RETRY_READY)


def test_sonic_fetch_dataset_contract(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    features = {
        "observation.images.ego_view": {"dtype": "video", "shape": [480, 640, 3]},
        "action.motion_token": {"dtype": "float64", "shape": [64]},
        "teleop.left_hand_joints": {"dtype": "float32", "shape": [7]},
        "teleop.right_hand_joints": {"dtype": "float32", "shape": [7]},
    }
    (meta / "info.json").write_text(json.dumps({"fps": 50, "features": features}))
    (meta / "modality.json").write_text(
        json.dumps(
            {
                "action": {
                    "motion_token": {"original_key": "action.motion_token"},
                    "left_hand_joints": {"original_key": "teleop.left_hand_joints"},
                    "right_hand_joints": {"original_key": "teleop.right_hand_joints"},
                }
            }
        )
    )
    (meta / "episodes.jsonl").write_text('{"episode_index": 0}\n')
    (meta / "tasks.jsonl").write_text('{"task": "fetch the red can"}\n')
    video = dataset / "videos/chunk-000/ego/episode_000000.mp4"
    parquet = dataset / "data/chunk-000/episode_000000.parquet"
    video.parent.mkdir(parents=True)
    parquet.parent.mkdir(parents=True)
    video.touch()
    parquet.touch()

    report, failures = audit(dataset, min_episodes=1, min_prompts=1)
    assert not failures
    assert report["ready_for_finetune"]
    assert report["action_dimensions"]["total"] == 78


def test_sonic_fetch_dataset_rejects_velocity_only_actions(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"fps": 50, "features": {}}))
    (meta / "modality.json").write_text(json.dumps({"action": {}}))
    (meta / "episodes.jsonl").write_text('{"episode_index": 0}\n')
    (meta / "tasks.jsonl").write_text('{"task": "fetch"}\n')

    _, failures = audit(dataset, min_episodes=1, min_prompts=1)
    assert any("motion_token" in failure for failure in failures)


def test_official_apple_to_plate_dataset_contract(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    meta.mkdir(parents=True)
    features = {
        "observation.images.ego_view": {"dtype": "video", "shape": [480, 640, 3]},
        "observation.state": {"dtype": "float32", "shape": [43]},
        "action": {"dtype": "float32", "shape": [43]},
        "action.navigate_command": {"dtype": "float32", "shape": [3]},
        "action.base_height_command": {"dtype": "float32", "shape": [1]},
    }
    (meta / "info.json").write_text(json.dumps({"robot_type": "unitree_g1", "fps": 30, "features": features}))
    (meta / "modality.json").write_text(
        json.dumps(
            {
                "action": {
                    "navigate_command": {"original_key": "action.navigate_command"},
                    "base_height_command": {"original_key": "action.base_height_command"},
                }
            }
        )
    )
    (meta / "episodes.jsonl").write_text('{"episode_index": 0}\n')
    (meta / "tasks.jsonl").write_text('{"task": "move the apple to the plate"}\n')
    (dataset / "new_embodiment_config_defaults.py").write_text("config = {}\n")
    video = dataset / "videos/chunk-000/ego/episode_000000.mp4"
    parquet = dataset / "data/chunk-000/episode_000000.parquet"
    video.parent.mkdir(parents=True)
    parquet.parent.mkdir(parents=True)
    video.touch()
    parquet.touch()

    report, failures = audit(dataset, min_episodes=1, min_prompts=1)
    assert not failures
    assert report["embodiment"] == "NEW_EMBODIMENT"
    assert report["action_dimensions"]["navigation"] == 3


def test_lightweight_policy_client_numeric_wire_contract() -> None:
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    decoded = MessageSerializer.from_bytes(MessageSerializer.to_bytes({"action": expected}))
    np.testing.assert_array_equal(decoded["action"], expected)


def test_lightweight_policy_client_rejects_pickle_arrays() -> None:
    unsafe = np.asarray([{"command": "never deserialize this"}], dtype=object)
    with pytest.raises(TypeError, match="object-dtype"):
        MessageSerializer.to_bytes({"action": unsafe})
