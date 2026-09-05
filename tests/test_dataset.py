import json
from pathlib import Path

from unitree_gr00t.dataset import validate_dataset


def _write_valid_dataset(root: Path) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "data").mkdir()
    video_dir = root / "videos" / "observation.images.ego_view"
    video_dir.mkdir(parents=True)
    features = {
        key: {"dtype": "float32", "shape": [1]}
        for key in (
            "observation.images.ego_view",
            "observation.state",
            "observation.projected_gravity",
            "action.motion_token",
            "teleop.left_hand_joints",
            "teleop.right_hand_joints",
        )
    }
    (root / "meta" / "info.json").write_text(
        json.dumps({"fps": 50, "features": features}), encoding="utf-8"
    )
    modality = {
        "state": {
            key: {}
            for key in (
                "left_leg",
                "right_leg",
                "waist",
                "left_arm",
                "right_arm",
                "left_hand",
                "right_hand",
                "projected_gravity",
            )
        },
        "action": {key: {} for key in ("motion_token", "left_hand_joints", "right_hand_joints")},
        "video": {"ego_view": {}},
        "annotation": {"human.task_description": {}},
    }
    (root / "meta" / "modality.json").write_text(json.dumps(modality), encoding="utf-8")
    (root / "meta" / "episodes.jsonl").write_text('{"episode_index": 0}\n', encoding="utf-8")
    (root / "meta" / "tasks.jsonl").write_text('{"task_index": 0}\n', encoding="utf-8")
    (root / "data" / "train-00000.parquet").touch()
    (video_dir / "episode_000000.mp4").touch()


def test_valid_sonic_dataset(tmp_path: Path) -> None:
    _write_valid_dataset(tmp_path)
    report = validate_dataset(tmp_path)
    assert report.valid
    assert report.episodes == 1
    assert report.video_files == 1


def test_missing_dataset_is_invalid(tmp_path: Path) -> None:
    report = validate_dataset(tmp_path / "missing")
    assert not report.valid
