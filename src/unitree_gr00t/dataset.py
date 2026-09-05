"""Validation for GEAR-SONIC LeRobot datasets before expensive fine-tuning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetIssue:
    level: str
    message: str


@dataclass(frozen=True)
class DatasetReport:
    path: Path
    issues: tuple[DatasetIssue, ...]
    episodes: int
    tasks: int
    parquet_files: int
    video_files: int

    @property
    def valid(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)


def _read_json(path: Path, issues: list[DatasetIssue]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("top-level value is not an object")
        return value
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        issues.append(DatasetIssue("error", f"Cannot parse {path.name}: {exc}"))
        return {}


def _count_jsonl(path: Path, issues: list[DatasetIssue]) -> int:
    count = 0
    _line_number = 0
    try:
        with path.open(encoding="utf-8") as handle:
            for _line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                json.loads(line)
                count += 1
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(DatasetIssue("error", f"Invalid {path.name} near line {_line_number}: {exc}"))
    return count


def validate_dataset(path: str | Path) -> DatasetReport:
    root = Path(path).expanduser().resolve()
    issues: list[DatasetIssue] = []
    if not root.is_dir():
        return DatasetReport(
            root, (DatasetIssue("error", "Dataset directory does not exist"),), 0, 0, 0, 0
        )

    required = [
        root / "meta" / "info.json",
        root / "meta" / "modality.json",
        root / "meta" / "episodes.jsonl",
        root / "meta" / "tasks.jsonl",
        root / "data",
        root / "videos" / "observation.images.ego_view",
    ]
    for item in required:
        if not item.exists():
            issues.append(DatasetIssue("error", f"Missing required path: {item.relative_to(root)}"))

    info_path = root / "meta" / "info.json"
    modality_path = root / "meta" / "modality.json"
    info = _read_json(info_path, issues) if info_path.is_file() else {}
    modality = _read_json(modality_path, issues) if modality_path.is_file() else {}

    features = info.get("features", {})
    required_features = {
        "observation.images.ego_view",
        "observation.state",
        "observation.projected_gravity",
        "action.motion_token",
        "teleop.left_hand_joints",
        "teleop.right_hand_joints",
    }
    if isinstance(features, dict):
        for key in sorted(required_features - features.keys()):
            issues.append(DatasetIssue("error", f"info.json is missing feature {key}"))
    else:
        issues.append(DatasetIssue("error", "info.json features must be an object"))

    expected_modality_keys = {
        "state": {
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "projected_gravity",
        },
        "action": {"motion_token", "left_hand_joints", "right_hand_joints"},
        "video": {"ego_view"},
        "annotation": {"human.task_description"},
    }
    for section, keys in expected_modality_keys.items():
        actual = modality.get(section, {})
        if not isinstance(actual, dict):
            issues.append(
                DatasetIssue("error", f"modality.json section {section} must be an object")
            )
            continue
        for key in sorted(keys - actual.keys()):
            issues.append(DatasetIssue("error", f"modality.json {section} is missing {key}"))

    fps = info.get("fps")
    if fps is not None and int(fps) != 50:
        issues.append(
            DatasetIssue("warning", f"Dataset fps is {fps}; SONIC collection normally uses 50 Hz")
        )

    episodes_path = root / "meta" / "episodes.jsonl"
    tasks_path = root / "meta" / "tasks.jsonl"
    episodes = _count_jsonl(episodes_path, issues) if episodes_path.is_file() else 0
    tasks = _count_jsonl(tasks_path, issues) if tasks_path.is_file() else 0
    parquet_files = len(list((root / "data").glob("*.parquet"))) if (root / "data").is_dir() else 0
    video_dir = root / "videos" / "observation.images.ego_view"
    video_files = len(list(video_dir.glob("*.mp4"))) if video_dir.is_dir() else 0

    if episodes == 0:
        issues.append(DatasetIssue("error", "Dataset has no episodes"))
    if tasks == 0:
        issues.append(DatasetIssue("error", "Dataset has no task prompts"))
    if parquet_files == 0:
        issues.append(DatasetIssue("error", "Dataset has no parquet shards"))
    if video_files == 0:
        issues.append(DatasetIssue("error", "Dataset has no ego-view MP4 files"))

    return DatasetReport(root, tuple(issues), episodes, tasks, parquet_files, video_files)
