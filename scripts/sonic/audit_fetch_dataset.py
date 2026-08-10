#!/usr/bin/env python3
"""Audit supported Unitree G1 fetch datasets before GR00T post-training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SONIC_REQUIRED_FEATURES = {
    "observation.images.ego_view": ("video", [480, 640, 3]),
    "action.motion_token": ("float64", [64]),
    "teleop.left_hand_joints": ("float32", [7]),
    "teleop.right_hand_joints": ("float32", [7]),
}
SONIC_REQUIRED_ACTION_MODALITIES = {
    "motion_token": "action.motion_token",
    "left_hand_joints": "teleop.left_hand_joints",
    "right_hand_joints": "teleop.right_hand_joints",
}

APPLE_TO_PLATE_REQUIRED_FEATURES = {
    "observation.images.ego_view": ("video", [480, 640, 3]),
    "observation.state": ("float32", [43]),
    "action": ("float32", [43]),
    "action.navigate_command": ("float32", [3]),
    "action.base_height_command": ("float32", [1]),
}
APPLE_TO_PLATE_REQUIRED_ACTION_MODALITIES = {
    "navigate_command": "action.navigate_command",
    "base_height_command": "action.base_height_command",
}


def detect_profile(dataset: Path, info: dict[str, Any] | None = None) -> str:
    """Return the registered embodiment contract represented by ``dataset``."""
    if info is None:
        info = _json(dataset / "meta/info.json")
    features = info.get("features", {})
    if "action.motion_token" in features:
        return "UNITREE_G1_SONIC"
    if (
        info.get("robot_type") == "unitree_g1"
        and features.get("observation.state", {}).get("shape") == [43]
        and features.get("action", {}).get("shape") == [43]
    ):
        return "NEW_EMBODIMENT"
    # Preserve the useful SONIC-specific failure messages for malformed input.
    return "UNITREE_G1_SONIC"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "datasets/sonic/g1_fetch_clean")
    parser.add_argument("--min-episodes", type=int, default=50)
    parser.add_argument("--min-prompts", type=int, default=1)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        records.append(value)
    return records


def audit(dataset: Path, *, min_episodes: int, min_prompts: int) -> tuple[dict[str, Any], list[str]]:
    if min(min_episodes, min_prompts) <= 0:
        raise ValueError("minimum counts must be positive")
    meta = dataset / "meta"
    required_files = [meta / name for name in ("info.json", "modality.json", "episodes.jsonl", "tasks.jsonl")]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        return {"dataset": str(dataset), "missing_files": missing}, [
            f"missing required file: {p}" for p in missing
        ]

    info = _json(meta / "info.json")
    modality = _json(meta / "modality.json")
    episodes = _jsonl(meta / "episodes.jsonl")
    tasks = _jsonl(meta / "tasks.jsonl")
    failures: list[str] = []
    profile = detect_profile(dataset, info)
    if profile == "NEW_EMBODIMENT":
        required_features = APPLE_TO_PLATE_REQUIRED_FEATURES
        required_action_modalities = APPLE_TO_PLATE_REQUIRED_ACTION_MODALITIES
        expected_fps = 30
        action_dimensions = {"joint_positions": 43, "navigation": 3, "base_height": 1}
        config_path = dataset / "new_embodiment_config_defaults.py"
        if not config_path.is_file():
            failures.append(f"missing NVIDIA modality config: {config_path}")
    else:
        required_features = SONIC_REQUIRED_FEATURES
        required_action_modalities = SONIC_REQUIRED_ACTION_MODALITIES
        expected_fps = 50
        action_dimensions = {"motion_token": 64, "left_hand": 7, "right_hand": 7, "total": 78}

    features = info.get("features", {})
    for key, (dtype, shape) in required_features.items():
        feature = features.get(key)
        if not isinstance(feature, dict):
            failures.append(f"missing feature {key}")
            continue
        if feature.get("dtype") != dtype:
            failures.append(f"{key} dtype is {feature.get('dtype')!r}, expected {dtype!r}")
        if list(feature.get("shape", [])) != shape:
            failures.append(f"{key} shape is {feature.get('shape')!r}, expected {shape!r}")

    action_modalities = modality.get("action", {})
    for key, original_key in required_action_modalities.items():
        entry = action_modalities.get(key)
        if not isinstance(entry, dict) or entry.get("original_key") != original_key:
            failures.append(f"action modality {key} must map to {original_key}")

    prompts = {
        str(record.get("task", record.get("task_description", ""))).strip()
        for record in tasks
        if str(record.get("task", record.get("task_description", ""))).strip()
    }
    videos = list((dataset / "videos").rglob("*.mp4")) if (dataset / "videos").is_dir() else []
    parquets = list((dataset / "data").rglob("*.parquet")) if (dataset / "data").is_dir() else []
    if len(episodes) < min_episodes:
        failures.append(f"only {len(episodes)} episodes; require at least {min_episodes}")
    if len(prompts) < min_prompts:
        failures.append(f"only {len(prompts)} unique prompts; require at least {min_prompts}")
    if not videos:
        failures.append("no ego-view MP4 files found")
    if not parquets:
        failures.append("no trajectory Parquet files found")
    if len(videos) != len(episodes):
        failures.append(f"found {len(videos)} videos for {len(episodes)} episodes")
    if len(parquets) != len(episodes):
        failures.append(f"found {len(parquets)} Parquet files for {len(episodes)} episodes")
    if info.get("fps") != expected_fps:
        failures.append(f"dataset fps is {info.get('fps')!r}, expected {expected_fps}")

    report = {
        "dataset": str(dataset.resolve()),
        "embodiment": profile,
        "episodes": len(episodes),
        "unique_prompts": len(prompts),
        "video_files": len(videos),
        "parquet_files": len(parquets),
        "fps": info.get("fps"),
        "action_dimensions": action_dimensions,
        "failures": failures,
        "ready_for_finetune": not failures,
    }
    return report, failures


def main() -> int:
    args = parse_args()
    report, failures = audit(args.dataset, min_episodes=args.min_episodes, min_prompts=args.min_prompts)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return int(args.strict and bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
