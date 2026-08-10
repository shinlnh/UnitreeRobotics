#!/usr/bin/env python3
"""Audit all four official G1 datasets before a multi-dataset fine-tune."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "unitree_rl_groot"))

from unitree_rl_groot.groot.multitask import G1_FRUIT_TASKS  # noqa: E402

EXPECTED_GROUPS = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "left_hand": (22, 29),
    "right_arm": (29, 36),
    "right_hand": (36, 43),
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _check_group_map(value: dict) -> bool:
    return value == {key: {"start": bounds[0], "end": bounds[1]} for key, bounds in EXPECTED_GROUPS.items()}


def audit_subdataset(path: Path, expected_prompt: str) -> tuple[dict, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    required = [
        path / "meta/info.json",
        path / "meta/modality.json",
        path / "meta/episodes.jsonl",
        path / "meta/tasks.jsonl",
    ]
    missing_meta = [str(item.relative_to(path)) for item in required if not item.is_file()]
    if missing_meta:
        return {"path": str(path), "complete": False}, [f"missing {', '.join(missing_meta)}"], warnings

    info = _json(required[0])
    modality = _json(required[1])
    episodes = _jsonl(required[2])
    tasks = _jsonl(required[3])
    expected_episodes = int(info.get("total_episodes", -1))
    reported_frames = int(info.get("total_frames", -1))
    fps = int(info.get("fps", -1))

    if info.get("codebase_version") != "v2.1":
        errors.append(f"expected LeRobot v2.1, got {info.get('codebase_version')!r}")
    if fps != 20:
        errors.append(f"expected 20 Hz, got {fps}")
    for feature in ("observation.state", "action"):
        shape = info.get("features", {}).get(feature, {}).get("shape")
        if shape != [43]:
            errors.append(f"{feature} must be 43D, got {shape}")
    video = info.get("features", {}).get("observation.images.ego_view", {})
    if video.get("dtype") != "video" or video.get("shape") != [480, 640, 3]:
        errors.append("observation.images.ego_view must be a 480x640 RGB video")
    if not _check_group_map(modality.get("state", {})):
        errors.append("state joint-group slices do not match the 43-DoF G1 contract")
    if not _check_group_map(modality.get("action", {})):
        errors.append("action joint-group slices do not match the 43-DoF G1 contract")
    if modality.get("video", {}).get("rs_view", {}).get("original_key") != "observation.images.ego_view":
        errors.append("video.rs_view mapping is missing")
    if modality.get("annotation", {}).get("human.task_description", {}).get("original_key") != "task_index":
        errors.append("language annotation mapping is missing")

    if len(episodes) != expected_episodes:
        errors.append(f"metadata has {len(episodes)} episodes, expected {expected_episodes}")
    indices = [int(row.get("episode_index", -1)) for row in episodes]
    if indices != list(range(max(expected_episodes, 0))):
        errors.append("episode indices are not contiguous")
    actual_frames = sum(int(row.get("length", 0)) for row in episodes)
    if actual_frames != reported_frames:
        warnings.append(
            f"upstream info.json reports total_frames={reported_frames}, but episodes.jsonl sums to {actual_frames}"
        )
    wrong_prompts = [row.get("episode_index") for row in episodes if row.get("tasks") != [expected_prompt]]
    if wrong_prompts:
        errors.append(f"{len(wrong_prompts)} episodes do not carry the expected object prompt")

    task_prompts = {row.get("task") for row in tasks}
    catalogue = {task.prompt for task in G1_FRUIT_TASKS}
    if expected_prompt not in task_prompts:
        errors.append("tasks.jsonl does not contain this subdataset's active prompt")
    if task_prompts != catalogue:
        warnings.append("upstream tasks.jsonl differs across subdatasets for an unused prompt")
    if info.get("total_tasks") != len(task_prompts):
        warnings.append(
            f"upstream info.json reports total_tasks={info.get('total_tasks')}, but tasks.jsonl contains {len(task_prompts)}"
        )

    data_template = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    video_template = info.get(
        "video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    )
    missing_data: list[int] = []
    missing_video: list[int] = []
    for index in range(max(expected_episodes, 0)):
        fields = {
            "episode_index": index,
            "episode_chunk": index // 1000,
            "video_key": "observation.images.ego_view",
        }
        if not (path / data_template.format(**fields)).is_file():
            missing_data.append(index)
        if not (path / video_template.format(**fields)).is_file():
            missing_video.append(index)
    if missing_data:
        errors.append(f"missing {len(missing_data)} parquet episodes (first: {missing_data[:5]})")
    if missing_video:
        errors.append(f"missing {len(missing_video)} video episodes (first: {missing_video[:5]})")
    actual_videos = expected_episodes - len(missing_video)
    if info.get("total_videos") != actual_videos:
        warnings.append(
            f"upstream info.json reports total_videos={info.get('total_videos')}; complete per-episode video count is {actual_videos}"
        )

    return (
        {
            "path": str(path),
            "episodes": expected_episodes,
            "frames": actual_frames,
            "videos": actual_videos,
            "prompt": expected_prompt,
            "complete": not errors,
        },
        errors,
        warnings,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "datasets/g1_fruits_multitask_hf")
    parser.add_argument("--strict", action="store_true", help="return non-zero for incomplete data")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    reports: list[dict] = []
    all_errors: list[str] = []
    all_warnings: list[str] = []
    for task in G1_FRUIT_TASKS:
        report, errors, warnings = audit_subdataset(args.root / task.dataset_directory, task.prompt)
        reports.append({"object": task.object_name, **report})
        all_errors.extend(f"{task.dataset_directory}: {message}" for message in errors)
        all_warnings.extend(f"{task.dataset_directory}: {message}" for message in warnings)

    result = {
        "dataset_root": str(args.root.resolve()),
        "subdatasets": reports,
        "totals": {
            "tasks": len(reports),
            "episodes": sum(int(report.get("episodes", 0)) for report in reports),
            "frames": sum(int(report.get("frames", 0)) for report in reports),
        },
        "errors": all_errors,
        "warnings": all_warnings,
        "ready_for_training": not all_errors,
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload + "\n", encoding="utf-8")
    return 1 if args.strict and all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
