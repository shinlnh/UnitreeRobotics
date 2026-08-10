#!/usr/bin/env python3
"""Convert Isaac Lab ``.npz`` episodes into GR00T/LeRobot v2.1.

Run this script inside the separately pinned GR00T environment. The converter
refuses to modify a non-empty output directory so a partial or good dataset is
never silently overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "source" / "unitree_rl_groot"
sys.path.insert(0, str(PACKAGE_ROOT))

from unitree_rl_groot.groot.dataset import RawNavigationEpisode, numeric_stats  # noqa: E402

CHUNK_SIZE = 1_000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    """Write RGB frames as an MP4 accepted by the GR00T torchcodec loader."""

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create an MP4; install a build with FFmpeg support")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"video encoder produced no data: {path}")


def _load_episodes(raw_dir: Path) -> list[RawNavigationEpisode]:
    paths = sorted(raw_dir.glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no episode_*.npz files found in {raw_dir}")
    episodes = [RawNavigationEpisode.load(path) for path in paths]
    reference_shape = episodes[0].rgb.shape[1:]
    state_dimension = episodes[0].state.shape[1]
    for index, episode in enumerate(episodes[1:], start=1):
        if episode.rgb.shape[1:] != reference_shape:
            raise ValueError(
                f"episode {index} image shape changed: {episode.rgb.shape[1:]} != {reference_shape}"
            )
        if episode.state.shape[1] != state_dimension:
            raise ValueError(
                f"episode {index} state dimension changed: {episode.state.shape[1]} != {state_dimension}"
            )
    return episodes


def convert(raw_dir: Path, output_dir: Path, fps: int) -> None:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    episodes = _load_episodes(raw_dir)
    height, width, channels = episodes[0].rgb.shape[1:]
    state_dimension = episodes[0].state.shape[1]
    tasks: list[str] = []
    task_indices: dict[str, int] = {}
    episode_rows: list[dict[str, Any]] = []
    navigation_layout_rows: list[dict[str, Any]] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    global_index = 0

    for episode_index, episode in enumerate(episodes):
        task_index = task_indices.setdefault(episode.language, len(task_indices))
        if task_index == len(tasks):
            tasks.append(episode.language)
        chunk = episode_index // CHUNK_SIZE
        data_file = output_dir / DATA_PATH.format(episode_chunk=chunk, episode_index=episode_index)
        video_file = output_dir / VIDEO_PATH.format(
            episode_chunk=chunk,
            episode_index=episode_index,
            video_key="observation.images.ego_view",
        )
        data_file.parent.mkdir(parents=True, exist_ok=True)
        length = episode.action.shape[0]
        frame_index = np.arange(length, dtype=np.int64)
        table = pd.DataFrame(
            {
                "observation.state": list(episode.state),
                "action": list(episode.action),
                "timestamp": frame_index.astype(np.float32) / np.float32(fps),
                "frame_index": frame_index,
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "index": np.arange(global_index, global_index + length, dtype=np.int64),
                "task_index": np.full(length, task_index, dtype=np.int64),
            }
        )
        table.to_parquet(data_file, index=False, engine="pyarrow")
        _write_video(video_file, episode.rgb, fps)
        episode_rows.append({"episode_index": episode_index, "tasks": [episode.language], "length": length})
        navigation_layout_rows.append(
            {
                "episode_index": episode_index,
                "goal_xy": episode.goal_xy.tolist() if episode.goal_xy is not None else None,
                "obstacles": episode.obstacles.tolist() if episode.obstacles is not None else None,
                "success": episode.success,
            }
        )
        all_states.append(episode.state)
        all_actions.append(episode.action)
        all_timestamps.append(table["timestamp"].to_numpy())
        global_index += length

    info = {
        "codebase_version": "v2.1",
        "robot_type": "unitree_g1_navigation",
        "total_episodes": len(episodes),
        "total_frames": global_index,
        "total_tasks": len(tasks),
        "total_videos": len(episodes),
        "total_chunks": (len(episodes) - 1) // CHUNK_SIZE + 1,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": DATA_PATH,
        "video_path": VIDEO_PATH,
        "features": {
            "observation.images.ego_view": {
                "dtype": "video",
                "shape": [height, width, channels],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.fps": fps,
                    "video.codec": "mpeg4",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dimension],
                "names": [f"proprio_{index}" for index in range(state_dimension)],
            },
            "action": {
                "dtype": "float32",
                "shape": [3],
                "names": ["vx_body_mps", "vy_body_mps", "yaw_rate_radps"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    modality = {
        "state": {"proprio": {"start": 0, "end": state_dimension}},
        "action": {"navigate_command": {"start": 0, "end": 3}},
        "video": {"ego_view": {"original_key": "observation.images.ego_view"}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    stats = {
        "observation.state": numeric_stats(np.concatenate(all_states)),
        "action": numeric_stats(np.concatenate(all_actions)),
        "timestamp": numeric_stats(np.concatenate(all_timestamps)),
    }
    _write_json(meta_dir / "info.json", info)
    _write_json(meta_dir / "modality.json", modality)
    _write_json(meta_dir / "stats.json", stats)
    _write_jsonl(meta_dir / "episodes.jsonl", episode_rows)
    _write_jsonl(meta_dir / "navigation_layouts.jsonl", navigation_layout_rows)
    _write_jsonl(
        meta_dir / "tasks.jsonl",
        [{"task_index": index, "task": task} for index, task in enumerate(tasks)],
    )
    print(f"Converted {len(episodes)} episodes / {global_index} frames to {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    convert(args.raw_dir.resolve(), args.output_dir.resolve(), args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
