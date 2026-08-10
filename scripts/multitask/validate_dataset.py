#!/usr/bin/env python3
"""Read real samples from every G1 fruit dataset with GR00T's N1.7 loader."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "unitree_rl_groot"))

from unitree_rl_groot.groot.multitask import G1_FRUIT_TASKS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "datasets/g1_fruits_multitask_hf")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/groot/g1_fruits_multitask_config.py",
    )
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    modality = runpy.run_path(str(args.config.resolve()))["config"]
    validated = 0
    for task in G1_FRUIT_TASKS:
        dataset_path = args.root / task.dataset_directory
        loader = LeRobotEpisodeLoader(dataset_path=str(dataset_path.resolve()), modality_configs=modality)
        count = min(args.samples, len(loader.episode_lengths))
        if count < 1:
            raise RuntimeError(f"GR00T loader returned no episodes for {dataset_path}")
        for index in range(count):
            episode = loader[index]
            if episode.empty:
                raise RuntimeError(f"empty sample {index} in {dataset_path}")
        validated += count
        print(f"validated {count} {task.object_name} episodes")
    print(f"GR00T loader validated {validated} episodes across one four-task training mixture")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
