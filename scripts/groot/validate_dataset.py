#!/usr/bin/env python3
"""Load several windows through GR00T's real dataset loader."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/groot/g1_navigation_config.py",
    )
    args = parser.parse_args()
    namespace = runpy.run_path(str(args.config.resolve()))
    dataset = LeRobotEpisodeLoader(
        dataset_path=str(args.dataset.resolve()),
        modality_configs=namespace["g1_navigation_config"],
    )
    count = min(args.samples, len(dataset.episode_lengths))
    if count <= 0:
        raise RuntimeError("dataset loader returned no samples")
    for index in range(count):
        episode = dataset[index]
        if episode.empty:
            raise RuntimeError(f"empty sample at index {index}")
    print(f"GR00T loader validated {count} episodes from {args.dataset.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
