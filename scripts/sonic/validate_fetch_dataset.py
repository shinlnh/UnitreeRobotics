#!/usr/bin/env python3
"""Load fetch episodes through the exact GR00T modality configuration used for training."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from audit_fetch_dataset import detect_profile
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader


def load_dataset_config(dataset_path: Path) -> str:
    profile = detect_profile(dataset_path)
    if profile == "NEW_EMBODIMENT":
        config_path = dataset_path / "new_embodiment_config_defaults.py"
        spec = importlib.util.spec_from_file_location("unitree_fetch_dataset_config", config_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import modality config: {config_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return "new_embodiment"
    return "unitree_g1_sonic"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    dataset_path = args.dataset.resolve()
    embodiment = load_dataset_config(dataset_path)
    loader = LeRobotEpisodeLoader(
        dataset_path=str(dataset_path),
        modality_configs=MODALITY_CONFIGS[embodiment],
    )
    count = min(args.samples, len(loader.episode_lengths))
    if count <= 0:
        raise RuntimeError("GR00T loader found no SONIC episodes")
    for index in range(count):
        episode = loader[index]
        if episode.empty:
            raise RuntimeError(f"GR00T loader returned empty episode {index}")
    print(f"GR00T {embodiment} loader validated {count} episodes from {dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
