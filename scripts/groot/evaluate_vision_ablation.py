#!/usr/bin/env python3
"""Measure whether a served GR00T checkpoint changes actions when vision is ablated."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from unitree_rl_groot.groot.client import GrootPolicyClient
from unitree_rl_groot.groot.dataset import RawNavigationEpisode
from unitree_rl_groot.groot.observations import build_navigation_observation


def _commands(action: dict, horizon: int) -> np.ndarray:
    if "navigate_command" not in action:
        raise KeyError(f"GR00T response has no navigate_command: {sorted(action)}")
    commands = np.asarray(action["navigate_command"], dtype=np.float32)
    if commands.ndim == 3:
        commands = commands[0]
    if commands.ndim != 2 or commands.shape[1] != 3:
        raise ValueError(f"unexpected navigation action shape: {commands.shape}")
    return commands[:horizon]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--execution-horizon", type=int, default=4)
    parser.add_argument("--min-action-delta", type=float, default=0.02)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.execution_horizon) <= 0:
        raise ValueError("sample count and horizon must be positive")
    episodes = [RawNavigationEpisode.load(path) for path in sorted(args.raw_dir.glob("episode_*.npz"))]
    if len(episodes) < 2:
        raise ValueError("vision ablation requires at least two episodes")
    rng = np.random.default_rng(42)
    original_black = []
    original_shuffled = []
    with GrootPolicyClient(args.server_host, args.server_port) as client:
        if not client.ping():
            raise ConnectionError("GR00T policy server is not ready")
        for _ in range(args.samples):
            episode_index = int(rng.integers(len(episodes)))
            episode = episodes[episode_index]
            frame_index = int(rng.integers(len(episode.rgb)))
            other = episodes[(episode_index + 1) % len(episodes)]
            other_index = min(frame_index, len(other.rgb) - 1)

            variants = (
                episode.rgb[frame_index],
                np.zeros_like(episode.rgb[frame_index]),
                other.rgb[other_index],
            )
            predictions = []
            for image in variants:
                client.reset()
                observation = build_navigation_observation(
                    image,
                    episode.state[frame_index],
                    episode.language,
                )
                action, _ = client.get_action(observation)
                predictions.append(_commands(action, args.execution_horizon))
            original_black.append(float(np.mean(np.abs(predictions[0] - predictions[1]))))
            original_shuffled.append(float(np.mean(np.abs(predictions[0] - predictions[2]))))
    report = {
        "samples": args.samples,
        "execution_horizon": args.execution_horizon,
        "mean_original_vs_black_action_delta": float(np.mean(original_black)),
        "mean_original_vs_shuffled_action_delta": float(np.mean(original_shuffled)),
        "min_required_delta": args.min_action_delta,
    }
    report["vision_sensitive"] = (
        min(
            report["mean_original_vs_black_action_delta"],
            report["mean_original_vs_shuffled_action_delta"],
        )
        >= args.min_action_delta
    )
    print(json.dumps(report, indent=2))
    return 1 if args.strict and not report["vision_sensitive"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
