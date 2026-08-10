#!/usr/bin/env python3
"""Audit raw navigation demonstrations before conversion or GR00T fine-tuning."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from unitree_rl_groot.groot.dataset import RawNavigationEpisode


def build_report(episodes: list[RawNavigationEpisode]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("at least one episode is required")
    actions = np.concatenate([episode.action for episode in episodes])
    goals = [episode.goal_xy for episode in episodes if episode.goal_xy is not None]
    obstacle_counts = [len(episode.obstacles) for episode in episodes if episode.obstacles is not None]
    frame_differences = []
    for episode in episodes:
        sampled = episode.rgb[:: max(1, len(episode.rgb) // 16)].astype(np.float32)
        if len(sampled) > 1:
            frame_differences.append(float(np.abs(np.diff(sampled, axis=0)).mean()))
    goal_array = np.stack(goals) if goals else None
    return {
        "episodes": len(episodes),
        "frames": int(sum(len(episode.action) for episode in episodes)),
        "successful_episodes": int(sum(episode.success is True for episode in episodes)),
        "episodes_with_layout_metadata": int(
            sum(episode.goal_xy is not None and episode.obstacles is not None for episode in episodes)
        ),
        "instruction_counts": dict(Counter(episode.language for episode in episodes)),
        "action_mean": actions.mean(axis=0).tolist(),
        "action_std": actions.std(axis=0).tolist(),
        "action_min": actions.min(axis=0).tolist(),
        "action_max": actions.max(axis=0).tolist(),
        "mean_temporal_pixel_difference": float(np.mean(frame_differences)) if frame_differences else 0.0,
        "goal_span_xy": np.ptp(goal_array, axis=0).tolist() if goal_array is not None else None,
        "obstacle_count_range": [min(obstacle_counts), max(obstacle_counts)] if obstacle_counts else None,
    }


def quality_failures(report: dict[str, Any], *, min_episodes: int) -> list[str]:
    failures = []
    if report["episodes"] < min_episodes:
        failures.append(f"episodes={report['episodes']} < {min_episodes}")
    if report["successful_episodes"] != report["episodes"]:
        failures.append("dataset contains unsuccessful expert episodes")
    if report["episodes_with_layout_metadata"] != report["episodes"]:
        failures.append("goal/obstacle metadata is incomplete")
    if len(report["instruction_counts"]) < 3:
        failures.append("fewer than three instruction paraphrases")
    action_std = np.asarray(report["action_std"])
    if np.any(action_std < np.asarray([0.08, 0.03, 0.08])):
        failures.append(f"insufficient action diversity: std={action_std.tolist()}")
    if report["mean_temporal_pixel_difference"] < 1.0:
        failures.append("camera stream has insufficient temporal variation")
    goal_span = report["goal_span_xy"]
    if goal_span is None or max(goal_span) < 2.0:
        failures.append(f"goal positions have insufficient spatial coverage: span={goal_span}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--min-episodes", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    paths = sorted(args.raw_dir.glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no raw episodes below {args.raw_dir}")
    report = build_report([RawNavigationEpisode.load(path) for path in paths])
    failures = quality_failures(report, min_episodes=args.min_episodes)
    report["quality_failures"] = failures
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
