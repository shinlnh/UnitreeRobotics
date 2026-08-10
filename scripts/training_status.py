#!/usr/bin/env python3
"""Print a compact status report for the newest RSL-RL experiment run."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = "unitree_g1_rough_robust"
METRICS = {
    "mean_episode_length": "Train/mean_episode_length",
    "mean_reward": "Train/mean_reward",
    "success_rate": "Metrics/success_rate",
    "xy_tracking_error_mps": "Metrics/base_velocity/error_vel_xy",
    "yaw_tracking_error_radps": "Metrics/base_velocity/error_vel_yaw",
    "base_contact_rate": "Episode_Termination/base_contact",
    "terrain_level": "Curriculum/terrain_levels",
    "action_std": "Policy/mean_std",
    "fps": "Perf/total_fps",
    "learning_rate": "Loss/learning_rate",
    "value_loss": "Loss/value",
    "surrogate_loss": "Loss/surrogate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def natural_checkpoint_key(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def newest_run(experiment: str) -> Path:
    root = PROJECT_ROOT / "logs" / "rsl_rl" / experiment
    candidates = [path.parent for path in root.glob("*/events.out.tfevents.*")]
    if not candidates:
        raise FileNotFoundError(f"no TensorBoard runs found below {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def configured_iterations(run: Path) -> int | None:
    agent_path = run / "params" / "agent.yaml"
    if not agent_path.is_file():
        return None
    match = re.search(r"^max_iterations:\s*(\d+)\s*$", agent_path.read_text(), re.MULTILINE)
    return int(match.group(1)) if match else None


def main() -> int:
    args = parse_args()
    run = newest_run(args.experiment)
    accumulator = EventAccumulator(str(run))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags()["scalars"])
    anchor_tag = "Perf/total_fps"
    if anchor_tag not in scalar_tags:
        raise RuntimeError(f"run has no completed iterations yet: {run}")
    anchors = accumulator.Scalars(anchor_tag)
    current_iteration = anchors[-1].step
    first_iteration = anchors[0].step
    additional_iterations = configured_iterations(run)
    final_iteration = (
        first_iteration + additional_iterations - 1 if additional_iterations is not None else None
    )
    window_start = anchors[max(0, len(anchors) - 101)]
    iteration_delta = anchors[-1].step - window_start.step
    seconds_per_iteration = (
        (anchors[-1].wall_time - window_start.wall_time) / iteration_delta if iteration_delta else None
    )
    remaining = max(0, final_iteration - current_iteration) if final_iteration is not None else None
    eta_seconds = (
        remaining * seconds_per_iteration if remaining is not None and seconds_per_iteration else None
    )

    checkpoints = sorted(run.glob("model_*.pt"), key=natural_checkpoint_key)
    report: dict[str, object] = {
        "run": str(run.relative_to(PROJECT_ROOT)),
        "iteration": current_iteration,
        "final_iteration": final_iteration,
        "progress": current_iteration / final_iteration if final_iteration else None,
        "seconds_per_iteration": seconds_per_iteration,
        "eta_seconds": eta_seconds,
        "latest_checkpoint": (str(checkpoints[-1].relative_to(PROJECT_ROOT)) if checkpoints else None),
    }
    for name, tag in METRICS.items():
        if tag in scalar_tags:
            value = accumulator.Scalars(tag)[-1].value
            report[name] = value if math.isfinite(value) else str(value)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        eta = "unknown" if eta_seconds is None else f"{eta_seconds / 60:.1f} min"
        progress = (
            "unknown" if final_iteration is None else f"{100 * current_iteration / final_iteration:.1f}%"
        )
        print(f"Run:        {report['run']}")
        print(f"Iteration:  {current_iteration}/{final_iteration} ({progress})")
        print(f"Checkpoint: {report['latest_checkpoint']}")
        print(f"ETA:        {eta}")
        for name in METRICS:
            if name in report:
                print(f"{name:22s} {report[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
