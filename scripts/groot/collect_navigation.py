#!/usr/bin/env python3
"""Collect vision-conditioned G1 navigation demonstrations in randomized scenes."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import unitree_rl_groot.tasks  # noqa: F401
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
from packaging import version
from unitree_rl_groot.groot.command_filter import VelocityCommandFilter
from unitree_rl_groot.groot.dataset import RawNavigationEpisode
from unitree_rl_groot.groot.isaac_runtime import (
    load_low_level_policy,
    place_navigation_layout,
    read_navigation_inputs,
    resolve_checkpoint,
    robot_planar_pose,
    set_velocity_command,
)
from unitree_rl_groot.groot.navigation import NavigationLayoutSampler, WaypointFollower, plan_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Unitree-G1-GR00T-Navigation")
    parser.add_argument("--checkpoint", default="latest", help="PPO .pt path, latest, or best")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "datasets/raw/g1_navigation")
    parser.add_argument("--episodes", type=int, default=200, help="Number of successful episodes to save")
    parser.add_argument("--max-attempts", type=int, default=600)
    parser.add_argument("--episode-seconds", type=float, default=18.0)
    parser.add_argument("--goal-hold-seconds", type=float, default=1.6)
    parser.add_argument("--dataset-fps", type=float, default=10.0)
    parser.add_argument("--obstacle-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])
    args, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra_args]
    return args


def _next_episode_index(directory: Path) -> int:
    indices: list[int] = []
    for path in directory.glob("episode_*.npz"):
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return max(indices, default=-1) + 1


def main() -> int:
    args = parse_args()
    positive_values = (
        args.episodes,
        args.max_attempts,
        args.episode_seconds,
        args.goal_hold_seconds,
        args.dataset_fps,
    )
    if min(positive_values) <= 0:
        raise ValueError("episode counts, durations, and dataset FPS must be positive")
    if not 0 <= args.obstacle_count <= 8:
        raise ValueError("--obstacle-count must be between 0 and 8")

    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", play_mode=True)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_index = _next_episode_index(args.output_dir)
    layout_sampler = NavigationLayoutSampler(obstacle_count=args.obstacle_count, seed=args.seed)

    with launch_simulation(env_cfg, args):
        raw_env = gym.make(args.task, cfg=env_cfg)
        checkpoint = resolve_checkpoint(agent_cfg, args.checkpoint, project_root=PROJECT_ROOT)
        env, runner, policy = load_low_level_policy(raw_env, agent_cfg, checkpoint)
        rsl_version = version.parse(metadata.version("rsl-rl-lib"))
        dt = float(env.unwrapped.step_dt)
        collection_interval = round(1.0 / (dt * args.dataset_fps))
        if collection_interval <= 0:
            raise ValueError("--dataset-fps cannot exceed low-level control frequency")
        low_level_steps = round(args.episode_seconds / dt)
        hold_steps = round(args.goal_hold_seconds / dt)
        command_filter = VelocityCommandFilter()
        print(f"[INFO] PPO checkpoint: {checkpoint}")
        print(f"[INFO] Writing perception-conditioned episodes to {args.output_dir.resolve()}")

        saved = 0
        attempts = 0
        try:
            while saved < args.episodes and attempts < args.max_attempts:
                attempts += 1
                obs, _ = env.reset()
                command_filter.reset()
                done_mask = torch.ones(1, dtype=torch.long, device=env.unwrapped.device)
                policy.reset(done_mask)
                start_xy, start_yaw = robot_planar_pose(env)
                layout = layout_sampler.sample(start_xy, start_yaw)
                path = plan_path(layout)
                follower = WaypointFollower(path)
                place_navigation_layout(env, layout)

                frames: list[np.ndarray] = []
                states: list[np.ndarray] = []
                commands: list[np.ndarray] = []
                timestamps: list[float] = []
                reached_at: int | None = None
                failed = False

                for step in range(low_level_steps):
                    position, yaw = robot_planar_pose(env)
                    if reached_at is None and follower.reached_goal(position):
                        reached_at = step
                    target = (
                        np.zeros(3, dtype=np.float32)
                        if reached_at is not None
                        else follower.command(position, yaw)
                    )
                    command = command_filter.step(target, dt)
                    set_velocity_command(env, command)
                    with torch.inference_mode():
                        action = policy(obs)
                        obs, _, dones, _ = env.step(action)
                        if rsl_version >= version.parse("4.0.0"):
                            policy.reset(dones)
                        else:
                            runner.alg.policy.reset(dones)
                    if step % collection_interval == 0:
                        rgb, proprio, _ = read_navigation_inputs(env, layout.instruction)
                        frames.append(rgb)
                        states.append(proprio)
                        commands.append(command.copy())
                        timestamps.append(len(timestamps) / args.dataset_fps)
                    if bool(dones[0]):
                        failed = reached_at is None
                        break
                    if reached_at is not None and step - reached_at >= hold_steps:
                        break

                success = reached_at is not None and not failed
                if not success or len(frames) < 16:
                    reason = "fall/contact" if failed else "goal timeout"
                    final_xy, _ = robot_planar_pose(env)
                    final_distance = float(np.linalg.norm(final_xy - layout.goal_xy))
                    print(
                        f"[SKIP] attempt {attempts}: {reason}; frames={len(frames)} "
                        f"goal_distance={final_distance:.2f}m"
                    )
                    continue
                episode = RawNavigationEpisode(
                    rgb=np.stack(frames),
                    state=np.stack(states),
                    action=np.stack(commands),
                    timestamp=np.asarray(timestamps, dtype=np.float32),
                    language=layout.instruction,
                    goal_xy=layout.goal_xy,
                    obstacles=layout.obstacle_array(),
                    success=True,
                )
                destination = args.output_dir / f"episode_{episode_index:06d}.npz"
                episode.save(destination)
                saved += 1
                episode_index += 1
                print(
                    f"[SAVE] {destination.name}: {saved}/{args.episodes}; "
                    f"frames={len(frames)} obstacles={len(layout.obstacles)} attempts={attempts}"
                )
        except KeyboardInterrupt:
            print("[INFO] Collection interrupted; completed episodes are intact.")
        finally:
            env.close()

        # AppLauncher closes the process when this context exits. Fail here so an
        # incomplete collection cannot be mistaken for a successful dataset run.
        if saved < args.episodes:
            raise RuntimeError(f"saved {saved}/{args.episodes} successful episodes after {attempts} attempts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
