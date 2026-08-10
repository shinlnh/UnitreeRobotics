#!/usr/bin/env python3
"""Closed-loop GR00T→PPO evaluation on held-out randomized navigation layouts."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import unitree_rl_groot.tasks  # noqa: F401
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
from packaging import version
from unitree_rl_groot.groot.client import GrootPolicyClient
from unitree_rl_groot.groot.command_filter import ActionChunk, VelocityCommandFilter
from unitree_rl_groot.groot.isaac_runtime import (
    load_low_level_policy,
    place_navigation_layout,
    read_navigation_inputs,
    resolve_checkpoint,
    robot_planar_pose,
    set_velocity_command,
)
from unitree_rl_groot.groot.navigation import NavigationLayoutSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Unitree-G1-GR00T-Navigation")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--execution-horizon", type=int, default=4)
    parser.add_argument("--high-level-hz", type=float, default=10.0)
    parser.add_argument("--obstacle-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=10_042, help="Held-out layout seed")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/navigation_eval.json")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=None)
    args, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra_args]
    return args


def main() -> int:
    args = parse_args()
    if min(args.episodes, args.max_steps, args.execution_horizon, args.high_level_hz) <= 0:
        raise ValueError("episode/step/horizon/frequency values must be positive")
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", play_mode=True)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    if args.device is not None:
        env_cfg.sim.device = args.device
    sampler = NavigationLayoutSampler(obstacle_count=args.obstacle_count, seed=args.seed)

    success_count = 0
    falls = 0
    final_distances = []
    episode_steps = []
    with launch_simulation(env_cfg, args):
        raw_env = gym.make(args.task, cfg=env_cfg)
        checkpoint = resolve_checkpoint(agent_cfg, args.checkpoint, project_root=PROJECT_ROOT)
        env, runner, policy = load_low_level_policy(raw_env, agent_cfg, checkpoint)
        rsl_version = version.parse(metadata.version("rsl-rl-lib"))
        dt = float(env.unwrapped.step_dt)
        high_level_interval = round(1.0 / (dt * args.high_level_hz))
        command_filter = VelocityCommandFilter()
        try:
            with GrootPolicyClient(args.server_host, args.server_port) as client:
                if not client.ping():
                    raise ConnectionError("GR00T policy server is not ready")
                for episode_index in range(args.episodes):
                    obs, _ = env.reset()
                    policy.reset(torch.ones(1, dtype=torch.long, device=env.unwrapped.device))
                    command_filter.reset()
                    client.reset()
                    start_xy, start_yaw = robot_planar_pose(env)
                    layout = sampler.sample(start_xy, start_yaw)
                    place_navigation_layout(env, layout)
                    chunk: ActionChunk | None = None
                    target = np.zeros(3, dtype=np.float32)
                    succeeded = False
                    fell = False
                    step = 0
                    for step in range(1, args.max_steps + 1):
                        position, _ = robot_planar_pose(env)
                        if np.linalg.norm(position - layout.goal_xy) <= 0.5:
                            succeeded = True
                            break
                        if (step - 1) % high_level_interval == 0:
                            if chunk is None or len(chunk) == 0:
                                _, _, observation = read_navigation_inputs(env, layout.instruction)
                                action, _ = client.get_action(observation)
                                chunk = ActionChunk.from_policy_action(
                                    action,
                                    execution_horizon=args.execution_horizon,
                                )
                            target = chunk.pop()
                        command = command_filter.step(target, dt)
                        set_velocity_command(env, command)
                        with torch.inference_mode():
                            low_level_action = policy(obs)
                            obs, _, dones, _ = env.step(low_level_action)
                            if rsl_version >= version.parse("4.0.0"):
                                policy.reset(dones)
                            else:
                                runner.alg.policy.reset(dones)
                        if bool(dones[0]):
                            fell = True
                            break
                    final_xy, _ = robot_planar_pose(env)
                    final_distance = float(np.linalg.norm(final_xy - layout.goal_xy))
                    success_count += int(succeeded)
                    falls += int(fell)
                    final_distances.append(final_distance)
                    episode_steps.append(step)
                    print(
                        f"[EVAL] {episode_index + 1}/{args.episodes} "
                        f"success={succeeded} fall={fell} distance={final_distance:.2f}m steps={step}"
                    )
        finally:
            env.close()

        # Persist metrics before AppLauncher closes Kit and terminates the process.
        report = {
            "episodes": args.episodes,
            "success_rate": success_count / args.episodes,
            "fall_rate": falls / args.episodes,
            "mean_final_goal_distance_m": float(np.mean(final_distances)),
            "mean_episode_steps": float(np.mean(episode_steps)),
            "seed": args.seed,
        }
        rendered = json.dumps(report, indent=2)
        print(rendered)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
