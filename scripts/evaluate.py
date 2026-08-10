#!/usr/bin/env python3
"""Evaluate a trained G1 PPO checkpoint with command-tracking metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import unitree_rl_groot.tasks  # noqa: F401
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
from unitree_rl_groot.groot.isaac_runtime import load_low_level_policy, resolve_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Unitree-G1-Velocity-Flat-Robust")
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=1_024)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--output", type=Path, default=None)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=None)
    args, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra_args]
    return args


def main() -> int:
    args = parse_args()
    if min(args.num_envs, args.episodes, args.max_steps) <= 0:
        raise ValueError("environment, episode, and step counts must be positive")
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", play_mode=False)
    env_cfg.scene.num_envs = args.num_envs
    if args.device is not None:
        env_cfg.sim.device = args.device

    with launch_simulation(env_cfg, args):
        raw_env = gym.make(args.task, cfg=env_cfg)
        checkpoint = resolve_checkpoint(agent_cfg, args.checkpoint, project_root=PROJECT_ROOT)
        env, _, policy = load_low_level_policy(raw_env, agent_cfg, checkpoint)
        obs = env.get_observations()
        episode_return = torch.zeros(args.num_envs, device=env.unwrapped.device)
        episode_length = torch.zeros(args.num_envs, device=env.unwrapped.device)
        returns: list[float] = []
        lengths: list[float] = []
        falls = 0
        xy_error_sum = 0.0
        yaw_error_sum = 0.0
        samples = 0

        try:
            for _ in range(args.max_steps):
                command = env.unwrapped.command_manager.get_command("base_velocity")
                robot = env.unwrapped.scene["robot"]
                xy_error_sum += (
                    torch.linalg.vector_norm(command[:, :2] - robot.data.root_lin_vel_b.torch[:, :2], dim=1)
                    .sum()
                    .item()
                )
                yaw_error_sum += torch.abs(command[:, 2] - robot.data.root_ang_vel_b.torch[:, 2]).sum().item()
                samples += args.num_envs
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, reward, dones, extras = env.step(actions)
                    policy.reset(dones)
                episode_return += reward
                episode_length += 1
                done_ids = dones.nonzero(as_tuple=False).flatten()
                if len(done_ids):
                    remaining = args.episodes - len(returns)
                    selected = done_ids[:remaining]
                    returns.extend(episode_return[selected].detach().cpu().tolist())
                    lengths.extend(episode_length[selected].detach().cpu().tolist())
                    timeouts = extras.get("time_outs")
                    if timeouts is not None:
                        falls += int((~timeouts[selected].bool()).sum().item())
                    episode_return[done_ids] = 0.0
                    episode_length[done_ids] = 0.0
                if len(returns) >= args.episodes:
                    break
        finally:
            env.close()

        # Kit's app close terminates the process, so reporting must happen before
        # leaving the launch context.
        if not returns:
            raise RuntimeError("no episode completed before --max-steps; increase the limit")
        result = {
            "task": args.task,
            "checkpoint": str(checkpoint),
            "episodes": len(returns),
            "mean_return": float(np.mean(returns)),
            "mean_episode_steps": float(np.mean(lengths)),
            "mean_xy_tracking_error_mps": xy_error_sum / samples,
            "mean_yaw_tracking_error_radps": yaw_error_sum / samples,
            "fall_rate": falls / len(returns),
        }
        rendered = json.dumps(result, indent=2)
        print(rendered)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
