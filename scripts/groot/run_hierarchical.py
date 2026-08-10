#!/usr/bin/env python3
"""Run GR00T high-level navigation over a trained G1 PPO controller."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch
import unitree_rl_groot.tasks  # noqa: F401
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli
from packaging import version
from unitree_rl_groot.groot.client import GrootPolicyClient
from unitree_rl_groot.groot.command_filter import ActionChunk, VelocityCommandFilter
from unitree_rl_groot.groot.isaac_runtime import (
    load_low_level_policy,
    read_navigation_inputs,
    resolve_checkpoint,
    set_velocity_command,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Unitree-G1-GR00T-Navigation")
    parser.add_argument("--checkpoint", default="latest", help="PPO .pt path, latest, or best")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5555)
    parser.add_argument("--server-timeout-ms", type=int, default=30_000)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--execution-horizon", type=int, default=4)
    parser.add_argument("--high-level-hz", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--real-time", action="store_true")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=["kit"])
    args, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra_args]
    return args


def main() -> int:
    args = parse_args()
    if args.execution_horizon <= 0:
        raise ValueError("--execution-horizon must be positive")
    env_cfg, agent_cfg = resolve_task_config(args.task, "rsl_rl_cfg_entry_point", play_mode=True)
    env_cfg.scene.num_envs = 1
    if args.device is not None:
        env_cfg.sim.device = args.device

    with launch_simulation(env_cfg, args):
        raw_env = gym.make(args.task, cfg=env_cfg)
        checkpoint = resolve_checkpoint(agent_cfg, args.checkpoint, project_root=PROJECT_ROOT)
        env, runner, policy = load_low_level_policy(raw_env, agent_cfg, checkpoint)
        rsl_version = version.parse(metadata.version("rsl-rl-lib"))
        dt = float(env.unwrapped.step_dt)
        high_level_interval = round(1.0 / (dt * args.high_level_hz))
        if args.high_level_hz <= 0.0 or high_level_interval <= 0:
            raise ValueError("--high-level-hz must be positive and no faster than low-level control")
        command_filter = VelocityCommandFilter()
        chunk: ActionChunk | None = None
        target_command = command_filter.command
        print(f"[INFO] PPO checkpoint: {checkpoint}")
        print(f"[INFO] Low-level control: {1.0 / dt:.1f} Hz; GR00T chunk: {args.execution_horizon} steps")

        try:
            with GrootPolicyClient(
                args.server_host,
                args.server_port,
                timeout_ms=args.server_timeout_ms,
            ) as client:
                if not client.ping():
                    raise ConnectionError(
                        f"GR00T server is not ready at {args.server_host}:{args.server_port}"
                    )
                client.reset()
                obs = env.get_observations()
                step = 0
                while args.max_steps is None or step < args.max_steps:
                    started = time.perf_counter()
                    with torch.inference_mode():
                        if step % high_level_interval == 0:
                            if chunk is None or len(chunk) == 0:
                                _, _, groot_observation = read_navigation_inputs(env, args.instruction)
                                action, _ = client.get_action(groot_observation)
                                chunk = ActionChunk.from_policy_action(
                                    action,
                                    execution_horizon=args.execution_horizon,
                                )
                            target_command = chunk.pop()
                        command = command_filter.step(target_command, dt)
                        set_velocity_command(env, command)
                        low_level_action = policy(obs)
                        obs, _, dones, _ = env.step(low_level_action)
                        if rsl_version >= version.parse("4.0.0"):
                            policy.reset(dones)
                        else:
                            runner.alg.policy.reset(dones)
                        if bool(dones[0]):
                            chunk = None
                            command_filter.reset()
                            target_command = command_filter.command
                            client.reset()
                    step += 1
                    remaining = dt - (time.perf_counter() - started)
                    if args.real_time and remaining > 0.0:
                        time.sleep(remaining)
        except KeyboardInterrupt:
            pass
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
