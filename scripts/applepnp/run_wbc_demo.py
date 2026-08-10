#!/usr/bin/env python3
"""Drive the official Apple-to-Plate RoboCasa task through GR00T and GEAR WBC."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import decoupled_wbc.control.envs.robocasa.sync_env  # noqa: E402, F401
from decoupled_wbc.control.utils.n1_utils import WholeBodyControlWrapper  # noqa: E402

from scripts.sonic.policy_client import PolicyClient  # noqa: E402

TASK = "gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc"
STATE_NAMES = ("left_leg", "right_leg", "waist", "left_arm", "right_arm", "left_hand", "right_hand")
ACTION_NAMES = (
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "waist",
    "navigate_command",
    "base_height_command",
)


def policy_observation(observation: dict) -> dict[str, np.ndarray]:
    result = {"video.ego_view": np.asarray(observation["video.ego_view"], dtype=np.uint8)}
    result.update(
        {f"state.{name}": np.asarray(observation[f"state.{name}"], dtype=np.float32) for name in STATE_NAMES}
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5550)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--policy-hz", type=float, default=30.0)
    args = parser.parse_args()

    renderer = "mujoco" if args.headless else "mjviewer"
    env = gym.make(
        TASK,
        onscreen=not args.headless,
        offscreen=True,
        renderer=renderer,
        camera_names=["robot0_oak_egoview"],
        camera_heights=[480],
        camera_widths=[640],
        randomize_cameras=False,
        enable_waist=True,
        enable_gravity_compensation=True,
        gravity_compensation_joints=["arms"],
        control_freq=int(args.control_hz),
        disable_env_checker=True,
    )
    env = WholeBodyControlWrapper(
        env,
        {
            "interface": "sim",
            "enable_waist": True,
            "enable_onscreen": not args.headless,
            "enable_offscreen": True,
            "verbose": False,
        },
    )
    client = PolicyClient(args.host, args.port, timeout_ms=120_000)
    try:
        if not client.ping():
            raise RuntimeError(f"ApplePnP server is not reachable at {args.host}:{args.port}")
        observation, _ = env.reset(seed=args.seed)
        client.reset({"seed": args.seed})
        chunk: dict[str, np.ndarray] | None = None
        chunk_step = 0
        chunk_control_steps = math.ceil(16 * args.control_hz / args.policy_hz)
        for step in range(args.steps):
            if chunk is None or chunk_step >= chunk_control_steps:
                chunk, info = client.get_action(policy_observation(observation))
                chunk = {name: np.asarray(chunk[f"action.{name}"], dtype=np.float32) for name in ACTION_NAMES}
                chunk_step = 0
                print(
                    f"GR00T chunk {info['call_index']} ready in {info['inference_seconds']:.3f}s",
                    flush=True,
                )
            action_index = min(int(chunk_step * args.policy_hz / args.control_hz), 15)
            action = {f"action.{name}": values[action_index] for name, values in chunk.items()}
            observation, reward, terminated, truncated, _ = env.step(action)
            chunk_step += 1
            success = bool(env.unwrapped.is_success().get("task", False))
            if step % int(args.control_hz) == 0:
                nav = action["action.navigate_command"]
                print(f"step={step} reward={reward:.3f} nav={nav.round(3)} success={success}")
            if not args.headless:
                env.render()
            if success:
                print(f"Apple-to-Plate succeeded at control step {step}.")
                return 0
            if terminated or truncated:
                print(f"Episode ended at control step {step} without task success.")
                return 1
        print(f"Reached {args.steps} control steps without task success.")
        return 1
    finally:
        client.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
