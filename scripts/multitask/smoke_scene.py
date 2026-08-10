#!/usr/bin/env python3
"""Build and step each local G1 multi-fruit scene without loading GR00T."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.multitask.g1_fruit_env import ensure_registered  # noqa: E402

UPPER_GROUPS = ("left_arm", "left_hand", "right_arm", "right_hand", "waist")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    task_ids = ensure_registered()

    from decoupled_wbc.control.utils.n1_utils import WholeBodyControlWrapper

    for object_name, task_id in task_ids.items():
        env = gym.make(
            task_id,
            onscreen=False,
            offscreen=True,
            renderer="mujoco",
            camera_names=["robot0_oak_egoview"],
            camera_heights=[480],
            camera_widths=[640],
            randomize_cameras=False,
            enable_waist=True,
            control_freq=50,
            disable_env_checker=True,
        )
        env = WholeBodyControlWrapper(
            env,
            {
                "interface": "sim",
                "enable_waist": True,
                "enable_onscreen": False,
                "enable_offscreen": True,
                "verbose": False,
            },
        )
        try:
            observation, _ = env.reset(seed=0)
            assert observation["video.ego_view"].shape == (480, 640, 3)
            assert observation["annotation.human.task_description"]
            hold = {
                f"action.{group}": np.asarray(observation[f"state.{group}"], dtype=np.float32)
                for group in UPPER_GROUPS
            }
            for _ in range(args.steps):
                observation, *_ = env.step(hold)
            print(f"PASS {object_name}: {task_id}")
        finally:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
