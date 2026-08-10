#!/usr/bin/env python3
"""Run either the generalist or post-trained GR00T G1 policy through GEAR WBC."""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "source" / "unitree_rl_groot"))

from unitree_rl_groot.groot.multitask import TASK_BY_OBJECT, resolve_fruit_task  # noqa: E402

from scripts.multitask.g1_fruit_env import ensure_registered  # noqa: E402
from scripts.sonic.policy_client import PolicyClient  # noqa: E402

JOINT_STATE_NAMES = (
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
)
BASE_STATE_NAMES = ("left_hand", "right_hand", "left_arm", "right_arm", "waist")
POSTTRAIN_ACTIONS = JOINT_STATE_NAMES
BASE_ACTIONS = (
    "left_wrist_eef_9d",
    "right_wrist_eef_9d",
    "left_hand",
    "right_hand",
    "left_arm",
    "right_arm",
    "waist",
    "base_height_command",
    "navigate_command",
)
POSTTRAIN_EXECUTED_ACTIONS = ("left_arm", "left_hand", "right_arm", "right_hand", "waist")
BASE_EXECUTED_ACTIONS = (
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
    "waist",
    "base_height_command",
    "navigate_command",
)


def _pose7_to_eef9d(pose: np.ndarray) -> np.ndarray:
    """Convert Pinocchio's XYZ+WXYZ wrist pose to GR00T XYZ+rot6d."""

    pose = np.asarray(pose, dtype=np.float64)
    matrix = Rotation.from_quat(pose[3:7], scalar_first=True).as_matrix()
    return np.concatenate((pose[:3], matrix[:2].reshape(6))).astype(np.float32)


def policy_observation(
    observation: dict,
    prompt: str,
    *,
    contract: str,
    video_history: deque[np.ndarray],
) -> dict:
    """Translate WBC observations to the selected N1.7 sim interface."""

    current_image = np.asarray(observation["video.ego_view"], dtype=np.uint8)
    if contract == "base-real-g1":
        # REAL_G1 was pretrained with image deltas [-20, 0], or roughly one
        # second between the two images at its source rate.  The deque spans
        # one control-loop second and duplicates the reset image on startup.
        previous_image = video_history[0] if video_history else current_image
        result: dict[str, object] = {
            "video.ego_view": np.stack((previous_image, current_image), axis=0)[None],
            "annotation.human.task_description": [prompt],
        }
        wrist_pose = np.asarray(observation["wrist_pose"], dtype=np.float32)
        result["state.left_wrist_eef_9d"] = _pose7_to_eef9d(wrist_pose[:7])[None, None]
        result["state.right_wrist_eef_9d"] = _pose7_to_eef9d(wrist_pose[7:])[None, None]
        state_names = BASE_STATE_NAMES
    else:
        result = {
            "video.rs_view": current_image[None, None],
            "annotation.human.task_description": [prompt],
        }
        state_names = JOINT_STATE_NAMES
    for name in state_names:
        result[f"state.{name}"] = np.asarray(observation[f"state.{name}"], dtype=np.float32)[None, None]
    return result


def unbatch_action_chunk(action: dict, action_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name in action_names:
        key = f"action.{name}"
        values = np.asarray(action[key], dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != 1:
            raise ValueError(f"{key} must have shape (1, horizon, dof), got {values.shape}")
        result[name] = values[0]
    horizons = {values.shape[0] for values in result.values()}
    if len(horizons) != 1:
        raise ValueError(f"GR00T returned inconsistent action horizons: {sorted(horizons)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    command = parser.add_mutually_exclusive_group()
    command.add_argument("--instruction", help="English or Vietnamese request naming one supported fruit")
    command.add_argument("--object", choices=tuple(TASK_BY_OBJECT), help="canonical target shortcut")
    parser.add_argument(
        "--raw-language",
        action="store_true",
        help="send exact phrasing to the post-trained policy instead of its canonical prompt",
    )
    parser.add_argument(
        "--policy-contract",
        choices=("base-real-g1", "posttrain-43d"),
        default="base-real-g1",
        help="N1.7 generalist base contract, or the jointly post-trained 43-DoF contract",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5550)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--policy-hz", type=float, default=20.0)
    parser.add_argument("--execution-horizon", type=int, default=8)
    args = parser.parse_args()

    user_instruction = args.instruction or args.object or "apple"
    task = TASK_BY_OBJECT[args.object] if args.object else resolve_fruit_task(user_instruction)
    use_raw_language = args.raw_language or args.policy_contract == "base-real-g1"
    prompt = user_instruction if use_raw_language and args.instruction else task.prompt
    task_ids = ensure_registered()

    # The base contract can predict navigation; the post-trained 43-DoF
    # dataset is stationary.  In both modes GEAR WBC owns balance and converts
    # the learned upper-body/nav targets into safe low-level G1 actions.
    from decoupled_wbc.control.utils.n1_utils import WholeBodyControlWrapper

    renderer = "mujoco" if args.headless else "mjviewer"
    env = gym.make(
        task_ids[task.object_name],
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
            raise RuntimeError(f"multitask GR00T server is not reachable at {args.host}:{args.port}")
        observation, _ = env.reset(seed=args.seed)
        client.reset({"seed": args.seed})
        print(f"User command : {user_instruction}")
        print(f"Scene target : {task.object_name}")
        print(f"GR00T prompt : {prompt}")
        print(f"Policy mode  : {args.policy_contract}")

        chunk: dict[str, np.ndarray] | None = None
        control_index = 0
        executed_horizon = 0
        history_gap = max(1, round(args.control_hz))
        video_history: deque[np.ndarray] = deque(maxlen=history_gap + 1)
        video_history.append(np.asarray(observation["video.ego_view"], dtype=np.uint8).copy())
        action_names = BASE_ACTIONS if args.policy_contract == "base-real-g1" else POSTTRAIN_ACTIONS
        executed_actions = (
            BASE_EXECUTED_ACTIONS if args.policy_contract == "base-real-g1" else POSTTRAIN_EXECUTED_ACTIONS
        )
        for step in range(args.steps):
            if chunk is None or control_index >= math.ceil(
                executed_horizon * args.control_hz / args.policy_hz
            ):
                raw_chunk, info = client.get_action(
                    policy_observation(
                        observation,
                        prompt,
                        contract=args.policy_contract,
                        video_history=video_history,
                    )
                )
                chunk = unbatch_action_chunk(raw_chunk, action_names)
                model_horizon = next(iter(chunk.values())).shape[0]
                executed_horizon = min(args.execution_horizon, model_horizon)
                if executed_horizon < 1:
                    raise ValueError("--execution-horizon must be positive")
                control_index = 0
                latency = info.get("inference_seconds") if isinstance(info, dict) else None
                suffix = f" ({latency:.3f}s)" if isinstance(latency, (float, int)) else ""
                print(f"GR00T predicted {model_horizon} steps; executing {executed_horizon}{suffix}")

            action_index = min(int(control_index * args.policy_hz / args.control_hz), executed_horizon - 1)
            wbc_action = {f"action.{name}": chunk[name][action_index] for name in executed_actions}
            observation, reward, terminated, truncated, _ = env.step(wbc_action)
            video_history.append(np.asarray(observation["video.ego_view"], dtype=np.uint8).copy())
            control_index += 1
            success = bool(env.unwrapped.is_success().get("task", False))
            if step % max(int(args.control_hz), 1) == 0:
                print(f"step={step} reward={float(reward):.3f} success={success}")
            if not args.headless:
                env.render()
            if success:
                print(f"SUCCESS: {task.object_name} reached the plate at control step {step}.")
                return 0
            if terminated or truncated:
                print(f"Episode ended at control step {step} without success.")
                return 1
        print(f"Reached {args.steps} control steps without success.")
        return 1
    finally:
        client.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
