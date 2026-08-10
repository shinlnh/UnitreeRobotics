#!/usr/bin/env python3
"""Run command -> GR00T -> whole-body G1 pick/navigate/place inside Isaac Sim."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher
from unitree_rl_groot.groot.locomanipulation import GrootN15PolicyClient, parse_commanded_fetch

ACTION_KEYS = (
    ("action.left_hand_pose", 7),
    ("action.right_hand_pose", 7),
    ("action.left_hand_joint_positions", 7),
    ("action.right_hand_joint_positions", 7),
    ("action.base_velocity", 3),
    ("action.base_height", 1),
)
POSE_STATE_KEYS = (
    "state.left_hand_pose",
    "state.right_hand_pose",
    "state.object_pose",
    "state.goal_pose",
    "state.end_fixture_pose",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INITIAL_STATE_DATASET = PROJECT_ROOT / "datasets/g1_locomanip_hf/dataset_annotated_g1_locomanip.hdf5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instruction",
        default="Hãy lấy vô lăng trên bàn đầu và mang tới bàn giao.",
        help="Supported command: fetch the steering wheel from the pickup table to the drop-off table.",
    )
    parser.add_argument("--task", default="Isaac-G1-SteeringWheel-Locomanipulation")
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=5556)
    parser.add_argument("--policy-timeout-ms", type=int, default=120_000)
    parser.add_argument("--policy-quat-format", choices=("xyzw", "wxyz"), default="wxyz")
    parser.add_argument("--initial-state-dataset", type=Path, default=DEFAULT_INITIAL_STATE_DATASET)
    parser.add_argument(
        "--demo",
        default="demo_1",
        help="Official annotated initial-state demo; demo_1 is the validated deterministic showcase.",
    )
    parser.add_argument("--max-seconds", type=float, default=50.0)
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Closed-loop rollout attempts; cycles official initial states and diffusion seeds.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Pass after at least one real GR00T inference and one Isaac Sim control step.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=0,
        help="Reset the GR00T diffusion RNG before this rollout for reproducible evaluation.",
    )
    parser.add_argument(
        "--success-mode",
        choices=("whole-table", "official-target"),
        default="whole-table",
        help="Use command-level delivery on the whole table, or Isaac Lab's smaller benchmark target box.",
    )
    parser.add_argument(
        "--delivery-stability-seconds",
        type=float,
        default=0.5,
        help="How long the released object must remain stable on the destination table.",
    )
    AppLauncher.add_app_launcher_args(parser)
    # NVIDIA's Isaac Lab 3.0 reference rollout uses CPU simulation.  Keeping
    # contact-rich physics on CPU makes the grasp reproducible while the 3B
    # GR00T policy remains on the GPU in its isolated server process.
    parser.set_defaults(visualizer=["kit"], device="cpu")
    return parser.parse_args()


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(value.detach().cpu().numpy())


def _convert_pose_quat(pose: torch.Tensor, to_format: str) -> torch.Tensor:
    if to_format == "xyzw":
        return pose
    from isaaclab.utils.math import convert_quat

    return torch.cat((pose[..., :3], convert_quat(pose[..., 3:7], to="wxyz")), dim=-1)


def _convert_action_pose_quats_to_env(action: torch.Tensor, policy_quat_format: str) -> None:
    if policy_quat_format == "xyzw":
        return
    from isaaclab.utils.math import convert_quat

    for start in (0, 7):
        action[..., start + 3 : start + 7] = convert_quat(action[..., start + 3 : start + 7], to="xyzw")


def _make_base_goal(env, input_episode_data):
    from isaaclab_mimic.locomanipulation_sdg.scene_utils import RelativePose
    from isaaclab_mimic.locomanipulation_sdg.transform_utils import transform_inv, transform_mul

    initial_input = env.load_input_data(input_episode_data, 0)
    if initial_input is None:
        raise ValueError("the selected demonstration has no initial action/state")
    relative_pose = transform_mul(transform_inv(initial_input.fixture_pose), initial_input.base_pose)
    return RelativePose(relative_pose=relative_pose, parent=env.get_end_fixture())


def _build_model_input(
    env, base_goal_pose: torch.Tensor, end_fixture_pose: torch.Tensor, policy_quat_format: str
) -> dict[str, np.ndarray]:
    from isaaclab_mimic.locomanipulation_sdg.transform_utils import transform_inv, transform_mul

    obs = env.obs_buf["policy"]
    left_hand_pose = torch.cat((obs["left_eef_pos"], obs["left_eef_quat"]), dim=-1)
    right_hand_pose = torch.cat((obs["right_eef_pos"], obs["right_eef_quat"]), dim=-1)
    hand_joints = obs["hand_joint_state"]
    base_pose_inv = transform_inv(env.get_base().get_pose())

    model_input = {
        "video.ego_view": obs["robot_pov_cam"],
        "state.left_hand_pose": transform_mul(base_pose_inv, left_hand_pose),
        "state.right_hand_pose": transform_mul(base_pose_inv, right_hand_pose),
        "state.left_hand_joint_positions": hand_joints[:, 0:7],
        "state.right_hand_joint_positions": hand_joints[:, 7:14],
        "state.object_pose": transform_mul(base_pose_inv, env.get_object().get_pose()),
        "state.goal_pose": transform_mul(base_pose_inv, base_goal_pose),
        "state.end_fixture_pose": transform_mul(base_pose_inv, end_fixture_pose),
    }
    # The published 2026-01-29 checkpoint was trained from the premade dataset,
    # whose Isaac Lab instructions explicitly require the legacy WXYZ format.
    for key in POSE_STATE_KEYS:
        model_input[key] = _convert_pose_quat(model_input[key], policy_quat_format)
    return {key: _to_numpy(value) for key, value in model_input.items()}


def _assemble_action_chunk(action_dict: dict[str, np.ndarray], device: str) -> torch.Tensor:
    chunks: list[np.ndarray] = []
    horizon: int | None = None
    for key, width in ACTION_KEYS:
        if key not in action_dict:
            raise KeyError(f"GR00T action is missing {key!r}")
        value = np.asarray(action_dict[key], dtype=np.float32)
        while value.ndim > 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim == 1:
            value = value[None, :]
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"{key} must have shape [horizon, {width}], got {value.shape}")
        horizon = value.shape[0] if horizon is None else horizon
        if value.shape[0] != horizon:
            raise ValueError("GR00T action modalities have inconsistent horizons")
        chunks.append(value)
    action = torch.as_tensor(np.concatenate(chunks, axis=-1), device=device)
    if action.shape[1] != 32 or not torch.isfinite(action).all():
        raise ValueError(f"invalid assembled action chunk: {tuple(action.shape)}")
    return action


def _action_in_world_frame(env, relative_action: torch.Tensor, policy_quat_format: str) -> torch.Tensor:
    from isaaclab_mimic.locomanipulation_sdg.transform_utils import transform_mul

    action = relative_action.clone()
    _convert_action_pose_quats_to_env(action, policy_quat_format)
    base_pose = env.get_base().get_pose()
    action[:, 0:7] = transform_mul(base_pose, action[:, 0:7])
    action[:, 7:14] = transform_mul(base_pose, action[:, 7:14])
    # Preserve the checkpoint's complete base command.  In particular, the
    # learned transition into navigation uses a fast pivot (about 2.2 rad/s);
    # clamping it desynchronizes base-relative wrist targets from locomotion.
    return action


def _phase_status(env, end_fixture_pose: torch.Tensor) -> tuple[str, str]:
    base_xy = env.get_base().get_pose()[0, :2]
    object_pose = env.get_object().get_pose()[0]
    end_xy = end_fixture_pose[0, :2]
    robot_distance = float(torch.linalg.vector_norm(object_pose[:2] - base_xy))
    destination_distance = float(torch.linalg.vector_norm(object_pose[:2] - end_xy))
    object_height = float(object_pose[2])
    if destination_distance < 1.0:
        phase = "PLACE"
    elif object_height > 0.78:
        phase = "CARRY_TO_DESTINATION"
    elif robot_distance < 0.75:
        phase = "GRASP"
    else:
        phase = "NAVIGATE_TO_OBJECT"
    detail = (
        f"robot-object={robot_distance:.2f}m object-destination={destination_distance:.2f}m "
        f"object-z={object_height:.2f}m"
    )
    return phase, detail


def _print_success_diagnostics(env) -> None:
    """Print the official Isaac Lab placement predicate without changing it."""
    term_cfg = env.termination_manager.get_term_cfg("success")
    params = dict(term_cfg.params)
    params["debug"] = True
    term_cfg.func(env, **params)


def _whole_table_delivery_candidate(env, table_pose: torch.Tensor) -> tuple[bool, str]:
    """Check command semantics: released, stable object anywhere on the drop-off tabletop."""
    from isaaclab_mimic.locomanipulation_sdg.transform_utils import transform_inv, transform_mul

    object_pose = env.get_object().get_pose()
    object_in_table = transform_mul(transform_inv(table_pose), object_pose)[0, :3]
    base_xy = env.get_base().get_pose()[0, :2]
    robot_distance = torch.linalg.vector_norm(object_pose[0, :2] - base_xy)
    object_speed = torch.linalg.vector_norm(env.scene["object"].data.root_vel_w.torch[0, :3])

    # PackingTable's useful tabletop is larger than Isaac Lab's benchmark target
    # rectangle.  Height rejects an object still held above the surface; base
    # distance rejects a grasp that has not yet been released/retracted.
    delivered = bool(
        (-0.85 < object_in_table[0] < 0.85)
        and (-0.50 < object_in_table[1] < 0.50)
        and (0.90 < object_in_table[2] < 1.08)
        and (object_speed < 0.10)
        and (robot_distance > 0.62)
    )
    detail = (
        f"table_xyz=({object_in_table[0]:+.3f},{object_in_table[1]:+.3f},"
        f"{object_in_table[2]:+.3f}) speed={object_speed:.3f}m/s "
        f"robot-object={robot_distance:.3f}m"
    )
    return delivered, detail


def main() -> int:
    args = parse_args()
    mission = parse_commanded_fetch(args.instruction)
    if args.max_seconds <= 0:
        raise ValueError("--max-seconds must be positive")
    if args.delivery_stability_seconds <= 0:
        raise ValueError("--delivery-stability-seconds must be positive")
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive")
    if not args.initial_state_dataset.is_file():
        raise FileNotFoundError(
            f"initial-state dataset is missing: {args.initial_state_dataset}; "
            "run `make download-locomanip-dataset`"
        )

    # Fail before starting Kit when the expensive VLA process is not ready.
    with GrootN15PolicyClient(
        args.policy_host, args.policy_port, timeout_ms=min(args.policy_timeout_ms, 2_000)
    ) as probe:
        if not probe.ping():
            raise RuntimeError(
                f"GR00T policy server is not ready at {args.policy_host}:{args.policy_port}; "
                "run `make serve-locomanip` first"
            )

    print(f"[COMMAND] {mission.instruction}")
    print(
        "[MISSION] steering wheel: pickup table -> drop-off table | "
        "perception=RGB+sim object pose | controller=GR00T 32D+AGILE+Pink IK"
    )

    # Camera tasks must request the rendering experience explicitly.  Isaac Lab
    # 3.0 removed the old CLI flag, so this intent is passed as a launcher kwarg.
    app_launcher = AppLauncher(args, enable_cameras=True)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_mimic.locomanipulation_sdg.envs  # noqa: F401
    from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg
    from isaaclab.utils.datasets import HDF5DatasetFileHandler
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = args.max_seconds
    env_cfg.recorders = RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
    # The SDG configuration overrides ``__post_init__`` without invoking its
    # locomanipulation parent.  Enable PhysX contact reporters explicitly so
    # the official left/right-hand contact sensors can resolve their bodies.
    env_cfg.scene.robot.spawn.activate_contact_sensors = True

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    input_dataset = HDF5DatasetFileHandler()
    input_dataset.open(str(args.initial_state_dataset))
    episode_names = list(input_dataset.get_episode_names())
    if args.demo not in episode_names:
        raise ValueError(f"unknown demo {args.demo!r}; available: {episode_names}")
    demo_start = episode_names.index(args.demo)
    demo_order = episode_names[demo_start:] + episode_names[:demo_start]
    max_steps = round(args.max_seconds / float(env.step_dt))
    inference_calls = 0
    steps_executed = 0
    started = time.perf_counter()
    success = False
    official_success = False
    required_stable_steps = max(1, round(args.delivery_stability_seconds / float(env.step_dt)))
    attempts_to_run = 1 if args.smoke else args.attempts

    try:
        with (
            GrootN15PolicyClient(
                args.policy_host, args.policy_port, timeout_ms=args.policy_timeout_ms
            ) as policy_client,
            torch.inference_mode(),
        ):
            for attempt in range(attempts_to_run):
                demo_name = demo_order[attempt % len(demo_order)]
                input_episode_data = input_dataset.load_episode(demo_name, env.device)
                env.reset_to(
                    state=input_episode_data.get_initial_state(),
                    env_ids=torch.tensor([0], device=env.device),
                    is_relative=False,
                )
                base_goal = _make_base_goal(env, input_episode_data)
                base_goal_pose = base_goal.get_pose().clone()
                end_fixture_pose = env.get_end_fixture().get_pose().clone()
                print(
                    f"[RESET {attempt + 1}/{attempts_to_run}] {args.initial_state_dataset.name}:{demo_name}",
                    flush=True,
                )

                if args.policy_seed is not None:
                    attempt_seed = args.policy_seed + attempt
                    policy_client.set_seed(attempt_seed)
                    print(f"[POLICY SEED] {attempt_seed}", flush=True)

                action_chunk: torch.Tensor | None = None
                action_index = 0
                delivery_stable_steps = 0
                attempt_ended = "time limit"

                for step in range(max_steps):
                    if not simulation_app.is_running() or simulation_app.is_exiting():
                        attempt_ended = "application closed"
                        break
                    if action_chunk is None or action_index >= action_chunk.shape[0]:
                        inference_started = time.perf_counter()
                        action_dict = policy_client.get_action(
                            _build_model_input(env, base_goal_pose, end_fixture_pose, args.policy_quat_format)
                        )
                        action_chunk = _assemble_action_chunk(action_dict, env.device)
                        action_index = 0
                        inference_calls += 1
                        phase, detail = _phase_status(env, end_fixture_pose)
                        base_command = action_chunk[:, 28:31].mean(dim=0)
                        print(
                            f"[POLICY {inference_calls:03d}] attempt={attempt + 1} phase={phase} "
                            f"horizon={action_chunk.shape[0]} "
                            f"inference={time.perf_counter() - inference_started:.2f}s "
                            f"base-cmd=({base_command[0]:+.2f},{base_command[1]:+.2f},"
                            f"{base_command[2]:+.2f}) {detail}",
                            flush=True,
                        )
                        if phase == "PLACE":
                            _print_success_diagnostics(env)

                    action = _action_in_world_frame(
                        env,
                        action_chunk[action_index : action_index + 1],
                        args.policy_quat_format,
                    )
                    _, _, terminated, timed_out, _ = env.step(action)
                    action_index += 1
                    steps_executed += 1

                    official_success = bool(env.termination_manager.get_term("success")[0])
                    if args.success_mode == "official-target":
                        success = official_success
                    else:
                        delivery_candidate, delivery_detail = _whole_table_delivery_candidate(
                            env, end_fixture_pose
                        )
                        delivery_stable_steps = delivery_stable_steps + 1 if delivery_candidate else 0
                        success = delivery_stable_steps >= required_stable_steps
                    if success:
                        suffix = (
                            "official Isaac Lab target"
                            if official_success
                            else f"whole-table delivery stable for {args.delivery_stability_seconds:.1f}s; {delivery_detail}"
                        )
                        print(
                            f"[SUCCESS] Mission completed on attempt {attempt + 1} in "
                            f"{step * env.step_dt:.1f}s simulated time ({suffix}).",
                            flush=True,
                        )
                        break
                    if bool(terminated[0]):
                        active = [
                            name
                            for name, value in env.termination_manager.get_active_iterable_terms(0)
                            if value[0]
                        ]
                        attempt_ended = f"termination {active}"
                        break
                    if bool(timed_out[0]):
                        break

                if success or args.smoke:
                    break
                print(
                    f"[RETRY] Attempt {attempt + 1}/{attempts_to_run} ended by {attempt_ended}; "
                    "resetting the simulated scene.",
                    flush=True,
                )
    finally:
        env.close()
        input_dataset.close()
        elapsed = time.perf_counter() - started
        if args.smoke and inference_calls >= 1 and steps_executed >= 1:
            print(
                f"[SMOKE PASS] {inference_calls} real GR00T inference call(s), "
                f"{steps_executed} Isaac Sim control step(s).",
                flush=True,
            )
        elif not success:
            print(
                f"[FAIL] Mission did not complete after {attempts_to_run} attempt(s), "
                f"{args.max_seconds:.1f}s simulated time each "
                f"({elapsed:.1f}s wall time, {inference_calls} policy calls).",
                flush=True,
            )
        simulation_app.close()

    if args.smoke:
        return 0 if inference_calls >= 1 and steps_executed >= 1 else 1
    if not success:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
