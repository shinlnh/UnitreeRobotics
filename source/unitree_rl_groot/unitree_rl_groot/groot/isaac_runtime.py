"""Isaac Lab side of the hierarchical GR00T/PPO runtime.

Imports that require Isaac Lab are deliberately local. Importing the top-level
package therefore remains possible in the standalone GR00T environment and in
unit tests.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np

from .navigation import NavigationLayout, yaw_from_quat_xyzw
from .observations import build_navigation_observation, build_proprio

NAVIGATION_OBSTACLE_COUNT = 8


def _inference_actor_state(checkpoint_payload: dict[str, Any]) -> dict[str, Any]:
    """Return an actor state compatible with the current bounded distribution.

    Flat locomotion checkpoints produced before the numerical-stability update
    store a directly optimized ``std_param``.  The current actor optimizes
    ``log_std_param`` instead.  The policy mean weights are identical, so the
    legacy scale can be converted exactly (up to the deployable lower bound)
    without retraining the locomotion policy.
    """

    if "actor_state_dict" not in checkpoint_payload:
        raise KeyError("checkpoint does not contain actor_state_dict")
    actor_state = dict(checkpoint_payload["actor_state_dict"])
    old_key = "distribution.std_param"
    new_key = "distribution.log_std_param"
    if old_key in actor_state and new_key not in actor_state:
        import torch

        std = torch.as_tensor(actor_state.pop(old_key))
        std = torch.nan_to_num(std, nan=0.8, posinf=2.0, neginf=0.05).clamp(0.05, 2.0)
        actor_state[new_key] = std.log()
    return actor_state


def resolve_checkpoint(agent_cfg: Any, checkpoint: str | None, *, project_root: Path) -> Path:
    """Resolve an explicit checkpoint or the most recent checkpoint for a run."""

    if checkpoint and checkpoint not in {"latest", "best"}:
        candidate = Path(checkpoint).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {candidate}")
        return candidate
    from isaaclab_tasks.utils import get_checkpoint_path

    log_root = project_root / "logs" / "rsl_rl" / agent_cfg.experiment_name
    pattern = r"model_.*\.pt" if checkpoint != "best" else r"model_best\.pt"
    try:
        result = get_checkpoint_path(str(log_root), agent_cfg.load_run, pattern)
    except ValueError as exc:
        raise FileNotFoundError(
            f"no PPO checkpoint found below {log_root}; train locomotion first or pass --checkpoint"
        ) from exc
    return Path(result).resolve()


def load_low_level_policy(env: Any, agent_cfg: Any, checkpoint: Path) -> tuple[Any, Any, Any]:
    """Wrap the environment, restore only the actor, and return inference policy.

    Loading only the actor is intentional: navigation uses the same deployable
    observation/action contract as locomotion, while the training-only critic
    may have a terrain scanner and therefore a different input shape.
    """

    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    installed_version = importlib.metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    payload = torch.load(checkpoint, map_location=wrapped.unwrapped.device, weights_only=False)
    actor_state = _inference_actor_state(payload)
    runner.alg.get_policy().load_state_dict(actor_state, strict=True)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)
    return wrapped, runner, policy


def set_velocity_command(env: Any, command: np.ndarray) -> None:
    """Override the built-in command generator for environment zero."""

    import torch

    value = np.asarray(command, dtype=np.float32)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("velocity command must contain three finite values")
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term.vel_command_b[0].copy_(torch.as_tensor(value, device=term.vel_command_b.device))
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[0] = False


def robot_planar_pose(env: Any) -> tuple[np.ndarray, float]:
    """Read robot ``(x, y, yaw)`` in world coordinates for expert planning and evaluation."""

    robot = env.unwrapped.scene["robot"]
    position = robot.data.root_pos_w.torch[0, :2].detach().cpu().numpy().astype(np.float32)
    quaternion = robot.data.root_quat_w.torch[0].detach().cpu().numpy()
    return position, yaw_from_quat_xyzw(quaternion)


def place_navigation_layout(env: Any, layout: NavigationLayout) -> None:
    """Move the kinematic goal and obstacle assets to a sampled layout."""

    import torch

    if len(layout.obstacles) > NAVIGATION_OBSTACLE_COUNT:
        raise ValueError(
            f"layout has {len(layout.obstacles)} obstacles but scene supports {NAVIGATION_OBSTACLE_COUNT}"
        )
    scene = env.unwrapped.scene
    device = scene["robot"].data.root_pos_w.torch.device

    def move(name: str, xyz: tuple[float, float, float]) -> None:
        asset = scene[name]
        pose = torch.tensor([[*xyz, 0.0, 0.0, 0.0, 1.0]], device=device, dtype=torch.float32)
        velocity = torch.zeros((1, 6), device=device, dtype=torch.float32)
        asset.write_root_pose_to_sim_index(root_pose=pose)
        asset.write_root_velocity_to_sim_index(root_velocity=velocity)

    for index in range(NAVIGATION_OBSTACLE_COUNT):
        if index < len(layout.obstacles):
            x, y = layout.obstacles[index].center
            position = float(x), float(y), 0.6
        else:
            position = 100.0 + index, 100.0, 0.6
        move(f"navigation_obstacle_{index}", position)
    move("navigation_goal", (float(layout.goal_xy[0]), float(layout.goal_xy[1]), 0.7))


def read_navigation_inputs(env: Any, instruction: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read synchronized RGB and G1 proprioception from the current simulation step."""

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    camera = base_env.scene["head_camera"]
    rgb = camera.data.output["rgb"].torch[0].detach().cpu().numpy()[..., :3]
    proprio = build_proprio(
        robot.data.joint_pos.torch[0].detach().cpu().numpy(),
        robot.data.joint_vel.torch[0].detach().cpu().numpy(),
        robot.data.root_lin_vel_b.torch[0].detach().cpu().numpy(),
        robot.data.root_ang_vel_b.torch[0].detach().cpu().numpy(),
        robot.data.projected_gravity_b.torch[0].detach().cpu().numpy(),
    )
    return rgb, proprio, build_navigation_observation(rgb, proprio, instruction)
