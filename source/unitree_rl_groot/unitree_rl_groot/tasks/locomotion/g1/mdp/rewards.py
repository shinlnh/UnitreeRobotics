"""Additional rewards for robust locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import RayCaster


def _bounded_square_sum(values: torch.Tensor, *, limit: float, dim: int = 1) -> torch.Tensor:
    """Square finite physical values after clipping outlier magnitude.

    Non-finite values deliberately remain non-finite so the rollout guard reports
    a broken simulation instead of silently converting it into training data.
    """

    bounded = torch.where(torch.isfinite(values), values.clamp(-limit, limit), values)
    return torch.sum(torch.square(bounded), dim=dim)


def safe_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Penalize base roll/pitch rates without allowing finite overflow."""

    asset: Articulation = env.scene[asset_cfg.name]
    return _bounded_square_sum(asset.data.root_ang_vel_b.torch[:, :2], limit=50.0)


def safe_joint_torques_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Penalize joint torque with headroom above the G1 actuator limits."""

    asset: Articulation = env.scene[asset_cfg.name]
    values = asset.data.applied_torque.torch[:, asset_cfg.joint_ids]
    return _bounded_square_sum(values, limit=500.0)


def safe_joint_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Penalize joint acceleration while keeping its square representable."""

    asset: Articulation = env.scene[asset_cfg.name]
    values = asset.data.joint_acc.torch[:, asset_cfg.joint_ids]
    return _bounded_square_sum(values, limit=10_000.0)


def safe_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize policy action changes with a conservative numeric bound."""

    delta = env.action_manager.action - env.action_manager.prev_action
    return _bounded_square_sum(delta, limit=20.0)


def safe_base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Height penalty that ignores ray-caster no-hit sentinels.

    Isaac Lab's stock term averages every ray directly. A single ``inf`` no-hit
    value therefore makes the entire environment reward infinite even though the
    clipped height-scan observation remains valid.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    root_height = asset.data.root_pos_w.torch[:, 2]
    if sensor_cfg is None:
        terrain_height: torch.Tensor | float = 0.0
    else:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_height = sensor.data.ray_hits_w.torch[..., 2]
        finite = torch.isfinite(ray_height)
        finite_count = finite.sum(dim=1)
        finite_sum = torch.where(finite, ray_height, 0.0).sum(dim=1)
        # If all rays miss, make this term neutral for that step; termination and
        # the strict rollout guard still handle genuinely invalid robot state.
        terrain_height = torch.where(
            finite_count > 0,
            finite_sum / finite_count.clamp_min(1),
            root_height - target_height,
        )
    error = root_height - (target_height + terrain_height)
    return torch.square(torch.where(torch.isfinite(error), error.clamp(-3.0, 3.0), error))


def joint_power_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Penalize absolute mechanical joint power in watts.

    This is more physically meaningful for energy shaping than torque alone and
    remains differentiable almost everywhere.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque.torch[:, asset_cfg.joint_ids].clamp(-500.0, 500.0)
    velocity = asset.data.joint_vel.torch[:, asset_cfg.joint_ids].clamp(-100.0, 100.0)
    return torch.sum(torch.abs(torque * velocity), dim=1)
