"""Privileged observations used only by the critic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the contact-force magnitude for each configured foot body."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_w = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :]
    return torch.linalg.vector_norm(force_w, dim=-1)
