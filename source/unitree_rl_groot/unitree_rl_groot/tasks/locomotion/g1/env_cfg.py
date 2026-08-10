"""Manager-based Unitree G1 locomotion environments for Isaac Lab 3.0."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.core.velocity import mdp as velocity_mdp
from isaaclab_tasks.core.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1Rewards, G1RoughEnvCfg
from isaaclab_tasks.core.velocity.velocity_env_cfg import ObservationsCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from . import mdp

FOOT_LINK_PATTERN = ".*_ankle_roll_link"
LOWER_BODY_JOINT_PATTERN = [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*", "torso_joint"]
NAVIGATION_OBSTACLE_COUNT = 8


def _navigation_obstacle_cfg(index: int) -> RigidObjectCfg:
    colors = ((0.9, 0.12, 0.08), (0.08, 0.25, 0.9), (0.95, 0.55, 0.05), (0.5, 0.12, 0.75))
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/NavigationObstacle{index}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(100.0 + index, 100.0, 0.6)),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 1.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=colors[index % len(colors)],
                roughness=0.7,
            ),
        ),
    )


def _navigation_goal_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/NavigationGoal",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(100.0, 100.0, 0.7)),
        spawn=sim_utils.CylinderCfg(
            radius=0.24,
            height=1.4,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.02, 0.95, 0.08),
                emissive_color=(0.0, 0.2, 0.0),
                roughness=0.4,
            ),
        ),
    )


@configclass
class G1PrivilegedCriticCfg(ObsGroup):
    """Noise-free state available to the critic but not required at deployment."""

    base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=base_mdp.projected_gravity)
    velocity_commands = ObsTerm(func=base_mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)
    joint_effort = ObsTerm(func=base_mdp.joint_effort)
    actions = ObsTerm(func=base_mdp.last_action)
    feet_contact = ObsTerm(
        func=mdp.feet_contact_force,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_PATTERN)},
    )
    height_scan = ObsTerm(
        func=base_mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class G1AsymmetricObservationsCfg(ObservationsCfg):
    """Noisy actor observations and privileged critic observations."""

    critic: G1PrivilegedCriticCfg = G1PrivilegedCriticCfg()


@configclass
class G1RobustRewardsCfg(G1Rewards):
    """Tracking, stability, energy, and standing rewards."""

    base_height_l2 = RewTerm(
        func=base_mdp.base_height_l2,
        weight=-5.0,
        params={"target_height": 0.74},
    )
    stand_still = RewTerm(
        func=velocity_mdp.stand_still_joint_deviation_l1,
        weight=-0.25,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.08,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINT_PATTERN),
        },
    )
    joint_power = RewTerm(
        func=mdp.joint_power_l1,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINT_PATTERN)},
    )


class _G1RobustSettings:
    """Shared post-init modifications kept out of configclass inheritance."""

    def _apply_robust_settings(self, *, rough: bool) -> None:
        self.episode_length_s = 20.0
        self.scene.num_envs = 4096

        # Broader omnidirectional command distribution, including a meaningful
        # standing subset so the learned controller does not shuffle at zero.
        command = self.commands.base_velocity
        command.rel_standing_envs = 0.10
        command.rel_heading_envs = 0.0
        command.heading_command = False
        command.ranges.lin_vel_x = (-0.5, 1.5)
        command.ranges.lin_vel_y = (-0.6, 0.6)
        command.ranges.ang_vel_z = (-1.2, 1.2)

        # Dynamics randomization for robustness. Gains are startup-only because
        # Isaac Sim implicit actuator properties are CPU-backed.
        self.events.physics_material.params.update(
            {
                "static_friction_range": (0.45, 1.25),
                "dynamic_friction_range": (0.35, 1.05),
                "restitution_range": (0.0, 0.15),
            }
        )
        self.events.add_base_mass.params.update(
            {"mass_distribution_params": (0.85, 1.15), "operation": "scale", "distribution": "uniform"}
        )
        self.events.base_com = EventTerm(
            func=base_mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {
                    "x": (-0.025, 0.025),
                    "y": (-0.025, 0.025),
                    "z": (-0.015, 0.015),
                },
            },
        )
        self.events.push_robot.interval_range_s = (8.0, 12.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.75, 0.75),
            "y": (-0.75, 0.75),
            "yaw": (-0.5, 0.5),
        }
        self.events.actuator_gains = EventTerm(
            func=base_mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINT_PATTERN),
                "stiffness_distribution_params": (0.9, 1.1),
                "damping_distribution_params": (0.9, 1.1),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # Keep the torso upright and make actuation smooth enough for transfer.
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.ang_vel_xy_l2.func = mdp.safe_ang_vel_xy_l2
        self.rewards.dof_torques_l2.func = mdp.safe_joint_torques_l2
        self.rewards.dof_acc_l2.func = mdp.safe_joint_acc_l2
        self.rewards.action_rate_l2.func = mdp.safe_action_rate_l2
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.feet_slide.weight = -0.2
        self.rewards.base_height_l2.func = mdp.safe_base_height_l2
        self.rewards.base_height_l2.params["sensor_cfg"] = SceneEntityCfg("height_scanner") if rough else None

        if rough:
            # 8192 G1 instances exceed PhysX's default GPU broad-phase buffers.
            # Undersized buffers drop contacts and eventually produce non-finite
            # simulation state, so keep power-of-two headroom above measured use.
            physics = self.sim.physics.isaacsim_physx
            physics.gpu_found_lost_pairs_capacity = 2**26
            physics.gpu_total_aggregate_pairs_capacity = 2**22
            self.scene.terrain.max_init_terrain_level = 2
        else:
            self.observations.critic.height_scan = None

    def play_mode(self) -> None:
        super().play_mode()
        self.scene.num_envs = min(self.scene.num_envs, 16)
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)


@configclass
class G1RoughRobustEnvCfg(_G1RobustSettings, G1RoughEnvCfg):
    observations: G1AsymmetricObservationsCfg = G1AsymmetricObservationsCfg()
    rewards: G1RobustRewardsCfg = G1RobustRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self._apply_robust_settings(rough=True)


@configclass
class G1FlatRobustEnvCfg(_G1RobustSettings, G1FlatEnvCfg):
    observations: G1AsymmetricObservationsCfg = G1AsymmetricObservationsCfg()
    rewards: G1RobustRewardsCfg = G1RobustRewardsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self._apply_robust_settings(rough=False)


@configclass
class G1GrootNavigationEnvCfg(G1FlatRobustEnvCfg):
    """Single-env camera configuration for GR00T closed-loop navigation."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 4.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)

        # Kinematic scene objects are repositioned for every collected/evaluated episode.
        # Their colours and geometry make the camera stream the only observation that
        # reveals the goal and obstacle layout to GR00T.
        for index in range(NAVIGATION_OBSTACLE_COUNT):
            setattr(self.scene, f"navigation_obstacle_{index}", _navigation_obstacle_cfg(index))
        self.scene.navigation_goal = _navigation_goal_cfg()

        # Calibrated G1 head camera from the official Isaac Lab locomanipulation task.
        self.scene.head_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_link/RobotHeadCam",
            update_period=self.decimation * self.sim.dt,
            height=224,
            width=224,
            data_types=["rgb"],
            renderer_cfg=MultiBackendRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=15.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.04485, 0.0, 0.35325),
                rot=(-0.62721, 0.62721, -0.32651, 0.32651),
                convention="ros",
            ),
        )

        # Evaluation and data collection must be deterministic by default.
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.actuator_gains = None
