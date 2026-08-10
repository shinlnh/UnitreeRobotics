"""PPO configs for asymmetric robust G1 locomotion."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def _actor() -> RslRlMLPModelCfg:
    return RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        # Optimize log(std), so the Gaussian scale remains strictly positive.
        # The scalar parameterization can cross zero during rough-terrain PPO.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            class_name="unitree_rl_groot.rl:BoundedGaussianDistribution",
            init_std=0.8,
            std_type="log",
        ),
    )


def _critic() -> RslRlMLPModelCfg:
    return RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )


def _algorithm() -> RslRlPpoAlgorithmCfg:
    return RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1RoughRobustPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_g1_rough_robust"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = _actor()
    critic = _critic()
    algorithm = _algorithm()


@configclass
class G1FlatRobustPPORunnerCfg(G1RoughRobustPPORunnerCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.max_iterations = 2000
        self.experiment_name = "unitree_g1_flat_robust"
