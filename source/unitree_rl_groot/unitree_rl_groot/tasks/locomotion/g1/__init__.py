"""Register robust Unitree G1 locomotion and GR00T-camera environments."""

import gymnasium as gym

from . import agents


def _register(task_id: str, env_cfg: str, agent_cfg: str) -> None:
    if task_id in gym.registry:
        return
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{agent_cfg}",
        },
    )


_register("Unitree-G1-Velocity-Flat-Robust", "G1FlatRobustEnvCfg", "G1FlatRobustPPORunnerCfg")
_register("Unitree-G1-Velocity-Rough-Robust", "G1RoughRobustEnvCfg", "G1RoughRobustPPORunnerCfg")
_register("Unitree-G1-GR00T-Navigation", "G1GrootNavigationEnvCfg", "G1FlatRobustPPORunnerCfg")
