"""Project-local RoboCasa scene containing every GR00T G1 fruit task."""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from robocasa.environments.locomanipulation.locomanip import LMSimpleEnv
from robocasa.environments.locomanipulation.locomanip_dc import LabEnvMixin
from robocasa.utils.scene.configs import ObjectConfig, SamplingConfig
from robocasa.utils.scene.scene import SceneObject
from robocasa.utils.scene.success_criteria import IsInContact, SuccessCriteria

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "unitree_rl_groot"))

from unitree_rl_groot.groot.multitask import TASK_BY_OBJECT  # noqa: E402

ASSET_ROOT = PROJECT_ROOT / "assets/multitask"


class _LMMultiFruitToPlateDC(LabEnvMixin, LMSimpleEnv):
    """One table, four visible fruit choices and one destination plate."""

    TARGET_OBJECT = "apple"
    _FRUIT_PATHS = {
        "apple": "objects/omniverse/locomanip/apple_0/model.xml",
        "pear": str(ASSET_ROOT / "pear.xml"),
        "grapes": str(ASSET_ROOT / "grapes.xml"),
        "starfruit": str(ASSET_ROOT / "starfruit.xml"),
    }
    # Staggered placement keeps all four objects visible between the hands and
    # preserves enough clearance for the larger grape/starfruit collisions.
    _FRUIT_XY = {
        "apple": (0.35, -0.15),
        "pear": (0.42, -0.05),
        "grapes": (0.35, 0.05),
        "starfruit": (0.42, 0.15),
    }

    def _get_objects(self) -> list[SceneObject]:
        self.table = SceneObject(
            ObjectConfig(
                name="table",
                mjcf_path="objects/omniverse/locomanip/lab_table/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.01, 0.01]),
                    y_range=np.array([-0.01, 0.01]),
                    reference_pos=np.array([0.5, 0.0, 0.0]),
                    rotation=np.array([np.pi * 0.5, np.pi * 0.5]),
                ),
            )
        )
        self.plate = SceneObject(
            ObjectConfig(
                name="plate",
                mjcf_path="objects/omniverse/locomanip/plate_1/model.xml",
                static=True,
                sampler_config=SamplingConfig(
                    x_range=np.array([-0.01, 0.01]),
                    y_range=np.array([-0.01, 0.01]),
                    rotation=np.array([-0.05, 0.05]),
                    reference_pos=np.array([0.62, 0.0, self.table.mj_obj.top_offset[2]]),
                ),
            )
        )
        self.fruits: dict[str, SceneObject] = {}
        for object_name, (x_position, y_position) in self._FRUIT_XY.items():
            self.fruits[object_name] = SceneObject(
                ObjectConfig(
                    name=object_name,
                    mjcf_path=self._FRUIT_PATHS[object_name],
                    static=False,
                    density=85,
                    friction=(1.2, 0.3, 0.1),
                    sampler_config=SamplingConfig(
                        x_range=np.array([-0.015, 0.015]),
                        y_range=np.array([-0.012, 0.012]),
                        rotation=np.array([-np.pi, np.pi]),
                        reference_pos=np.array(
                            [x_position, y_position, self.table.mj_obj.top_offset[2]], dtype=float
                        ),
                    ),
                )
            )
        return [self.table, self.plate, *self.fruits.values()]

    @property
    def target(self) -> SceneObject:
        return self.fruits[self.TARGET_OBJECT]

    def _get_success_criteria(self) -> SuccessCriteria:
        return IsInContact(self.target, self.plate)

    def _get_instruction(self) -> str:
        return TASK_BY_OBJECT[self.TARGET_OBJECT].prompt

    def get_object(self) -> dict:
        result = {
            name: {"obj_name": fruit.mj_obj.root_body, "obj_type": "body"}
            for name, fruit in self.fruits.items()
        }
        result["plate"] = {"obj_name": self.plate.mj_obj.root_body, "obj_type": "body"}
        return result


class LMMultiFruitAppleToPlateDC(_LMMultiFruitToPlateDC):
    TARGET_OBJECT = "apple"


class LMMultiFruitPearToPlateDC(_LMMultiFruitToPlateDC):
    TARGET_OBJECT = "pear"


class LMMultiFruitGrapesToPlateDC(_LMMultiFruitToPlateDC):
    TARGET_OBJECT = "grapes"


class LMMultiFruitStarfruitToPlateDC(_LMMultiFruitToPlateDC):
    TARGET_OBJECT = "starfruit"


CLASS_BY_OBJECT = {
    "apple": LMMultiFruitAppleToPlateDC,
    "pear": LMMultiFruitPearToPlateDC,
    "grapes": LMMultiFruitGrapesToPlateDC,
    "starfruit": LMMultiFruitStarfruitToPlateDC,
}


def ensure_registered() -> dict[str, str]:
    """Import WBC registration after these project-local classes exist."""

    from decoupled_wbc.control.envs.robocasa import sync_env

    task_ids: dict[str, str] = {}
    for object_name, env_class in CLASS_BY_OBJECT.items():
        task_id = f"gr00tlocomanip_g1_sim/{env_class.__name__}_G1_gear_wbc"
        if task_id not in gym.registry:
            sync_env.create_gym_sync_env_class(env_class.__name__, "G1", "g1_sim", "gear_wbc")
        task_ids[object_name] = task_id
    return task_ids
