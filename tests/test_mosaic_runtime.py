import random

import numpy as np
import pytest

from unitree_gr00t.mosaic_runtime import (
    InitialFlowNoiseSteering,
    capture_random_tape,
    random_tape_sha256,
    restore_random_tape,
)
from unitree_gr00t.ours import OursContractError


def test_complete_random_tape_replays_global_and_dynamic_rng() -> None:
    original = capture_random_tape(
        dynamic_state=None,
        absolute_step=0,
        random_module=random,
        np=np,
    )
    try:
        random.seed(17)
        np.random.seed(19)
        dynamic_state = {"rng": random.Random(23), "cursor": 4, "toggle": -1}
        snapshot = capture_random_tape(
            dynamic_state=dynamic_state,
            absolute_step=32,
            random_module=random,
            np=np,
        )
        expected = (
            random.random(),
            float(np.random.random()),
            dynamic_state["rng"].random(),
        )
        random.random()
        np.random.random()
        dynamic_state["rng"].random()

        restored_dynamic, restored_step = restore_random_tape(
            snapshot,
            random_module=random,
            np=np,
        )
        actual = (
            random.random(),
            float(np.random.random()),
            restored_dynamic["rng"].random(),
        )
        assert actual == expected
        assert restored_step == 32
        assert restored_dynamic["cursor"] == 4
        assert restored_dynamic["toggle"] == -1
        assert len(random_tape_sha256(snapshot, np=np)) == 64
    finally:
        restore_random_tape(original, random_module=random, np=np)


def test_random_tape_requires_torch_to_restore_torch_state() -> None:
    snapshot = capture_random_tape(
        dynamic_state=None,
        absolute_step=0,
        random_module=random,
        np=np,
    )
    object.__setattr__(snapshot, "torch_cpu_state", object())
    with pytest.raises(OursContractError, match="requires torch"):
        restore_random_tape(snapshot, random_module=random, np=np)


def test_noise_steering_rejects_unhookable_action_head() -> None:
    with pytest.raises(OursContractError, match="hookable"):
        InitialFlowNoiseSteering(
            object(),
            basis=np.eye(2),
            code=np.zeros(2),
            maximum_angle=0.2,
        )
