from types import SimpleNamespace

import numpy as np
import pytest

from unitree_gr00t.ours import OursContractError, RecoveryOption
from unitree_gr00t.ours_counterfactual import evaluate_counterfactual_options


class _State:
    def __init__(self, value: np.ndarray):
        self.value = value

    def flatten(self) -> np.ndarray:
        return self.value.copy()


class _Sim:
    def __init__(self):
        self.value = np.zeros(2, dtype=np.float64)
        self.data = SimpleNamespace(
            ctrl=np.zeros(1),
            mocap_pos=np.zeros((1, 3)),
            mocap_quat=np.zeros((1, 4)),
        )

    def get_state(self) -> _State:
        return _State(self.value)

    def set_state_from_flattened(self, value: np.ndarray) -> None:
        self.value[...] = value

    def forward(self) -> None:
        pass


class _Env:
    def __init__(self):
        self.sim = _Sim()
        self._state_progress = {"goal": 0}
        self.timestep = 0
        self.done = False

    def step(self, action: np.ndarray) -> tuple[None, int, bool, dict]:
        self.sim.value[0] += float(action[0])
        self.timestep += 1
        return None, 0, False, {}

    def _check_success(self, goal: object) -> tuple[None, int, bool]:
        complete = int(self.sim.value[0] >= 1.0)
        return None, complete, bool(complete)

    def _eval_predicate(self, predicate: object) -> bool:
        return bool(self.sim.value[0] >= 1.0)

    def _post_process(self) -> None:
        pass

    def _update_observables(self, force: bool) -> None:
        assert force

    def _get_observations(self) -> dict:
        return {}


def test_counterfactual_options_branch_and_restore_exact_state() -> None:
    env = _Env()
    branches = evaluate_counterfactual_options(
        env,
        goal={"goal": [["predicate"]]},
        option_actions={
            RecoveryOption.ACCEPT_B: [np.asarray([0.0])],
            RecoveryOption.CONSENSUS_PREFIX: [np.asarray([1.0])],
        },
        completed_subtasks_before=0,
        base_seed=10007,
        state_index=3,
        np=np,
    )
    assert [branch.option for branch in branches] == ["ACCEPT_B", "CONSENSUS_PREFIX"]
    assert branches[1].return_value > branches[0].return_value
    assert env.sim.value.tolist() == [0.0, 0.0]
    assert env.timestep == 0


def test_counterfactual_options_reject_final_seed() -> None:
    with pytest.raises(OursContractError, match="train seeds"):
        evaluate_counterfactual_options(
            _Env(),
            goal={"goal": [["predicate"]]},
            option_actions={RecoveryOption.ACCEPT_B: []},
            completed_subtasks_before=0,
            base_seed=7,
            state_index=0,
            np=np,
        )
