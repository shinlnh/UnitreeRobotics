from types import SimpleNamespace

import numpy as np
import pytest

from unitree_gr00t.ours import OursContractError, RecoveryOption
from unitree_gr00t.ours_counterfactual import (
    EFFICIENCY_SHAPED_RETURN_TARGET,
    OUTCOME_FIRST_RETURN_TARGET,
    counterfactual_return,
    evaluate_counterfactual_options,
    evaluate_counterfactual_rollouts,
)


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
        self._last_regions = {"goal": "table"}
        self.timestep = 0
        self.cur_time = 0.0
        self.done = False
        controller = SimpleNamespace(
            sim=self.sim,
            goal_pos=np.zeros(3),
            goal_ori=np.eye(3),
            new_update=False,
        )
        self.robots = [SimpleNamespace(controller=controller, recent_actions=[np.zeros(1)])]

    def step(self, action: np.ndarray) -> tuple[None, int, bool, dict]:
        self.sim.value[0] += float(action[0])
        self.timestep += 1
        self.cur_time += 0.05
        self.robots[0].controller.goal_pos[0] += float(action[0])
        self.robots[0].recent_actions.append(np.asarray(action).copy())
        self._last_regions["goal"] = "moved"
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


def test_outcome_first_counterfactual_return_excludes_efficiency_costs() -> None:
    shaped = counterfactual_return(
        progress_gain=1,
        final_success=False,
        predicate_gain=1,
        executed_steps=75,
        policy_calls=8,
        return_target=EFFICIENCY_SHAPED_RETURN_TARGET,
    )
    outcome_first = counterfactual_return(
        progress_gain=1,
        final_success=False,
        predicate_gain=1,
        executed_steps=75,
        policy_calls=8,
        return_target=OUTCOME_FIRST_RETURN_TARGET,
    )

    assert shaped == pytest.approx(8.77)
    assert outcome_first == 9.0
    with pytest.raises(OursContractError, match="unsupported"):
        counterfactual_return(
            progress_gain=0,
            final_success=False,
            predicate_gain=0,
            executed_steps=0,
            policy_calls=0,
            return_target="unknown",
        )


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
    assert len({branch.branch_seed for branch in branches}) == 1
    assert branches[1].return_value > branches[0].return_value
    assert branches[1].policy_calls == 0
    assert env.sim.value.tolist() == [0.0, 0.0]
    assert env.timestep == 0
    assert env.cur_time == 0.0
    assert env.robots[0].controller.goal_pos.tolist() == [0.0, 0.0, 0.0]
    assert len(env.robots[0].recent_actions) == 1
    assert env._last_regions == {"goal": "table"}


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


def test_counterfactual_policy_rollouts_restore_same_state() -> None:
    env = _Env()

    def advance(_observation: object, _seed: int) -> tuple[int, int]:
        env.step(np.asarray([2.0], dtype=np.float32))
        return 1, 3

    branches = evaluate_counterfactual_rollouts(
        env,
        goal={"object": [["reached", "target"]]},
        option_rollouts={RecoveryOption.RETRY_CURRENT: advance},
        completed_subtasks_before=0,
        base_seed=10007,
        state_index=4,
        np=np,
    )

    assert env.sim.get_state().flatten().tolist() == [0.0, 0.0]
    assert branches[0].executed_steps == 1
    assert branches[0].policy_calls == 3
    assert branches[0].return_value > 0.0
