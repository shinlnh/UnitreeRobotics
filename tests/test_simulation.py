from pathlib import Path

from unitree_gr00t.config import load_config
from unitree_gr00t.policy import HeuristicGeneralistPolicy
from unitree_gr00t.simulation import run_suite
from unitree_gr00t.tasks import load_tasks

ROOT = Path(__file__).resolve().parents[1]


def test_all_generalist_tasks_succeed_across_randomized_scenes() -> None:
    config = load_config(ROOT / "configs" / "project.toml")
    tasks = load_tasks(ROOT / "configs" / "tasks.toml")
    policy = HeuristicGeneralistPolicy(horizon=40, execution_horizon=4)
    results = run_suite(tasks, episodes=4, seed=101, policy=policy, safety_config=config.safety)
    assert len(results) == 20
    assert all(result.success for result in results)
    assert all(result.steps <= config.safety.max_episode_steps for result in results)


def test_policy_observation_exposes_sonic_modality_contract() -> None:
    from unitree_gr00t.simulation import TabletopWorld

    task = load_tasks(ROOT / "configs" / "tasks.toml")[0]
    payload = TabletopWorld(task, seed=1).observe().as_sonic_state()
    assert set(payload) == {"video", "state", "language"}
    assert len(payload["state"]["left_leg"][0]) == 6
    assert len(payload["state"]["left_hand"][0]) == 7
    assert payload["language"]["annotation.human.task_description"] == [[task.instruction]]
