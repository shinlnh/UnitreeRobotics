import numpy as np

from unitree_gr00t.ours_development_search import (
    paired_task_macro_bootstrap,
    task_macro_subtask_rate,
)


def _row(
    task: str, case: str, complete: int, possible: int = 2
) -> dict[str, object]:
    return {
        "task_type": task,
        "case": case,
        "trial": 0,
        "agent_completed_subtasks": complete,
        "possible_subtasks": possible,
    }


def test_task_macro_and_paired_bootstrap_preserve_task_balance() -> None:
    baseline = [_row("A", "case1", 0), _row("A", "case2", 1), _row("B", "case1", 0)]
    candidate = [_row("A", "case1", 1), _row("A", "case2", 1), _row("B", "case1", 1)]
    assert task_macro_subtask_rate(baseline) == 0.125
    assert task_macro_subtask_rate(candidate) == 0.5
    result = paired_task_macro_bootstrap(
        baseline,
        candidate,
        resamples=100,
        seed=20007,
        np=np,
    )
    assert result["delta"] == 0.375
    assert result["ci95_lower"] <= result["delta"] <= result["ci95_upper"]


def test_paired_bootstrap_matches_macro_rate_with_unequal_denominators() -> None:
    baseline = [
        _row("A", "case1", 0, 1),
        _row("A", "case2", 0, 3),
        _row("B", "case1", 0, 2),
    ]
    candidate = [
        _row("A", "case1", 1, 1),
        _row("A", "case2", 0, 3),
        _row("B", "case1", 1, 2),
    ]

    result = paired_task_macro_bootstrap(
        baseline,
        candidate,
        resamples=100,
        seed=20007,
        np=np,
    )

    expected = task_macro_subtask_rate(candidate) - task_macro_subtask_rate(baseline)
    assert result["delta"] == expected == 0.375
