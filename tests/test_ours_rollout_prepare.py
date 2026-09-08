import numpy as np

from unitree_gr00t.ours_rollout_prepare import build_failure_targets


def test_failure_targets_use_injection_or_failed_segment_outcome() -> None:
    targets = build_failure_targets(
        subgoals=[0, 0, 0, 1, 1, 1, 2],
        elapsed=[0, 75, 100, 0, 80, 90, 0],
        complete=[False, False, False, False, True, True, False],
        failure_after_injection=[False, False, False, False, False, False, True],
        failure_onset_steps=75,
        np=np,
    )
    assert targets.tolist() == [False, True, True, False, False, False, True]
