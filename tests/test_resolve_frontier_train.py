import hashlib

import numpy as np

from unitree_gr00t.resolve_frontier_train import (
    _development_groups,
    _executable_chunks,
)


def test_frontier_split_is_group_disjoint_and_keeps_both_strata() -> None:
    rows = [
        {"source_manifest_index": group, "positive": positive}
        for positive, groups in ((False, range(10)), (True, range(10, 20)))
        for group in groups
    ]
    development = _development_groups(rows, fraction=0.2, seed=7)
    assert len(development) == 4
    assert any(group < 10 for group in development)
    assert any(group >= 10 for group in development)
    assert development == _development_groups(rows[::-1], fraction=0.2, seed=7)
    expected = {
        group
        for positive, groups in ((False, range(10)), (True, range(10, 20)))
        for group in sorted(
            groups,
            key=lambda item: hashlib.sha256(
                f"resolve-frontier-split-v1:7:{item}".encode()
            ).digest(),
        )[:2]
    }
    assert development == expected


def test_frontier_chunks_match_executable_action_box() -> None:
    values = np.zeros((16, 7), dtype=np.float32)
    values[0] = [-2, 2, -0.5, 0.5, 3, -3, -1]
    values[1, -1] = 2
    bounded = _executable_chunks(values, np=np)
    assert np.array_equal(bounded[0], np.asarray([-1, 1, -0.5, 0.5, 1, -1, 0], dtype=np.float32))
    assert bounded[1, -1] == 1
