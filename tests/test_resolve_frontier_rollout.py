import numpy as np
import pytest

from unitree_gr00t.ours import OursContractError
from unitree_gr00t.resolve_frontier_rollout import (
    expert_model_chunk,
    flatten_frontier_anchors,
)


def test_frontier_anchors_expand_in_stable_hard_to_easy_order() -> None:
    rows = [
        {
            "source_manifest_index": 3,
            "subgoal_index": 2,
            "split": "train",
            "first_true_frame": 100,
            "anchor_frames": [84, 36, 68],
        },
        {
            "source_manifest_index": 4,
            "subgoal_index": 1,
            "split": "development",
            "first_true_frame": 40,
            "anchor_frames": [24],
        },
    ]
    selected = flatten_frontier_anchors(
        rows,
        split="train",
        anchor_offset=None,
        selection_seed=None,
        start=0,
        limit=2,
    )
    assert [row["anchor_offset"] for row in selected] == [64, 32]
    assert all(row["frontier_index"] == 0 for row in selected)


def test_frontier_anchors_filter_one_offset_across_frontiers() -> None:
    rows = [
        {
            "source_manifest_index": 1,
            "subgoal_index": 0,
            "split": "train",
            "first_true_frame": 100,
            "anchor_frames": [36, 68, 84],
        },
        {
            "source_manifest_index": 2,
            "subgoal_index": 0,
            "split": "train",
            "first_true_frame": 80,
            "anchor_frames": [16, 48, 64],
        },
    ]

    selected = flatten_frontier_anchors(
        rows,
        split="train",
        anchor_offset=32,
        selection_seed=None,
        start=0,
        limit=None,
    )

    assert [(row["source_manifest_index"], row["anchor_frame"]) for row in selected] == [
        (1, 68),
        (2, 48),
    ]


def test_frontier_hash_selection_is_stable_and_not_manifest_order() -> None:
    rows = [
        {
            "source_manifest_index": index,
            "subgoal_index": 0,
            "split": "train",
            "first_true_frame": 100,
            "anchor_frames": [68],
            "predicate": ["in", f"object_{index}", "region"],
        }
        for index in range(10)
    ]
    first = flatten_frontier_anchors(
        rows,
        split="train",
        anchor_offset=32,
        selection_seed=17,
        start=0,
        limit=4,
    )
    second = flatten_frontier_anchors(
        rows[::-1],
        split="train",
        anchor_offset=32,
        selection_seed=17,
        start=0,
        limit=4,
    )
    assert [row["source_manifest_index"] for row in first] == [
        row["source_manifest_index"] for row in second
    ]
    assert [row["source_manifest_index"] for row in first] != list(range(4))


def test_frontier_selection_start_makes_disjoint_stable_shards() -> None:
    rows = [
        {
            "source_manifest_index": index,
            "subgoal_index": 0,
            "split": "train",
            "first_true_frame": 100,
            "anchor_frames": [68],
            "predicate": ["in", f"object_{index}", "region"],
        }
        for index in range(10)
    ]
    left = flatten_frontier_anchors(
        rows,
        split="train",
        anchor_offset=32,
        selection_seed=19,
        start=0,
        limit=5,
    )
    right = flatten_frontier_anchors(
        rows,
        split="train",
        anchor_offset=32,
        selection_seed=19,
        start=5,
        limit=5,
    )
    assert {row["source_manifest_index"] for row in left}.isdisjoint(
        row["source_manifest_index"] for row in right
    )
    assert len(left) == len(right) == 5


def test_expert_chunk_converts_gripper_and_pads_episode_end() -> None:
    actions = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, -1],
            [1, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    chunk = expert_model_chunk(actions, frame=1, horizon=3, np=np)
    assert chunk.shape == (3, 7)
    assert np.array_equal(chunk[:, -1], np.zeros(3))
    assert np.array_equal(chunk[1], chunk[0])
    with pytest.raises(OursContractError, match="out of range"):
        expert_model_chunk(actions, frame=2, horizon=3, np=np)
