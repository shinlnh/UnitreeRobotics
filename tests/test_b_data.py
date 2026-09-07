from pathlib import Path

import numpy as np

from unitree_gr00t.b_data import (
    SelectorEpisode,
    _sample_rows,
    map_raw_boundary,
    parse_annotated_subgoals,
    retained_source_indices,
)


def test_parse_annotated_subgoals_requires_contiguous_ranges() -> None:
    parsed = parse_annotated_subgoals(
        "Task: Put items away\n"
        "Step: Pick up item\n[0, 4]\nRelated Objects: item\n"
        "Step: Put item in box\n[4, 9]\nRelated Objects: item, box\n"
    )
    assert parsed == (("Pick up item", 0, 4), ("Put item in box", 4, 9))


def test_raw_boundaries_replay_a1_noop_filter() -> None:
    actions = np.asarray(
        [
            [0, 0, 0, 0, 0, 0, -1],
            [1, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, -1],
            [0, 0, 0, 0, 0, 0, 1],
            [2, 0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    retained = retained_source_indices(actions, np)
    assert retained.tolist() == [1, 3, 4]
    assert map_raw_boundary(0, retained, np) == 0
    assert map_raw_boundary(3, retained, np) == 1
    assert map_raw_boundary(5, retained, np) == 3


def test_sample_index_keeps_boundary_and_post_boundary_examples(tmp_path: Path) -> None:
    from unitree_gr00t.b_data import AnnotatedSubgoal

    episode = SelectorEpisode(
        episode_index=0,
        source_manifest_index=0,
        scene="scene",
        case="case",
        split="train",
        instruction="task",
        demonstration=str(tmp_path / "demo.hdf5"),
        demonstration_sha256="a" * 64,
        parquet=str(tmp_path / "episode.parquet"),
        agent_video=str(tmp_path / "agent.mp4"),
        wrist_video=str(tmp_path / "wrist.mp4"),
        raw_frames=20,
        filtered_frames=10,
        retained_source_indices_sha256="b" * 64,
        subgoals=(AnnotatedSubgoal("pick", 0, 10, 0, 5),),
    )
    rows = _sample_rows([episode], stride=4, post_boundary_steps=3)
    assert [row["frame_index"] for row in rows] == [0, 4, 5]
    assert all(row["completion_step"] == 5 for row in rows)
