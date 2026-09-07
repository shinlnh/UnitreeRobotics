from unitree_gr00t.b_train import anchor_sample_map


def test_anchor_sample_map_uses_earliest_frame_per_subgoal() -> None:
    samples = [
        {"episode_index": 3, "subgoal_index": 1, "frame_index": 18, "sample_index": 8},
        {"episode_index": 3, "subgoal_index": 0, "frame_index": 0, "sample_index": 2},
        {"episode_index": 3, "subgoal_index": 1, "frame_index": 10, "sample_index": 7},
    ]
    assert anchor_sample_map(samples) == {(3, 0): 2, (3, 1): 7}
