from unitree_gr00t.b_features import episode_feature_seed


def test_episode_feature_seed_is_stable_and_episode_specific() -> None:
    assert episode_feature_seed(7, 12) == episode_feature_seed(7, 12)
    assert episode_feature_seed(7, 12) != episode_feature_seed(7, 13)
