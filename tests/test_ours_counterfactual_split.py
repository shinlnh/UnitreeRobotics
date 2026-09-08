import pytest

from unitree_gr00t.ours import OursContractError
from unitree_gr00t.ours_counterfactual_split import (
    option_episode_partition,
    residual_stratified_episode_partition,
)


def test_option_split_holds_out_two_cases_per_task_block() -> None:
    split = option_episode_partition(list(range(60)), modulus=5, remainder=4)

    assert len(split["train"]) == 48
    assert split["development"] == list(range(4, 60, 5))
    assert not set(split["train"]) & set(split["development"])


def test_option_split_rejects_empty_partition() -> None:
    with pytest.raises(OursContractError, match="empty"):
        option_episode_partition([0], modulus=5, remainder=4)


def test_residual_stratified_split_preserves_both_episode_classes() -> None:
    labels = {episode_id: episode_id >= 10 for episode_id in range(15)}

    split = residual_stratified_episode_partition(labels, split_seed=10007)

    assert len(split["development"]) == 3
    assert not set(split["train"]) & set(split["development"])
    assert set(split["train"]) | set(split["development"]) == set(labels)
    for partition in split.values():
        assert {labels[episode_id] for episode_id in partition} == {False, True}


def test_residual_stratified_split_rejects_single_positive_episode() -> None:
    labels = {0: False, 1: False, 2: True}
    with pytest.raises(OursContractError, match="two episodes in each class"):
        residual_stratified_episode_partition(labels, split_seed=10007)
