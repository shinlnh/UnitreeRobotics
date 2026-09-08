import pytest

from unitree_gr00t.ours import OursContractError
from unitree_gr00t.ours_counterfactual_split import option_episode_partition


def test_option_split_holds_out_two_cases_per_task_block() -> None:
    split = option_episode_partition(list(range(60)), modulus=5, remainder=4)

    assert len(split["train"]) == 48
    assert split["development"] == list(range(4, 60, 5))
    assert not set(split["train"]) & set(split["development"])


def test_option_split_rejects_empty_partition() -> None:
    with pytest.raises(OursContractError, match="empty"):
        option_episode_partition([0], modulus=5, remainder=4)
