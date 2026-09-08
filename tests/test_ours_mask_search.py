from unitree_gr00t.ours_mask_search import (
    enumerate_override_masks,
    rank_mask_variants,
)


def _row(name: str, beneficial: int, rate: float, recall: float) -> dict:
    return {
        "variant_id": name,
        "option_validation": {
            "selective_recovery": {
                "beneficial_recovery_states": beneficial,
                "beneficial_recovery_rate_all_states": rate,
                "true_recovery_rate": recall,
                "mean_decision_regret": 0.1,
            }
        },
    }


def test_mask_search_registers_every_nonempty_library() -> None:
    masks = enumerate_override_masks()
    assert len(masks) == 15
    assert ("CONSENSUS_PREFIX",) in masks
    assert len(masks[-1]) == 4


def test_mask_search_ranks_absolute_benefit_before_conditional_rate() -> None:
    rows = [
        _row("one-perfect", 1, 0.1, 1.0),
        _row("two-partial", 2, 0.05, 0.5),
    ]
    ranked = rank_mask_variants(rows)
    assert ranked[0]["variant_id"] == "two-partial"
    assert [row["offline_option_rank"] for row in ranked] == [1, 2]
