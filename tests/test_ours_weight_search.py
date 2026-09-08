from unitree_gr00t.ours_weight_search import rank_weight_variants


def _variant(name: str, beneficial: float, recall: float, regret: float) -> dict:
    return {
        "variant_id": name,
        "checkpoint": f"/checkpoint/{name}",
        "option_validation": {
            "selective_recovery": {
                "beneficial_recovery_rate": beneficial,
                "true_recovery_rate": recall,
                "mean_decision_regret": regret,
            }
        },
    }


def test_weight_search_ranks_all_models_with_unique_ids() -> None:
    registries = {
        "w005": {"variants": [_variant("linear", 0.5, 0.5, 0.2)]},
        "w010": {"variants": [_variant("linear", 0.5, 0.75, 0.3)]},
    }

    ranked = rank_weight_variants(registries)

    assert [row["variant_id"] for row in ranked] == ["w010-linear", "w005-linear"]
    assert [row["offline_option_rank"] for row in ranked] == [1, 2]
