from types import SimpleNamespace

import numpy as np
import pytest

from unitree_gr00t.ours_train import (
    calibrate_threshold,
    causal_anchor_histories,
    evaluate_failure,
    evaluate_residual_options,
    merge_corpora,
    sample_training_ids,
    validate_residual_advantage_manifests,
)


def _residual_advantage_manifest() -> dict:
    return {
        "counterfactual_sampling": {
            "residual_retry_baseline": True,
            "return_target": "outcome-first-physical-v1",
            "cost_terms_in_target": False,
            "global_step_budget_contract": (
                "min-configured-rollout-and-source-global-steps-remaining-v1"
            ),
        }
    }


def test_residual_advantage_manifest_requires_physical_budgeted_targets() -> None:
    manifest = _residual_advantage_manifest()
    validate_residual_advantage_manifests([manifest])

    for field, invalid in (
        ("residual_retry_baseline", False),
        ("return_target", "efficiency-shaped-v1"),
        ("cost_terms_in_target", True),
        ("global_step_budget_contract", "configured-rollout-only"),
    ):
        modified = _residual_advantage_manifest()
        modified["counterfactual_sampling"][field] = invalid
        with pytest.raises(ValueError, match="outcome-first"):
            validate_residual_advantage_manifests([modified])


def test_residual_advantage_manifest_requires_a_corpus() -> None:
    with pytest.raises(ValueError, match="require residual counterfactual corpora"):
        validate_residual_advantage_manifests([])


def test_training_histories_reset_at_every_runtime_anchor() -> None:
    histories = causal_anchor_histories(
        np.arange(6, dtype=np.int64),
        np.asarray([0, 0, 0, 3, 3, 5], dtype=np.int64),
        3,
        np,
    )

    assert histories.tolist() == [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 2],
        [3, 3, 3],
        [3, 3, 4],
        [5, 5, 5],
    ]


def test_completion_calibration_maximizes_recall_under_false_positive_cap() -> None:
    probabilities = np.asarray([0.9, 0.8, 0.7, 0.2, 0.1], dtype=np.float32)
    labels = np.asarray([True, False, True, False, False], dtype=np.bool_)
    threshold, metrics = calibrate_threshold(
        probabilities, labels, max_false_positive_rate=0.0, np=np
    )
    assert threshold == probabilities[0]
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["true_positive_rate"] == 0.5


def test_training_sampler_reserves_live_quota_without_losing_demo_balance() -> None:
    corpus = SimpleNamespace(
        train_ids=np.arange(12, dtype=np.int64),
        live_ids=np.arange(8, 12, dtype=np.int64),
        target_complete=np.asarray(
            [True] * 4 + [False] * 4 + [True] + [False] * 3,
            dtype=np.bool_,
        ),
        target_option_valid=np.zeros((12, 6), dtype=np.bool_),
        target_option_values=np.zeros((12, 6), dtype=np.float32),
    )
    ids = sample_training_ids(
        corpus,
        batch_size=8,
        live_batch_fraction=0.25,
        generator=np.random.default_rng(10007),
        np=np,
    )
    assert len(ids) == 8
    assert int((ids >= 8).sum()) == 2
    assert int(corpus.target_complete[ids[ids >= 8]].sum()) == 1
    demo = ids[ids < 8]
    assert int(corpus.target_complete[demo].sum()) == 3

    demo_only = sample_training_ids(
        corpus,
        batch_size=8,
        live_batch_fraction=0.0,
        generator=np.random.default_rng(10007),
        np=np,
    )
    assert int((demo_only >= 8).sum()) == 0


def test_training_sampler_guarantees_counterfactual_option_quota() -> None:
    option_valid = np.zeros((16, 6), dtype=np.bool_)
    option_valid[12:14, 0] = True
    option_valid[12:14, 4] = True
    option_values = np.zeros((16, 6), dtype=np.float32)
    option_values[12, 0] = 1.0
    option_values[13, 4] = 1.0
    corpus = SimpleNamespace(
        train_ids=np.arange(16, dtype=np.int64),
        live_ids=np.arange(8, 16, dtype=np.int64),
        target_complete=np.asarray([True, False] * 8, dtype=np.bool_),
        target_option_valid=option_valid,
        target_option_values=option_values,
    )
    ids = sample_training_ids(
        corpus,
        batch_size=8,
        live_batch_fraction=0.5,
        option_batch_fraction=0.25,
        generator=np.random.default_rng(10007),
        np=np,
    )
    assert len(ids) == 8
    assert int((ids >= 8).sum()) == 4
    selected_options = ids[np.isin(ids, [12, 13])]
    assert len(selected_options) >= 2
    assert {12, 13}.issubset(set(selected_options.tolist()))


def test_training_sampler_does_not_treat_ties_as_first_class_wins() -> None:
    option_valid = np.zeros((16, 6), dtype=np.bool_)
    option_valid[12:15, :2] = True
    option_values = np.zeros((16, 6), dtype=np.float32)
    option_values[12, 0] = 1.0
    option_values[13, 1] = 1.0
    corpus = SimpleNamespace(
        train_ids=np.arange(16, dtype=np.int64),
        live_ids=np.arange(8, 16, dtype=np.int64),
        target_complete=np.asarray([True, False] * 8, dtype=np.bool_),
        target_option_valid=option_valid,
        target_option_values=option_values,
    )

    ids = sample_training_ids(
        corpus,
        batch_size=8,
        live_batch_fraction=0.5,
        option_batch_fraction=0.25,
        generator=np.random.default_rng(10007),
        np=np,
    )

    assert {12, 13}.issubset(set(ids.tolist()))


def test_residual_sampler_balances_overrides_with_retry_and_ties() -> None:
    option_valid = np.zeros((16, 6), dtype=np.bool_)
    option_valid[12:16, 1:6] = True
    option_values = np.zeros((16, 6), dtype=np.float32)
    option_values[12, 1] = 1.0
    option_values[13, 5] = 1.0
    corpus = SimpleNamespace(
        train_ids=np.arange(16),
        live_ids=np.arange(8, 16),
        live_groups=(np.arange(8, 16),),
        target_complete=np.asarray([True, False] * 8),
        target_option_valid=option_valid,
        target_option_values=option_values,
    )

    ids = sample_training_ids(
        corpus,
        batch_size=8,
        live_batch_fraction=0.5,
        option_batch_fraction=0.5,
        option_baseline_index=2,
        generator=np.random.default_rng(10007),
        np=np,
    )

    assert len(set(ids.tolist()) & {12, 13}) == 2
    assert len(set(ids.tolist()) & {14, 15}) >= 1


def test_repeated_corpus_merge_preserves_all_live_training_ids() -> None:
    def corpus(size: int, *, train_ids: list[int], live_ids: list[int]) -> SimpleNamespace:
        return SimpleNamespace(
            contexts=np.zeros((size, 2), dtype=np.float16),
            anchor_ids=np.arange(size),
            action_chunks=np.zeros((size, 1, 1), dtype=np.float16),
            selector_features=np.zeros((size, 2), dtype=np.float16),
            scalars=np.zeros((size, 1), dtype=np.float16),
            histories=np.arange(size)[:, None],
            target_progress=np.zeros(size),
            target_progress_valid=np.ones(size, dtype=np.bool_),
            target_complete=np.zeros(size, dtype=np.bool_),
            target_failure=np.zeros(size, dtype=np.bool_),
            target_failure_valid=np.ones(size, dtype=np.bool_),
            target_option_values=np.zeros((size, 6)),
            target_option_valid=np.zeros((size, 6), dtype=np.bool_),
            selector_candidates=np.zeros(size, dtype=np.int8),
            train_ids=np.asarray(train_ids),
            development_ids=np.asarray([], dtype=np.int64),
            live_ids=np.asarray(live_ids),
        )

    primary = corpus(4, train_ids=[0, 1, 2], live_ids=[])
    first = merge_corpora(primary, corpus(3, train_ids=[0, 1], live_ids=[0, 1]), np)
    combined = merge_corpora(first, corpus(2, train_ids=[0], live_ids=[0]), np)

    assert combined.train_ids.tolist() == [0, 1, 2, 4, 5, 7]
    assert combined.live_ids.tolist() == [4, 5, 7]
    assert [group.tolist() for group in combined.live_groups] == [[4, 5], [7]]


def test_option_sampler_balances_seed_and_winner_strata() -> None:
    option_valid = np.zeros((16, 6), dtype=np.bool_)
    option_valid[[8, 9, 12, 13], :2] = True
    option_values = np.zeros((16, 6), dtype=np.float32)
    option_values[[8, 12], 0] = 1.0
    option_values[[9, 13], 1] = 1.0
    corpus = SimpleNamespace(
        train_ids=np.arange(16),
        live_ids=np.arange(8, 16),
        live_groups=(np.arange(8, 12), np.arange(12, 16)),
        target_complete=np.asarray([True, False] * 8),
        target_option_valid=option_valid,
        target_option_values=option_values,
    )

    ids = sample_training_ids(
        corpus,
        batch_size=8,
        live_batch_fraction=0.5,
        option_batch_fraction=0.5,
        generator=np.random.default_rng(10007),
        np=np,
    )

    assert {8, 9, 12, 13}.issubset(set(ids.tolist()))


def test_failure_calibration_uses_valid_heldout_rows() -> None:
    torch = pytest.importorskip("torch")

    size = 4
    corpus = SimpleNamespace(
        contexts=np.zeros((size, 2), dtype=np.float16),
        anchor_ids=np.arange(size),
        action_chunks=np.zeros((size, 1, 1), dtype=np.float16),
        selector_features=np.zeros((size, 2), dtype=np.float16),
        scalars=np.asarray([[-3.0], [-2.0], [2.0], [3.0]], dtype=np.float16),
        histories=np.arange(size)[:, None],
        target_progress=np.zeros(size),
        target_progress_valid=np.ones(size, dtype=np.bool_),
        target_complete=np.zeros(size, dtype=np.bool_),
        target_failure=np.asarray([False, False, True, True]),
        target_failure_valid=np.ones(size, dtype=np.bool_),
        target_option_values=np.zeros((size, 6)),
        target_option_valid=np.zeros((size, 6), dtype=np.bool_),
    )

    class ScalarFailure(torch.nn.Module):
        def forward(self, contexts, anchors, actions, selector, scalars):
            del contexts, anchors, actions, selector
            return {"failure_logit": scalars[:, -1, 0]}

    result = evaluate_failure(
        ScalarFailure(),
        corpus,
        np.arange(size),
        batch_size=2,
        device="cpu",
        max_false_positive_rate=0.0,
        np=np,
        torch=torch,
    )

    assert result is not None
    assert result["failure"]["false_positive_rate"] == 0.0
    assert result["failure"]["true_positive_rate"] == 1.0
    assert result["failure_calibration_samples"] == size


def test_residual_checkpoint_metric_uses_heldout_option_advantage() -> None:
    torch = pytest.importorskip("torch")

    targets = np.asarray(
        [
            [0.0, 0.0, 3.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, 3.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, -1.0, 3.0, 0.0],
            [0.0, 0.0, 1.0, -1.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(targets, dtype=np.bool_)
    predictions = np.asarray(
        [
            [0.0, -1.0, 0.0, -1.0, 0.3, -1.0],
            [0.0, -1.0, 0.0, -1.0, -0.1, -1.0],
            [0.0, -1.0, 0.0, -1.0, 0.5, -1.0],
            [0.0, -1.0, 0.0, -1.0, 0.4, -1.0],
        ],
        dtype=np.float32,
    )
    size = len(targets)
    corpus = SimpleNamespace(
        contexts=np.zeros((size, 2), dtype=np.float16),
        anchor_ids=np.arange(size),
        action_chunks=np.zeros((size, 1, 1), dtype=np.float16),
        selector_features=np.zeros((size, 2), dtype=np.float16),
        scalars=predictions.astype(np.float16),
        histories=np.arange(size)[:, None],
        target_progress=np.zeros(size),
        target_progress_valid=np.ones(size, dtype=np.bool_),
        target_complete=np.zeros(size, dtype=np.bool_),
        target_failure=np.zeros(size, dtype=np.bool_),
        target_failure_valid=np.zeros(size, dtype=np.bool_),
        target_option_values=targets,
        target_option_valid=valid,
    )

    class ScalarOptions(torch.nn.Module):
        def forward(self, contexts, anchors, actions, selector, scalars):
            del contexts, anchors, actions, selector
            return {"option_values": scalars[:, -1, :6]}

    result = evaluate_residual_options(
        ScalarOptions(),
        corpus,
        np.arange(size),
        batch_size=2,
        device="cpu",
        max_false_recovery_rate=0.0,
        np=np,
        torch=torch,
    )

    assert result["baseline_option"] == "RETRY_CURRENT"
    assert result["selective_recovery"]["false_recovery_rate"] == 0.0
    assert result["selective_recovery"]["beneficial_recovery_rate"] == 1.0
