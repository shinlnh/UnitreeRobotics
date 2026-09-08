from types import SimpleNamespace

import numpy as np

from unitree_gr00t.ours_train import calibrate_threshold, sample_training_ids


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
        target_complete=np.asarray([True] * 4 + [False] * 8, dtype=np.bool_),
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
