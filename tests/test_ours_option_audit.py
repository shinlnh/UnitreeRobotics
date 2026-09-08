import warnings

import numpy as np

from unitree_gr00t.ours_option_audit import summarize_option_predictions


def test_option_audit_reports_multiclass_and_accept_recover_errors() -> None:
    targets = np.asarray(
        [
            [3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
            [0.0, 3.0, 2.0, 1.0, -1.0, -2.0],
            [0.0, 1.0, 3.0, 2.0, -1.0, -2.0],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(targets, dtype=np.bool_)
    predictions = np.asarray(
        [
            [2.0, 1.0, 0.0, -1.0, -2.0, -3.0],
            [4.0, 3.0, 2.0, 1.0, 0.0, -1.0],
            [0.0, 1.0, 3.0, 2.0, -1.0, -2.0],
        ],
        dtype=np.float32,
    )

    result = summarize_option_predictions(targets, valid, predictions, np=np)

    assert result["strict_states"] == 3
    assert result["top1_accuracy_strict"] == 2 / 3
    assert result["accept_recover"]["accuracy"] == 2 / 3
    assert result["accept_recover"]["missed_recovery_rate"] == 0.5
    assert result["mean_decision_regret"] == 1.0


def test_option_audit_does_not_subtract_masked_infinities() -> None:
    targets = np.asarray([[2.0, 1.0, 0.0, -1.0, -2.0, -3.0]], dtype=np.float32)
    valid = np.asarray([[True, True, False, False, False, False]])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = summarize_option_predictions(targets, valid, targets, np=np)

    assert result["pairwise_comparisons"] == 1
    assert result["pairwise_ranking_accuracy"] == 1.0
