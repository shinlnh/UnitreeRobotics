import warnings

import numpy as np

from unitree_gr00t.ours_option_audit import (
    summarize_option_predictions,
    summarize_residual_predictions,
)


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


def test_option_audit_calibrates_a_safe_recovery_margin() -> None:
    targets = np.asarray(
        [
            [2.0, 1.0, 0.0, 0.0, -1.0, 0.0],
            [2.0, 1.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(targets, dtype=np.bool_)
    predictions = np.asarray(
        [
            [0.0, 0.1, -1.0, -1.0, -1.0, -1.0],
            [0.0, -0.1, -1.0, -1.0, -1.0, -1.0],
            [0.0, 0.3, -1.0, -1.0, -1.0, -1.0],
            [0.0, 0.2, -1.0, -1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )

    result = summarize_option_predictions(
        targets,
        valid,
        predictions,
        np=np,
        max_false_recovery_rate=0.0,
    )

    calibration = result["selective_recovery"]
    assert np.isclose(calibration["option_value_margin"], 0.2)
    assert calibration["false_recovery_rate"] == 0.0
    assert calibration["true_recovery_rate"] == 1.0
    assert calibration["beneficial_recovery_rate"] == 1.0


def test_residual_audit_calibrates_against_retry_on_confirmed_stops() -> None:
    targets = np.asarray(
        [
            [1.0, 0.0, 3.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 3.0, -1.0, 0.0, 0.0],
            [3.0, 0.0, 1.0, -1.0, 2.0, 0.0],
            [3.0, 0.0, 1.0, -1.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(targets, dtype=np.bool_)
    predictions = np.asarray(
        [
            [0.0, 0.0, 0.0, -1.0, 0.3, 0.0],
            [0.0, -0.1, 0.0, -1.0, -0.1, -0.1],
            [0.0, 0.0, 0.0, -1.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, -1.0, 0.4, 0.0],
        ],
        dtype=np.float32,
    )

    result = summarize_residual_predictions(
        targets,
        valid,
        predictions,
        np=np,
        max_false_recovery_rate=0.0,
    )

    assert result["baseline_option"] == "RETRY_CURRENT"
    assert result["target_retry_states"] == 2
    assert result["target_override_states"] == 2
    assert np.isclose(result["selective_recovery"]["option_value_margin"], 0.4)
    assert result["selective_recovery"]["false_recovery_rate"] == 0.0
    assert result["selective_recovery"]["beneficial_recovery_rate"] == 1.0


def test_residual_audit_counts_retry_ties_as_false_override_negatives() -> None:
    targets = np.asarray(
        [
            [-1.0, -1.0, 0.0, -1.0, 0.0, -1.0],
            [-1.0, -1.0, 0.0, -1.0, 1.0, -1.0],
        ],
        dtype=np.float32,
    )
    predictions = np.asarray(
        [
            [-1.0, -1.0, 0.0, -1.0, 0.3, -1.0],
            [-1.0, -1.0, 0.0, -1.0, 0.4, -1.0],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(targets, dtype=np.bool_)

    result = summarize_residual_predictions(
        targets,
        valid,
        predictions,
        np=np,
        max_false_recovery_rate=0.0,
    )

    assert result["strict_states"] == 1
    assert result["target_retry_states"] == 1
    assert result["target_tie_states"] == 1
    assert np.isclose(result["selective_recovery"]["option_value_margin"], 0.4)
    assert result["selective_recovery"]["false_recovery_rate"] == 0.0
    assert result["selective_recovery"]["true_recovery_rate"] == 1.0
