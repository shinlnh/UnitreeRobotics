import numpy as np

from unitree_gr00t.ours_rollout_audit import calibrate_rollout_signal


def test_rollout_calibration_uses_only_stop_proposals() -> None:
    result = calibrate_rollout_signal(
        np.asarray([0.9, 0.8, 0.7, 0.1], dtype=np.float32),
        np.asarray([True, False, True, False], dtype=np.bool_),
        np.asarray([True, True, False, True], dtype=np.bool_),
        max_false_positive_rate=0.0,
        np=np,
    )
    assert result["calibratable"]
    assert result["stop_samples"] == 3
    assert result["completion_positives"] == 1
    assert result["threshold"] == np.float32(0.9)


def test_rollout_calibration_reports_degenerate_slice() -> None:
    result = calibrate_rollout_signal(
        np.asarray([0.2, 0.3], dtype=np.float32),
        np.asarray([False, False], dtype=np.bool_),
        np.asarray([True, True], dtype=np.bool_),
        max_false_positive_rate=0.05,
        np=np,
    )
    assert not result["calibratable"]
    assert result["completion_positives"] == 0
