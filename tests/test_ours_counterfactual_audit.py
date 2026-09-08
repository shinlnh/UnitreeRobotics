import numpy as np

from unitree_gr00t.ours_counterfactual_audit import summarize_option_targets


def test_counterfactual_audit_detects_diverse_strict_preferences() -> None:
    values = np.zeros((3, 6), dtype=np.float32)
    valid = np.zeros((3, 6), dtype=np.bool_)
    valid[:, [0, 4, 5]] = True
    values[0, [0, 4, 5]] = [2.0, 1.0, 0.0]
    values[1, [0, 4, 5]] = [0.0, 3.0, 1.0]
    values[2, [0, 4, 5]] = [1.0, 1.0, 1.0]

    audit = summarize_option_targets(values, valid, np=np)

    assert audit["labeled_states"] == 3
    assert audit["strict_preferences"] == 2
    assert audit["ties"] == 1
    assert audit["winning_options"] == {"ACCEPT_B": 1, "ADVANCE": 1}
    assert audit["distinct_winning_options"] == 2


def test_counterfactual_audit_rejects_single_valid_option_rows() -> None:
    values = np.zeros((2, 6), dtype=np.float32)
    valid = np.zeros((2, 6), dtype=np.bool_)
    valid[:, 5] = True

    audit = summarize_option_targets(values, valid, np=np)

    assert audit["labeled_states"] == 0
    assert audit["distinct_winning_options"] == 0
