import pytest

from unitree_gr00t.ours import OursContractError
from unitree_gr00t.resolve_pilot_audit import summarize_rows


def arm(reach: int, tag: str) -> dict[str, object]:
    return {
        "reachability": [reach],
        "action_trace_sha256": f"action-{tag}",
        "final_state_sha256": f"state-{tag}",
    }


def test_audit_distinguishes_rescue_redundancy_and_regression() -> None:
    rows = [
        {
            "base_seed": 1,
            "sample_index": 2,
            "recovery": arm(1, "r0"),
            "deletions": [arm(0, "d0")],
            "baselines": [arm(0, "d0")],
            "program_crb": [1.0],
        },
        {
            "base_seed": 1,
            "sample_index": 2,
            "recovery": arm(0, "r1"),
            "deletions": [arm(0, "d0")],
            "baselines": [arm(0, "d0")],
            "program_crb": [0.0],
        },
    ]
    report = summarize_rows(rows, ("subtask",))
    assert report["anchors"] == 1
    assert report["milestones"]["subtask"] == {
        "programs": 2,
        "recovery_reaches": 1,
        "baseline_zero_reaches": 0,
        "strict_crb_positive": 1,
        "recovery_but_not_deletion_minimal": 0,
        "baseline_regressions": 0,
        "joint_exclusive_rescues": 1,
    }


def test_audit_rejects_noncanonical_identical_arms() -> None:
    row = {
        "base_seed": 1,
        "sample_index": 2,
        "recovery": arm(1, "r"),
        "deletions": [arm(0, "d")],
        "baselines": [arm(0, "b")],
        "program_crb": [1.0],
    }
    with pytest.raises(OursContractError, match="D_last"):
        summarize_rows([row], ("subtask",))
