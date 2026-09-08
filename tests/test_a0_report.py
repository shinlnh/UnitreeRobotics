import json
from pathlib import Path

from unitree_gr00t.a0_report import (
    _paired_delta,
    _scan_decisions,
    cluster_bootstrap_mean,
    cluster_bootstrap_ratio,
    metric_block,
    wilson_interval,
)


def _row(case: str, completed: int, final: bool) -> dict[str, object]:
    return {
        "task_type": "Ideal",
        "case": case,
        "agent_completed_subtasks": completed,
        "possible_subtasks": 4,
        "reached_success": final,
        "final_success": final,
        "post_success_reactivation": False,
        "policy_calls": 2,
        "steps": 16,
        "predicted_actions": 32,
        "policy_inference_seconds": 0.2,
        "elapsed_seconds": 1.0,
        "total_simulator_steps": 20,
        "control_frequency_hz": 20,
        "injection_count": 0,
        "steps_after_first_success": 0,
        "first_success_step": 16 if final else None,
    }


def test_wilson_interval_handles_extreme_binary_rate() -> None:
    interval = wilson_interval(0, 10)
    assert interval["estimate"] == 0.0
    assert interval["lower_95"] == 0.0
    assert 0.27 < interval["upper_95"] < 0.29


def test_cluster_bootstrap_is_deterministic() -> None:
    rows = [_row("case1", 1, False), _row("case2", 3, True)]
    first = cluster_bootstrap_mean(
        rows,
        lambda row: row["agent_completed_subtasks"] / row["possible_subtasks"],
        lambda row: str(row["case"]),
        samples=1000,
        seed=7,
    )
    second = cluster_bootstrap_mean(
        rows,
        lambda row: row["agent_completed_subtasks"] / row["possible_subtasks"],
        lambda row: str(row["case"]),
        samples=1000,
        seed=7,
    )
    assert first == second
    assert first["estimate"] == 0.5


def test_cluster_bootstrap_ratio_uses_pooled_denominator() -> None:
    rows = [_row("case1", 1, False), _row("case2", 4, True)]
    rows[0]["possible_subtasks"] = 2
    interval = cluster_bootstrap_ratio(
        rows,
        lambda row: float(row["agent_completed_subtasks"]),
        lambda row: float(row["possible_subtasks"]),
        lambda row: str(row["case"]),
        samples=1000,
        seed=7,
    )
    assert interval["estimate"] == 5 / 6


def test_metric_block_separates_partial_and_strict_success() -> None:
    block = metric_block([_row("case1", 1, False), _row("case2", 4, True)], samples=1000, seed=7)
    assert block["paper_subtask_success_rate"]["estimate"] == 0.625
    assert block["micro_subtask_completion_rate"] == 0.625
    assert block["reference_pooled_subtask_success_rate"]["estimate"] == 0.625
    assert block["strict_full_task_success_rate"]["estimate"] == 0.5
    assert block["action_chunk_utilization"] == 0.5


def test_metric_block_keeps_task_macro_and_pooled_rates_separate() -> None:
    rows = [_row("case1", 1, False), _row("case2", 4, True)]
    rows[0]["possible_subtasks"] = 2
    block = metric_block(rows, samples=1000, seed=7)
    assert block["paper_subtask_success_rate"]["estimate"] == 0.75
    assert block["reference_pooled_subtask_success_rate"]["estimate"] == 5 / 6


def test_b_chunk_utilization_excludes_post_success_controller_holds() -> None:
    row = _row("case1", 4, True)
    row["selector_executed_actions"] = 8
    block = metric_block([row], samples=1000, seed=7)
    assert block["total_executed_steps"] == 16
    assert block["total_selector_executed_actions"] == 8
    assert block["action_chunk_utilization"] == 0.25


def test_stability_is_conditioned_on_ordered_goal_reach() -> None:
    terminal_only = _row("case1", 1, True)
    terminal_only["reached_success"] = False
    terminal_only["first_success_step"] = None
    block = metric_block([terminal_only], samples=1000, seed=7)
    assert block["strict_full_task_success_rate"]["estimate"] == 1.0
    assert block["ever_reached_full_success_rate"]["estimate"] == 0.0
    assert block["stable_success_given_reached"]["estimate"] is None


def test_decision_scan_counts_controller_monitoring_as_steps_not_policy_calls(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "policy_latency_seconds": 0.1,
            "selected_prefix_length": 2,
            "prefix_selection": "learned_unified_stop_prefix",
            "active_subgoal_index_before": 0,
            "transitions": [{}, {}],
        },
        {
            "policy_invoked": False,
            "policy_latency_seconds": 0.0,
            "selected_prefix_length": 3,
            "prefix_selection": "post_success_controller_hold",
            "transitions": [{}, {}, {}],
        },
    ]
    path = tmp_path / "decisions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    metrics = _scan_decisions(path)
    assert metrics["policy_calls"] == 1
    assert metrics["executed_transitions"] == 5
    assert metrics["selected_prefix_histogram"] == {2: 1}


def test_paired_delta_recognizes_request_local_diffusion_seeds() -> None:
    row = _row("case1", 2, False) | {"trial": 0}
    derivation = "sha256(B-decision-v1:episode_seed:decision_index)[:31-bit]"
    left = {
        "label": "H16",
        "manifest": {"decision_seed_derivation": derivation},
        "episodes": [row],
    }
    right = {
        "label": "H8",
        "manifest": {"decision_seed_derivation": derivation},
        "episodes": [dict(row)],
    }
    comparison = _paired_delta(left, right, samples=1000, seed=7)
    assert "request-local policy diffusion seeds" in comparison["pairing_note"]
