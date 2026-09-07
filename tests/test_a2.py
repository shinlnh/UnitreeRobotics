import json
from pathlib import Path

import pytest

from unitree_gr00t.a0 import BenchmarkCase, TaskDescription
from unitree_gr00t.a0_merge import _compatible, _create_manifest
from unitree_gr00t.a2 import (
    A2_PLAN_SOURCE,
    A2ContractError,
    audit_fixed_hierarchy,
    build_fixed_plan,
    hierarchy_audit_payload,
    select_fixed_prefix_length,
    select_subgoal,
)


def _task() -> TaskDescription:
    return TaskDescription(
        instruction="Put both objects away.",
        steps=("Pick up object", "Put object in box"),
        start_indices=(0, 42),
    )


def test_fixed_hierarchy_switches_only_at_frozen_anchors() -> None:
    plan = build_fixed_plan(_task(), 150)

    first = select_subgoal(plan, 149)
    second = select_subgoal(plan, 150)

    assert first.subgoal_index == 0
    assert first.subgoal_instruction == "Pick up object"
    assert first.steps_remaining_in_segment == 1
    assert second.subgoal_index == 1
    assert second.subgoal_instruction == "Put object in box"
    assert second.steps_remaining_in_segment == 150
    assert select_fixed_prefix_length(first, 16, 151) == 1
    assert select_fixed_prefix_length(second, 16, 150) == 16


def test_fixed_plan_is_deterministic_and_rejects_contract_drift() -> None:
    left = build_fixed_plan(_task(), 150)
    right = build_fixed_plan(_task(), 150)
    assert left.sha256 == right.sha256
    assert len(left.sha256) == 64
    assert left.source == A2_PLAN_SOURCE

    with pytest.raises(A2ContractError, match="source is not frozen"):
        build_fixed_plan(_task(), 150, source="generated online")
    with pytest.raises(A2ContractError, match="outside"):
        select_subgoal(left, left.max_steps)


def test_hierarchy_audit_reads_canonical_benchmark_steps(tmp_path: Path) -> None:
    case_root = tmp_path / "Ideal" / "case1"
    case_root.mkdir(parents=True)
    (case_root / "task_description.txt").write_text(
        "Task: Put both objects away.\n"
        "Step: Pick up object\n"
        "[0, 42]\n"
        "Step: Put object in box\n"
        "[42, 99]\n",
        encoding="utf-8",
    )
    case = BenchmarkCase("Ideal", "case1", case_root)

    audit = audit_fixed_hierarchy([case], 150)

    assert audit.valid
    assert audit.plans == 1
    assert audit.subgoals == 2
    assert audit.unique_plans == 1
    assert len(audit.plan_sha256) == 1
    assert hierarchy_audit_payload(audit)["plan_sha256"] == list(audit.plan_sha256)


def test_hierarchy_audit_fails_closed_on_malformed_plan(tmp_path: Path) -> None:
    case_root = tmp_path / "Ideal" / "case1"
    case_root.mkdir(parents=True)
    (case_root / "task_description.txt").write_text(
        json.dumps({"not": "a task description"}), encoding="utf-8"
    )

    audit = audit_fixed_hierarchy([BenchmarkCase("Ideal", "case1", case_root)], 150)

    assert not audit.valid
    assert audit.plans == 0
    assert len(audit.issues) == 1


def test_shard_manifest_merge_aggregates_hierarchy_audit(tmp_path: Path) -> None:
    common = {
        "experiment_id": "A2",
        "trials_per_case": 10,
        "hierarchy": True,
        "checkpoint": "/checkpoint",
    }
    left = common | {
        "task_types": ["Ideal"],
        "cases": ["Ideal/case1"],
        "expected_episodes": 10,
        "output_dir": "/left",
        "hierarchy_audit": {
            "cases": 1,
            "plans": 1,
            "subgoals": 2,
            "unique_plans": 1,
            "plan_sha256": ["a"],
            "valid": True,
            "issues": [],
            "planner": "fixed",
        },
    }
    right = common | {
        "task_types": ["Memory_Execution"],
        "cases": ["Memory_Execution/case1"],
        "expected_episodes": 10,
        "output_dir": "/right",
        "hierarchy_audit": {
            "cases": 1,
            "plans": 1,
            "subgoals": 3,
            "unique_plans": 1,
            "plan_sha256": ["b"],
            "valid": True,
            "issues": [],
            "planner": "fixed",
        },
    }

    assert _compatible(left, right)
    merged = _create_manifest(tmp_path, [left, right])
    assert merged["expected_episodes"] == 20
    assert merged["hierarchy_audit"]["cases"] == 2
    assert merged["hierarchy_audit"]["subgoals"] == 5
    assert merged["hierarchy_audit"]["plan_sha256"] == ["a", "b"]
