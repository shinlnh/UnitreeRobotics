import json
from pathlib import Path

import pytest

from unitree_gr00t.a1 import sha256_file
from unitree_gr00t.b import (
    B_CHECKPOINT_PROVENANCE,
    B_ID,
    B_METHOD,
    B_PAPER,
    B_VARIANT,
    BContractError,
    StopConfirmationState,
    build_ordinal_targets,
    candidate_validity,
    confirm_stop,
    inspect_selector_checkpoint,
    select_unified_candidate,
    verify_a1_weight_hashes,
)


def test_candidate_validity_masks_trajectory_end_and_h8() -> None:
    assert (
        candidate_validity(decision_step=7, trajectory_steps=10, horizon=16, max_prefix=8)
        == (True, True, True, True) + (False,) * 13
    )


def test_successful_ordinal_targets_prefer_nearest_covering_prefix() -> None:
    targets = build_ordinal_targets(
        decision_step=10,
        trajectory_steps=100,
        horizon=16,
        successful=True,
        completion_step=14,
    )
    assert targets.stop_label == 0
    assert targets.stop_weight == 3.0
    assert targets.priorities[4] > targets.priorities[3]
    assert targets.priorities[4] > targets.priorities[5]
    assert (0, 4) in targets.comparable_pairs


def test_successful_post_boundary_targets_prefer_stop() -> None:
    targets = build_ordinal_targets(
        decision_step=42,
        trajectory_steps=100,
        horizon=16,
        successful=True,
        completion_step=42,
    )
    assert targets.stop_label == 1
    assert targets.priorities[0] == 17
    assert targets.priorities[0] > max(targets.priorities[1:])
    assert targets.stop_weight == 3.0


def test_failed_targets_tie_prefixes_above_stop() -> None:
    targets = build_ordinal_targets(
        decision_step=90,
        trajectory_steps=100,
        horizon=16,
        successful=False,
        completion_step=None,
    )
    assert targets.rank_weight == 0.1
    assert targets.stop_weight == 1.5
    assert set(
        priority
        for priority, valid in zip(targets.priorities[1:], targets.valid[1:], strict=True)
        if valid
    ) == {1}
    assert all(0 in pair for pair in targets.comparable_pairs)


def test_unified_selection_masks_invalid_and_confirms_stop() -> None:
    assert select_unified_candidate([100.0, 1.0, 2.0], [False, True, True]) == 2
    first = confirm_stop(0, StopConfirmationState(), confirmation_window=2)
    second = confirm_stop(0, first.state, confirmation_window=2)
    action = confirm_stop(3, first.state, confirmation_window=2)
    assert first.stop_pending and not first.stop_committed
    assert second.stop_committed and second.state.streak == 0
    assert action.executed_prefix_length == 3 and action.state.streak == 0


def test_selector_checkpoint_fails_closed_on_weight_drift(tmp_path: Path) -> None:
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"selector")
    provenance = {
        "experiment_id": B_ID,
        "variant": B_VARIANT,
        "method": B_METHOD,
        "paper": B_PAPER,
        "action_horizon": 16,
        "context_width": 2048,
        "weights_sha256": sha256_file(weights),
    }
    (tmp_path / B_CHECKPOINT_PROVENANCE).write_text(json.dumps(provenance), encoding="utf-8")
    audit, _ = inspect_selector_checkpoint(
        tmp_path, expected_action_horizon=16, expected_context_width=2048
    )
    assert audit.valid
    weights.write_bytes(b"drift")
    with pytest.raises(BContractError, match="weight hash"):
        inspect_selector_checkpoint(
            tmp_path, expected_action_horizon=16, expected_context_width=2048
        )


def test_b_parent_weight_hashes_fail_closed_on_drift(tmp_path: Path) -> None:
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"a1")
    contract = type(
        "Contract",
        (),
        {"checkpoint_dir": tmp_path, "weight_shards": (shard.name,)},
    )()
    expected = {shard.name: sha256_file(shard)}
    assert verify_a1_weight_hashes(contract, expected) == expected
    shard.write_bytes(b"drift")
    with pytest.raises(BContractError, match="differ"):
        verify_a1_weight_hashes(contract, expected)
