import hashlib
import json
from pathlib import Path

import pytest

from unitree_gr00t.ours import OursContractError
from unitree_gr00t.ours_data import (
    audit_recovery_corpus,
    build_demonstration_targets,
    temporal_predecessors,
)
from unitree_gr00t.ours_prepare import ordered_episode_rows


def test_demonstration_targets_and_histories_are_causal_and_subgoal_local() -> None:
    samples = [
        {
            "sample_index": 10,
            "episode_index": 2,
            "subgoal_index": 0,
            "frame_index": 4,
            "completion_step": 12,
        },
        {
            "sample_index": 11,
            "episode_index": 2,
            "subgoal_index": 0,
            "frame_index": 12,
            "completion_step": 12,
        },
        {
            "sample_index": 12,
            "episode_index": 2,
            "subgoal_index": 0,
            "frame_index": 20,
            "completion_step": 12,
        },
        {
            "sample_index": 13,
            "episode_index": 2,
            "subgoal_index": 1,
            "frame_index": 20,
            "completion_step": 30,
        },
    ]
    subgoals = [
        {"filtered_start": 4, "filtered_end": 12},
        {"filtered_start": 20, "filtered_end": 30},
    ]
    targets = build_demonstration_targets(samples, subgoals)
    assert [target.progress for target in targets] == [0.0, 1.0, 1.0, 0.0]
    assert [target.complete for target in targets] == [False, True, True, False]
    assert temporal_predecessors(targets, 3) == (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 2),
        (3, 3, 3),
    )


def _write_corpus(root: Path) -> None:
    features = root / "features"
    features.mkdir(parents=True)
    payload = b"compact"
    (features / "episode_000001.npz").write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "experiment_id": "Ours",
        "variant": "GR00T-RC-SparkVLA-counterfactual-temporal-recovery",
        "method": "counterfactual-temporal-recovery",
        "parent_experiment": "B",
        "selector_weights_sha256": "selector",
        "consumed_rollout_base_seeds": [10007, 20007],
        "files": 1,
        "samples": 3,
        "files_sha256": {"episode_000001.npz": hashlib.sha256(payload).hexdigest()},
        "split_episodes": {"train": [1], "development": [2]},
        "label_counts": {"completion_negative": 2, "completion_positive": 1},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_corpus_audit_binds_selector_hash_splits_and_final_seed(tmp_path: Path) -> None:
    _write_corpus(tmp_path)
    audit = audit_recovery_corpus(tmp_path, expected_selector_sha256="selector")
    assert audit.valid
    assert audit.samples == 3

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["consumed_rollout_base_seeds"].append(7)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(OursContractError, match="final held-out"):
        audit_recovery_corpus(tmp_path)


def test_ordered_episode_rows_rejects_noncausal_source() -> None:
    samples = [
        {"episode_index": 2, "sample_index": 1, "subgoal_index": 0, "frame_index": 8},
        {"episode_index": 2, "sample_index": 0, "subgoal_index": 0, "frame_index": 16},
    ]
    with pytest.raises(ValueError, match="not causal"):
        ordered_episode_rows(samples, 2)
