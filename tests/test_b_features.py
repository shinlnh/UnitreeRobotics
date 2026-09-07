import json
from pathlib import Path

from unitree_gr00t.b_features import episode_feature_seed, feature_run_contract


def test_episode_feature_seed_is_stable_and_episode_specific() -> None:
    assert episode_feature_seed(7, 12) == episode_feature_seed(7, 12)
    assert episode_feature_seed(7, 12) != episode_feature_seed(7, 13)


def test_feature_run_contract_binds_index_checkpoint_and_extraction(tmp_path: Path) -> None:
    index = tmp_path / "index"
    checkpoint = tmp_path / "checkpoint"
    index.mkdir()
    checkpoint.mkdir()
    (index / "manifest.json").write_text(json.dumps({"experiment_id": "B"}), encoding="utf-8")
    (checkpoint / "a1_training_provenance.json").write_text(
        json.dumps({"experiment_id": "A1"}), encoding="utf-8"
    )
    checkpoint_contract = type(
        "Contract",
        (),
        {"checkpoint_dir": checkpoint, "action_horizon": 16},
    )()
    payload = feature_run_contract(
        index=index,
        index_manifest={"episodes_sha256": "episodes", "samples_sha256": "samples"},
        checkpoint_contract=checkpoint_contract,
        checkpoint_provenance={"training_dataset_revision": "revision"},
        checkpoint_weight_hashes={"model.safetensors": "weights"},
        device="cuda:0",
        batch_size=64,
        seed=7,
    )
    assert payload["experiment_id"] == "B"
    assert payload["samples_sha256"] == "samples"
    assert payload["checkpoint_weight_shards_sha256"] == {"model.safetensors": "weights"}
    assert payload["batch_size"] == 64
