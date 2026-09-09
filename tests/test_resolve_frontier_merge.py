import json
from argparse import Namespace

import numpy as np
import pytest

from unitree_gr00t.a1 import sha256_file
from unitree_gr00t.ours import OursContractError
from unitree_gr00t.resolve_frontier_merge import merge


def _shard(root, selection_index: int) -> None:
    features = root / "features"
    features.mkdir(parents=True)
    filename = "anchor_000000.npz"
    np.savez(features / filename, value=np.asarray([selection_index]))
    row = {
        "anchor_index": 0,
        "selection_index": selection_index,
        "source_manifest_index": selection_index,
        "expert_reached": selection_index == 0,
        "baseline": {"reached": False},
        "expert_superiority": int(selection_index == 0),
        "feature_file": filename,
    }
    records = root / "anchors.jsonl"
    records.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest = {
        "experiment_id": "Ours",
        "stage": "RESOLVE train-only expert/B frontier arms",
        "frontier_manifest_sha256": "f" * 64,
        "runtime_manifest_sha256": "r" * 64,
        "runtime_provenance": {"model": "frozen"},
        "split": "train",
        "anchor_offset": 32,
        "selection_seed": 7,
        "anchors": 1,
        "anchors_sha256": sha256_file(records),
        "files_sha256": {filename: sha256_file(features / filename)},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_frontier_merge_preserves_selection_order_and_hashes(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _shard(left, 0)
    _shard(right, 1)
    output = tmp_path / "merged"
    manifest = merge(Namespace(shards=[right, left], output=output))
    assert manifest["anchors"] == 2
    assert manifest["positive_gaps"] == 1
    assert manifest["selected_anchors"] == 2
    rows = [json.loads(line) for line in (output / "anchors.jsonl").read_text().splitlines()]
    assert [row["selection_index"] for row in rows] == [0, 1]
    assert all(
        sha256_file(output / "features" / name) == digest
        for name, digest in manifest["files_sha256"].items()
    )


def test_frontier_merge_rejects_overlapping_shards(tmp_path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _shard(left, 0)
    _shard(right, 0)
    with pytest.raises(OursContractError, match="overlap"):
        merge(Namespace(shards=[left, right], output=tmp_path / "merged"))
