"""Audit and rank a registered Ours research sweep."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .a1 import sha256_file
from .ours import OURS_ID, OURS_METHOD, OURS_PARENT, OURS_VARIANT, OursContractError
from .ours_train import OURS_CHECKPOINT_PROVENANCE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-variants", type=int, default=12)
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_r0_registry(checkpoint_root: Path, expected_variants: int = 12) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    dataset_hashes: set[str] = set()
    for path in sorted(checkpoint_root.glob(f"*/{OURS_CHECKPOINT_PROVENANCE}")):
        record = json.loads(path.read_text(encoding="utf-8"))
        identity = (
            record.get("experiment_id"),
            record.get("variant"),
            record.get("method"),
            record.get("parent_experiment"),
        )
        if identity != (OURS_ID, OURS_VARIANT, OURS_METHOD, OURS_PARENT):
            raise OursContractError(f"invalid Ours search identity: {path}")
        weights_path = path.parent / "model.safetensors"
        if sha256_file(weights_path) != record.get("weights_sha256"):
            raise OursContractError(f"Ours search weight hash mismatch: {weights_path}")
        dataset_hashes.add(str(record["dataset_manifest_sha256"]))
        metrics = record["development_metrics"]
        completion = metrics["completion"]
        records.append(
            {
                "variant_id": path.parent.name,
                "checkpoint": str(path.parent.resolve()),
                "weights_sha256": record["weights_sha256"],
                "selected_step": int(record["selected_step"]),
                "parameter_count": int(record["parameter_count"]),
                "encoder": record["model"]["encoder"],
                "history_length": int(record["model"]["history_length"]),
                "temporal_layers": int(record["model"]["temporal_layers"]),
                "completion_threshold": float(metrics["completion_threshold"]),
                "false_positive_rate": float(completion["false_positive_rate"]),
                "true_positive_rate": float(completion["true_positive_rate"]),
                "precision": float(completion["precision"]),
                "brier": float(metrics["brier"]),
                "ece_15": float(metrics["ece_15"]),
                "progress_mae": float(metrics["progress_mae"]),
                "b_false_stop_proposals": int(metrics["b_false_stop_proposals"]),
                "gated_false_stops": int(metrics["gated_false_stops"]),
                "b_true_stop_proposals": int(metrics["b_true_stop_proposals"]),
                "gated_true_stops": int(metrics["gated_true_stops"]),
                "elapsed_seconds": float(record["elapsed_seconds"]),
            }
        )
    if len(records) != expected_variants:
        raise OursContractError(f"expected {expected_variants} R0 variants, found {len(records)}")
    if len(dataset_hashes) != 1:
        raise OursContractError("R0 variants do not share one frozen corpus")
    records.sort(
        key=lambda row: (
            -row["true_positive_rate"],
            row["brier"],
            row["progress_mae"],
            row["variant_id"],
        )
    )
    for rank, record in enumerate(records, start=1):
        record["offline_rank"] = rank
        record["false_stop_rejection_rate"] = 1.0 - (
            record["gated_false_stops"] / record["b_false_stop_proposals"]
        )
    return {
        "schema_version": 1,
        "experiment_id": OURS_ID,
        "round": "R0",
        "selection_scope": "offline sanity only; no rollout winner selected",
        "ranking": "TPR at FPR<=5%, then Brier, progress MAE, registered id",
        "dataset_manifest_sha256": next(iter(dataset_hashes)),
        "variants": records,
        "complete": True,
    }


def main() -> int:
    args = _parser().parse_args()
    payload = build_r0_registry(args.checkpoints, args.expected_variants)
    _write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
