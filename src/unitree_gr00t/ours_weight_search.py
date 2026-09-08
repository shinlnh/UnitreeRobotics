"""Rank the frozen residual binary-loss sweep without rollout-seed access."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .a1 import sha256_file
from .ours import OursContractError
from .ours_train import OURS_CHECKPOINT_PROVENANCE

FROZEN_OVERRIDE_WEIGHTS = {
    "w005": 0.05,
    "w010": 0.10,
    "w025": 0.25,
    "w050": 0.50,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-variants-per-weight", type=int, default=6)
    return parser


def rank_weight_variants(
    registries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for weight_id, registry in sorted(registries.items()):
        for source in registry["variants"]:
            row = dict(source)
            source_variant_id = str(row["variant_id"])
            row.update(
                {
                    "variant_id": f"{weight_id}-{source_variant_id}",
                    "source_variant_id": source_variant_id,
                    "override_weight_id": weight_id,
                    "option_override_weight": FROZEN_OVERRIDE_WEIGHTS[weight_id],
                }
            )
            rows.append(row)
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["option_validation"]["selective_recovery"][
                "beneficial_recovery_rate"
            ],
            -row["option_validation"]["selective_recovery"]["true_recovery_rate"],
            row["option_validation"]["selective_recovery"]["mean_decision_regret"],
            row["variant_id"],
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["offline_option_rank"] = rank
    return ranked


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_variants_per_weight < 1:
        raise ValueError("expected variants per weight must be positive")
    root = args.search_root.expanduser().resolve()
    registries: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    found = {path.parent.name: path for path in root.glob("*/residual_registry.json")}
    if set(found) != set(FROZEN_OVERRIDE_WEIGHTS):
        raise OursContractError("residual override-weight registry inventory is incomplete")
    for weight_id, path in sorted(found.items()):
        registry = json.loads(path.read_text(encoding="utf-8"))
        if (
            not bool(registry.get("complete"))
            or not bool(registry.get("residual_retry_baseline"))
            or len(registry.get("variants", [])) != args.expected_variants_per_weight
        ):
            raise OursContractError(f"invalid residual registry: {path}")
        for row in registry["variants"]:
            checkpoint = Path(row["checkpoint"]).expanduser().resolve()
            provenance_path = checkpoint / OURS_CHECKPOINT_PROVENANCE
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if (
                float(provenance.get("option_override_weight", -1.0))
                != FROZEN_OVERRIDE_WEIGHTS[weight_id]
                or not bool(provenance.get("residual_option_advantages"))
                or provenance.get("option_sampling")
                != "balanced-residual-override-and-retry-tie-v1"
            ):
                raise OursContractError(
                    f"checkpoint does not match residual weight sweep: {checkpoint}"
                )
        registries[weight_id] = registry
        source_hashes[str(path)] = sha256_file(path)
    ranked = rank_weight_variants(registries)
    selected = ranked[0]
    result = {
        "schema_version": 1,
        "experiment_id": "Ours",
        "stage": "R0-residual-binary-loss-weight-search",
        "scope": "train-seed held-out episodes only; no rollout development seed",
        "frozen_override_weights": FROZEN_OVERRIDE_WEIGHTS,
        "selection_rule": (
            "beneficial recovery at false override <=5%, then recovery recall, "
            "decision regret, registered id"
        ),
        "source_registry_sha256": source_hashes,
        "selected_variant": selected["variant_id"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_option_value_margin": selected["option_validation"][
            "selective_recovery"
        ]["option_value_margin"],
        "variants": ranked,
        "complete": True,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(output)
    return result


def main() -> int:
    print(json.dumps(run(_parser().parse_args()), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
