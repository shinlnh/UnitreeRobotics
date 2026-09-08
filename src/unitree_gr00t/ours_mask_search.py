"""Search all fixed residual recovery-option libraries on train-seed holdouts."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any

from .ours_option_audit import (
    bound_counterfactual_datasets,
    summarize_residual_predictions,
)
from .ours_train import _batch, inspect_recovery_checkpoint, load_corpus
from .ours_model import TemporalRecoveryModelConfig, build_temporal_recovery_model
from .ours_weight_search import FROZEN_OVERRIDE_WEIGHTS

OVERRIDE_OPTIONS = (
    "REOBSERVE",
    "BACKTRACK_ONE",
    "ADVANCE",
    "CONSENSUS_PREFIX",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-checkpoints", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    return parser


def enumerate_override_masks() -> list[tuple[str, ...]]:
    return [
        tuple(mask)
        for size in range(1, len(OVERRIDE_OPTIONS) + 1)
        for mask in itertools.combinations(OVERRIDE_OPTIONS, size)
    ]


def rank_mask_variants(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["option_validation"]["selective_recovery"][
                "beneficial_recovery_states"
            ],
            -row["option_validation"]["selective_recovery"][
                "beneficial_recovery_rate_all_states"
            ],
            -row["option_validation"]["selective_recovery"]["true_recovery_rate"],
            row["option_validation"]["selective_recovery"]["mean_decision_regret"],
            row["variant_id"],
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["offline_option_rank"] = rank
    return ranked


def _mask_id(mask: tuple[str, ...]) -> str:
    aliases = {
        "REOBSERVE": "reobserve",
        "BACKTRACK_ONE": "backtrack",
        "ADVANCE": "advance",
        "CONSENSUS_PREFIX": "consensus",
    }
    return "+".join(aliases[option] for option in mask)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.expected_checkpoints, args.batch_size) < 1:
        raise ValueError("mask search counts must be positive")
    root = args.checkpoint_root.expanduser().resolve()
    checkpoints = sorted(path.parent for path in root.glob("*/*/model.safetensors"))
    if len(checkpoints) != args.expected_checkpoints:
        raise ValueError(
            f"expected {args.expected_checkpoints} residual checkpoints, found {len(checkpoints)}"
        )

    import numpy as np
    import torch
    from safetensors.torch import load_file

    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        weight_id = checkpoint.parent.name
        audit, provenance = inspect_recovery_checkpoint(checkpoint)
        if (
            weight_id not in FROZEN_OVERRIDE_WEIGHTS
            or float(provenance.get("option_override_weight", -1.0))
            != FROZEN_OVERRIDE_WEIGHTS[weight_id]
            or provenance.get("option_sampling")
            != "balanced-residual-override-and-retry-tie-v1"
        ):
            raise ValueError(f"checkpoint is outside the frozen mask search: {checkpoint}")
        config = TemporalRecoveryModelConfig(**provenance["model"])
        corpora = [
            load_corpus(path, config.history_length, np)
            for path in bound_counterfactual_datasets(provenance)
        ]
        model = build_temporal_recovery_model(config).to(args.device)
        model.load_state_dict(
            load_file(checkpoint / "model.safetensors", device=args.device)
        )
        model.eval()
        targets: list[Any] = []
        validity: list[Any] = []
        predictions: list[Any] = []
        with torch.inference_mode():
            for corpus in corpora:
                ids = corpus.development_ids[
                    corpus.target_option_valid[corpus.development_ids].sum(axis=1) >= 2
                ]
                targets.append(corpus.target_option_values[ids])
                validity.append(corpus.target_option_valid[ids])
                for start in range(0, len(ids), args.batch_size):
                    selected = ids[start : start + args.batch_size]
                    batch = _batch(
                        corpus, selected, device=args.device, np=np, torch=torch
                    )
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=args.device.startswith("cuda"),
                    ):
                        output = model(
                            batch["contexts"],
                            batch["anchors"],
                            batch["actions"],
                            batch["selector"],
                            batch["scalars"],
                        )
                    predictions.append(output["option_values"].float().cpu().numpy())
        target = np.concatenate(targets)
        valid = np.concatenate(validity)
        prediction = np.concatenate(predictions)
        for mask in enumerate_override_masks():
            option_validation = summarize_residual_predictions(
                target,
                valid,
                prediction,
                np=np,
                max_false_recovery_rate=0.05,
                allowed_override_options=mask,
            )
            mask_id = _mask_id(mask)
            rows.append(
                {
                    "variant_id": f"{weight_id}-{checkpoint.name}-{mask_id}",
                    "checkpoint_variant": checkpoint.name,
                    "checkpoint": str(checkpoint),
                    "weights_sha256": audit.weights_sha256,
                    "override_weight_id": weight_id,
                    "option_override_weight": FROZEN_OVERRIDE_WEIGHTS[weight_id],
                    "option_mask_id": mask_id,
                    "allowed_override_options": list(mask),
                    "option_validation": option_validation,
                }
            )
        del model, corpora
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    ranked = rank_mask_variants(rows)
    selected = ranked[0]
    beneficial = selected["option_validation"]["selective_recovery"][
        "beneficial_recovery_states"
    ]
    result = {
        "schema_version": 1,
        "experiment_id": "Ours",
        "stage": "R0-residual-option-library-search",
        "scope": "train-seed held-out episodes only; no rollout development seed",
        "registered_masks": len(enumerate_override_masks()),
        "registered_checkpoints": len(checkpoints),
        "selection_rule": (
            "beneficial recovery count at false override <=5%, normalized benefit, "
            "recovery recall, regret, registered id"
        ),
        "selected_variant": selected["variant_id"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_allowed_override_options": selected["allowed_override_options"],
        "selected_option_value_margin": selected["option_validation"][
            "selective_recovery"
        ]["option_value_margin"],
        "offline_positive_candidate": beneficial > 0,
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
