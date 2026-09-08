"""Audit train-only counterfactual option heads without consuming rollout seeds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .ours import RECOVERY_OPTIONS
from .ours_model import TemporalRecoveryModelConfig, build_temporal_recovery_model
from .ours_train import _batch, inspect_recovery_checkpoint, load_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    return parser


def summarize_option_predictions(
    target_values: Any,
    target_valid: Any,
    predictions: Any,
    *,
    np: Any,
    max_false_recovery_rate: float = 0.05,
) -> dict[str, Any]:
    """Summarize strict multiclass and runtime accept/recover decisions."""

    if (
        target_values.ndim != 2
        or target_values.shape != target_valid.shape
        or target_values.shape != predictions.shape
        or target_values.shape[1] != len(RECOVERY_OPTIONS)
        or not 0.0 <= max_false_recovery_rate <= 1.0
    ):
        raise ValueError("option audit arrays have incompatible shapes")
    labeled = target_valid.sum(axis=1) >= 2
    if not labeled.any():
        raise ValueError("option audit requires at least one multiply-labeled state")
    target = np.where(target_valid[labeled], target_values[labeled], -np.inf)
    prediction = np.where(target_valid[labeled], predictions[labeled], -np.inf)
    ordered = np.sort(target, axis=1)
    strict = ordered[:, -1] - ordered[:, -2] > 1e-6
    target_winner = target.argmax(axis=1)
    predicted_winner = prediction.argmax(axis=1)
    strict_target = target_winner[strict]
    strict_prediction = predicted_winner[strict]
    supports = np.bincount(strict_target, minlength=len(RECOVERY_OPTIONS))
    correct = np.bincount(
        strict_target[strict_target == strict_prediction],
        minlength=len(RECOVERY_OPTIONS),
    )
    recalls = np.divide(
        correct,
        supports,
        out=np.full(len(RECOVERY_OPTIONS), np.nan),
        where=supports > 0,
    )

    accept_ids = np.asarray(
        [RECOVERY_OPTIONS.index("ACCEPT_B"), RECOVERY_OPTIONS.index("ADVANCE")]
    )
    recovery_ids = np.asarray(
        [index for index in range(len(RECOVERY_OPTIONS)) if index not in accept_ids]
    )
    target_accept = target[:, accept_ids].max(axis=1)
    target_recovery = target[:, recovery_ids].max(axis=1)
    predicted_accept = prediction[:, accept_ids].max(axis=1)
    predicted_recovery = prediction[:, recovery_ids].max(axis=1)
    strict_group = np.abs(target_accept - target_recovery) > 1e-6
    target_recover = target_recovery > target_accept
    predicted_recover = predicted_recovery > predicted_accept

    pair_correct = 0
    pair_total = 0
    for left in range(len(RECOVERY_OPTIONS)):
        for right in range(left + 1, len(RECOVERY_OPTIONS)):
            valid_pair = target_valid[labeled, left] & target_valid[labeled, right]
            # Invalid options are represented by -inf.  Subtract only where
            # both options are valid so the audit never evaluates -inf - -inf.
            difference = np.zeros(len(target), dtype=target.dtype)
            difference[valid_pair] = (
                target[valid_pair, left] - target[valid_pair, right]
            )
            pair = valid_pair & (np.abs(difference) > 1e-6)
            pair_total += int(pair.sum())
            pair_correct += int(
                (
                    np.sign(prediction[pair, left] - prediction[pair, right])
                    == np.sign(difference[pair])
                ).sum()
            )

    selected_target = target[np.arange(len(target)), predicted_winner]
    target_best = target.max(axis=1)
    strict_target_accept = strict_group & ~target_recover
    strict_target_recover = strict_group & target_recover
    recovery_gap = predicted_recovery - predicted_accept
    predicted_recovery_id = recovery_ids[prediction[:, recovery_ids].argmax(axis=1)]
    predicted_accept_id = accept_ids[prediction[:, accept_ids].argmax(axis=1)]
    nonnegative_gaps = recovery_gap[
        np.isfinite(recovery_gap) & (recovery_gap >= 0.0)
    ]
    reject_all_margin = np.nextafter(
        np.float32(max(float(nonnegative_gaps.max()) if len(nonnegative_gaps) else 0.0, 0.0)),
        np.float32(np.inf),
    )
    margins = np.unique(
        np.concatenate((np.asarray([0.0, reject_all_margin]), nonnegative_gaps))
    )
    selected_calibration: tuple[tuple[float, float, float], dict[str, Any]] | None = None
    for margin in margins:
        calibrated_recover = recovery_gap >= margin
        false_recovery_rate = float(
            calibrated_recover[strict_target_accept].mean()
            if strict_target_accept.any()
            else 0.0
        )
        if false_recovery_rate > max_false_recovery_rate + 1e-12:
            continue
        true_recovery_rate = float(
            calibrated_recover[strict_target_recover].mean()
            if strict_target_recover.any()
            else 0.0
        )
        beneficial_recovery = calibrated_recover & (
            target[np.arange(len(target)), predicted_recovery_id] > target_accept + 1e-6
        )
        beneficial_recovery_rate = float(
            beneficial_recovery[strict_target_recover].mean()
            if strict_target_recover.any()
            else 0.0
        )
        accuracy = float(
            (calibrated_recover[strict_group] == target_recover[strict_group]).mean()
        )
        selected = np.where(
            calibrated_recover,
            predicted_recovery_id,
            predicted_accept_id,
        )
        regret = float((target_best - target[np.arange(len(target)), selected]).mean())
        key = (beneficial_recovery_rate, true_recovery_rate, accuracy, -float(margin))
        payload = {
            "max_false_recovery_rate": max_false_recovery_rate,
            "option_value_margin": float(margin),
            "false_recovery_rate": false_recovery_rate,
            "true_recovery_rate": true_recovery_rate,
            "beneficial_recovery_rate": beneficial_recovery_rate,
            "accuracy": accuracy,
            "mean_decision_regret": regret,
        }
        if selected_calibration is None or key > selected_calibration[0]:
            selected_calibration = (key, payload)
    if selected_calibration is None:
        raise ValueError("option margin calibration has no feasible operating point")
    return {
        "labeled_states": int(labeled.sum()),
        "strict_states": int(strict.sum()),
        "top1_accuracy_strict": float((strict_prediction == strict_target).mean()),
        "balanced_recall_strict": float(np.nanmean(recalls)),
        "per_option": {
            option: {
                "support": int(supports[index]),
                "recall": float(recalls[index]) if supports[index] else None,
            }
            for index, option in enumerate(RECOVERY_OPTIONS)
        },
        "pairwise_ranking_accuracy": float(pair_correct / pair_total),
        "pairwise_comparisons": pair_total,
        "mean_decision_regret": float((target_best - selected_target).mean()),
        "accept_recover": {
            "strict_states": int(strict_group.sum()),
            "accuracy": float(
                (predicted_recover[strict_group] == target_recover[strict_group]).mean()
            ),
            "target_accept_states": int(strict_target_accept.sum()),
            "false_recovery_rate": float(
                predicted_recover[strict_target_accept].mean()
                if strict_target_accept.any()
                else 0.0
            ),
            "target_recovery_states": int(strict_target_recover.sum()),
            "missed_recovery_rate": float(
                (~predicted_recover[strict_target_recover]).mean()
                if strict_target_recover.any()
                else 0.0
            ),
        },
        "selective_recovery": selected_calibration[1],
    }


def _audit_model_partitions(
    model: Any,
    partitions: list[tuple[Any, Any]],
    *,
    batch_size: int,
    device: str,
    np: Any,
    torch: Any,
) -> dict[str, Any] | None:
    targets: list[Any] = []
    validity: list[Any] = []
    predictions: list[Any] = []
    with torch.inference_mode():
        for corpus, raw_ids in partitions:
            ids = raw_ids[corpus.target_option_valid[raw_ids].sum(axis=1) >= 2]
            if not len(ids):
                continue
            targets.append(corpus.target_option_values[ids])
            validity.append(corpus.target_option_valid[ids])
            for start in range(0, len(ids), batch_size):
                selected = ids[start : start + batch_size]
                batch = _batch(corpus, selected, device=device, np=np, torch=torch)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=device.startswith("cuda"),
                ):
                    outputs = model(
                        batch["contexts"],
                        batch["anchors"],
                        batch["actions"],
                        batch["selector"],
                        batch["scalars"],
                    )
                predictions.append(outputs["option_values"].float().cpu().numpy())
    if not targets:
        return None
    return summarize_option_predictions(
        np.concatenate(targets),
        np.concatenate(validity),
        np.concatenate(predictions),
        np=np,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("option audit batch size must be positive")
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    checkpoints = sorted(path.parent for path in checkpoint_root.glob("*/model.safetensors"))
    if not checkpoints:
        raise FileNotFoundError(f"no Ours checkpoints under {checkpoint_root}")

    import numpy as np
    import torch
    from safetensors.torch import load_file

    results: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        audit, provenance = inspect_recovery_checkpoint(checkpoint)
        config = TemporalRecoveryModelConfig(**provenance["model"])
        additional_paths = provenance.get("additional_train_datasets") or []
        if not additional_paths and provenance.get("additional_train_dataset"):
            additional_paths = [provenance["additional_train_dataset"]]
        if not additional_paths:
            raise ValueError(f"checkpoint lacks counterfactual training data: {checkpoint}")
        additional_corpora = [
            load_corpus(Path(path), config.history_length, np) for path in additional_paths
        ]
        model = build_temporal_recovery_model(config).to(args.device)
        model.load_state_dict(load_file(checkpoint / "model.safetensors", device=args.device))
        model.eval()
        result = _audit_model_partitions(
            model,
            [(corpus, corpus.train_ids) for corpus in additional_corpora],
            batch_size=args.batch_size,
            device=args.device,
            np=np,
            torch=torch,
        )
        if result is None:
            raise ValueError(f"checkpoint has no counterfactual fit states: {checkpoint}")
        result["option_validation"] = _audit_model_partitions(
            model,
            [(corpus, corpus.development_ids) for corpus in additional_corpora],
            batch_size=args.batch_size,
            device=args.device,
            np=np,
            torch=torch,
        )
        result.update(
            {
                "variant_id": checkpoint.name,
                "checkpoint": str(checkpoint),
                "weights_sha256": audit.weights_sha256,
                "selected_step": int(provenance["selected_step"]),
                "scope": (
                    "train-seed counterfactual fit plus held-out episodes; "
                    "no rollout development seed"
                ),
            }
        )
        results.append(result)
        del model, additional_corpora
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    ranked = sorted(
        results,
        key=lambda result: (
            -result["option_validation"]["selective_recovery"][
                "beneficial_recovery_rate"
            ],
            -result["option_validation"]["selective_recovery"]["true_recovery_rate"],
            result["option_validation"]["selective_recovery"]["mean_decision_regret"],
            -result["option_validation"]["balanced_recall_strict"],
            result["variant_id"],
        ),
    )
    for rank, result in enumerate(ranked, start=1):
        result["offline_option_rank"] = rank
    payload = {
        "schema_version": 1,
        "experiment_id": "Ours",
        "stage": "R0-counterfactual-option-fit-audit",
        "scope": "train-only diagnostic; rollout development seeds were not consumed",
        "selection_rule": (
            "max beneficial recovery at calibrated false-recovery<=5%, then recovery "
            "recall, regret, balanced recall, registered id"
        ),
        "selected_variant": ranked[0]["variant_id"],
        "selected_option_value_margin": ranked[0]["option_validation"][
            "selective_recovery"
        ]["option_value_margin"],
        "variants": results,
        "complete": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(f"{args.output.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(args.output)
    return payload


def main() -> int:
    print(json.dumps(run(_parser().parse_args()), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
