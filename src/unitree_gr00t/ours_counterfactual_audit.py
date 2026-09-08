"""Audit whether counterfactual recovery labels contain a learnable choice."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .ours import RECOVERY_OPTIONS, OursContractError
from .ours_data import OURS_CORPUS_MANIFEST, audit_recovery_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-labeled-states", type=int, default=32)
    parser.add_argument("--min-winning-options", type=int, default=2)
    parser.add_argument("--min-strict-preference-rate", type=float, default=0.1)
    return parser


def summarize_option_targets(
    values: Any,
    valid: Any,
    *,
    np: Any,
    tie_tolerance: float = 1e-6,
) -> dict[str, Any]:
    if values.ndim != 2 or valid.shape != values.shape or values.shape[1] != len(
        RECOVERY_OPTIONS
    ):
        raise OursContractError("counterfactual option arrays have the wrong shape")
    if not np.isfinite(values).all():
        raise OursContractError("counterfactual option returns must be finite")
    selected = valid.sum(axis=1) >= 2
    values = values[selected]
    valid = valid[selected]
    winners: Counter[str] = Counter()
    ties = 0
    spreads: list[float] = []
    per_option: dict[str, dict[str, float | int | None]] = {}
    for option_index, option in enumerate(RECOVERY_OPTIONS):
        option_values = values[:, option_index][valid[:, option_index]]
        per_option[option.value] = {
            "count": int(len(option_values)),
            "mean": float(option_values.mean()) if len(option_values) else None,
            "standard_deviation": float(option_values.std()) if len(option_values) else None,
            "minimum": float(option_values.min()) if len(option_values) else None,
            "maximum": float(option_values.max()) if len(option_values) else None,
        }
    for row, mask in zip(values, valid, strict=True):
        indices = np.flatnonzero(mask)
        ranked = row[indices]
        maximum = float(ranked.max())
        best = indices[np.abs(ranked - maximum) <= tie_tolerance]
        sorted_values = np.sort(ranked)
        spreads.append(float(sorted_values[-1] - sorted_values[-2]))
        if len(best) == 1:
            winners[RECOVERY_OPTIONS[int(best[0])].value] += 1
        else:
            ties += 1
    strict = sum(winners.values())
    return {
        "labeled_states": int(len(values)),
        "valid_option_targets": int(valid.sum()),
        "strict_preferences": strict,
        "strict_preference_rate": strict / len(values) if len(values) else 0.0,
        "ties": ties,
        "winning_options": dict(sorted(winners.items())),
        "distinct_winning_options": len(winners),
        "mean_best_margin": float(np.mean(spreads)) if spreads else 0.0,
        "per_option": per_option,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        min(args.min_labeled_states, args.min_winning_options) < 1
        or not 0.0 <= args.min_strict_preference_rate <= 1.0
    ):
        raise ValueError("counterfactual audit thresholds are invalid")
    import numpy as np

    root = args.corpus.expanduser().resolve()
    corpus_audit = audit_recovery_corpus(
        root,
        verify_hashes=True,
        require_development=False,
        require_both_completion_classes=False,
    )
    manifest = json.loads((root / OURS_CORPUS_MANIFEST).read_text(encoding="utf-8"))
    values: list[Any] = []
    masks: list[Any] = []
    for filename in sorted(manifest["files_sha256"]):
        with np.load(root / "features" / filename, allow_pickle=False) as episode:
            if "target_option_values" not in episode or "target_option_valid" not in episode:
                raise OursContractError(f"counterfactual targets are missing: {filename}")
            values.append(episode["target_option_values"].astype(np.float32, copy=False))
            masks.append(episode["target_option_valid"].astype(np.bool_, copy=False))
    summary = summarize_option_targets(np.concatenate(values), np.concatenate(masks), np=np)
    checks = {
        "minimum_labeled_states": summary["labeled_states"] >= args.min_labeled_states,
        "multiple_winning_options": (
            summary["distinct_winning_options"] >= args.min_winning_options
        ),
        "strict_preference_rate": (
            summary["strict_preference_rate"] >= args.min_strict_preference_rate
        ),
    }
    result = {
        "schema_version": 1,
        "corpus": str(root),
        "corpus_files": corpus_audit.files,
        "corpus_samples": corpus_audit.samples,
        "thresholds": {
            "min_labeled_states": args.min_labeled_states,
            "min_winning_options": args.min_winning_options,
            "min_strict_preference_rate": args.min_strict_preference_rate,
        },
        "option_targets": summary,
        "checks": checks,
        "valid": all(checks.values()),
    }
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> int:
    result = run(_parser().parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
