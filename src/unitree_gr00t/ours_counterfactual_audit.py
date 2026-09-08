"""Audit whether counterfactual recovery labels contain a learnable choice."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .a1 import sha256_file
from .ours import RECOVERY_OPTIONS, OursContractError
from .ours_counterfactual import (
    COUNTERFACTUAL_RETURN_TARGETS,
    EFFICIENCY_SHAPED_RETURN_TARGET,
)
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


def residual_contract_checks(
    manifest: dict[str, Any],
    branches: list[dict[str, Any]],
    *,
    branches_sha256: str,
) -> dict[str, bool]:
    """Verify that residual labels estimate a one-step deviation from B-retry."""

    sampling = manifest.get("counterfactual_sampling", {})
    if not bool(sampling.get("residual_retry_baseline")):
        return {}
    observed_options = Counter(str(row.get("option")) for row in branches)
    source_contract = sampling.get("source_behavior_contract", {})
    return_target = sampling.get(
        "return_target", EFFICIENCY_SHAPED_RETURN_TARGET
    )
    expected_options = {
        str(option): int(count)
        for option, count in sampling.get("options", {}).items()
        if int(count)
    }
    states: dict[tuple[int, int], set[str]] = {}
    state_seeds: dict[tuple[int, int], set[int]] = {}
    for row in branches:
        key = (int(row.get("episode_index", -1)), int(row.get("sample_index", -1)))
        states.setdefault(key, set()).add(str(row.get("option")))
        state_seeds.setdefault(key, set()).add(int(row.get("branch_seed", -1)))
    return {
        "residual_source_is_abstaining_b_retry": source_contract
        == {
            "decision_schedule": "counterfactual-residual-over-b-retry-v1",
            "residual_retry_baseline": True,
            "option_value_margin": 1_000_000.0,
            "recovery_triggers": 0,
            "capture_training_context": True,
            "collection_force_boundary_steps": None,
            "stagnation_boundary_steps": None,
        },
        "residual_confirmed_stop_sources": bool(sampling.get("require_stop_pending"))
        and all(bool(row.get("source_stop_pending")) for row in branches),
        "residual_b_retry_continuation": sampling.get("continuation_policy")
        == "B-retry-confirmed-stop-one-retry-per-subtask"
        and bool(sampling.get("consensus_source_proposal_included")),
        "residual_branch_hash": branches_sha256
        == manifest.get("counterfactual_branches_sha256"),
        "residual_branch_inventory": len(branches) == int(sampling.get("branch_count", -1))
        and len(states) == int(sampling.get("state_count", -1))
        and dict(observed_options) == expected_options,
        "residual_distinct_actions": observed_options.get("ACCEPT_B", 0) == 0
        and all(
            {"RETRY_CURRENT", "ADVANCE"}.issubset(options)
            for options in states.values()
        ),
        "residual_common_random_numbers": sampling.get("randomness_coupling")
        == "common-random-numbers-per-state-v1"
        and sampling.get("rollouts_per_option") == 1
        and all(len(seeds) == 1 and -1 not in seeds for seeds in state_seeds.values()),
        "residual_return_target": return_target in COUNTERFACTUAL_RETURN_TARGETS
        and bool(
            sampling.get(
                "cost_terms_in_target",
                return_target == EFFICIENCY_SHAPED_RETURN_TARGET,
            )
        )
        == (return_target == EFFICIENCY_SHAPED_RETURN_TARGET),
    }


def summarize_residual_branch_mechanisms(
    branches: list[dict[str, Any]], *, tie_tolerance: float = 1e-6
) -> dict[str, Any]:
    """Separate physical recovery gains from return gains due only to cost."""

    states: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in branches:
        key = (int(row.get("episode_index", -1)), int(row.get("sample_index", -1)))
        states.setdefault(key, []).append(row)
    physical_benefit = 0
    return_benefit = 0
    efficiency_only = 0
    physical_regression = 0
    physical_winners: Counter[str] = Counter()
    for rows in states.values():
        retry_rows = [row for row in rows if row.get("option") == "RETRY_CURRENT"]
        alternatives = [row for row in rows if row.get("option") != "RETRY_CURRENT"]
        if len(retry_rows) != 1 or not alternatives:
            raise OursContractError("residual mechanism audit has an invalid branch inventory")
        retry = retry_rows[0]

        def physical_key(row: dict[str, Any]) -> tuple[int, int, int]:
            return (
                int(bool(row["final_success"])),
                int(row["completed_subtasks_after"]),
                int(row["predicate_count_after"]),
            )

        retry_physical = physical_key(retry)
        best_physical = max(physical_key(row) for row in alternatives)
        if best_physical > retry_physical:
            physical_benefit += 1
            for row in alternatives:
                if physical_key(row) == best_physical:
                    physical_winners[str(row["option"])] += 1
        best_return = max(alternatives, key=lambda row: float(row["return_value"]))
        if float(best_return["return_value"]) > float(retry["return_value"]) + tie_tolerance:
            return_benefit += 1
            if physical_key(best_return) == retry_physical:
                efficiency_only += 1
            elif physical_key(best_return) < retry_physical:
                physical_regression += 1
    count = len(states)
    return {
        "states": count,
        "physical_beneficial_override_states": physical_benefit,
        "physical_beneficial_override_rate": physical_benefit / count if count else 0.0,
        "return_beneficial_override_states": return_benefit,
        "efficiency_only_return_override_states": efficiency_only,
        "return_override_with_physical_regression_states": physical_regression,
        "physical_winning_options": dict(sorted(physical_winners.items())),
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
    branch_path = root / "counterfactual_branches.jsonl"
    branches = (
        [json.loads(line) for line in branch_path.read_text(encoding="utf-8").splitlines()]
        if branch_path.is_file()
        else []
    )
    values: list[Any] = []
    masks: list[Any] = []
    for filename in sorted(manifest["files_sha256"]):
        with np.load(root / "features" / filename, allow_pickle=False) as episode:
            if "target_option_values" not in episode or "target_option_valid" not in episode:
                raise OursContractError(f"counterfactual targets are missing: {filename}")
            values.append(episode["target_option_values"].astype(np.float32, copy=False))
            masks.append(episode["target_option_valid"].astype(np.bool_, copy=False))
    summary = summarize_option_targets(np.concatenate(values), np.concatenate(masks), np=np)
    residual_mechanisms = (
        summarize_residual_branch_mechanisms(branches)
        if manifest.get("counterfactual_sampling", {}).get("residual_retry_baseline")
        else None
    )
    checks = {
        "minimum_labeled_states": summary["labeled_states"] >= args.min_labeled_states,
        "multiple_winning_options": (
            summary["distinct_winning_options"] >= args.min_winning_options
        ),
        "strict_preference_rate": (
            summary["strict_preference_rate"] >= args.min_strict_preference_rate
        ),
        **residual_contract_checks(
            manifest,
            branches,
            branches_sha256=(sha256_file(branch_path) if branch_path.is_file() else ""),
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
        "residual_branch_mechanisms": residual_mechanisms,
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
