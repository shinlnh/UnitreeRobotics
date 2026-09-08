"""Rank complete Ours development rollouts against the paired B-retry control."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from .a1 import sha256_file
from .ours import OURS_ID, OURS_METHOD, OURS_VARIANT, OursContractError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-variants", type=int, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20007)
    parser.add_argument("--run-prefix", default="r1-")
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["task_type"]), str(row["case"]), int(row["trial"])


def task_macro_subtask_rate(rows: list[dict[str, Any]]) -> float:
    by_task: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        values = by_task[str(row["task_type"])]
        values[0] += int(row["agent_completed_subtasks"])
        values[1] += int(row["possible_subtasks"])
    if not by_task or any(possible == 0 for _, possible in by_task.values()):
        raise OursContractError("development rows have an invalid task inventory")
    return sum(completed / possible for completed, possible in by_task.values()) / len(by_task)


def paired_task_macro_bootstrap(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
    np: Any,
) -> dict[str, float]:
    baseline_by_key = {_key(row): row for row in baseline}
    candidate_by_key = {_key(row): row for row in candidate}
    if baseline_by_key.keys() != candidate_by_key.keys() or resamples < 1:
        raise OursContractError("development variants do not share a valid paired inventory")
    by_task: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for key in sorted(baseline_by_key):
        base = baseline_by_key[key]
        current = candidate_by_key[key]
        if int(base["possible_subtasks"]) != int(current["possible_subtasks"]):
            raise OursContractError("paired development episode denominators differ")
        possible = int(base["possible_subtasks"])
        by_task[key[0]].append(
            (
                int(current["agent_completed_subtasks"])
                - int(base["agent_completed_subtasks"]),
                possible,
            )
        )
    observed = sum(
        sum(delta for delta, _ in values) / sum(possible for _, possible in values)
        for values in by_task.values()
    ) / len(by_task)
    generator = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    arrays = [np.asarray(values, dtype=np.int64) for _, values in sorted(by_task.items())]
    for index in range(resamples):
        task_means = []
        for values in arrays:
            selected = values[generator.integers(0, len(values), size=len(values))]
            task_means.append(float(selected[:, 0].sum() / selected[:, 1].sum()))
        samples[index] = sum(task_means) / len(task_means)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {"delta": observed, "ci95_lower": float(lower), "ci95_upper": float(upper)}


def _complete_run(
    root: Path, *, expected_experiment: str | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "run_manifest.json"
    summary_path = root / "summary.json"
    episodes_path = root / "episodes.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(episodes_path)
    if (
        not bool(summary.get("complete"))
        or len(rows) != int(summary.get("episodes", -1))
        or len({_key(row) for row in rows}) != len(rows)
        or (
            expected_experiment is not None
            and manifest.get("experiment_id") != expected_experiment
        )
    ):
        raise OursContractError(f"incomplete development run: {root}")
    return manifest, rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if min(args.expected_variants, args.bootstrap_resamples) < 1:
        raise ValueError("development search counts must be positive")
    baseline_root = args.baseline.expanduser().resolve()
    runs_root = args.runs.expanduser().resolve()
    output = args.output.expanduser().resolve()
    baseline_manifest, baseline = _complete_run(baseline_root, expected_experiment="B-retry")
    if int(baseline_manifest["base_seed"]) != args.seed:
        raise OursContractError("B-retry development control uses the wrong base seed")

    import numpy as np

    variants: list[dict[str, Any]] = []
    if not args.run_prefix or "/" in args.run_prefix or ".." in args.run_prefix:
        raise ValueError("development run prefix is invalid")
    run_dirs = sorted(
        path.parent for path in runs_root.glob(f"{args.run_prefix}*/summary.json")
    )
    if len(run_dirs) != args.expected_variants:
        raise OursContractError(
            f"expected {args.expected_variants} development variants, found {len(run_dirs)}"
        )
    baseline_calls = sum(int(row["policy_calls"]) for row in baseline) / len(baseline)
    baseline_steps = sum(int(row["steps"]) for row in baseline) / len(baseline)
    for variant_index, root in enumerate(run_dirs):
        manifest, rows = _complete_run(root, expected_experiment=OURS_ID)
        identity = (manifest.get("variant"), manifest.get("method"), int(manifest["base_seed"]))
        if identity != (OURS_VARIANT, OURS_METHOD, args.seed):
            raise OursContractError(f"invalid Ours development identity: {root}")
        belief_path = root / "belief_audit.json"
        belief = json.loads(belief_path.read_text(encoding="utf-8"))
        paired = paired_task_macro_bootstrap(
            baseline,
            rows,
            resamples=args.bootstrap_resamples,
            seed=args.seed + variant_index,
            np=np,
        )
        mean_calls = sum(int(row["policy_calls"]) for row in rows) / len(rows)
        mean_steps = sum(int(row["steps"]) for row in rows) / len(rows)
        policy_call_ratio = mean_calls / baseline_calls
        simulator_step_ratio = mean_steps / baseline_steps
        false_recovery_rate = float(belief["false_recovery_rate"])
        pareto_valid = (
            paired["delta"] > 0.0
            and false_recovery_rate <= 0.05
            and policy_call_ratio <= 1.2
            and simulator_step_ratio <= 1.2
        )
        failure_rows = [
            row
            for row in rows
            if row["task_type"] != "Ideal" or int(row.get("injection_count", 0)) > 0
        ]
        variants.append(
            {
                "variant_id": root.name,
                "rollout": str(root),
                "checkpoint": manifest["recovery_checkpoint"],
                "recovery_weights_sha256": manifest["recovery_weights_sha256"],
                "decision_schedule": manifest.get("decision_schedule"),
                "gate_signal": manifest["gate_signal"],
                "gate_threshold": manifest["gate_threshold"],
                "consensus_hypotheses": manifest["consensus_hypotheses"],
                "failure_threshold": manifest["failure_threshold"],
                "consensus_cooldown_decisions": manifest["consensus_cooldown_decisions"],
                "max_recovery_attempts": manifest.get("max_recovery_attempts"),
                "min_recovery_elapsed_steps": manifest.get(
                    "min_recovery_elapsed_steps"
                ),
                "stagnation_boundary_steps": manifest["stagnation_boundary_steps"],
                "task_macro_subtask_rate": task_macro_subtask_rate(rows),
                "paired_task_macro": paired,
                "final_success_rate": sum(int(row["final_success"]) for row in rows) / len(rows),
                "conditional_recovery_success": (
                    sum(int(row["final_success"]) for row in failure_rows) / len(failure_rows)
                    if failure_rows
                    else 0.0
                ),
                "false_recovery_rate": false_recovery_rate,
                "mean_policy_calls": mean_calls,
                "policy_call_ratio_vs_b_retry": policy_call_ratio,
                "mean_simulator_steps": mean_steps,
                "simulator_step_ratio_vs_b_retry": simulator_step_ratio,
                "pareto_valid": pareto_valid,
                "run_manifest_sha256": sha256_file(root / "run_manifest.json"),
                "summary_sha256": sha256_file(root / "summary.json"),
                "episodes_sha256": sha256_file(root / "episodes.jsonl"),
                "belief_audit_sha256": sha256_file(belief_path),
            }
        )
    variants.sort(
        key=lambda row: (
            not row["pareto_valid"],
            -row["paired_task_macro"]["ci95_lower"],
            -row["conditional_recovery_success"],
            -row["final_success_rate"],
            row["false_recovery_rate"],
            row["mean_simulator_steps"],
            row["variant_id"],
        )
    )
    for rank, variant in enumerate(variants, 1):
        variant["development_rank"] = rank
    result = {
        "schema_version": 1,
        "experiment_id": OURS_ID,
        "round": "R1",
        "base_seed": args.seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "baseline": {
            "rollout": str(baseline_root),
            "task_macro_subtask_rate": task_macro_subtask_rate(baseline),
            "mean_policy_calls": baseline_calls,
            "mean_simulator_steps": baseline_steps,
            "run_manifest_sha256": sha256_file(baseline_root / "run_manifest.json"),
            "summary_sha256": sha256_file(baseline_root / "summary.json"),
            "episodes_sha256": sha256_file(baseline_root / "episodes.jsonl"),
        },
        "ranking_rule": "Pareto gate, paired CI lower bound, recovery/final success, false recovery, efficiency, id",
        "variants": variants,
        "complete": True,
    }
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
