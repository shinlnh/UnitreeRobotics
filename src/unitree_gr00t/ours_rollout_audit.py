"""Audit live Ours beliefs and calibrate STOP gates from rollout-only labels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .a1 import sha256_file
from .ours import OURS_ID, OURS_METHOD, OURS_VARIANT, OursContractError
from .ours_train import calibrate_threshold


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.05)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["task_type"]), str(row["case"]), int(row["trial"])


def calibrate_rollout_signal(
    probabilities: Any,
    labels: Any,
    stop_proposals: Any,
    *,
    max_false_positive_rate: float,
    np: Any,
) -> dict[str, Any]:
    selected_probability = probabilities[stop_proposals]
    selected_labels = labels[stop_proposals]
    if not len(selected_labels) or not selected_labels.any() or selected_labels.all():
        return {
            "calibratable": False,
            "stop_samples": int(len(selected_labels)),
            "completion_negatives": int((~selected_labels).sum()),
            "completion_positives": int(selected_labels.sum()),
        }
    threshold, metrics = calibrate_threshold(
        selected_probability,
        selected_labels,
        max_false_positive_rate=max_false_positive_rate,
        np=np,
    )
    return {
        "calibratable": True,
        "stop_samples": int(len(selected_labels)),
        "completion_negatives": int((~selected_labels).sum()),
        "completion_positives": int(selected_labels.sum()),
        "threshold": threshold,
        **metrics,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.max_false_positive_rate <= 0.05:
        raise ValueError("rollout calibration false-positive cap must be inside (0, 0.05]")
    rollout = args.rollout.expanduser().resolve()
    output = args.output.expanduser().resolve()
    manifest_path = rollout / "run_manifest.json"
    summary_path = rollout / "summary.json"
    decisions_path = rollout / "decisions.jsonl"
    episodes_path = rollout / "episodes.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment_id"),
        manifest.get("variant"),
        manifest.get("method"),
    ) != (OURS_ID, OURS_VARIANT, OURS_METHOD):
        raise OursContractError("rollout audit source is not Ours")
    if not bool(summary.get("complete")):
        raise OursContractError("rollout audit requires a complete run")

    import numpy as np

    episodes = _read_jsonl(episodes_path)
    excluded = {_key(row): int(row["excluded_subtasks"]) for row in episodes}
    if len(excluded) != int(summary["episodes"]):
        raise OursContractError("rollout episode inventory is incomplete or duplicated")

    completion_labels: list[bool] = []
    stop_proposals: list[bool] = []
    signals: dict[str, list[float]] = {
        "completion": [],
        "progress": [],
        "maximum": [],
    }
    failure_labels: list[bool] = []
    failure_probabilities: list[float] = []
    injection_seen: dict[tuple[str, str, int], bool] = {}
    recovery_triggers = 0
    false_recovery_triggers = 0
    for row in _read_jsonl(decisions_path):
        if row.get("policy_invoked") is False:
            continue
        episode_key = _key(row)
        if episode_key not in excluded:
            raise OursContractError("rollout decisions contain an incomplete episode")
        completed = max(
            0,
            int(row["completed_subtasks_before"]) - excluded[episode_key],
        )
        completion = completed > int(row["active_subgoal_index_before"])
        completion_probability = float(row["recovery_completion_probability"])
        progress_probability = float(row["recovery_progress_probability"])
        completion_labels.append(completion)
        stop_proposals.append(int(row["selector_candidate_before_recovery"]) == 0)
        signals["completion"].append(completion_probability)
        signals["progress"].append(progress_probability)
        signals["maximum"].append(max(completion_probability, progress_probability))

        task_type = episode_key[0]
        failure_active = (
            injection_seen.get(episode_key, False)
            if task_type == "Random_Disturbance"
            else task_type != "Ideal"
        )
        if "recovery_failure_probability" in row:
            failure_labels.append(failure_active)
            failure_probabilities.append(float(row["recovery_failure_probability"]))
        triggered = bool(row.get("recovery_triggered"))
        recovery_triggers += int(triggered)
        false_recovery_triggers += int(triggered and not failure_active)
        injection_seen[episode_key] = injection_seen.get(episode_key, False) or any(
            transition.get("injection") is not None for transition in row["transitions"]
        )

    labels = np.asarray(completion_labels, dtype=np.bool_)
    stops = np.asarray(stop_proposals, dtype=np.bool_)
    calibration = {
        name: calibrate_rollout_signal(
            np.asarray(values, dtype=np.float32),
            labels,
            stops,
            max_false_positive_rate=args.max_false_positive_rate,
            np=np,
        )
        for name, values in signals.items()
    }
    failure_calibration = None
    if failure_probabilities:
        failure_calibration = calibrate_rollout_signal(
            np.asarray(failure_probabilities, dtype=np.float32),
            np.asarray(failure_labels, dtype=np.bool_),
            np.ones(len(failure_labels), dtype=np.bool_),
            max_false_positive_rate=args.max_false_positive_rate,
            np=np,
        )
    result = {
        "schema_version": 1,
        "experiment_id": OURS_ID,
        "source_rollout": str(rollout),
        "base_seed": int(manifest["base_seed"]),
        "episodes": len(episodes),
        "policy_decisions": len(labels),
        "completion_positives": int(labels.sum()),
        "completion_negatives": int((~labels).sum()),
        "selector_stop_proposals": int(stops.sum()),
        "max_false_positive_rate": args.max_false_positive_rate,
        "gate_calibration_on_stop_proposals": calibration,
        "failure_calibration": failure_calibration,
        "recovery_triggers": recovery_triggers,
        "false_recovery_triggers": false_recovery_triggers,
        "false_recovery_rate": (
            false_recovery_triggers / recovery_triggers if recovery_triggers else 0.0
        ),
        "run_manifest_sha256": sha256_file(manifest_path),
        "summary_sha256": sha256_file(summary_path),
        "decisions_sha256": sha256_file(decisions_path),
        "episodes_sha256": sha256_file(episodes_path),
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
