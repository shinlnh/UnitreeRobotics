"""Audit paired RESOLVE physical-program corpora and summarize causal outcomes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .a1 import sha256_file
from .ours import OursContractError
from .ours_rollout_prepare import _write_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def summarize_rows(rows: list[dict[str, Any]], milestone_names: tuple[str, ...]) -> dict[str, Any]:
    if not rows or not milestone_names:
        raise OursContractError("RESOLVE audit received no physical programs")
    metrics: dict[str, dict[str, int]] = {name: defaultdict(int) for name in milestone_names}
    anchors: set[tuple[int, int]] = set()
    baseline_zero_hashes: dict[tuple[int, int], tuple[str, str]] = {}
    for row in rows:
        state = (int(row.get("base_seed", 0)), int(row["sample_index"]))
        anchors.add(state)
        recovery = row["recovery"]
        deletions = row["deletions"]
        baselines = row["baselines"]
        if not deletions or len(deletions) != len(baselines):
            raise OursContractError("RESOLVE audit found malformed R/D/B arms")
        last_d = deletions[-1]
        last_b = baselines[-1]
        if (
            last_d["action_trace_sha256"] != last_b["action_trace_sha256"]
            or last_d["final_state_sha256"] != last_b["final_state_sha256"]
        ):
            raise OursContractError("RESOLVE audit D_last/B_last invariant failed")
        baseline_zero = (
            baselines[0]["action_trace_sha256"],
            baselines[0]["final_state_sha256"],
        )
        previous = baseline_zero_hashes.setdefault(state, baseline_zero)
        if previous != baseline_zero:
            raise OursContractError("RESOLVE audit B_0 reuse invariant failed")
        for index, name in enumerate(milestone_names):
            target = metrics[name]
            r = int(bool(recovery["reachability"][index]))
            d = [int(bool(arm["reachability"][index])) for arm in deletions]
            b = [int(bool(arm["reachability"][index])) for arm in baselines]
            crb = float(row["program_crb"][index])
            target["programs"] += 1
            target["recovery_reaches"] += r
            target["baseline_zero_reaches"] += b[0]
            target["strict_crb_positive"] += int(crb > 0.0)
            target["recovery_but_not_deletion_minimal"] += int(r == 1 and crb <= 0.0)
            target["baseline_regressions"] += int(r == 0 and b[0] == 1)
            target["joint_exclusive_rescues"] += int(r == 1 and not any(d) and not any(b))
    return {
        "anchors": len(anchors),
        "programs": len(rows),
        "milestones": {name: dict(value) for name, value in metrics.items()},
        "canonical_identical_arm_invariants": True,
    }


def audit(corpus: Path) -> dict[str, Any]:
    root = corpus.expanduser().resolve()
    manifest_path = root / "manifest.json"
    record_path = root / "programs.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(record_path) != manifest["program_records_sha256"]:
        raise OursContractError("RESOLVE program-record hash mismatch")
    rows = [json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines()]
    base_seed = int(manifest["base_seed"])
    for row in rows:
        row["base_seed"] = base_seed
        feature_path = root / "features" / row["features"]
        if sha256_file(feature_path) != manifest["files_sha256"][row["features"]]:
            raise OursContractError("RESOLVE feature hash mismatch")
    if len(rows) != int(manifest["files"]):
        raise OursContractError("RESOLVE manifest file count mismatch")
    return {
        "schema_version": 1,
        "corpus": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "protocol": manifest["pairing"],
        "supersedes_incomplete_snapshot_artifacts": bool(
            manifest.get("supersedes_incomplete_snapshot_artifacts")
        ),
    } | summarize_rows(rows, tuple(manifest["milestones"]))


def main() -> int:
    args = _parser().parse_args()
    report = audit(args.corpus)
    if args.output is not None:
        _write_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
