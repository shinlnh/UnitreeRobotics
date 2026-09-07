"""Merge disjoint A0 benchmark shards into one validated artifact directory."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .a0 import inspect_checkpoint
from .a0_eval import _build_summary, _clean_partial_trace, _read_jsonl, _write_json

CONDITION_ORDER = {
    name: index
    for index, name in enumerate(
        (
            "Ideal",
            "Memory_Execution",
            "Memory_Exploration",
            "Mix",
            "Observation_Mismatching",
            "Random_Disturbance",
        )
    )
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge A0 benchmark shards")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--create-target", action="store_true")
    return parser


def _key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["task_type"]), str(row["case"]), int(row["trial"])


def _natural(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def _episode_sort(row: dict[str, Any]) -> tuple[Any, ...]:
    task_type, case, trial = _key(row)
    return CONDITION_ORDER[task_type], _natural(case), trial


def _load_manifest(root: Path) -> dict[str, Any]:
    value = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid run manifest: {root}")
    return value


def _compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    ignored = {"task_types", "cases", "expected_episodes", "output_dir", "hierarchy_audit"}
    return {key: value for key, value in left.items() if key not in ignored} == {
        key: value for key, value in right.items() if key not in ignored
    }


def _create_manifest(target: Path, shard_manifests: list[dict[str, Any]]) -> dict[str, Any]:
    base = dict(shard_manifests[0])
    if any(not _compatible(base, manifest) for manifest in shard_manifests[1:]):
        raise ValueError("Shard manifests do not share one frozen A0 configuration")
    cases = {case for manifest in shard_manifests for case in manifest["cases"]}
    ordered_cases = sorted(
        cases,
        key=lambda item: (
            CONDITION_ORDER[item.split("/", 1)[0]],
            _natural(item.split("/", 1)[1]),
        ),
    )
    base["task_types"] = list(CONDITION_ORDER)
    base["cases"] = ordered_cases
    base["expected_episodes"] = len(ordered_cases) * int(base["trials_per_case"])
    base["output_dir"] = str(target)
    hierarchy_audits = [manifest.get("hierarchy_audit") for manifest in shard_manifests]
    if all(isinstance(audit, dict) for audit in hierarchy_audits):
        typed_audits = [audit for audit in hierarchy_audits if isinstance(audit, dict)]
        hashes = sorted(
            {str(value) for audit in typed_audits for value in audit.get("plan_sha256", [])}
        )
        base["hierarchy_audit"] = {
            **typed_audits[0],
            "cases": sum(int(audit["cases"]) for audit in typed_audits),
            "plans": sum(int(audit["plans"]) for audit in typed_audits),
            "subgoals": sum(int(audit["subgoals"]) for audit in typed_audits),
            "unique_plans": len(hashes),
            "plan_sha256": hashes,
            "valid": all(bool(audit["valid"]) for audit in typed_audits),
            "issues": [issue for audit in typed_audits for issue in audit.get("issues", [])],
        }
    return base


def merge(
    target: Path,
    shards: list[Path],
    create_target: bool,
    *,
    summary_builder: Any = _build_summary,
) -> dict[str, Any]:
    target = target.expanduser().resolve()
    shard_roots = [path.expanduser().resolve() for path in shards]
    shard_manifests = [_load_manifest(root) for root in shard_roots]
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "run_manifest.json"
    if create_target:
        if manifest_path.exists() or (target / "episodes.jsonl").exists():
            raise FileExistsError(f"--create-target requires an empty target: {target}")
        manifest = _create_manifest(target, shard_manifests)
        _write_json(manifest_path, manifest)
    else:
        manifest = _load_manifest(target)
        if any(not _compatible(manifest, value) for value in shard_manifests):
            raise ValueError("A shard does not match the target's frozen A0 configuration")

    episodes = _read_jsonl(target / "episodes.jsonl")
    completed = {_key(row) for row in episodes}
    if len(completed) != len(episodes):
        raise ValueError("Target contains duplicate episodes before merge")
    _clean_partial_trace(target, completed)
    decision_path = target / "decisions.jsonl"
    decision_path.parent.mkdir(parents=True, exist_ok=True)

    added_episodes: list[dict[str, Any]] = []
    added_decisions = 0
    for shard, shard_manifest in zip(shard_roots, shard_manifests, strict=True):
        if not _compatible(manifest, shard_manifest):
            raise ValueError(f"Incompatible shard: {shard}")
        shard_episodes = _read_jsonl(shard / "episodes.jsonl")
        shard_keys = [_key(row) for row in shard_episodes]
        if len(shard_keys) != len(set(shard_keys)):
            raise ValueError(f"Shard contains duplicate episode keys: {shard}")
        selected = {key for key in shard_keys if key not in completed}
        added_episodes.extend(row for row in shard_episodes if _key(row) in selected)
        completed.update(selected)

        with (
            (shard / "decisions.jsonl").open(encoding="utf-8") as source,
            decision_path.open("a", encoding="utf-8") as destination,
        ):
            for line in source:
                if not line.strip():
                    continue
                decision = json.loads(line)
                if _key(decision) not in selected:
                    continue
                destination.write(line)
                added_decisions += 1
                frame_bundle = decision.get("frame_bundle")
                if frame_bundle:
                    source_frame = shard / str(frame_bundle)
                    target_frame = target / str(frame_bundle)
                    target_frame.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_frame, target_frame)

    episodes.extend(added_episodes)
    episodes.sort(key=_episode_sort)
    temporary = target / "episodes.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for row in episodes:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(target / "episodes.jsonl")

    expected_keys = {
        (*case.split("/", 1), trial)
        for case in manifest["cases"]
        for trial in range(int(manifest["trials_per_case"]))
    }
    actual_keys = {_key(row) for row in episodes}
    if len(actual_keys) != len(episodes):
        raise ValueError("Merged target contains duplicate episode keys")
    if not actual_keys <= expected_keys:
        raise ValueError(
            f"Merged target contains unexpected episode keys: {actual_keys - expected_keys}"
        )

    contract = inspect_checkpoint(manifest["checkpoint"])
    summary = summary_builder(manifest, contract, episodes)
    _write_json(target / "progress.json", summary)
    _write_json(target / "summary.json", summary)
    return {
        "target": str(target),
        "episodes": len(episodes),
        "expected_episodes": len(expected_keys),
        "complete": actual_keys == expected_keys and len(episodes) == len(expected_keys),
        "added_episodes": len(added_episodes),
        "added_decisions": added_decisions,
    }


def main() -> int:
    args = _parser().parse_args()
    result = merge(args.target, args.shard, args.create_target)
    print(json.dumps(result, indent=2))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
