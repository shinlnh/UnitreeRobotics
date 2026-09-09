"""Merge deterministic RESOLVE frontier shards into one hash-bound corpus."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from .a1 import sha256_file
from .ours import OursContractError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def merge(args: argparse.Namespace) -> dict[str, Any]:
    roots = [path.expanduser().resolve() for path in args.shards]
    destination = args.output.expanduser().resolve()
    if len(roots) < 2 or len(set(roots)) != len(roots):
        raise OursContractError("RESOLVE merge requires distinct shards")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite RESOLVE merge: {destination}")
    feature_dir = destination / "features"
    feature_dir.mkdir(parents=True)
    manifests: list[dict[str, Any]] = []
    records: list[tuple[int, dict[str, Any], Path, str]] = []
    exclusions: list[dict[str, Any]] = []
    binding_keys = (
        "experiment_id",
        "stage",
        "frontier_manifest_sha256",
        "runtime_manifest_sha256",
        "runtime_provenance",
        "split",
        "anchor_offset",
        "selection_seed",
    )
    reference = None
    for root in roots:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows_path = root / "anchors.jsonl"
        if sha256_file(rows_path) != manifest.get("anchors_sha256"):
            raise OursContractError("RESOLVE shard record hash mismatch")
        binding = {key: manifest.get(key) for key in binding_keys}
        if reference is None:
            reference = binding
        elif binding != reference:
            raise OursContractError("RESOLVE shards have incompatible run contracts")
        hashes = manifest.get("files_sha256")
        if not isinstance(hashes, dict):
            raise OursContractError("RESOLVE shard feature registry is invalid")
        rows = _read_jsonl(rows_path)
        if len(rows) != int(manifest["anchors"]):
            raise OursContractError("RESOLVE shard row count mismatch")
        for row in rows:
            filename = str(row["feature_file"])
            source = root / "features" / filename
            if filename not in hashes or sha256_file(source) != hashes[filename]:
                raise OursContractError("RESOLVE shard feature hash mismatch")
            selection_index = int(row["selection_index"])
            records.append((selection_index, row, source, hashes[filename]))
        exclusions.extend(manifest.get("exclusions", []))
        manifests.append(manifest)
    records.sort(key=lambda item: item[0])
    selection_indices = [item[0] for item in records]
    if len(set(selection_indices)) != len(selection_indices):
        raise OursContractError("RESOLVE shards overlap in selection index")
    covered = sorted(selection_indices + [int(item["selection_index"]) for item in exclusions])
    if len(set(covered)) != len(covered) or covered != list(range(covered[0], covered[-1] + 1)):
        raise OursContractError("RESOLVE shards leave a selection-index gap")

    output_rows: list[dict[str, Any]] = []
    files_sha256: dict[str, str] = {}
    for merged_index, (selection_index, row, source, digest) in enumerate(records):
        filename = f"anchor_{merged_index:06d}.npz"
        shutil.copyfile(source, feature_dir / filename)
        if sha256_file(feature_dir / filename) != digest:
            raise OursContractError("RESOLVE merged feature copy changed bytes")
        files_sha256[filename] = digest
        output_rows.append(
            row
            | {
                "anchor_index": merged_index,
                "selection_index": selection_index,
                "feature_file": filename,
            }
        )
    records_path = destination / "anchors.jsonl"
    _write_jsonl(records_path, output_rows)
    assert reference is not None
    manifest = {
        "schema_version": 1,
        **reference,
        "novelty_claim": False,
        "merged_shards": [str(root) for root in roots],
        "merged_shard_manifest_sha256": [sha256_file(root / "manifest.json") for root in roots],
        "anchor_start": selection_indices[0],
        "limited": False,
        "anchors": len(output_rows),
        "selected_anchors": len(output_rows) + len(exclusions),
        "exclusions": sorted(exclusions, key=lambda item: int(item["selection_index"])),
        "expert_reached": sum(int(row["expert_reached"]) for row in output_rows),
        "baseline_reached": sum(int(row["baseline"]["reached"]) for row in output_rows),
        "positive_gaps": sum(row["expert_superiority"] > 0 for row in output_rows),
        "negative_gaps": sum(row["expert_superiority"] < 0 for row in output_rows),
        "anchors_sha256": sha256_file(records_path),
        "files_sha256": files_sha256,
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def main() -> int:
    try:
        result = merge(_parser().parse_args())
    except (FileNotFoundError, OursContractError, ValueError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
