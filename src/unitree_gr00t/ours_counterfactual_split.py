"""Create a deterministic train-only option-label validation split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .ours import OursContractError
from .ours_data import OURS_CORPUS_MANIFEST, audit_recovery_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--modulus", type=int, default=5)
    parser.add_argument("--remainder", type=int, default=4)
    return parser


def option_episode_partition(
    episode_ids: list[int], *, modulus: int, remainder: int
) -> dict[str, list[int]]:
    if modulus < 2 or not 0 <= remainder < modulus or not episode_ids:
        raise OursContractError("counterfactual option split rule is invalid")
    ordered = sorted(set(int(value) for value in episode_ids))
    development = [value for value in ordered if value % modulus == remainder]
    train = [value for value in ordered if value % modulus != remainder]
    if not train or not development:
        raise OursContractError("counterfactual option split is empty")
    return {"train": train, "development": development}


def run(corpus: Path, *, modulus: int, remainder: int) -> dict[str, Any]:
    root = corpus.expanduser().resolve()
    audit_recovery_corpus(
        root,
        verify_hashes=True,
        require_development=False,
        require_both_completion_classes=True,
    )
    path = root / OURS_CORPUS_MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("source_kind") != "counterfactual_live_rollout":
        raise OursContractError("option split requires a counterfactual rollout corpus")
    episode_ids = [
        int(name.removeprefix("episode_").removesuffix(".npz"))
        for name in manifest["files_sha256"]
    ]
    split = option_episode_partition(episode_ids, modulus=modulus, remainder=remainder)
    manifest["split_episodes"] = split
    manifest["counterfactual_option_split"] = {
        "scope": "train-seed episode holdout; not rollout development or final",
        "rule": "episode_index modulo modulus equals remainder",
        "modulus": modulus,
        "remainder": remainder,
        "train_episodes": len(split["train"]),
        "validation_episodes": len(split["development"]),
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    audit_recovery_corpus(
        root,
        verify_hashes=True,
        require_development=True,
        require_both_completion_classes=True,
    )
    return manifest


def main() -> int:
    args = _parser().parse_args()
    payload = run(args.corpus, modulus=args.modulus, remainder=args.remainder)
    print(json.dumps(payload["counterfactual_option_split"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
