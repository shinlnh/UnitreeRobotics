"""Create a deterministic train-only option-label validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from .ours import RECOVERY_OPTIONS, OursContractError, RecoveryOption
from .ours_counterfactual import OUTCOME_FIRST_RETURN_TARGET
from .ours_data import OURS_CORPUS_MANIFEST, audit_recovery_corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--modulus", type=int, default=5)
    parser.add_argument("--remainder", type=int, default=4)
    parser.add_argument(
        "--stratify-residual-overrides",
        action="store_true",
        help="Hold out deterministic episodes within physical override/no-override strata",
    )
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


def residual_stratified_episode_partition(
    episode_has_override: dict[int, bool],
    *,
    split_seed: int,
    validation_fraction: float = 0.2,
) -> dict[str, list[int]]:
    """Keep both sparse physical-label classes in train and validation."""

    if split_seed < 0 or not 0.0 < validation_fraction < 1.0:
        raise OursContractError("residual stratified split settings are invalid")
    if not episode_has_override:
        raise OursContractError("residual stratified split is empty")
    train: list[int] = []
    development: list[int] = []
    for label in (False, True):
        episode_ids = sorted(
            int(episode_id)
            for episode_id, value in episode_has_override.items()
            if bool(value) is label
        )
        if len(episode_ids) < 2:
            raise OursContractError(
                "residual stratified split requires two episodes in each class"
            )
        ranked = sorted(
            episode_ids,
            key=lambda episode_id: (
                hashlib.sha256(
                    f"ctr-residual-stratified-v1:{split_seed}:{episode_id}".encode()
                ).hexdigest(),
                episode_id,
            ),
        )
        validation_count = min(
            len(ranked) - 1,
            max(1, math.ceil(len(ranked) * validation_fraction)),
        )
        development.extend(ranked[:validation_count])
        train.extend(ranked[validation_count:])
    return {"train": sorted(train), "development": sorted(development)}


def _episode_override_strata(
    root: Path, manifest: dict[str, Any], np: Any
) -> dict[int, bool]:
    retry_index = RECOVERY_OPTIONS.index(RecoveryOption.RETRY_CURRENT)
    accept_index = RECOVERY_OPTIONS.index(RecoveryOption.ACCEPT_B)
    alternative_indices = [
        index
        for index in range(len(RECOVERY_OPTIONS))
        if index not in {accept_index, retry_index}
    ]
    result: dict[int, bool] = {}
    for filename in sorted(manifest["files_sha256"]):
        episode_id = int(filename.removeprefix("episode_").removesuffix(".npz"))
        with np.load(root / "features" / filename, allow_pickle=False) as episode:
            values = episode["target_option_values"].astype(np.float32, copy=False)
            valid = episode["target_option_valid"].astype(np.bool_, copy=False)
        if values.shape != valid.shape or values.shape[1] != len(RECOVERY_OPTIONS):
            raise OursContractError("residual option split targets are invalid")
        labeled = (valid.sum(axis=1) >= 2) & valid[:, retry_index]
        alternatives = np.where(
            valid[:, alternative_indices],
            values[:, alternative_indices],
            -np.inf,
        ).max(axis=1)
        result[episode_id] = bool(
            np.any(labeled & (alternatives > values[:, retry_index] + 1e-6))
        )
    return result


def run(
    corpus: Path,
    *,
    modulus: int,
    remainder: int,
    stratify_residual_overrides: bool = False,
) -> dict[str, Any]:
    root = corpus.expanduser().resolve()
    audit_recovery_corpus(
        root,
        verify_hashes=True,
        require_development=False,
        require_both_completion_classes=False,
    )
    path = root / OURS_CORPUS_MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("source_kind") != "counterfactual_live_rollout":
        raise OursContractError("option split requires a counterfactual rollout corpus")
    episode_ids = [
        int(name.removeprefix("episode_").removesuffix(".npz"))
        for name in manifest["files_sha256"]
    ]
    if stratify_residual_overrides:
        sampling = manifest.get("counterfactual_sampling", {})
        if (
            not bool(sampling.get("residual_retry_baseline"))
            or sampling.get("return_target") != OUTCOME_FIRST_RETURN_TARGET
        ):
            raise OursContractError(
                "residual stratification requires outcome-first residual labels"
            )
        import numpy as np

        consumed_seeds = manifest.get("consumed_rollout_base_seeds")
        if (
            not isinstance(consumed_seeds, list)
            or len(consumed_seeds) != 1
            or not isinstance(consumed_seeds[0], int)
        ):
            raise OursContractError("residual stratification requires one train seed")
        strata = _episode_override_strata(root, manifest, np)
        split = residual_stratified_episode_partition(
            strata,
            split_seed=consumed_seeds[0],
        )
        rule = "sha256 rank within physical-override episode strata"
        details = {
            "stratified_residual_overrides": True,
            "positive_episodes": sum(strata.values()),
            "negative_episodes": len(strata) - sum(strata.values()),
            "validation_positive_episodes": sum(
                strata[episode_id] for episode_id in split["development"]
            ),
            "validation_negative_episodes": sum(
                not strata[episode_id] for episode_id in split["development"]
            ),
            "hash_salt": "ctr-residual-stratified-v1",
        }
    else:
        split = option_episode_partition(episode_ids, modulus=modulus, remainder=remainder)
        rule = "episode_index modulo modulus equals remainder"
        details = {"stratified_residual_overrides": False}
    manifest["split_episodes"] = split
    manifest["counterfactual_option_split"] = {
        "scope": "train-seed episode holdout; not rollout development or final",
        "rule": rule,
        "modulus": modulus,
        "remainder": remainder,
        "train_episodes": len(split["train"]),
        "validation_episodes": len(split["development"]),
        **details,
    }
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    audit_recovery_corpus(
        root,
        verify_hashes=True,
        require_development=True,
        require_both_completion_classes=False,
    )
    return manifest


def main() -> int:
    args = _parser().parse_args()
    payload = run(
        args.corpus,
        modulus=args.modulus,
        remainder=args.remainder,
        stratify_residual_overrides=args.stratify_residual_overrides,
    )
    print(json.dumps(payload["counterfactual_option_split"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
