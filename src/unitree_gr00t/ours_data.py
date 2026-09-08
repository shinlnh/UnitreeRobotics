"""Audited compact temporal corpus for Ours."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .a1 import sha256_file
from .ours import OURS_ID, OURS_METHOD, OURS_PARENT, OURS_VARIANT, OursContractError

OURS_CORPUS_SCHEMA = 1
OURS_CORPUS_MANIFEST = "manifest.json"


@dataclass(frozen=True)
class DemonstrationTarget:
    sample_index: int
    episode_index: int
    subgoal_index: int
    frame_index: int
    subgoal_start: int
    completion_step: int
    elapsed_steps: int
    duration_steps: int
    progress: float
    complete: bool


@dataclass(frozen=True)
class RecoveryCorpusAudit:
    root: Path
    files: int
    samples: int
    train_episodes: int
    development_episodes: int
    completion_negatives: int
    completion_positives: int
    valid: bool


def build_demonstration_targets(
    samples: Sequence[Mapping[str, Any]],
    subgoals: Sequence[Mapping[str, Any]],
) -> tuple[DemonstrationTarget, ...]:
    """Derive causal progress labels from frozen demonstration annotations."""

    targets: list[DemonstrationTarget] = []
    for sample in samples:
        subgoal_index = int(sample["subgoal_index"])
        if not 0 <= subgoal_index < len(subgoals):
            raise OursContractError("demonstration sample has an invalid subgoal index")
        subgoal = subgoals[subgoal_index]
        start = int(subgoal["filtered_start"])
        annotated_end = int(subgoal["filtered_end"])
        completion = int(sample["completion_step"])
        frame = int(sample["frame_index"])
        if completion != annotated_end or completion < start or frame < start:
            raise OursContractError("demonstration progress annotation is inconsistent")
        duration = max(1, completion - start)
        elapsed = frame - start
        targets.append(
            DemonstrationTarget(
                sample_index=int(sample["sample_index"]),
                episode_index=int(sample["episode_index"]),
                subgoal_index=subgoal_index,
                frame_index=frame,
                subgoal_start=start,
                completion_step=completion,
                elapsed_steps=elapsed,
                duration_steps=duration,
                progress=min(1.0, max(0.0, elapsed / duration)),
                complete=frame >= completion,
            )
        )
    return tuple(targets)


def temporal_predecessors(
    targets: Sequence[DemonstrationTarget], history_length: int
) -> tuple[tuple[int, ...], ...]:
    """Return left-padded causal sample positions without crossing a subgoal."""

    if history_length < 1:
        raise OursContractError("history length must be positive")
    histories: list[tuple[int, ...]] = []
    seen: dict[tuple[int, int], list[int]] = defaultdict(list)
    for position, target in enumerate(targets):
        key = (target.episode_index, target.subgoal_index)
        candidates = (seen[key] + [position])[-history_length:]
        histories.append(tuple([candidates[0]] * (history_length - len(candidates)) + candidates))
        seen[key].append(position)
    return tuple(histories)


def _read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OursContractError("Ours corpus manifest must be a JSON object")
    return value


def audit_recovery_corpus(
    root: str | Path,
    *,
    expected_selector_sha256: str | None = None,
    verify_hashes: bool = True,
    require_development: bool = True,
    require_both_completion_classes: bool = True,
) -> RecoveryCorpusAudit:
    """Verify identities, split isolation, inventories, and hashes for Ours data."""

    corpus = Path(root).expanduser().resolve()
    manifest_path = corpus / OURS_CORPUS_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Ours corpus manifest not found: {manifest_path}")
    manifest = _read_manifest(manifest_path)
    identity = (
        manifest.get("experiment_id"),
        manifest.get("variant"),
        manifest.get("method"),
        manifest.get("parent_experiment"),
    )
    if identity != (OURS_ID, OURS_VARIANT, OURS_METHOD, OURS_PARENT):
        raise OursContractError("Ours corpus identity or parent is invalid")
    if int(manifest.get("schema_version", -1)) != OURS_CORPUS_SCHEMA:
        raise OursContractError("unsupported Ours corpus schema")
    if expected_selector_sha256 is not None and (
        manifest.get("selector_weights_sha256") != expected_selector_sha256
    ):
        raise OursContractError("Ours corpus does not bind the frozen B selector")
    if 7 in set(int(value) for value in manifest.get("consumed_rollout_base_seeds", [])):
        raise OursContractError("Ours corpus consumed the final held-out base seed")

    inventory = manifest.get("files_sha256")
    if not isinstance(inventory, dict) or not inventory:
        raise OursContractError("Ours corpus has no complete file inventory")
    if int(manifest.get("files", -1)) != len(inventory):
        raise OursContractError("Ours corpus file count does not match its inventory")
    if any(Path(name).name != name for name in inventory):
        raise OursContractError("Ours corpus inventory contains an unsafe filename")
    actual = {path.name for path in (corpus / "features").glob("episode_*.npz")}
    if actual != set(inventory):
        raise OursContractError("Ours corpus file inventory differs from disk")
    if verify_hashes:
        for name, expected in inventory.items():
            if sha256_file(corpus / "features" / name) != expected:
                raise OursContractError(f"Ours corpus hash mismatch: {name}")

    split_episodes = manifest.get("split_episodes")
    if not isinstance(split_episodes, dict):
        raise OursContractError("Ours corpus split inventory is missing")
    train = {int(value) for value in split_episodes.get("train", [])}
    development = {int(value) for value in split_episodes.get("development", [])}
    if not train or (require_development and not development) or train & development:
        raise OursContractError("Ours train/development episode splits overlap or are empty")
    counts = manifest.get("label_counts")
    if not isinstance(counts, dict):
        raise OursContractError("Ours corpus label counts are missing")
    negatives = int(counts.get("completion_negative", -1))
    positives = int(counts.get("completion_positive", -1))
    samples = int(manifest.get("samples", -1))
    if (
        min(negatives, positives) < int(require_both_completion_classes)
        or negatives + positives != samples
    ):
        raise OursContractError("Ours completion labels are degenerate or inconsistent")

    return RecoveryCorpusAudit(
        root=corpus,
        files=len(inventory),
        samples=samples,
        train_episodes=len(train),
        development_episodes=len(development),
        completion_negatives=negatives,
        completion_positives=positives,
        valid=True,
    )


def audit_payload(audit: RecoveryCorpusAudit) -> dict[str, Any]:
    return asdict(audit) | {"root": str(audit.root)}


def validate_finite(values: Iterable[float], *, field: str) -> None:
    if not all(math.isfinite(float(value)) for value in values):
        raise OursContractError(f"Ours corpus contains non-finite {field}")
