"""Contracts for A1: a shared RoboCerebra-post-trained GR00T checkpoint."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .a0 import CheckpointContract, inspect_checkpoint, parse_task_description

A1_ID = "A1"
A1_VARIANT = "GR00T-RC"
A1_PROVENANCE_FILE = "a1_training_provenance.json"


class A1ContractError(ValueError):
    """Raised when an A1 training or checkpoint contract is violated."""


@dataclass(frozen=True)
class TrainingRecord:
    episode_index: int
    scene: str
    case: str
    instruction: str
    task_description: str
    directory: Path
    demonstration: Path
    bddl: Path


@dataclass(frozen=True)
class TrainingSourceAudit:
    source_root: Path
    manifest: Path
    manifest_sha256: str
    expected_manifest_sha256: str
    manifest_rows: int
    expected_manifest_rows: int
    usable_episodes: int
    expected_usable_episodes: int
    unique_source_cases: int
    unique_training_prompts: int
    benchmark_prompts: int
    exact_prompt_overlaps: tuple[str, ...]
    missing_demonstrations: tuple[str, ...]
    missing_bddl_cases: tuple[str, ...]
    resolved_ambiguous_bddl_cases: tuple[str, ...]
    malformed_demonstrations: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return (
            self.manifest_rows == self.expected_manifest_rows
            and self.manifest_sha256 == self.expected_manifest_sha256
            and self.usable_episodes == self.expected_usable_episodes
            and self.unique_source_cases == self.manifest_rows
            and not self.exact_prompt_overlaps
            and not self.malformed_demonstrations
        )


@dataclass(frozen=True)
class ConvertedDatasetAudit:
    root: Path
    episodes: int
    frames: int
    parquet_files: int
    video_files: int
    fps: int
    source_revision: str
    exact_prompt_overlaps: int
    valid: bool
    issues: tuple[str, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_prompt(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise A1ContractError(f"Cannot read A1 training manifest {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise A1ContractError("A1 training manifest must be a JSON array of objects")
    return value


def _demonstration_metadata(path: Path) -> tuple[str, str]:
    try:
        import h5py
    except ImportError as exc:
        raise A1ContractError(
            "h5py is required to inspect A1 demonstrations; run with the .venv-a0 runtime"
        ) from exc

    try:
        with h5py.File(path, "r") as handle:
            attributes = handle["data"].attrs
            problem_info = json.loads(str(attributes["problem_info"]))
            instruction = str(problem_info["language_instruction"]).strip()
            bddl_name = Path(str(attributes["bddl_file_name"])).name
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise A1ContractError(f"Cannot inspect demonstration metadata {path}: {exc}") from exc
    if not instruction or not bddl_name:
        raise A1ContractError(f"Incomplete demonstration metadata in {path}")
    return instruction, bddl_name


def load_training_records(source_root: str | Path, manifest: str | Path) -> list[TrainingRecord]:
    """Load only replayable rows, using HDF5 metadata as the label/BDDL authority."""

    root = Path(source_root).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    records: list[TrainingRecord] = []
    for episode_index, row in enumerate(_load_manifest_rows(manifest_path)):
        scene = str(row.get("scene", "")).strip()
        case = str(row.get("case", "")).strip()
        description = str(row.get("task_description", "")).strip()
        if not scene or not case or not description:
            raise A1ContractError(f"Incomplete A1 training row at index {episode_index}")
        directory = root / scene / case
        demonstration = directory / "demo.hdf5"
        if not demonstration.is_file():
            continue
        instruction, bddl_name = _demonstration_metadata(demonstration)
        bddl = directory / bddl_name
        if not bddl.is_file():
            continue
        records.append(
            TrainingRecord(
                episode_index=episode_index,
                scene=scene,
                case=case,
                instruction=instruction,
                task_description=description,
                directory=directory,
                demonstration=demonstration,
                bddl=bddl,
            )
        )
    return records


def _benchmark_prompts(benchmark_root: Path) -> set[str]:
    prompts: set[str] = set()
    for path in sorted(benchmark_root.glob("*/*/task_description.txt")):
        prompts.add(normalize_prompt(parse_task_description(path).instruction))
    return prompts


def audit_training_source(
    source_root: str | Path,
    manifest: str | Path,
    benchmark_root: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_manifest_rows: int,
    expected_usable_episodes: int,
) -> TrainingSourceAudit:
    root = Path(source_root).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    rows = _load_manifest_rows(manifest_path)
    records = load_training_records(root, manifest_path)
    source_keys = {
        (str(row.get("scene", "")).strip(), str(row.get("case", "")).strip()) for row in rows
    }
    training_prompts = {normalize_prompt(record.instruction) for record in records}
    benchmark_prompts = _benchmark_prompts(Path(benchmark_root).expanduser().resolve())
    missing: list[str] = []
    missing_bddl: list[str] = []
    ambiguous: list[str] = []
    malformed: list[str] = []
    for row in rows:
        scene = str(row.get("scene", "")).strip()
        case = str(row.get("case", "")).strip()
        key = f"{scene}/{case}"
        demonstration = root / scene / case / "demo.hdf5"
        if not demonstration.is_file():
            missing.append(key)
            continue
        try:
            _, bddl_name = _demonstration_metadata(demonstration)
        except A1ContractError:
            malformed.append(key)
            continue
        bddl_files = tuple(sorted((root / scene / case).glob("*.bddl")))
        if not (root / scene / case / bddl_name).is_file():
            missing_bddl.append(key)
        elif len(bddl_files) > 1:
            ambiguous.append(f"{key}:{bddl_name}")
    return TrainingSourceAudit(
        source_root=root,
        manifest=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        expected_manifest_sha256=expected_manifest_sha256,
        manifest_rows=len(rows),
        expected_manifest_rows=expected_manifest_rows,
        usable_episodes=len(records),
        expected_usable_episodes=expected_usable_episodes,
        unique_source_cases=len(source_keys),
        unique_training_prompts=len(training_prompts),
        benchmark_prompts=len(benchmark_prompts),
        exact_prompt_overlaps=tuple(sorted(training_prompts & benchmark_prompts)),
        missing_demonstrations=tuple(missing),
        missing_bddl_cases=tuple(missing_bddl),
        resolved_ambiguous_bddl_cases=tuple(ambiguous),
        malformed_demonstrations=tuple(malformed),
    )


def validate_converted_dataset(
    root: str | Path,
    *,
    expected_episodes: int | None = None,
    expected_revision: str | None = None,
) -> ConvertedDatasetAudit:
    dataset_root = Path(root).expanduser().resolve()
    issues: list[str] = []

    def read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(f"cannot read {path.relative_to(dataset_root)}: {exc}")
            return {}
        if not isinstance(value, dict):
            issues.append(f"{path.relative_to(dataset_root)} is not an object")
            return {}
        return value

    info = read_json(dataset_root / "meta" / "info.json")
    provenance = read_json(dataset_root / "meta" / "a1_source_audit.json")
    required = (
        dataset_root / "meta" / "modality.json",
        dataset_root / "meta" / "tasks.jsonl",
        dataset_root / "meta" / "episodes.jsonl",
        dataset_root / "meta" / "stats.json",
    )
    for path in required:
        if not path.is_file():
            issues.append(f"missing {path.relative_to(dataset_root)}")
    episodes = int(info.get("total_episodes", 0))
    frames = int(info.get("total_frames", 0))
    fps = int(info.get("fps", 0))
    parquets = len(tuple((dataset_root / "data").glob("*/*.parquet")))
    videos = len(tuple((dataset_root / "videos").glob("*/*/*.mp4")))
    if expected_episodes is not None and episodes != expected_episodes:
        issues.append(f"expected {expected_episodes} episodes, found {episodes}")
    if parquets != episodes:
        issues.append(f"expected {episodes} parquet files, found {parquets}")
    if videos != episodes * 2:
        issues.append(f"expected {episodes * 2} videos, found {videos}")
    if frames <= episodes * 16:
        issues.append("dataset does not retain at least one native H16 window per episode")
    if fps != 20:
        issues.append(f"expected 20 fps, found {fps}")
    source_revision = str(provenance.get("source_revision", ""))
    if expected_revision is not None and source_revision != expected_revision:
        issues.append(
            f"source revision mismatch: expected {expected_revision}, found {source_revision or 'none'}"
        )
    overlap_count = int(provenance.get("exact_prompt_overlap_count", -1))
    if overlap_count != 0:
        issues.append(f"training/benchmark exact prompt overlap count is {overlap_count}")
    return ConvertedDatasetAudit(
        root=dataset_root,
        episodes=episodes,
        frames=frames,
        parquet_files=parquets,
        video_files=videos,
        fps=fps,
        source_revision=source_revision,
        exact_prompt_overlaps=overlap_count,
        valid=not issues,
        issues=tuple(issues),
    )


def write_checkpoint_provenance(
    checkpoint_dir: str | Path,
    *,
    training_dataset: str,
    training_dataset_revision: str,
    converted_dataset: ConvertedDatasetAudit,
    base_checkpoint: str | Path,
    max_steps: int,
    effective_batch_size: int,
) -> Path:
    root = Path(checkpoint_dir).expanduser().resolve()
    contract = inspect_checkpoint(root)
    payload = {
        "schema_version": 1,
        "experiment_id": A1_ID,
        "variant": A1_VARIANT,
        "training_dataset": training_dataset,
        "training_dataset_revision": training_dataset_revision,
        "converted_dataset": asdict(converted_dataset) | {"root": str(converted_dataset.root)},
        "base_checkpoint": str(Path(base_checkpoint).expanduser().resolve()),
        "max_steps": max_steps,
        "effective_batch_size": effective_batch_size,
        "checkpoint_contract": asdict(contract) | {"checkpoint_dir": str(root)},
    }
    path = root / A1_PROVENANCE_FILE
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def inspect_a1_checkpoint(
    checkpoint_dir: str | Path, *, expected_training_revision: str | None = None
) -> tuple[CheckpointContract, dict[str, Any]]:
    root = Path(checkpoint_dir).expanduser().resolve()
    contract = inspect_checkpoint(root)
    provenance_path = root / A1_PROVENANCE_FILE
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise A1ContractError(f"Cannot read A1 checkpoint provenance: {exc}") from exc
    if provenance.get("experiment_id") != A1_ID or provenance.get("variant") != A1_VARIANT:
        raise A1ContractError("Checkpoint is not marked as the frozen A1 GR00T-RC variant")
    actual_revision = provenance.get("training_dataset_revision")
    if expected_training_revision is not None and actual_revision != expected_training_revision:
        raise A1ContractError(
            f"A1 training revision mismatch: {actual_revision!r} != {expected_training_revision!r}"
        )
    converted = provenance.get("converted_dataset", {})
    if converted.get("exact_prompt_overlaps") != 0 or not converted.get("valid"):
        raise A1ContractError("A1 checkpoint provenance does not contain a clean dataset audit")
    return contract, provenance
