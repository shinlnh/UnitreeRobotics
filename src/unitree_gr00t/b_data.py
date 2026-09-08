"""Build the leakage-checked offline selector index for experiment B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .a1 import (
    audit_training_source,
    load_training_records,
    sha256_file,
    validate_converted_dataset,
)
from .a1_data import is_noop

_STEP_PATTERN = re.compile(r"^Step:\s*(.+?)\s*$", re.MULTILINE)
_RANGE_PATTERN = re.compile(r"^\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*$", re.MULTILINE)


@dataclass(frozen=True)
class AnnotatedSubgoal:
    instruction: str
    raw_start: int
    raw_end: int
    filtered_start: int
    filtered_end: int


@dataclass(frozen=True)
class SelectorEpisode:
    episode_index: int
    source_manifest_index: int
    scene: str
    case: str
    split: str
    instruction: str
    demonstration: str
    demonstration_sha256: str
    parquet: str
    agent_video: str
    wrist_video: str
    raw_frames: int
    filtered_frames: int
    retained_source_indices_sha256: str
    subgoals: tuple[AnnotatedSubgoal, ...]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--converted-dataset", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-manifest-rows", type=int, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--sample-stride", type=int, default=8)
    parser.add_argument("--post-boundary-steps", type=int, default=80)
    parser.add_argument("--development-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int)
    return parser


def parse_annotated_subgoals(description: str) -> tuple[tuple[str, int, int], ...]:
    """Parse canonical Step/range pairs and reject ambiguous annotations."""

    steps = _STEP_PATTERN.findall(description)
    ranges = [(int(start), int(end)) for start, end in _RANGE_PATTERN.findall(description)]
    if not steps or len(steps) != len(ranges):
        raise ValueError("task description must contain matching Step and frame-range entries")
    parsed: list[tuple[str, int, int]] = []
    previous_end = 0
    for index, (instruction, (start, end)) in enumerate(zip(steps, ranges, strict=True)):
        if not instruction.strip() or start < 0 or end <= start:
            raise ValueError(f"invalid subgoal annotation at index {index}")
        if index and start != previous_end:
            raise ValueError(f"non-contiguous subgoal annotation at index {index}")
        parsed.append((instruction.strip(), start, end))
        previous_end = end
    return tuple(parsed)


def retained_source_indices(actions: Any, np: Any) -> Any:
    """Replay the exact A1 no-op filter and retain raw indices for boundary mapping."""

    retained: list[int] = []
    previous = None
    for index, action in enumerate(actions):
        if not is_noop(action, previous, np):
            retained.append(index)
        previous = action
    return np.asarray(retained, dtype=np.int64)


def map_raw_boundary(raw_index: int, retained: Any, np: Any) -> int:
    """Map a raw half-open boundary to the filtered A1 timeline."""

    if raw_index < 0:
        raise ValueError("raw boundary cannot be negative")
    return int(np.searchsorted(retained, raw_index, side="left"))


def _split(key: str, development_fraction: float, seed: int) -> str:
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development fraction must be inside (0, 1)")
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "development" if value < development_fraction else "train"


def _paths(dataset: Path, episode_index: int) -> tuple[Path, Path, Path]:
    chunk = episode_index // 1000
    parquet = dataset / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    agent = (
        dataset
        / "videos"
        / f"chunk-{chunk:03d}"
        / "observation.images.image"
        / f"episode_{episode_index:06d}.mp4"
    )
    wrist = (
        dataset
        / "videos"
        / f"chunk-{chunk:03d}"
        / "observation.images.wrist_image"
        / f"episode_{episode_index:06d}.mp4"
    )
    return parquet, agent, wrist


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    temporary.replace(path)


def _sample_rows(
    episodes: list[SelectorEpisode], *, stride: int, post_boundary_steps: int
) -> list[dict[str, Any]]:
    if stride < 1 or post_boundary_steps < 0:
        raise ValueError("sample stride must be positive and post-boundary steps non-negative")
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        for subgoal_index, subgoal in enumerate(episode.subgoals):
            end = min(episode.filtered_frames, subgoal.filtered_end + post_boundary_steps)
            indices = set(range(subgoal.filtered_start, end, stride))
            indices.add(subgoal.filtered_start)
            if subgoal.filtered_end < episode.filtered_frames:
                indices.add(subgoal.filtered_end)
            if subgoal.filtered_end > subgoal.filtered_start:
                indices.add(subgoal.filtered_end - 1)
            for frame_index in sorted(
                index for index in indices if index < episode.filtered_frames
            ):
                rows.append(
                    {
                        "sample_index": len(rows),
                        "episode_index": episode.episode_index,
                        "split": episode.split,
                        "subgoal_index": subgoal_index,
                        "frame_index": frame_index,
                        "completion_step": subgoal.filtered_end,
                        "trajectory_steps": episode.filtered_frames,
                        "subgoal_instruction": subgoal.instruction,
                    }
                )
    return rows


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.action_horizon < 1 or args.sample_stride < 1 or args.post_boundary_steps < 0:
        raise ValueError("B selector sampling counts are invalid")
    source_audit = audit_training_source(
        args.source,
        args.manifest,
        args.benchmark_dir,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_manifest_rows=args.expected_manifest_rows,
        expected_usable_episodes=args.expected_episodes,
    )
    if not source_audit.valid:
        raise ValueError(f"B source leakage/audit failed: {asdict(source_audit)}")
    converted = validate_converted_dataset(
        args.converted_dataset,
        expected_episodes=args.expected_episodes,
        expected_revision=args.source_revision,
    )
    if not converted.valid:
        raise ValueError(f"B converted dataset audit failed: {converted.issues}")

    records = load_training_records(args.source, args.manifest)
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(records):
            raise ValueError("--limit must select available episodes")
        records = records[: args.limit]
    dataset = args.converted_dataset.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    if any(
        (destination / name).exists()
        for name in ("manifest.json", "episodes.jsonl", "samples.jsonl")
    ):
        raise FileExistsError(f"refusing to overwrite B selector index: {destination}")

    import h5py
    import numpy as np
    import pandas as pd

    episodes: list[SelectorEpisode] = []
    annotation_exclusions: list[dict[str, Any]] = []
    for converted_index, record in enumerate(records):
        parquet, agent_video, wrist_video = _paths(dataset, converted_index)
        if not (parquet.is_file() and agent_video.is_file() and wrist_video.is_file()):
            raise FileNotFoundError(f"converted episode {converted_index} is incomplete")
        filtered_frames = len(pd.read_parquet(parquet, columns=["frame_index"]))
        with h5py.File(record.demonstration, "r") as handle:
            actions = handle["data"]["demo_1"]["actions"][:]
        retained = retained_source_indices(actions, np)
        if len(retained) != filtered_frames:
            raise ValueError(
                f"no-op replay mismatch for episode {converted_index}: "
                f"{len(retained)} != {filtered_frames}"
            )
        try:
            parsed = parse_annotated_subgoals(record.task_description)
        except ValueError as exc:
            annotation_exclusions.append(
                {
                    "episode_index": converted_index,
                    "source_manifest_index": record.episode_index,
                    "source_case": f"{record.scene}/{record.case}",
                    "reason": str(exc),
                }
            )
            continue
        subgoals = tuple(
            AnnotatedSubgoal(
                instruction=instruction,
                raw_start=raw_start,
                raw_end=raw_end,
                filtered_start=map_raw_boundary(raw_start, retained, np),
                filtered_end=map_raw_boundary(raw_end, retained, np),
            )
            for instruction, raw_start, raw_end in parsed
        )
        if any(item.filtered_end <= item.filtered_start for item in subgoals):
            annotation_exclusions.append(
                {
                    "episode_index": converted_index,
                    "source_manifest_index": record.episode_index,
                    "source_case": f"{record.scene}/{record.case}",
                    "reason": "at least one annotated subgoal is empty after A1 no-op filtering",
                }
            )
            continue
        retained_hash = hashlib.sha256(retained.astype("<i8", copy=False).tobytes()).hexdigest()
        episodes.append(
            SelectorEpisode(
                episode_index=converted_index,
                source_manifest_index=record.episode_index,
                scene=record.scene,
                case=record.case,
                split=_split(
                    f"{record.scene}/{record.case}:{sha256_file(record.demonstration)}",
                    args.development_fraction,
                    args.seed,
                ),
                instruction=record.instruction,
                demonstration=str(record.demonstration),
                demonstration_sha256=sha256_file(record.demonstration),
                parquet=str(parquet),
                agent_video=str(agent_video),
                wrist_video=str(wrist_video),
                raw_frames=len(actions),
                filtered_frames=filtered_frames,
                retained_source_indices_sha256=retained_hash,
                subgoals=subgoals,
            )
        )

    samples = _sample_rows(
        episodes, stride=args.sample_stride, post_boundary_steps=args.post_boundary_steps
    )
    episode_rows = [
        asdict(episode) | {"subgoals": [asdict(value) for value in episode.subgoals]}
        for episode in episodes
    ]
    _write_jsonl(destination / "episodes.jsonl", episode_rows)
    _write_jsonl(destination / "samples.jsonl", samples)
    split_counts = {
        split: sum(episode.split == split for episode in episodes)
        for split in ("train", "development")
    }
    sample_split_counts = {
        split: sum(row["split"] == split for row in samples) for split in ("train", "development")
    }
    if min(split_counts.values()) < 1 or min(sample_split_counts.values()) < 1:
        raise ValueError("deterministic split produced an empty train or development partition")
    manifest = {
        "schema_version": 1,
        "experiment_id": "B",
        "dataset": "SparkVLA-style selector offline index",
        "source_revision": args.source_revision,
        "source_manifest": str(args.manifest.expanduser().resolve()),
        "source_manifest_sha256": sha256_file(args.manifest.expanduser().resolve()),
        "converted_dataset": str(dataset),
        "converted_dataset_provenance_sha256": sha256_file(
            dataset / "meta" / "a1_source_audit.json"
        ),
        "benchmark_exact_prompt_overlaps": len(source_audit.exact_prompt_overlaps),
        "action_horizon": args.action_horizon,
        "sample_stride": args.sample_stride,
        "post_boundary_steps": args.post_boundary_steps,
        "development_fraction": args.development_fraction,
        "split_seed": args.seed,
        "source_episodes": len(records),
        "episodes": len(episodes),
        "annotation_exclusion_count": len(annotation_exclusions),
        "annotation_exclusions": annotation_exclusions,
        "episode_split_counts": split_counts,
        "samples": len(samples),
        "sample_split_counts": sample_split_counts,
        "successful_demonstrations_only": True,
        "policy_rollouts_included": False,
        "episodes_sha256": sha256_file(destination / "episodes.jsonl"),
        "samples_sha256": sha256_file(destination / "samples.jsonl"),
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def main() -> int:
    args = _parser().parse_args()
    random.seed(args.seed)
    summary = build_dataset(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
