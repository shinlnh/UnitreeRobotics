"""Extract frozen A1 backbone contexts and proposed chunks for B training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .a0 import LANGUAGE_KEY
from .a1 import inspect_a1_checkpoint, sha256_file
from .b_runtime import BackboneCapture, flat_action_chunk


def episode_feature_seed(base_seed: int, episode_index: int) -> int:
    """Derive an order-independent seed so resumed diffusion chunks replay."""

    payload = f"B-feature-v1:{base_seed}:{episode_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFF_FFFF


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _observation(image: Any, wrist: Any, state: Any, instruction: str, np: Any) -> dict[str, Any]:
    def scalar(value: float) -> Any:
        return np.asarray(value, dtype=np.float32).reshape(1, 1, 1)

    state = np.asarray(state, dtype=np.float32)
    return {
        "video.image": np.ascontiguousarray(image, dtype=np.uint8)[None, None],
        "video.wrist_image": np.ascontiguousarray(wrist, dtype=np.uint8)[None, None],
        "state.x": scalar(state[0]),
        "state.y": scalar(state[1]),
        "state.z": scalar(state[2]),
        "state.roll": scalar(state[3]),
        "state.pitch": scalar(state[4]),
        "state.yaw": scalar(state[5]),
        "state.gripper": state[6:8].reshape(1, 1, -1),
        LANGUAGE_KEY: [instruction],
    }


def _batch(observations: list[dict[str, Any]], np: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in observations[0]:
        if key == LANGUAGE_KEY:
            result[key] = [observation[key][0] for observation in observations]
        else:
            result[key] = np.concatenate([observation[key] for observation in observations], axis=0)
    return result


def _episode_features(
    episode: dict[str, Any],
    samples: list[dict[str, Any]],
    policy: Any,
    capture: BackboneCapture,
    *,
    batch_size: int,
    extraction_seed: int,
    np: Any,
) -> dict[str, Any]:
    import cv2
    import pandas as pd

    frame = pd.read_parquet(episode["parquet"], columns=["observation.state"])
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_frame[int(sample["frame_index"])].append(sample)
    agent = cv2.VideoCapture(episode["agent_video"])
    wrist = cv2.VideoCapture(episode["wrist_video"])
    if not agent.isOpened() or not wrist.isOpened():
        raise RuntimeError(f"cannot open selector videos for episode {episode['episode_index']}")
    observations: list[dict[str, Any]] = []
    sample_ids: list[int] = []
    contexts: list[Any] = []
    chunks: list[Any] = []

    def flush() -> None:
        if not observations:
            return
        action, _ = policy.get_action(_batch(observations, np))
        contexts.append(capture.pop_context().cpu().numpy().astype(np.float16))
        chunks.append(flat_action_chunk(action, np).astype(np.float16))
        observations.clear()

    try:
        for frame_index in range(max(by_frame) + 1):
            ok_agent, agent_bgr = agent.read()
            ok_wrist, wrist_bgr = wrist.read()
            if not ok_agent or not ok_wrist:
                raise RuntimeError(
                    f"video ended before requested frame {frame_index} in episode "
                    f"{episode['episode_index']}"
                )
            if frame_index not in by_frame:
                continue
            agent_rgb = cv2.cvtColor(agent_bgr, cv2.COLOR_BGR2RGB)
            wrist_rgb = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)
            state = frame.iloc[frame_index]["observation.state"]
            for sample in by_frame[frame_index]:
                observations.append(
                    _observation(
                        agent_rgb,
                        wrist_rgb,
                        state,
                        str(sample["subgoal_instruction"]),
                        np,
                    )
                )
                sample_ids.append(int(sample["sample_index"]))
                if len(observations) == batch_size:
                    flush()
        flush()
    finally:
        agent.release()
        wrist.release()
    return {
        "sample_indices": np.asarray(sample_ids, dtype=np.int64),
        "contexts": np.concatenate(contexts, axis=0),
        "action_chunks": np.concatenate(chunks, axis=0),
        "extraction_seed": np.asarray(extraction_seed, dtype=np.int64),
        "batch_size": np.asarray(batch_size, dtype=np.int64),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("feature batch size must be positive")
    index = args.index.expanduser().resolve()
    index_manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    for name in ("episodes.jsonl", "samples.jsonl"):
        if sha256_file(index / name) != index_manifest[f"{name.removesuffix('.jsonl')}_sha256"]:
            raise ValueError(f"B selector index hash mismatch: {name}")
    contract, checkpoint_provenance = inspect_a1_checkpoint(
        args.checkpoint, expected_training_revision=args.model_revision
    )
    if contract.action_horizon != int(index_manifest["action_horizon"]):
        raise ValueError("A1 action horizon does not match B selector index")

    import numpy as np
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    base = Gr00tPolicy(
        EmbodimentTag.LIBERO_PANDA,
        str(args.checkpoint.expanduser().resolve()),
        device=args.device,
        strict=True,
    )
    policy = Gr00tSimPolicyWrapper(base)
    capture = BackboneCapture(base.model.backbone)
    episodes = _read_jsonl(index / "episodes.jsonl")
    samples = _read_jsonl(index / "samples.jsonl")
    if args.limit_episodes is not None:
        if args.limit_episodes < 1:
            raise ValueError("--limit-episodes must be positive")
        episodes = episodes[: args.limit_episodes]
    selected_ids = {int(episode["episode_index"]) for episode in episodes}
    samples_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        episode_index = int(sample["episode_index"])
        if episode_index in selected_ids:
            samples_by_episode[episode_index].append(sample)

    destination = args.destination.expanduser().resolve()
    features_dir = destination / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    completed: list[int] = []
    extracted_samples = 0
    try:
        for position, episode in enumerate(episodes, start=1):
            episode_index = int(episode["episode_index"])
            output = features_dir / f"episode_{episode_index:06d}.npz"
            expected_ids = [
                int(sample["sample_index"]) for sample in samples_by_episode[episode_index]
            ]
            extraction_seed = episode_feature_seed(args.seed, episode_index)
            if args.resume and output.is_file():
                with np.load(output) as existing:
                    actual_ids = existing["sample_indices"].tolist()
                    metadata_match = (
                        "extraction_seed" in existing
                        and "batch_size" in existing
                        and int(existing["extraction_seed"]) == extraction_seed
                        and int(existing["batch_size"]) == args.batch_size
                    )
                    ids_match = len(actual_ids) == len(set(actual_ids)) and sorted(
                        actual_ids
                    ) == sorted(expected_ids)
                if not metadata_match or not ids_match:
                    raise ValueError(
                        f"resume feature contract mismatch for episode {episode_index}"
                    )
                completed.append(episode_index)
                extracted_samples += len(expected_ids)
                continue
            random.seed(extraction_seed)
            np.random.seed(extraction_seed)
            torch.manual_seed(extraction_seed)
            torch.cuda.manual_seed_all(extraction_seed)
            value = _episode_features(
                episode,
                samples_by_episode[episode_index],
                policy,
                capture,
                batch_size=args.batch_size,
                extraction_seed=extraction_seed,
                np=np,
            )
            temporary = output.with_suffix(f".tmp-{os.getpid()}.npz")
            np.savez_compressed(temporary, **value)
            temporary.replace(output)
            completed.append(episode_index)
            extracted_samples += len(expected_ids)
            progress = {
                "complete": position == len(episodes),
                "episodes": len(completed),
                "expected_episodes": len(episodes),
                "samples": extracted_samples,
            }
            _write_json(destination / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    finally:
        capture.close()

    manifest = {
        "schema_version": 1,
        "experiment_id": "B",
        "feature_contract": "frozen GR00T-RC last-valid backbone token plus proposed H16 chunk",
        "selector_index": str(index),
        "selector_index_manifest_sha256": sha256_file(index / "manifest.json"),
        "checkpoint": str(contract.checkpoint_dir),
        "checkpoint_training_revision": checkpoint_provenance["training_dataset_revision"],
        "checkpoint_provenance_sha256": sha256_file(
            contract.checkpoint_dir / "a1_training_provenance.json"
        ),
        "context_width": 2048,
        "action_horizon": contract.action_horizon,
        "action_dim": 7,
        "dtype": "float16",
        "seed": args.seed,
        "seed_derivation": "sha256(B-feature-v1:base_seed:episode_index)[:31-bit]",
        "batch_size": args.batch_size,
        "episodes": len(completed),
        "samples": extracted_samples,
        "complete": len(completed) == len(episodes),
        "limited": args.limit_episodes is not None,
        "feature_files": len(completed),
        "feature_files_sha256": {
            f"episode_{episode_index:06d}.npz": sha256_file(
                features_dir / f"episode_{episode_index:06d}.npz"
            )
            for episode_index in completed
        },
    }
    _write_json(
        destination / "progress.json",
        {
            "complete": manifest["complete"],
            "episodes": manifest["episodes"],
            "expected_episodes": len(episodes),
            "samples": manifest["samples"],
        },
    )
    _write_json(destination / "feature_manifest.json", manifest)
    return manifest


def main() -> int:
    summary = run(_parser().parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
