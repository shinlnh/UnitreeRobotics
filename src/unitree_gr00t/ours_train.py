"""Train and calibrate Ours temporal completion/failure recovery heads."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .a1 import sha256_file
from .ours import OURS_ID, OURS_METHOD, OURS_PARENT, OURS_VARIANT
from .ours_data import OURS_CORPUS_MANIFEST, audit_recovery_corpus
from .ours_model import (
    TemporalRecoveryModelConfig,
    build_temporal_recovery_model,
    completion_progress_loss,
    count_trainable_parameters,
)

OURS_CHECKPOINT_PROVENANCE = "ours_recovery_provenance.json"


@dataclass(frozen=True)
class LoadedCorpus:
    contexts: Any
    anchor_ids: Any
    action_chunks: Any
    selector_features: Any
    scalars: Any
    histories: Any
    target_progress: Any
    target_progress_valid: Any
    target_complete: Any
    target_failure: Any
    target_failure_valid: Any
    target_option_values: Any
    target_option_valid: Any
    selector_candidates: Any
    train_ids: Any
    development_ids: Any
    live_ids: Any


@dataclass(frozen=True)
class RecoveryCheckpointAudit:
    checkpoint_dir: Path
    weights_sha256: str
    provenance_sha256: str
    parameter_count: int
    encoder: str
    history_length: int
    completion_threshold: float
    valid: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--additional-train-dataset",
        type=Path,
        help="Optional train-only live-rollout corpus; the primary development split stays frozen",
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--encoder", choices=("linear", "mlp", "gru", "transformer"), required=True)
    parser.add_argument("--history-length", type=int, choices=(4, 8, 16), required=True)
    parser.add_argument("--temporal-width", type=int, default=256)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--temporal-heads", type=int, default=8)
    parser.add_argument("--feedforward-width", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--validate-steps", type=int, default=250)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-parameters", type=int, default=15_000_000)
    parser.add_argument("--max-false-positive-rate", type=float, default=0.05)
    parser.add_argument(
        "--live-batch-fraction",
        type=float,
        default=0.25,
        help="Fraction of each batch sampled from the optional live-rollout corpus",
    )
    parser.add_argument(
        "--option-batch-fraction",
        type=float,
        default=0.125,
        help="Minimum batch fraction carrying counterfactual option supervision",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=10007)
    return parser


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def normalized_selector_features(scores: Any, valid: Any, np: Any) -> Any:
    masked = np.where(valid, scores, 0.0).astype(np.float32, copy=False)
    count = valid.sum(axis=1, keepdims=True).clip(min=1)
    mean = masked.sum(axis=1, keepdims=True) / count
    centered = np.where(valid, scores - mean, 0.0)
    variance = (centered * centered).sum(axis=1, keepdims=True) / count
    normalized = np.where(valid, centered / np.sqrt(variance + 1e-6), 0.0)
    return np.concatenate((normalized, valid.astype(np.float32)), axis=1).astype(np.float16)


def runtime_scalars(
    *, elapsed_steps: Any, candidates: Any, chunks: Any, scores: Any, valid: Any, np: Any
) -> Any:
    horizon = chunks.shape[1]
    valid_nonstop = valid[:, 1:]
    nonstop = np.where(valid_nonstop, scores[:, 1:], -np.inf).max(axis=1)
    margin = np.clip(scores[:, 0] - nonstop, -20.0, 20.0) / 20.0
    magnitude = np.linalg.norm(chunks.astype(np.float32), axis=2)
    return np.stack(
        (
            np.clip(elapsed_steps.astype(np.float32) / 150.0, 0.0, 4.0),
            candidates.astype(np.float32) / horizon,
            (candidates == 0).astype(np.float32),
            margin.astype(np.float32),
            magnitude.mean(axis=1),
            magnitude.max(axis=1),
        ),
        axis=1,
    ).astype(np.float16)


def load_corpus(
    root: Path, history_length: int, np: Any, *, require_development: bool = True
) -> LoadedCorpus:
    """Load compact files into a one-gigabyte indexed CPU cache."""

    manifest = json.loads((root / OURS_CORPUS_MANIFEST).read_text(encoding="utf-8"))
    samples = int(manifest["samples"])
    context_width = int(manifest["context_width"])
    horizon = int(manifest["action_horizon"])
    action_dim = int(manifest["action_dim"])
    contexts = np.empty((samples, context_width), dtype=np.float16)
    anchor_ids = np.empty(samples, dtype=np.int64)
    action_chunks = np.empty((samples, horizon, action_dim), dtype=np.float16)
    selector_features = np.empty((samples, 2 * (horizon + 1)), dtype=np.float16)
    scalars = np.empty((samples, 6), dtype=np.float16)
    histories = np.empty((samples, history_length), dtype=np.int64)
    target_progress = np.empty(samples, dtype=np.float32)
    target_progress_valid = np.empty(samples, dtype=np.bool_)
    target_complete = np.empty(samples, dtype=np.bool_)
    target_failure = np.empty(samples, dtype=np.bool_)
    target_failure_valid = np.empty(samples, dtype=np.bool_)
    target_option_values = np.zeros((samples, 6), dtype=np.float32)
    target_option_valid = np.zeros((samples, 6), dtype=np.bool_)
    selector_candidates = np.empty(samples, dtype=np.int8)
    seen = np.zeros(samples, dtype=np.bool_)
    train_episode_ids = set(int(value) for value in manifest["split_episodes"]["train"])
    development_episode_ids = set(int(value) for value in manifest["split_episodes"]["development"])
    train_ids: list[Any] = []
    development_ids: list[Any] = []
    source_is_live = manifest.get("source_kind") in {
        "live_rollout",
        "counterfactual_live_rollout",
    }

    for filename in sorted(manifest["files_sha256"]):
        episode_index = int(filename.removeprefix("episode_").removesuffix(".npz"))
        with np.load(root / "features" / filename) as value:
            ids = value["sample_indices"].astype(np.int64, copy=False)
            if np.any(ids < 0) or np.any(ids >= samples) or np.any(seen[ids]):
                raise ValueError(f"invalid or duplicate Ours sample ids: {filename}")
            episode_contexts = value["contexts"].astype(np.float16, copy=False)
            episode_chunks = value["action_chunks"].astype(np.float16, copy=False)
            scores = value["selector_scores"].astype(np.float32, copy=False)
            valid = value["selector_valid"].astype(np.bool_, copy=False)
            candidates = value["selector_candidates"].astype(np.int8, copy=False)
            local_anchors = value["anchor_positions"].astype(np.int64, copy=False)
            anchor_contexts = (
                value["anchor_contexts"].astype(np.float16, copy=False)
                if "anchor_contexts" in value
                else None
            )
            subgoals = value["subgoal_indices"].astype(np.int64, copy=False)
            elapsed = value["elapsed_steps"].astype(np.int64, copy=False)
            progress = value["target_progress"].astype(np.float32, copy=False)
            progress_valid = (
                value["target_progress_valid"].astype(np.bool_, copy=False)
                if "target_progress_valid" in value
                else np.ones(len(ids), dtype=np.bool_)
            )
            complete = value["target_complete"].astype(np.bool_, copy=False)
            failure = (
                value["target_failure"].astype(np.bool_, copy=False)
                if "target_failure" in value
                else np.zeros(len(ids), dtype=np.bool_)
            )
            failure_valid = (
                np.ones(len(ids), dtype=np.bool_)
                if "target_failure" in value
                else np.zeros(len(ids), dtype=np.bool_)
            )
            option_values = (
                value["target_option_values"].astype(np.float32, copy=False)
                if "target_option_values" in value
                else np.zeros((len(ids), 6), dtype=np.float32)
            )
            option_valid = (
                value["target_option_valid"].astype(np.bool_, copy=False)
                if "target_option_valid" in value
                else np.zeros((len(ids), 6), dtype=np.bool_)
            )
        row_count = len(ids)
        shapes = (
            len(episode_contexts),
            len(episode_chunks),
            len(scores),
            len(valid),
            len(candidates),
            len(local_anchors),
            len(subgoals),
            len(elapsed),
            len(progress),
            len(progress_valid),
            len(complete),
            len(failure),
            len(failure_valid),
            len(option_values),
            len(option_valid),
        )
        expected_shapes = (
            (row_count, context_width),
            (row_count, horizon, action_dim),
            (row_count, horizon + 1),
            (row_count, horizon + 1),
        )
        actual_shapes = (
            episode_contexts.shape,
            episode_chunks.shape,
            scores.shape,
            valid.shape,
        )
        candidate_out_of_bounds = np.any((candidates < 0) | (candidates > horizon))
        invalid_candidate = bool(candidate_out_of_bounds) or np.any(
            ~valid[np.arange(row_count), candidates.astype(np.int64)]
        )
        if (
            any(size != row_count for size in shapes)
            or actual_shapes != expected_shapes
            or np.any((local_anchors < 0) | (local_anchors >= row_count))
            or invalid_candidate
            or not np.isfinite(episode_contexts).all()
            or not np.isfinite(episode_chunks).all()
            or not np.isfinite(scores).all()
            or option_values.shape != (row_count, 6)
            or option_valid.shape != (row_count, 6)
            or not np.isfinite(option_values).all()
        ):
            raise ValueError(f"inconsistent Ours feature rows: {filename}")
        if anchor_contexts is not None:
            if anchor_contexts.shape != episode_contexts.shape or not np.array_equal(
                anchor_contexts,
                episode_contexts[local_anchors],
            ):
                raise ValueError(f"noncausal Ours anchor contexts: {filename}")
        contexts[ids] = episode_contexts
        anchor_ids[ids] = ids[local_anchors]
        action_chunks[ids] = episode_chunks
        selector_features[ids] = normalized_selector_features(scores, valid, np)
        scalars[ids] = runtime_scalars(
            elapsed_steps=elapsed,
            candidates=candidates,
            chunks=episode_chunks,
            scores=scores,
            valid=valid,
            np=np,
        )
        target_progress[ids] = progress
        target_progress_valid[ids] = progress_valid
        target_complete[ids] = complete
        target_failure[ids] = failure
        target_failure_valid[ids] = failure_valid
        target_option_values[ids] = option_values
        target_option_valid[ids] = option_valid
        selector_candidates[ids] = candidates
        seen[ids] = True

        for subgoal in np.unique(subgoals):
            positions = np.flatnonzero(subgoals == subgoal)
            ordered_ids = ids[positions]
            for offset, sample_id in enumerate(ordered_ids):
                start = max(0, offset - history_length + 1)
                previous = ordered_ids[start : offset + 1]
                histories[sample_id] = np.pad(
                    previous,
                    (history_length - len(previous), 0),
                    mode="constant",
                    constant_values=previous[0],
                )
        if episode_index in train_episode_ids:
            train_ids.append(ids)
        elif episode_index in development_episode_ids:
            development_ids.append(ids)
        else:
            raise ValueError(f"Ours corpus episode is absent from both splits: {episode_index}")

    if not seen.all() or not train_ids or (require_development and not development_ids):
        raise ValueError("Ours corpus cache is incomplete or has an empty split")
    concatenated_train = np.concatenate(train_ids)
    concatenated_development = (
        np.concatenate(development_ids) if development_ids else np.empty(0, dtype=np.int64)
    )
    return LoadedCorpus(
        contexts=contexts,
        anchor_ids=anchor_ids,
        action_chunks=action_chunks,
        selector_features=selector_features,
        scalars=scalars,
        histories=histories,
        target_progress=target_progress,
        target_progress_valid=target_progress_valid,
        target_complete=target_complete,
        target_failure=target_failure,
        target_failure_valid=target_failure_valid,
        target_option_values=target_option_values,
        target_option_valid=target_option_valid,
        selector_candidates=selector_candidates,
        train_ids=concatenated_train,
        development_ids=concatenated_development,
        live_ids=concatenated_train if source_is_live else np.empty(0, dtype=np.int64),
    )


def merge_corpora(primary: LoadedCorpus, additional: LoadedCorpus, np: Any) -> LoadedCorpus:
    offset = len(primary.contexts)
    return LoadedCorpus(
        contexts=np.concatenate((primary.contexts, additional.contexts)),
        anchor_ids=np.concatenate((primary.anchor_ids, additional.anchor_ids + offset)),
        action_chunks=np.concatenate((primary.action_chunks, additional.action_chunks)),
        selector_features=np.concatenate((primary.selector_features, additional.selector_features)),
        scalars=np.concatenate((primary.scalars, additional.scalars)),
        histories=np.concatenate((primary.histories, additional.histories + offset)),
        target_progress=np.concatenate((primary.target_progress, additional.target_progress)),
        target_progress_valid=np.concatenate(
            (primary.target_progress_valid, additional.target_progress_valid)
        ),
        target_complete=np.concatenate((primary.target_complete, additional.target_complete)),
        target_failure=np.concatenate((primary.target_failure, additional.target_failure)),
        target_failure_valid=np.concatenate(
            (primary.target_failure_valid, additional.target_failure_valid)
        ),
        target_option_values=np.concatenate(
            (primary.target_option_values, additional.target_option_values)
        ),
        target_option_valid=np.concatenate(
            (primary.target_option_valid, additional.target_option_valid)
        ),
        selector_candidates=np.concatenate(
            (primary.selector_candidates, additional.selector_candidates)
        ),
        train_ids=np.concatenate((primary.train_ids, additional.train_ids + offset)),
        development_ids=primary.development_ids,
        live_ids=additional.train_ids + offset,
    )


def sample_training_ids(
    corpus: LoadedCorpus,
    *,
    batch_size: int,
    live_batch_fraction: float,
    option_batch_fraction: float = 0.0,
    generator: Any,
    np: Any,
) -> Any:
    """Sample a repeatable batch with live and counterfactual-label quotas."""

    live_count = (
        min(batch_size - 2, max(1, round(batch_size * live_batch_fraction)))
        if len(corpus.live_ids) and live_batch_fraction > 0.0
        else 0
    )
    demonstration_count = len(corpus.train_ids) - len(corpus.live_ids)
    demonstration_ids = corpus.train_ids[:demonstration_count]
    demo_count = batch_size - live_count
    demo_positive = demonstration_ids[corpus.target_complete[demonstration_ids]]
    demo_negative = demonstration_ids[~corpus.target_complete[demonstration_ids]]
    positive_count = demo_count // 2
    negative_count = demo_count - positive_count
    if not len(demo_positive) or not len(demo_negative):
        raise ValueError("primary Ours training split requires both completion classes")
    parts: tuple[Any, ...] = (
        generator.choice(demo_positive, size=positive_count, replace=True),
        generator.choice(demo_negative, size=negative_count, replace=True),
    )
    if live_count:
        option_ids = corpus.live_ids[
            corpus.target_option_valid[corpus.live_ids].sum(axis=1) >= 2
        ]
        option_count = (
            min(live_count, max(1, round(batch_size * option_batch_fraction)))
            if len(option_ids) and option_batch_fraction > 0.0
            else 0
        )
        if option_count:
            masked_values = np.where(
                corpus.target_option_valid[option_ids],
                corpus.target_option_values[option_ids],
                -np.inf,
            )
            winners = masked_values.argmax(axis=1)
            winner_groups = [
                option_ids[winners == winner] for winner in np.unique(winners)
            ]
            per_group, remainder = divmod(option_count, len(winner_groups))
            sampled_options = [
                generator.choice(
                    group,
                    size=per_group + int(index < remainder),
                    replace=True,
                )
                for index, group in enumerate(winner_groups)
                if per_group + int(index < remainder)
            ]
            parts += (np.concatenate(sampled_options),)
        remaining_live = live_count - option_count
        if remaining_live:
            live_positive = corpus.live_ids[corpus.target_complete[corpus.live_ids]]
            live_negative = corpus.live_ids[~corpus.target_complete[corpus.live_ids]]
            if len(live_positive) and len(live_negative):
                live_positive_count = remaining_live // 2
                parts += (
                    generator.choice(live_positive, size=live_positive_count, replace=True),
                    generator.choice(
                        live_negative,
                        size=remaining_live - live_positive_count,
                        replace=True,
                    ),
                )
            else:
                parts += (
                    generator.choice(corpus.live_ids, size=remaining_live, replace=True),
                )
    ids = np.concatenate(parts)
    generator.shuffle(ids)
    return ids


def _batch(corpus: LoadedCorpus, ids: Any, *, device: str, np: Any, torch: Any) -> dict[str, Any]:
    history_ids = corpus.histories[ids]
    anchor_history_ids = corpus.anchor_ids[history_ids]
    return {
        "contexts": torch.as_tensor(
            corpus.contexts[history_ids], device=device, dtype=torch.float32
        ),
        "anchors": torch.as_tensor(
            corpus.contexts[anchor_history_ids], device=device, dtype=torch.float32
        ),
        "actions": torch.as_tensor(
            corpus.action_chunks[history_ids], device=device, dtype=torch.float32
        ),
        "selector": torch.as_tensor(
            corpus.selector_features[history_ids], device=device, dtype=torch.float32
        ),
        "scalars": torch.as_tensor(corpus.scalars[history_ids], device=device, dtype=torch.float32),
        "progress": torch.as_tensor(
            corpus.target_progress[ids], device=device, dtype=torch.float32
        ),
        "progress_valid": torch.as_tensor(corpus.target_progress_valid[ids], device=device),
        "complete": torch.as_tensor(corpus.target_complete[ids], device=device),
        "failure": torch.as_tensor(corpus.target_failure[ids], device=device),
        "failure_valid": torch.as_tensor(corpus.target_failure_valid[ids], device=device),
        "option_values": torch.as_tensor(
            corpus.target_option_values[ids], device=device, dtype=torch.float32
        ),
        "option_valid": torch.as_tensor(corpus.target_option_valid[ids], device=device),
    }


def calibrate_threshold(
    probabilities: Any, labels: Any, *, max_false_positive_rate: float, np: Any
) -> tuple[float, dict[str, float]]:
    negatives = ~labels
    positives = labels
    if not negatives.any() or not positives.any():
        raise ValueError("completion calibration requires both classes")
    thresholds = np.unique(np.concatenate((probabilities, np.asarray([1.0]))))[::-1]
    best: tuple[float, float, float] | None = None
    for threshold in thresholds:
        predicted = probabilities >= threshold
        false_positive_rate = float((predicted & negatives).sum() / negatives.sum())
        true_positive_rate = float((predicted & positives).sum() / positives.sum())
        if false_positive_rate <= max_false_positive_rate + 1e-12:
            candidate = (true_positive_rate, float(threshold), false_positive_rate)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        best = (0.0, 1.0, 0.0)
    true_positive_rate, threshold, false_positive_rate = best
    predicted = probabilities >= threshold
    precision_denominator = int(predicted.sum())
    precision = (
        float((predicted & positives).sum() / precision_denominator)
        if precision_denominator
        else 1.0
    )
    return threshold, {
        "false_positive_rate": false_positive_rate,
        "true_positive_rate": true_positive_rate,
        "precision": precision,
        "accuracy": float((predicted == labels).mean()),
    }


def inspect_recovery_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_selector_sha256: str | None = None,
    expected_dataset_manifest_sha256: str | None = None,
) -> tuple[RecoveryCheckpointAudit, dict[str, Any]]:
    root = Path(checkpoint_dir).expanduser().resolve()
    weights_path = root / "model.safetensors"
    provenance_path = root / OURS_CHECKPOINT_PROVENANCE
    if not weights_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(f"incomplete Ours recovery checkpoint: {root}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    identity = (
        provenance.get("experiment_id"),
        provenance.get("variant"),
        provenance.get("method"),
        provenance.get("parent_experiment"),
    )
    if identity != (OURS_ID, OURS_VARIANT, OURS_METHOD, OURS_PARENT):
        raise ValueError("Ours recovery checkpoint identity or parent is invalid")
    weights_sha256 = sha256_file(weights_path)
    if weights_sha256 != provenance.get("weights_sha256"):
        raise ValueError("Ours recovery checkpoint weight hash mismatch")
    if expected_selector_sha256 is not None and (
        provenance.get("selector_weights_sha256") != expected_selector_sha256
    ):
        raise ValueError("Ours recovery checkpoint binds a different B selector")
    if expected_dataset_manifest_sha256 is not None and (
        provenance.get("dataset_manifest_sha256") != expected_dataset_manifest_sha256
    ):
        raise ValueError("Ours recovery checkpoint binds a different temporal corpus")
    parameter_count = int(provenance.get("parameter_count", -1))
    threshold = float(provenance["development_metrics"]["completion_threshold"])
    model = provenance.get("model")
    if (
        not isinstance(model, dict)
        or not 0 < parameter_count <= 15_000_000
        or not 0.0 <= threshold <= 1.0
        or int(provenance.get("selected_step", 0)) < 1
    ):
        raise ValueError("Ours recovery checkpoint contract is invalid")
    return (
        RecoveryCheckpointAudit(
            checkpoint_dir=root,
            weights_sha256=weights_sha256,
            provenance_sha256=sha256_file(provenance_path),
            parameter_count=parameter_count,
            encoder=str(model["encoder"]),
            history_length=int(model["history_length"]),
            completion_threshold=threshold,
            valid=True,
        ),
        provenance,
    )


def _ece(probabilities: Any, labels: Any, np: Any, bins: int = 15) -> float:
    total = len(labels)
    result = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        upper = lower + 1.0 / bins
        selected = (probabilities >= lower) & (
            probabilities <= upper if upper >= 1.0 else probabilities < upper
        )
        if selected.any():
            result += float(selected.sum() / total) * abs(
                float(probabilities[selected].mean()) - float(labels[selected].mean())
            )
    return result


def evaluate(
    model: Any,
    corpus: LoadedCorpus,
    ids: Any,
    *,
    batch_size: int,
    device: str,
    max_false_positive_rate: float,
    np: Any,
    torch: Any,
) -> dict[str, Any]:
    model.eval()
    probabilities: list[Any] = []
    progress: list[Any] = []
    failures: list[Any] = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            selected = ids[start : start + batch_size]
            batch = _batch(corpus, selected, device=device, np=np, torch=torch)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")
            ):
                outputs = model(
                    batch["contexts"],
                    batch["anchors"],
                    batch["actions"],
                    batch["selector"],
                    batch["scalars"],
                )
            probabilities.append(outputs["completion_logit"].sigmoid().float().cpu().numpy())
            progress.append(outputs["progress_logit"].sigmoid().float().cpu().numpy())
            failures.append(outputs["failure_logit"].sigmoid().float().cpu().numpy())
    probability = np.concatenate(probabilities)
    progress_prediction = np.concatenate(progress)
    failure_probability = np.concatenate(failures)
    labels = corpus.target_complete[ids]
    b_stop = corpus.selector_candidates[ids] == 0
    if not b_stop.any() or not labels[b_stop].any() or labels[b_stop].all():
        raise ValueError("development STOP proposals require both completion classes")
    threshold, calibration = calibrate_threshold(
        probability[b_stop],
        labels[b_stop],
        max_false_positive_rate=max_false_positive_rate,
        np=np,
    )
    progress_threshold, progress_calibration = calibrate_threshold(
        progress_prediction[b_stop],
        labels[b_stop],
        max_false_positive_rate=max_false_positive_rate,
        np=np,
    )
    maximum_probability = np.maximum(probability, progress_prediction)
    maximum_threshold, maximum_calibration = calibrate_threshold(
        maximum_probability[b_stop],
        labels[b_stop],
        max_false_positive_rate=max_false_positive_rate,
        np=np,
    )
    gated_stop = b_stop & (probability >= threshold)
    negative = ~labels
    positive = labels
    progress_valid = corpus.target_progress_valid[ids]
    progress_mae = (
        float(
            np.abs(
                progress_prediction[progress_valid] - corpus.target_progress[ids][progress_valid]
            ).mean()
        )
        if progress_valid.any()
        else None
    )
    failure_valid = corpus.target_failure_valid[ids]
    failure_labels = corpus.target_failure[ids][failure_valid]
    failure_threshold = None
    failure_calibration = None
    if len(failure_labels) and failure_labels.any() and not failure_labels.all():
        failure_threshold, failure_calibration = calibrate_threshold(
            failure_probability[failure_valid],
            failure_labels,
            max_false_positive_rate=max_false_positive_rate,
            np=np,
        )
    return {
        "samples": len(ids),
        "completion_threshold": threshold,
        "completion": calibration,
        "progress_gate_threshold": progress_threshold,
        "progress_completion": progress_calibration,
        "maximum_gate_threshold": maximum_threshold,
        "maximum_completion": maximum_calibration,
        "brier": float(np.square(probability - labels.astype(np.float32)).mean()),
        "ece_15": _ece(probability, labels, np),
        "progress_mae": progress_mae,
        "failure_threshold": failure_threshold,
        "failure": failure_calibration,
        "b_stop_proposals": int(b_stop.sum()),
        "b_false_stop_proposals": int((b_stop & negative).sum()),
        "b_true_stop_proposals": int((b_stop & positive).sum()),
        "gated_false_stops": int((gated_stop & negative).sum()),
        "gated_true_stops": int((gated_stop & positive).sum()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    counts = (
        args.steps,
        args.batch_size,
        args.temporal_width,
        args.temporal_layers,
        args.temporal_heads,
        args.feedforward_width,
        args.validate_steps,
        args.gradient_clip_norm,
    )
    if (
        min(counts) <= 0
        or not 0.0 < args.max_false_positive_rate <= 0.05
        or not 0.0 <= args.live_batch_fraction < 1.0
        or not 0.0 <= args.option_batch_fraction <= args.live_batch_fraction
    ):
        raise ValueError("Ours training counts or calibration limit are invalid")
    dataset = args.dataset.expanduser().resolve()
    additional_dataset = (
        args.additional_train_dataset.expanduser().resolve()
        if args.additional_train_dataset is not None
        else None
    )
    destination = args.destination.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite Ours checkpoint: {destination}")
    audit_recovery_corpus(dataset, verify_hashes=True)
    manifest_path = dataset / OURS_CORPUS_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    additional_manifest_path = None
    additional_manifest = None
    if additional_dataset is not None:
        audit_recovery_corpus(
            additional_dataset,
            verify_hashes=True,
            require_development=False,
            require_both_completion_classes=False,
        )
        additional_manifest_path = additional_dataset / OURS_CORPUS_MANIFEST
        additional_manifest = json.loads(additional_manifest_path.read_text(encoding="utf-8"))
        compatible_fields = (
            "context_width",
            "action_horizon",
            "action_dim",
            "selector_weights_sha256",
            "a1_checkpoint_weight_shards_sha256",
        )
        if additional_manifest.get("source_kind") not in {
            "live_rollout",
            "counterfactual_live_rollout",
        } or any(
            additional_manifest.get(field) != manifest.get(field) for field in compatible_fields
        ):
            raise ValueError("additional Ours corpus is not compatible with the primary corpus")

    # CUDA requires this workspace contract for deterministic CuBLAS kernels.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from safetensors.torch import save_file

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    corpus = load_corpus(dataset, args.history_length, np)
    if additional_dataset is not None:
        live_corpus = load_corpus(
            additional_dataset,
            args.history_length,
            np,
            require_development=False,
        )
        corpus = merge_corpora(corpus, live_corpus, np)
    model_config = TemporalRecoveryModelConfig(
        encoder=args.encoder,
        history_length=args.history_length,
        context_width=int(manifest["context_width"]),
        action_horizon=int(manifest["action_horizon"]),
        action_dim=int(manifest["action_dim"]),
        selector_candidates=int(manifest["action_horizon"]) + 1,
        temporal_width=args.temporal_width,
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        feedforward_width=args.feedforward_width,
        dropout=args.dropout,
    )
    model = build_temporal_recovery_model(model_config).to(args.device)
    parameter_count = count_trainable_parameters(model)
    if parameter_count > args.max_parameters:
        raise ValueError(
            f"Ours model has {parameter_count:,} trainable parameters, above {args.max_parameters:,}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = np.random.default_rng(args.seed)
    if args.batch_size < 2:
        raise ValueError("Ours batch size must be at least two")
    log: list[dict[str, Any]] = []
    selected_key: tuple[float, float, float] | None = None
    selected_step = 0
    selected_metrics: dict[str, Any] | None = None
    selected_state: dict[str, Any] | None = None
    started = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        ids = sample_training_ids(
            corpus,
            batch_size=args.batch_size,
            live_batch_fraction=args.live_batch_fraction,
            option_batch_fraction=args.option_batch_fraction,
            generator=generator,
            np=np,
        )
        batch = _batch(corpus, ids, device=args.device, np=np, torch=torch)
        warmup = min(1.0, step / args.warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * step / args.steps))
        learning_rate = args.learning_rate * warmup * cosine
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=args.device.startswith("cuda")
        ):
            outputs = model(
                batch["contexts"],
                batch["anchors"],
                batch["actions"],
                batch["selector"],
                batch["scalars"],
            )
            loss, losses = completion_progress_loss(
                outputs,
                batch["complete"],
                batch["progress"],
                batch["progress_valid"],
                batch["failure"],
                batch["failure_valid"],
                batch["option_values"],
                batch["option_valid"],
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
        optimizer.step()
        if step == 1 or step % args.validate_steps == 0 or step == args.steps:
            dev_metrics = evaluate(
                model,
                corpus,
                corpus.development_ids,
                batch_size=args.batch_size,
                device=args.device,
                max_false_positive_rate=args.max_false_positive_rate,
                np=np,
                torch=torch,
            )
            record = {
                "step": step,
                "learning_rate": learning_rate,
                "train_loss": float(losses["loss"]),
                "train_completion_loss": float(losses["completion_loss"]),
                "train_progress_loss": float(losses["progress_loss"]),
                "train_failure_loss": float(losses["failure_loss"]),
                "train_option_value_loss": float(losses["option_value_loss"]),
                "train_option_rank_loss": float(losses["option_rank_loss"]),
                "gradient_norm": float(gradient_norm),
                "development": dev_metrics,
                "elapsed_seconds": time.perf_counter() - started,
            }
            log.append(record)
            print(json.dumps(record), flush=True)
            candidate_key = (
                float(dev_metrics["completion"]["true_positive_rate"]),
                -float(dev_metrics["brier"]),
                -float(dev_metrics["progress_mae"] or 0.0),
            )
            if selected_key is None or candidate_key > selected_key:
                selected_key = candidate_key
                selected_step = step
                selected_metrics = dev_metrics
                selected_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            model.train()

    if selected_state is None or selected_metrics is None:
        raise RuntimeError("Ours training produced no selectable checkpoint")
    model.load_state_dict(selected_state)
    destination.mkdir(parents=True, exist_ok=True)
    weights_path = destination / "model.safetensors"
    save_file(
        {key: value.detach().cpu() for key, value in model.state_dict().items()}, weights_path
    )
    provenance = {
        "schema_version": 1,
        "experiment_id": OURS_ID,
        "variant": OURS_VARIANT,
        "method": OURS_METHOD,
        "parent_experiment": OURS_PARENT,
        "stage": (
            "R0-counterfactual-option-distillation"
            if additional_manifest is not None
            and bool(additional_manifest.get("contains_counterfactuals"))
            else "R0-live-adapted-temporal-gate"
            if additional_dataset is not None
            else "R0-successful-demonstration-temporal-gate"
        ),
        "model": model_config.payload(),
        "parameter_count": parameter_count,
        "dataset": str(dataset),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "additional_train_dataset": (
            str(additional_dataset) if additional_dataset is not None else None
        ),
        "additional_train_dataset_manifest_sha256": (
            sha256_file(additional_manifest_path) if additional_manifest_path is not None else None
        ),
        "live_training_samples": int(len(corpus.live_ids)),
        "counterfactual_training_states": int(
            (
                corpus.target_option_valid[corpus.live_ids].sum(axis=1) >= 2
            ).sum()
        ),
        "live_batch_fraction": args.live_batch_fraction if additional_dataset else 0.0,
        "option_batch_fraction": (
            args.option_batch_fraction if additional_dataset is not None else 0.0
        ),
        "selector_weights_sha256": manifest["selector_weights_sha256"],
        "a1_checkpoint_weight_shards_sha256": manifest["a1_checkpoint_weight_shards_sha256"],
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_false_positive_rate": args.max_false_positive_rate,
        "selection_rule": "max TPR at FPR<=limit, then min Brier/progress MAE, earliest tie",
        "selected_step": selected_step,
        "development_metrics": selected_metrics,
        "training_log": log,
        "elapsed_seconds": time.perf_counter() - started,
    }
    provenance["weights_sha256"] = sha256_file(weights_path)
    _write_json(destination / OURS_CHECKPOINT_PROVENANCE, provenance)
    return provenance


def main() -> int:
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
