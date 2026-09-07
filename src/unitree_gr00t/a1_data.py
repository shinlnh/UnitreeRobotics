"""Convert raw RoboCerebra demonstrations to the GR00T LeRobot v2 contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .a0 import BenchmarkCase, quat_to_axis_angle
from .a0_eval import _configure_environment_imports, _make_environment
from .a1 import audit_training_source, load_training_records, sha256_file

_AUTOLIMITS_PATCHED = False


@dataclass(frozen=True)
class ConversionJob:
    episode_index: int
    scene: str
    case: str
    instruction: str
    directory: str
    demonstration: str
    bddl: str
    destination: str
    robocerebra_source: str
    benchmark_dir: str
    fps: int


@dataclass(frozen=True)
class ConversionResult:
    episode_index: int
    scene: str
    case: str
    instruction: str
    source_frames: int
    filtered_frames: int
    removed_noops: int
    demonstration_sha256: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robocerebra-source", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--expected-manifest-rows", type=int, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def is_noop(action: Any, previous_action: Any | None, np: Any, threshold: float = 1e-4) -> bool:
    """Match RoboCerebra's published no-op removal rule."""

    if previous_action is None:
        return bool(np.linalg.norm(action[:-1]) < threshold)
    return bool(np.linalg.norm(action[:-1]) < threshold and action[-1] == previous_action[-1])


def model_action(raw_action: Any, np: Any) -> Any:
    """Convert the LIBERO environment gripper sign to GR00T's [0, 1] target."""

    action = np.asarray(raw_action, dtype=np.float32).copy()
    action[-1] = 0.5 * (1.0 - np.sign(action[-1]))
    return action


def mujoco_autolimits_xml(xml: str) -> str:
    """Enable modern MuJoCo inference for ranged joints in legacy LIBERO assets."""

    root = ET.fromstring(xml)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("autolimits", "true")
    return ET.tostring(root, encoding="unicode")


def _install_mujoco_autolimits_compatibility() -> None:
    global _AUTOLIMITS_PATCHED
    if _AUTOLIMITS_PATCHED:
        return
    from robosuite.utils import binding_utils

    original = binding_utils.MjSim.from_xml_string

    @classmethod
    def from_xml_string(cls: Any, xml: str) -> Any:
        del cls
        return original(mujoco_autolimits_xml(xml))

    binding_utils.MjSim.from_xml_string = from_xml_string
    _AUTOLIMITS_PATCHED = True


def _video_process(path: Path, fps: int) -> subprocess.Popen[bytes]:
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.mp4")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        (
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            "256x256",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-movflags",
            "+faststart",
            str(temporary),
        ),
        stdin=subprocess.PIPE,
    )
    process._a1_temporary = temporary  # type: ignore[attr-defined]
    process._a1_destination = path  # type: ignore[attr-defined]
    return process


def _finish_video(process: subprocess.Popen[bytes]) -> None:
    if process.stdin is None:
        raise RuntimeError("ffmpeg stdin was not created")
    process.stdin.close()
    return_code = process.wait()
    temporary = process._a1_temporary  # type: ignore[attr-defined]
    destination = process._a1_destination  # type: ignore[attr-defined]
    if return_code:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {destination}")
    temporary.replace(destination)


def _abort_video(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        process.wait()
    process._a1_temporary.unlink(missing_ok=True)  # type: ignore[attr-defined]


def _output_paths(destination: Path, episode_index: int) -> tuple[Path, Path, Path]:
    chunk = episode_index // 1000
    parquet = destination / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    agent = (
        destination
        / "videos"
        / f"chunk-{chunk:03d}"
        / "observation.images.image"
        / f"episode_{episode_index:06d}.mp4"
    )
    wrist = (
        destination
        / "videos"
        / f"chunk-{chunk:03d}"
        / "observation.images.wrist_image"
        / f"episode_{episode_index:06d}.mp4"
    )
    return parquet, agent, wrist


def _read_existing(job: ConversionJob) -> ConversionResult | None:
    import h5py
    import pandas as pd

    destination = Path(job.destination)
    parquet, agent, wrist = _output_paths(destination, job.episode_index)
    if not (parquet.is_file() and agent.is_file() and wrist.is_file()):
        return None
    frame = pd.read_parquet(parquet)
    with h5py.File(job.demonstration, "r") as handle:
        source_frames = len(handle["data"]["demo_1"]["actions"])
    return ConversionResult(
        episode_index=job.episode_index,
        scene=job.scene,
        case=job.case,
        instruction=job.instruction,
        source_frames=source_frames,
        filtered_frames=len(frame),
        removed_noops=max(0, source_frames - len(frame)),
        demonstration_sha256=sha256_file(Path(job.demonstration)),
    )


def _convert_one(job: ConversionJob) -> ConversionResult:
    import h5py
    import numpy as np
    import pandas as pd

    destination = Path(job.destination)
    parquet, agent_path, wrist_path = _output_paths(destination, job.episode_index)
    for path in (parquet, agent_path, wrist_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    np_module, bddl_utils, runtime = _configure_environment_imports(
        Path(job.robocerebra_source), Path(job.benchmark_dir)
    )
    if np_module is not np:
        raise RuntimeError("A1 converter imported inconsistent NumPy modules")
    _install_mujoco_autolimits_compatibility()
    case = BenchmarkCase("Training", job.case, Path(job.directory))
    env = _make_environment(case, bddl_utils, runtime, job.fps, Path(job.bddl))
    agent_encoder = _video_process(agent_path, job.fps)
    wrist_encoder = _video_process(wrist_path, job.fps)
    states_out: list[Any] = []
    actions_out: list[Any] = []
    source_frames = 0
    previous_action = None
    try:
        env.reset()
        with h5py.File(job.demonstration, "r") as handle:
            demo = handle["data"]["demo_1"]
            states = demo["states"]
            actions = demo["actions"]
            if len(states) != len(actions):
                raise ValueError(f"state/action length mismatch in {job.demonstration}")
            source_frames = len(actions)
            for state, raw_action in zip(states, actions, strict=True):
                if is_noop(raw_action, previous_action, np):
                    previous_action = raw_action
                    continue
                previous_action = raw_action
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                env._post_process()
                env._update_observables(force=True)
                observation = env._get_observations()
                position = np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
                rotation = quat_to_axis_angle(observation["robot0_eef_quat"], np)
                gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)
                states_out.append(np.concatenate((position, rotation, gripper)))
                actions_out.append(model_action(raw_action, np))
                agent_image = np.ascontiguousarray(
                    observation["agentview_image"][::-1, ::-1], dtype=np.uint8
                )
                wrist_image = np.ascontiguousarray(
                    observation["robot0_eye_in_hand_image"][::-1, ::-1], dtype=np.uint8
                )
                if agent_encoder.stdin is None or wrist_encoder.stdin is None:
                    raise RuntimeError("ffmpeg pipe closed while converting A1 data")
                agent_encoder.stdin.write(agent_image.tobytes())
                wrist_encoder.stdin.write(wrist_image.tobytes())
        if len(states_out) <= 16:
            raise ValueError(f"too few non-noop frames in {job.demonstration}: {len(states_out)}")
        _finish_video(agent_encoder)
        _finish_video(wrist_encoder)
        length = len(states_out)
        frame = pd.DataFrame(
            {
                "observation.state": states_out,
                "action": actions_out,
                "timestamp": np.arange(length, dtype=np.float32) / job.fps,
                "frame_index": np.arange(length, dtype=np.int64),
                "episode_index": np.full(length, job.episode_index, dtype=np.int64),
                "index": np.arange(length, dtype=np.int64),
                "task_index": np.full(length, job.episode_index, dtype=np.int64),
            }
        )
        temporary = parquet.with_name(f".{parquet.name}.tmp-{os.getpid()}")
        frame.to_parquet(temporary, index=False)
        temporary.replace(parquet)
    except BaseException:
        _abort_video(agent_encoder)
        _abort_video(wrist_encoder)
        parquet.unlink(missing_ok=True)
        agent_path.unlink(missing_ok=True)
        wrist_path.unlink(missing_ok=True)
        raise
    finally:
        env.close()
    return ConversionResult(
        episode_index=job.episode_index,
        scene=job.scene,
        case=job.case,
        instruction=job.instruction,
        source_frames=source_frames,
        filtered_frames=len(states_out),
        removed_noops=source_frames - len(states_out),
        demonstration_sha256=sha256_file(Path(job.demonstration)),
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _statistics(destination: Path, results: list[ConversionResult]) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    state_values: list[Any] = []
    action_values: list[Any] = []
    timestamp_values: list[Any] = []
    offset = 0
    for result in results:
        parquet, _, _ = _output_paths(destination, result.episode_index)
        frame = pd.read_parquet(parquet)
        length = len(frame)
        frame["index"] = np.arange(offset, offset + length, dtype=np.int64)
        temporary = parquet.with_name(f".{parquet.name}.stats-{os.getpid()}")
        frame.to_parquet(temporary, index=False)
        temporary.replace(parquet)
        offset += length
        state_values.append(np.stack(frame["observation.state"].to_numpy()))
        action_values.append(np.stack(frame["action"].to_numpy()))
        timestamp_values.append(frame["timestamp"].to_numpy(dtype=np.float32)[:, None])

    def block(values: list[Any]) -> dict[str, list[float]]:
        data = np.concatenate(values, axis=0).astype(np.float32, copy=False)
        return {
            "mean": np.mean(data, axis=0).tolist(),
            "std": np.std(data, axis=0).tolist(),
            "min": np.min(data, axis=0).tolist(),
            "max": np.max(data, axis=0).tolist(),
            "q01": np.quantile(data, 0.01, axis=0).tolist(),
            "q99": np.quantile(data, 0.99, axis=0).tolist(),
        }

    return {
        "observation.state": block(state_values),
        "action": block(action_values),
        "timestamp": block(timestamp_values),
    }


def _metadata(
    destination: Path,
    results: list[ConversionResult],
    source_audit: Any,
    source_revision: str,
    fps: int,
) -> None:
    total_frames = sum(result.filtered_frames for result in results)
    features = {
        "observation.images.wrist_image": {
            "dtype": "video",
            "shape": [256, 256, 3],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.height": 256,
                "video.width": 256,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "observation.images.image": {
            "dtype": "video",
            "shape": [256, 256, 3],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.height": 256,
                "video.width": 256,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [8],
            "names": {
                "motors": [
                    "x",
                    "y",
                    "z",
                    "axis_angle1",
                    "axis_angle2",
                    "axis_angle3",
                    "gripper",
                    "gripper",
                ]
            },
        },
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": {
                "motors": [
                    "x",
                    "y",
                    "z",
                    "axis_angle1",
                    "axis_angle2",
                    "axis_angle3",
                    "gripper",
                ]
            },
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    _write_json(
        destination / "meta" / "info.json",
        {
            "codebase_version": "v2.1",
            "robot_type": "franka",
            "total_episodes": len(results),
            "total_frames": total_frames,
            "total_tasks": len(results),
            "total_videos": len(results) * 2,
            "total_chunks": (len(results) + 999) // 1000,
            "chunks_size": 1000,
            "fps": fps,
            "splits": {"train": f"0:{len(results)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            ),
            "features": features,
        },
    )
    _write_json(
        destination / "meta" / "modality.json",
        {
            "state": {
                "x": {"start": 0, "end": 1},
                "y": {"start": 1, "end": 2},
                "z": {"start": 2, "end": 3},
                "roll": {"start": 3, "end": 4},
                "pitch": {"start": 4, "end": 5},
                "yaw": {"start": 5, "end": 6},
                "gripper": {"start": 6, "end": 8},
            },
            "action": {
                "x": {"start": 0, "end": 1},
                "y": {"start": 1, "end": 2},
                "z": {"start": 2, "end": 3},
                "roll": {"start": 3, "end": 4},
                "pitch": {"start": 4, "end": 5},
                "yaw": {"start": 5, "end": 6},
                "gripper": {"start": 6, "end": 7},
            },
            "video": {
                "image": {"original_key": "observation.images.image"},
                "wrist_image": {"original_key": "observation.images.wrist_image"},
            },
            "annotation": {"human.action.task_description": {"original_key": "task_index"}},
        },
    )
    _write_jsonl(
        destination / "meta" / "tasks.jsonl",
        [{"task_index": result.episode_index, "task": result.instruction} for result in results],
    )
    _write_jsonl(
        destination / "meta" / "episodes.jsonl",
        [
            {
                "episode_index": result.episode_index,
                "tasks": [result.instruction],
                "length": result.filtered_frames,
            }
            for result in results
        ],
    )
    _write_json(destination / "meta" / "stats.json", _statistics(destination, results))
    source_payload = asdict(source_audit)
    source_payload["source_root"] = str(source_audit.source_root)
    source_payload["manifest"] = str(source_audit.manifest)
    source_payload["source_revision"] = source_revision
    source_payload["exact_prompt_overlap_count"] = len(source_audit.exact_prompt_overlaps)
    source_payload["no_op_filter"] = (
        "norm(action[:-1]) < 1e-4 and gripper unchanged from previous action"
    )
    source_payload["fps"] = fps
    source_payload["source_frames"] = sum(result.source_frames for result in results)
    source_payload["filtered_frames"] = total_frames
    source_payload["removed_noops"] = sum(result.removed_noops for result in results)
    source_payload["demonstrations"] = [asdict(result) for result in results]
    _write_json(destination / "meta" / "a1_source_audit.json", source_payload)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1 or args.fps < 1:
        raise ValueError("--workers and --fps must be positive")
    expected = args.expected_episodes if args.limit is None else args.limit
    source_audit = audit_training_source(
        args.source,
        args.manifest,
        args.benchmark_dir,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_manifest_rows=args.expected_manifest_rows,
        expected_usable_episodes=args.expected_episodes,
    )
    if not source_audit.valid:
        raise ValueError(f"A1 source audit failed: {asdict(source_audit)}")
    records = load_training_records(args.source, args.manifest)
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(records):
            raise ValueError("--limit must select at least one available episode")
        records = records[: args.limit]
    destination = args.destination.expanduser().resolve()
    jobs = [
        ConversionJob(
            episode_index=index,
            scene=record.scene,
            case=record.case,
            instruction=record.instruction,
            directory=str(record.directory),
            demonstration=str(record.demonstration),
            bddl=str(record.bddl),
            destination=str(destination),
            robocerebra_source=str(args.robocerebra_source.expanduser().resolve()),
            benchmark_dir=str(args.benchmark_dir.expanduser().resolve()),
            fps=args.fps,
        )
        for index, record in enumerate(records)
    ]
    completed: dict[int, ConversionResult] = {}
    if args.resume:
        for job in jobs:
            existing = _read_existing(job)
            if existing is not None:
                completed[job.episode_index] = existing
    pending = [job for job in jobs if job.episode_index not in completed]
    if pending:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
            futures = {executor.submit(_convert_one, job): job for job in pending}
            for future in as_completed(futures):
                result = future.result()
                completed[result.episode_index] = result
                progress = {
                    "complete": len(completed) == expected,
                    "episodes": len(completed),
                    "expected_episodes": expected,
                    "frames": sum(value.filtered_frames for value in completed.values()),
                }
                _write_json(destination / "meta" / "conversion_progress.json", progress)
                print(json.dumps(progress), flush=True)
    results = [completed[index] for index in range(len(jobs))]
    _metadata(destination, results, source_audit, args.source_revision, args.fps)
    summary = {
        "complete": len(results) == expected,
        "episodes": len(results),
        "expected_episodes": expected,
        "source_frames": sum(result.source_frames for result in results),
        "frames": sum(result.filtered_frames for result in results),
        "removed_noops": sum(result.removed_noops for result in results),
        "destination": str(destination),
    }
    _write_json(destination / "meta" / "conversion_progress.json", summary)
    return summary


def main() -> int:
    summary = run(_parser().parse_args())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
