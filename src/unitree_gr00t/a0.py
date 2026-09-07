"""Contracts and adapters for experiment A0: unmodified GR00T N1.7 LIBERO."""

from __future__ import annotations

import io
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

A0_ID = "A0"
A0_VARIANT = "GR00T-N1.7-LIBERO-original"
A0_EMBODIMENT = "libero_sim"

VIDEO_KEYS = ("image", "wrist_image")
STATE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
ACTION_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
LANGUAGE_KEY = "annotation.human.action.task_description"

REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "embodiment_id.json",
    "model.safetensors.index.json",
    "processor_config.json",
    "statistics.json",
)


class A0ContractError(ValueError):
    """Raised when an asset violates the frozen A0 contract."""


@dataclass(frozen=True)
class CheckpointContract:
    checkpoint_dir: Path
    embodiment: str
    embodiment_id: int
    action_horizon: int
    video_keys: tuple[str, ...]
    state_keys: tuple[str, ...]
    action_keys: tuple[str, ...]
    language_key: str
    weight_shards: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkCase:
    task_type: str
    case_name: str
    path: Path


@dataclass(frozen=True)
class TaskDescription:
    instruction: str
    steps: tuple[str, ...]
    start_indices: tuple[int, ...]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise A0ContractError(f"Cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A0ContractError(f"Expected a JSON object in {path}")
    return value


def inspect_checkpoint(checkpoint_dir: str | Path) -> CheckpointContract:
    """Read the checkpoint metadata and freeze the embodiment-specific contract.

    ``config.json`` exposes the model-wide maximum horizon. A0 deliberately
    reads the ``libero_sim`` action delta indices from ``processor_config.json``
    because those indices define the actual checkpoint output horizon.
    """

    root = Path(checkpoint_dir).expanduser().resolve()
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (root / name).is_file()]
    if missing:
        raise A0ContractError(f"Checkpoint is missing required files: {', '.join(missing)}")

    processor = _read_json(root / "processor_config.json")
    try:
        modalities = processor["processor_kwargs"]["modality_configs"][A0_EMBODIMENT]
        video_keys = tuple(str(value) for value in modalities["video"]["modality_keys"])
        state_keys = tuple(str(value) for value in modalities["state"]["modality_keys"])
        action_keys = tuple(str(value) for value in modalities["action"]["modality_keys"])
        language_keys = tuple(str(value) for value in modalities["language"]["modality_keys"])
        action_horizon = len(modalities["action"]["delta_indices"])
    except (KeyError, TypeError) as exc:
        raise A0ContractError("Checkpoint does not contain a valid libero_sim contract") from exc

    expected = (VIDEO_KEYS, STATE_KEYS, ACTION_KEYS, (LANGUAGE_KEY,))
    actual = (video_keys, state_keys, action_keys, language_keys)
    if actual != expected:
        raise A0ContractError(f"Unexpected libero_sim modalities: {actual!r}")
    if action_horizon < 1:
        raise A0ContractError("libero_sim action horizon must be positive")

    embodiments = _read_json(root / "embodiment_id.json")
    if A0_EMBODIMENT not in embodiments:
        raise A0ContractError(f"{A0_EMBODIMENT!r} is absent from embodiment_id.json")

    index = _read_json(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise A0ContractError("model.safetensors.index.json has no weight map")
    shards = tuple(sorted({str(value) for value in weight_map.values()}))
    missing_shards = [name for name in shards if not (root / name).is_file()]
    if missing_shards:
        raise A0ContractError(f"Checkpoint is missing weight shards: {', '.join(missing_shards)}")

    return CheckpointContract(
        checkpoint_dir=root,
        embodiment=A0_EMBODIMENT,
        embodiment_id=int(embodiments[A0_EMBODIMENT]),
        action_horizon=action_horizon,
        video_keys=video_keys,
        state_keys=state_keys,
        action_keys=action_keys,
        language_key=language_keys[0],
        weight_shards=shards,
    )


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def discover_cases(
    benchmark_dir: str | Path,
    task_types: Iterable[str],
    case_names: Iterable[str] | None = None,
) -> list[BenchmarkCase]:
    """Discover exact benchmark case directories without aliasing task types."""

    root = Path(benchmark_dir).expanduser().resolve()
    wanted_cases = set(case_names or ())
    cases: list[BenchmarkCase] = []
    for task_type in task_types:
        source_type = (
            "Ideal" if task_type in {"Observation_Mismatching", "Random_Disturbance"} else task_type
        )
        task_root = root / source_type
        if not task_root.is_dir():
            raise A0ContractError(f"Benchmark task type is missing: {task_root}")
        candidates = sorted(
            (path for path in task_root.iterdir() if path.is_dir()),
            key=lambda path: _natural_key(path.name),
        )
        for path in candidates:
            if wanted_cases and path.name not in wanted_cases:
                continue
            if len(tuple(path.glob("*.bddl"))) != 1:
                continue
            for required in ("goal.json", "task_description.txt"):
                if not (path / required).is_file():
                    raise A0ContractError(f"Benchmark case is missing {required}: {path}")
            cases.append(BenchmarkCase(task_type, path.name, path))
    if not cases:
        raise A0ContractError("No benchmark cases matched the requested selection")
    if wanted_cases:
        found = {case.case_name for case in cases}
        missing = sorted(wanted_cases - found, key=_natural_key)
        if missing:
            raise A0ContractError(f"Requested cases were not found: {', '.join(missing)}")
    return cases


_RANGE_RE = re.compile(r"^\[\s*(\d+)\s*,\s*(\d+)\s*\]$")


def parse_task_description(path: str | Path) -> TaskDescription:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]
    task_lines = [line.split(":", 1)[1].strip() for line in lines if line.startswith("Task:")]
    if len(task_lines) != 1 or not task_lines[0]:
        raise A0ContractError(f"Expected exactly one non-empty Task line in {path}")

    steps: list[str] = []
    starts: list[int] = []
    for index, line in enumerate(lines):
        if not line.startswith("Step:"):
            continue
        if index + 1 >= len(lines):
            raise A0ContractError(f"Step has no frame range in {path}: {line}")
        match = _RANGE_RE.match(lines[index + 1])
        if match is None:
            raise A0ContractError(f"Step has an invalid frame range in {path}: {line}")
        steps.append(line.split(":", 1)[1].strip())
        starts.append(int(match.group(1)))
    if not steps:
        raise A0ContractError(f"No steps found in {path}")
    return TaskDescription(task_lines[0], tuple(steps), tuple(starts))


def load_goal(path: str | Path) -> dict[str, list[list[str]]]:
    raw = _read_json(Path(path))
    goal: dict[str, list[list[str]]] = {}
    for object_name, entries in raw.items():
        if not isinstance(entries, list):
            raise A0ContractError(f"Goal entries for {object_name} must be a list")
        states: list[list[str]] = []
        for entry in entries:
            state = entry.get("state_pair") if isinstance(entry, dict) else entry
            if not isinstance(state, list) or len(state) not in {2, 3}:
                raise A0ContractError(f"Invalid goal state for {object_name}: {state!r}")
            states.append(
                [
                    str(value).lower() if index == 0 else str(value)
                    for index, value in enumerate(state)
                ]
            )
        goal[str(object_name)] = states
    return goal


def load_goal_steps(path: str | Path) -> dict[str, list[int]]:
    raw = _read_json(Path(path))
    steps: dict[str, list[int]] = {}
    for object_name, entries in raw.items():
        if not isinstance(entries, list):
            raise A0ContractError(f"Goal entries for {object_name} must be a list")
        object_steps: list[int] = []
        for index, entry in enumerate(entries):
            value = entry.get("task_step", index) if isinstance(entry, dict) else index
            object_steps.append(int(value))
        steps[str(object_name)] = object_steps
    return steps


def quat_to_axis_angle(quaternion: Any, np: Any) -> Any:
    quat = np.asarray(quaternion, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - float(quat[3] * quat[3])))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * (2.0 * math.acos(float(quat[3]))) / denominator).astype(np.float32)


def build_policy_observation(
    observation: dict[str, Any], instruction: str, np: Any
) -> dict[str, Any]:
    """Convert a raw RoboCerebra/LIBERO observation to the official flat sim API."""

    position = np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
    rotation = quat_to_axis_angle(observation["robot0_eef_quat"], np)
    gripper = np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)

    def scalar(value: float) -> Any:
        return np.asarray(value, dtype=np.float32).reshape(1, 1, 1)

    return {
        "video.image": np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])[None, None],
        "video.wrist_image": np.ascontiguousarray(
            observation["robot0_eye_in_hand_image"][::-1, ::-1]
        )[None, None],
        "state.x": scalar(position[0]),
        "state.y": scalar(position[1]),
        "state.z": scalar(position[2]),
        "state.roll": scalar(rotation[0]),
        "state.pitch": scalar(rotation[1]),
        "state.yaw": scalar(rotation[2]),
        "state.gripper": gripper.reshape(1, 1, -1),
        LANGUAGE_KEY: [instruction],
    }


def unpack_action_chunk(action: dict[str, Any], np: Any) -> Any:
    """Convert the official flat action dictionary to one ``(T, 7)`` chunk."""

    columns = []
    horizon: int | None = None
    for key in ACTION_KEYS:
        flat_key = f"action.{key}"
        if flat_key not in action:
            raise A0ContractError(f"Policy response is missing {flat_key}")
        values = np.asarray(action[flat_key], dtype=np.float32)
        if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != 1:
            raise A0ContractError(f"{flat_key} must have shape (1, T, 1), got {values.shape}")
        horizon = values.shape[1] if horizon is None else horizon
        if values.shape[1] != horizon:
            raise A0ContractError("Policy action modalities have inconsistent horizons")
        columns.append(values[0])
    return np.concatenate(columns, axis=-1)


def to_libero_action(action: Any, np: Any) -> Any:
    """Apply NVIDIA's LIBERO gripper normalization/inversion to one action."""

    result = np.asarray(action, dtype=np.float32).copy()
    if result.shape != (7,):
        raise A0ContractError(f"LIBERO action must have shape (7,), got {result.shape}")
    result[-1] = -np.sign(2.0 * result[-1] - 1.0)
    return result


class RemotePolicyClient:
    """Small client for the official GR00T ZeroMQ server wire protocol."""

    def __init__(self, host: str, port: int, timeout_ms: int = 120_000):
        try:
            import msgpack
            import numpy as np
            import zmq
        except ImportError as exc:
            raise RuntimeError("A0 evaluator requires msgpack, numpy and pyzmq") from exc
        self._msgpack = msgpack
        self._np = np
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(f"tcp://{host}:{port}")

    def _encode(self, value: Any) -> Any:
        if isinstance(value, self._np.ndarray):
            output = io.BytesIO()
            self._np.save(output, value, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        raise TypeError(f"Cannot serialize {type(value)!r}")

    def _decode(self, value: Any) -> Any:
        if isinstance(value, dict) and "__ndarray_class__" in value:
            return self._np.load(io.BytesIO(value["as_npy"]), allow_pickle=False)
        return value

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        self._socket.send(self._msgpack.packb(request, default=self._encode))
        response = self._msgpack.unpackb(self._socket.recv(), object_hook=self._decode)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def ping(self) -> bool:
        return self.call("ping") == {"status": "ok", "message": "Server is running"}

    def reset(self) -> None:
        self.call("reset", {"options": None})

    def get_action(self, observation: dict[str, Any]) -> dict[str, Any]:
        response = self.call("get_action", {"observation": observation, "options": None})
        if (
            not isinstance(response, list)
            or len(response) != 2
            or not isinstance(response[0], dict)
        ):
            raise RuntimeError("GR00T server returned an invalid get_action response")
        return response[0]

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()

    def __enter__(self) -> RemotePolicyClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
