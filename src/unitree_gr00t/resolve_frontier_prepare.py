"""Index physically verified expert goal frontiers for RESOLVE collection.

This is a source audit, not a success label derived from language or trajectory
distance.  Every retained frontier is a false-to-true transition of an actual
BDDL predicate under a recorded MuJoCo demonstration state.  Later collectors
may branch B and recovery policies before these frontiers without touching the
held-out RoboCerebra benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .a1 import load_training_records, sha256_file
from .a1_data import _install_mujoco_autolimits_compatibility, mujoco_autolimits_xml
from .b_data import parse_annotated_subgoals
from .ours import OursContractError


@dataclass(frozen=True)
class GoalFrontier:
    source_manifest_index: int
    scene: str
    case: str
    split: str
    subgoal_index: int
    subgoal_instruction: str
    predicate: tuple[str, ...]
    subgoal_start_frame: int
    subgoal_end_frame: int
    first_true_frame: int
    anchor_frames: tuple[int, ...]
    demonstration: str
    demonstration_sha256: str
    bddl: str
    bddl_sha256: str
    source_model_xml_sha256: str
    runtime_model_xml_sha256: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selector-index", type=Path, required=True)
    parser.add_argument("--robocerebra-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor-offsets", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--scan-stride", type=int, default=8)
    parser.add_argument("--control-frequency-hz", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-episodes", type=int)
    return parser


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def controllable_goal_states(
    parsed_problem: dict[str, Any], object_names: tuple[str, ...] | list[str]
) -> tuple[tuple[str, ...], ...]:
    """Exclude fixture invariants and source-placement predicates from BDDL goals."""

    movable = set(object_names)
    fixtures = {
        str(name) for names in parsed_problem.get("fixtures", {}).values() for name in names
    }
    goals: list[tuple[str, ...]] = []
    for raw in parsed_problem.get("goal_state", []):
        goal = tuple(str(item) for item in raw)
        if (
            len(goal) not in {2, 3}
            or goal[1] not in movable
            or goal[1] in fixtures
            or any("init_region" in item for item in goal)
        ):
            continue
        goals.append(goal)
    if len(set(goals)) != len(goals):
        raise OursContractError("RESOLVE BDDL contains duplicate controllable goals")
    return tuple(goals)


_OBJECT_STOPWORDS = {
    "akita",
    "chefmate",
    "new",
    "one",
    "object",
    "the",
}


def goal_matches_instruction(goal: tuple[str, ...], instruction: str) -> bool:
    """Use language only to associate a physical predicate with its annotated step."""

    if len(goal) < 2:
        return False
    subject_tokens = {
        token
        for token in re.findall(r"[a-z]+", goal[1].casefold())
        if len(token) >= 3 and token not in _OBJECT_STOPWORDS
    }
    instruction_tokens = set(re.findall(r"[a-z]+", instruction.casefold()))
    return bool(subject_tokens & instruction_tokens)


def first_false_to_true(
    *,
    start: int,
    end: int,
    stride: int,
    evaluate: Callable[[int], bool],
) -> int | None:
    """Find the first exact false-to-true frame using a coarse-to-fine scan."""

    if start < 0 or end <= start or stride < 1:
        raise OursContractError("RESOLVE frontier scan interval is invalid")
    if evaluate(start):
        return None
    sampled = list(range(start + stride, end, stride))
    if end - 1 not in sampled:
        sampled.append(end - 1)
    previous = start
    for frame in sampled:
        if evaluate(frame):
            for exact in range(previous + 1, frame + 1):
                if evaluate(exact):
                    return exact
            raise AssertionError("coarse RESOLVE frontier hit vanished during exact scan")
        previous = frame
    return None


def valid_anchor_frames(
    *, frontier: int, subgoal_start: int, offsets: tuple[int, ...]
) -> tuple[int, ...]:
    if frontier <= subgoal_start or not offsets or any(offset < 1 for offset in offsets):
        raise OursContractError("RESOLVE frontier anchor request is invalid")
    return tuple(
        sorted({frontier - offset for offset in offsets if frontier - offset >= subgoal_start})
    )


def portable_model_xml(xml: str, *, libero_package_root: Path, robosuite_package_root: Path) -> str:
    """Relocate absolute collection-machine assets in a demonstration XML."""

    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise OursContractError(f"RESOLVE demonstration model XML is invalid: {exc}") from exc
    asset = root.find("asset")
    if asset is None:
        raise OursContractError("RESOLVE demonstration model XML has no asset section")
    for element in (*asset.findall("mesh"), *asset.findall("texture")):
        old = element.get("file")
        if old is None:
            continue
        normalized = old.replace("\\", "/")
        lowered = normalized.casefold()
        if "/robosuite/" in lowered:
            marker = lowered.rfind("/robosuite/")
            suffix = normalized[marker + len("/robosuite/") :]
            element.set("file", str(robosuite_package_root / suffix))
        elif "/libero/libero/" in lowered:
            marker = lowered.rfind("/libero/libero/")
            suffix = normalized[marker + len("/libero/libero/") :]
            element.set("file", str(libero_package_root / suffix))
    for joint in root.iter("joint"):
        actuator_limit = joint.get("actuatorfrclimited")
        if actuator_limit is not None:
            if actuator_limit.casefold() != "false":
                raise OursContractError("RESOLVE cannot drop an active legacy actuator-force limit")
            del joint.attrib["actuatorfrclimited"]
        if joint.get("range") is not None and joint.get("limited") is None:
            # This is the explicit legacy equivalent of MuJoCo autolimits and
            # survives robosuite's XML reconstruction path.
            joint.set("limited", "true")
    return ET.tostring(root, encoding="unicode")


def _split_map(selector_index: Path) -> dict[int, str]:
    rows = _read_jsonl(selector_index / "episodes.jsonl")
    mapping = {int(row["source_manifest_index"]): str(row["split"]) for row in rows}
    if not mapping or set(mapping.values()) != {"train", "development"}:
        raise OursContractError("RESOLVE selector split map is invalid")
    return mapping


def _configure_environment(robocerebra_source: Path) -> tuple[Any, Any, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import sys

    libero_root = robocerebra_source.expanduser().resolve() / "LIBERO"
    sys.path.insert(0, str(libero_root))
    config_dir = Path(os.environ.get("RESOLVE_LIBERO_CONFIG", ".resolve-libero-config"))
    config_dir = config_dir.expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(libero_root / "libero" / "libero"),
        "bddl_files": str(libero_root / "libero" / "libero" / "bddl_files"),
        "init_states": str(libero_root / "libero" / "libero" / "init_files"),
        "datasets": str(libero_root / "datasets"),
        "assets": str(libero_root / "libero" / "libero" / "assets"),
    }
    _write_json(config_dir / "config.yaml", config)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    import libero.libero.envs.bddl_utils as bddl_utils
    import numpy as np
    import robosuite
    from libero.libero.envs import TASK_MAPPING
    from robosuite import load_controller_config

    _install_mujoco_autolimits_compatibility()
    return (
        np,
        bddl_utils,
        (
            TASK_MAPPING,
            load_controller_config,
            libero_root / "libero" / "libero",
            Path(robosuite.__file__).resolve().parent,
        ),
    )


def _make_environment(
    bddl: Path,
    *,
    bddl_utils: Any,
    runtime: Any,
    control_frequency_hz: int,
    camera_observations: bool = False,
) -> Any:
    task_mapping, load_controller_config, *_ = runtime
    problem_name = bddl_utils.get_problem_info(str(bddl))["problem_name"]
    options: dict[str, Any] = {
        "bddl_file_name": str(bddl),
        "robots": ["Panda"],
        "controller_configs": load_controller_config(default_controller="OSC_POSE"),
        "has_renderer": False,
        "has_offscreen_renderer": camera_observations,
        "ignore_done": True,
        "use_camera_obs": camera_observations,
        "reward_shaping": True,
        "control_freq": control_frequency_hz,
    }
    if camera_observations:
        options |= {
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "camera_heights": 256,
            "camera_widths": 256,
        }
    return task_mapping[problem_name](
        **options,
    )


_WORKER_ENVIRONMENT: tuple[Any, Any, Any] | None = None


def _initialize_frontier_worker(robocerebra_source: str) -> None:
    global _WORKER_ENVIRONMENT
    _WORKER_ENVIRONMENT = _configure_environment(Path(robocerebra_source))


def _scan_record(
    item: tuple[Any, str, int, int, tuple[int, ...]],
) -> tuple[list[GoalFrontier], dict[str, Any] | None]:
    """Scan one demonstration inside an isolated MuJoCo worker."""

    record, split, scan_stride, control_frequency_hz, anchor_offsets = item
    if _WORKER_ENVIRONMENT is None:
        raise RuntimeError("RESOLVE frontier worker was not initialized")
    _, bddl_utils, runtime = _WORKER_ENVIRONMENT
    import h5py

    frontiers: list[GoalFrontier] = []
    env = None
    try:
        subgoals = parse_annotated_subgoals(record.task_description)
        env = _make_environment(
            record.bddl,
            bddl_utils=bddl_utils,
            runtime=runtime,
            control_frequency_hz=control_frequency_hz,
        )
        env.reset()
        goals = controllable_goal_states(env.parsed_problem, env.object_names)
        demonstration_sha256 = sha256_file(record.demonstration)
        bddl_sha256 = sha256_file(record.bddl)
        with h5py.File(record.demonstration, "r") as handle:
            states = handle["data"]["demo_1"]["states"]
            source_model_xml = str(handle["data"]["demo_1"].attrs["model_file"])
            runtime_model_xml = mujoco_autolimits_xml(
                portable_model_xml(
                    source_model_xml,
                    libero_package_root=runtime[2],
                    robosuite_package_root=runtime[3],
                )
            )
            env.reset_from_xml_string(runtime_model_xml)
            env.sim.reset()
            source_model_xml_sha256 = hashlib.sha256(source_model_xml.encode("utf-8")).hexdigest()
            runtime_model_xml_sha256 = hashlib.sha256(runtime_model_xml.encode("utf-8")).hexdigest()
            for subgoal_index, (instruction, raw_start, raw_end) in enumerate(subgoals):
                start = min(max(0, raw_start), len(states) - 1)
                end = min(max(start + 1, raw_end), len(states))
                for goal in goals:
                    if not goal_matches_instruction(goal, instruction):
                        continue

                    def evaluate(
                        frame: int,
                        *,
                        _goal: tuple[str, ...] = goal,
                        _env: Any = env,
                        _states: Any = states,
                    ) -> bool:
                        _env.sim.set_state_from_flattened(_states[frame])
                        _env.sim.forward()
                        return bool(_env._eval_predicate(list(_goal)))

                    frontier = first_false_to_true(
                        start=start,
                        end=end,
                        stride=scan_stride,
                        evaluate=evaluate,
                    )
                    if frontier is None:
                        continue
                    anchors = valid_anchor_frames(
                        frontier=frontier,
                        subgoal_start=start,
                        offsets=anchor_offsets,
                    )
                    if not anchors:
                        continue
                    frontiers.append(
                        GoalFrontier(
                            source_manifest_index=record.episode_index,
                            scene=record.scene,
                            case=record.case,
                            split=split,
                            subgoal_index=subgoal_index,
                            subgoal_instruction=instruction,
                            predicate=goal,
                            subgoal_start_frame=start,
                            subgoal_end_frame=end,
                            first_true_frame=frontier,
                            anchor_frames=anchors,
                            demonstration=str(record.demonstration),
                            demonstration_sha256=demonstration_sha256,
                            bddl=str(record.bddl),
                            bddl_sha256=bddl_sha256,
                            source_model_xml_sha256=source_model_xml_sha256,
                            runtime_model_xml_sha256=runtime_model_xml_sha256,
                        )
                    )
        return frontiers, None
    except Exception as exc:
        return [], {
            "source_manifest_index": record.episode_index,
            "scene": record.scene,
            "case": record.case,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if env is not None:
            env.close()


def build_frontier_index(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.scan_stride < 1
        or args.control_frequency_hz < 1
        or args.workers < 1
        or not args.anchor_offsets
        or any(offset < 1 for offset in args.anchor_offsets)
    ):
        raise ValueError("RESOLVE frontier arguments are invalid")
    destination = args.output.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite RESOLVE frontier index: {destination}")
    records = load_training_records(args.source, args.manifest)
    if args.limit_episodes is not None:
        if args.limit_episodes < 1:
            raise ValueError("--limit-episodes must be positive")
        records = records[: args.limit_episodes]
    splits = _split_map(args.selector_index.expanduser().resolve())
    frontiers: list[GoalFrontier] = []
    exclusions: list[dict[str, Any]] = [
        {"source_manifest_index": record.episode_index, "reason": "selector exclusion"}
        for record in records
        if record.episode_index not in splits
    ]
    work = [
        (
            record,
            splits[record.episode_index],
            args.scan_stride,
            args.control_frequency_hz,
            tuple(args.anchor_offsets),
        )
        for record in records
        if record.episode_index in splits
    ]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_initialize_frontier_worker,
        initargs=(str(args.robocerebra_source.expanduser().resolve()),),
    ) as executor:
        for found, exclusion in executor.map(_scan_record, work, chunksize=1):
            frontiers.extend(found)
            if exclusion is not None:
                exclusions.append(exclusion)

    rows = [asdict(item) for item in frontiers]
    index_path = destination / "frontiers.jsonl"
    _write_jsonl(index_path, rows)
    split_counts = {
        split: sum(item.split == split for item in frontiers) for split in ("train", "development")
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": "Ours",
        "stage": "RESOLVE expert physical-frontier source audit",
        "novelty_claim": False,
        "source_manifest": str(args.manifest.expanduser().resolve()),
        "source_manifest_sha256": sha256_file(args.manifest.expanduser().resolve()),
        "selector_index": str(args.selector_index.expanduser().resolve()),
        "selector_manifest_sha256": sha256_file(
            args.selector_index.expanduser().resolve() / "manifest.json"
        ),
        "episodes_scanned": len(records),
        "episodes_with_frontiers": len({item.source_manifest_index for item in frontiers}),
        "frontiers": len(frontiers),
        "anchors": sum(len(item.anchor_frames) for item in frontiers),
        "split_counts": split_counts,
        "scan_stride": args.scan_stride,
        "workers": args.workers,
        "anchor_offsets": args.anchor_offsets,
        "physical_label": "false-to-true BDDL predicate under recorded simulator state",
        "language_used_only_for_step_association": True,
        "exclusions": exclusions,
        "frontiers_sha256": sha256_file(index_path),
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def main() -> int:
    args = _parser().parse_args()
    result = build_frontier_index(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
