"""Command-line entry point for the Unitree G1 GR00T project."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .commands import (
    build_collect_command,
    build_deploy_command,
    build_open_loop_command,
    build_process_dataset_command,
    build_server_command,
    build_train_command,
    run_or_preview,
)
from .config import load_config
from .dataset import validate_dataset
from .doctor import doctor_ok, run_doctor
from .policy import HeuristicGeneralistPolicy
from .reporting import write_report
from .simulation import run_suite
from .tasks import load_tasks, select_tasks

REAL_ROBOT_ACK = "I_UNDERSTAND_G1_CAN_MOVE"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t-g1",
        description="GR00T N1.7 + GEAR-SONIC generalist robotics project for Unitree G1",
    )
    parser.add_argument("--config", default="configs/project.toml", help="Project TOML path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run the dependency-free generalist smoke demo")
    demo.add_argument("--tasks", default="all", help="all or comma-separated task ids")
    demo.add_argument("--episodes", type=int, default=1, help="Episodes per task")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--execution-horizon", type=int, default=4)
    demo.add_argument("--output", help="Report directory (default: artifacts/mock_demo)")

    doctor = subparsers.add_parser("doctor", help="Check prerequisites")
    doctor.add_argument("--profile", choices=("mock", "sim", "real"), default="mock")
    doctor.add_argument("--online", action="store_true", help="Also check gated model and services")

    dataset_check = subparsers.add_parser("dataset-check", help="Validate a SONIC LeRobot dataset")
    dataset_check.add_argument("path")

    process_dataset = subparsers.add_parser("process-dataset", help="Clean a recorded dataset")
    process_dataset.add_argument("--dataset", required=True)
    process_dataset.add_argument("--output", required=True)
    process_dataset.add_argument("--execute", action="store_true")

    train = subparsers.add_parser("train", help="Preview or run GR00T fine-tuning")
    train.add_argument("--dataset", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--num-gpus", type=int)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--execute", action="store_true")

    serve = subparsers.add_parser("serve", help="Preview or start the GR00T PolicyServer")
    serve.add_argument("--checkpoint", help="Fine-tuned UNITREE_G1_SONIC checkpoint")
    serve.add_argument("--execute", action="store_true")

    open_loop = subparsers.add_parser(
        "open-loop", help="Evaluate trajectories through PolicyServer"
    )
    open_loop.add_argument("--dataset", required=True)
    open_loop.add_argument("--traj-ids", default="0", help="Comma-separated trajectory ids")
    open_loop.add_argument("--execution-horizon", type=int, default=8)
    open_loop.add_argument("--execute", action="store_true")

    deploy = subparsers.add_parser("deploy", help="Preview or launch SONIC inference")
    deploy.add_argument("--mode", choices=("sim", "real"), default="sim")
    deploy.add_argument("--prompt", required=True)
    deploy.add_argument("--no-record", action="store_true")
    deploy.add_argument("--execute", action="store_true")
    deploy.add_argument("--acknowledge-real-robot-risk", default="")

    collect = subparsers.add_parser("collect", help="Preview or launch demonstration collection")
    collect.add_argument("--mode", choices=("sim", "real"), default="sim")
    collect.add_argument("--prompt", required=True)
    collect.add_argument("--dataset-name")
    collect.add_argument("--execute", action="store_true")
    collect.add_argument("--acknowledge-real-robot-risk", default="")

    subparsers.add_parser("show-config", help="Print resolved runtime configuration")
    return parser


def _require_real_ack(args: argparse.Namespace) -> None:
    if args.mode == "real" and args.execute and args.acknowledge_real_robot_risk != REAL_ROBOT_ACK:
        raise ValueError(
            "Real-robot execution is locked. First pass simulation, establish a 3 m safety zone, "
            "put an operator on the hardware E-stop and keyboard `O`, then pass "
            f"--acknowledge-real-robot-risk {REAL_ROBOT_ACK}"
        )


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)

    if args.command == "demo":
        task_catalog = load_tasks(config.root / "configs" / "tasks.toml")
        tasks = select_tasks(task_catalog, args.tasks)
        policy = HeuristicGeneralistPolicy(
            horizon=config.sonic.action_horizon,
            execution_horizon=args.execution_horizon,
        )
        results = run_suite(
            tasks=tasks,
            episodes=args.episodes,
            seed=args.seed,
            policy=policy,
            safety_config=config.safety,
            execution_horizon=args.execution_horizon,
        )
        output = (
            Path(args.output).expanduser() if args.output else config.artifact_dir / "mock_demo"
        )
        summary = write_report(results, output, backend="mock-sonic-shaped")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if summary["successes"] == summary["episodes"] else 2

    if args.command == "doctor":
        checks = run_doctor(config, args.profile, args.online)
        for check in checks:
            symbol = "OK" if check.ok else ("WARN" if not check.required else "FAIL")
            print(f"[{symbol:4}] {check.name}: {check.detail}")
        return 0 if doctor_ok(checks) else 2

    if args.command == "dataset-check":
        report = validate_dataset(args.path)
        print(f"Dataset: {report.path}")
        print(
            f"episodes={report.episodes} tasks={report.tasks} "
            f"parquet={report.parquet_files} videos={report.video_files}"
        )
        for issue in report.issues:
            print(f"[{issue.level.upper()}] {issue.message}")
        print("VALID" if report.valid else "INVALID")
        return 0 if report.valid else 2

    if args.command == "process-dataset":
        spec = build_process_dataset_command(config, args.dataset, args.output)
        return run_or_preview(spec, args.execute)

    if args.command == "train":
        if args.execute:
            report = validate_dataset(args.dataset)
            if not report.valid:
                raise ValueError(
                    "Dataset validation failed; run `gr00t-g1 dataset-check PATH` for details"
                )
        spec = build_train_command(config, args.dataset, args.output, args.num_gpus, args.max_steps)
        return run_or_preview(spec, args.execute)

    if args.command == "serve":
        return run_or_preview(build_server_command(config, args.checkpoint), args.execute)

    if args.command == "open-loop":
        try:
            trajectory_ids = [
                int(value.strip()) for value in args.traj_ids.split(",") if value.strip()
            ]
        except ValueError as exc:
            raise ValueError("--traj-ids must contain comma-separated integers") from exc
        spec = build_open_loop_command(config, args.dataset, trajectory_ids, args.execution_horizon)
        return run_or_preview(spec, args.execute)

    if args.command == "deploy":
        _require_real_ack(args)
        spec = build_deploy_command(config, args.mode, args.prompt, record=not args.no_record)
        return run_or_preview(spec, args.execute)

    if args.command == "collect":
        _require_real_ack(args)
        spec = build_collect_command(config, args.mode, args.prompt, args.dataset_name)
        return run_or_preview(spec, args.execute)

    if args.command == "show-config":
        resolved = {
            "project": config.name,
            "root": str(config.root),
            "isaac_gr00t": str(config.isaac_gr00t_dir),
            "sonic": str(config.sonic_dir),
            "base_model": config.model.base_model,
            "checkpoint": config.model.checkpoint or None,
            "embodiment": config.model.embodiment,
            "policy_server": f"{config.model.host}:{config.model.port}",
            "sonic_variant": config.sonic.variant,
        }
        print(json.dumps(resolved, indent=2))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
