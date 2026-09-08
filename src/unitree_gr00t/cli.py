"""Command-line entry point for the Unitree G1 GR00T project."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .a0 import discover_cases, inspect_checkpoint
from .a1 import (
    audit_training_source,
    inspect_a1_checkpoint,
    sha256_file,
    validate_converted_dataset,
)
from .a2 import audit_fixed_hierarchy, hierarchy_audit_payload
from .b import inspect_selector_checkpoint, verify_a1_weight_hashes
from .b_features import FEATURE_RUN_CONTRACT
from .b_retry import retry_contract_payload
from .commands import (
    build_a0_eval_command,
    build_a0_server_command,
    build_a1_eval_command,
    build_a1_prepare_command,
    build_a1_server_command,
    build_a1_train_command,
    build_a2_eval_command,
    build_a2_server_command,
    build_b_eval_command,
    build_b_features_command,
    build_b_prepare_command,
    build_b_retry_eval_command,
    build_b_retry_server_command,
    build_b_server_command,
    build_b_train_command,
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
from .ours import validate_recovery_config
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

    a0_check = subparsers.add_parser(
        "a0-check", help="Validate frozen A0 checkpoint and benchmark assets"
    )
    a0_check.add_argument("--task-types", nargs="+", default=["Ideal"])
    a0_check.add_argument("--cases", nargs="*", default=[])

    a0_server = subparsers.add_parser(
        "a0-server", help="Preview or start the original GR00T N1.7 LIBERO server"
    )
    a0_server.add_argument("--seed", type=int, default=7)
    a0_server.add_argument("--execute", action="store_true")

    a0_eval = subparsers.add_parser(
        "a0-eval", help="Preview or run the no-hierarchy A0 RoboCerebra evaluation"
    )
    a0_eval.add_argument("--task-types", nargs="+", default=["Ideal"])
    a0_eval.add_argument("--cases", nargs="*", default=[])
    a0_eval.add_argument("--trials", type=int, default=1)
    a0_eval.add_argument("--execution-horizon", type=int, default=16, choices=(8, 16))
    a0_eval.add_argument("--seed", type=int, default=7)
    a0_eval.add_argument("--output")
    a0_eval.add_argument("--no-trace-images", action="store_true")
    a0_eval.add_argument("--resume", action="store_true")
    a0_eval.add_argument("--execute", action="store_true")

    a1_check = subparsers.add_parser(
        "a1-check", help="Validate A1 source data, converted data, or frozen checkpoint"
    )
    a1_check.add_argument("--stage", choices=("source", "dataset", "checkpoint"), default="source")

    a1_prepare = subparsers.add_parser(
        "a1-prepare", help="Preview or convert RoboCerebra demonstrations to GR00T LeRobot v2"
    )
    a1_prepare.add_argument("--workers", type=int)
    a1_prepare.add_argument("--limit", type=int)
    a1_prepare.add_argument("--output")
    a1_prepare.add_argument("--no-resume", action="store_true")
    a1_prepare.add_argument("--execute", action="store_true")

    a1_train = subparsers.add_parser(
        "a1-train", help="Preview or post-train the shared A1 GR00T-RC checkpoint"
    )
    a1_train.add_argument("--max-steps", type=int)
    a1_train.add_argument("--dataset")
    a1_train.add_argument("--output")
    a1_train.add_argument("--artifact")
    a1_train.add_argument("--execute", action="store_true")

    a1_server = subparsers.add_parser(
        "a1-server", help="Preview or start the post-trained A1 GR00T-RC server"
    )
    a1_server.add_argument("--seed", type=int, default=7)
    a1_server.add_argument("--execute", action="store_true")

    a1_eval = subparsers.add_parser(
        "a1-eval", help="Preview or run the no-hierarchy A1 RoboCerebra evaluation"
    )
    a1_eval.add_argument("--task-types", nargs="+", default=["Ideal"])
    a1_eval.add_argument("--cases", nargs="*", default=[])
    a1_eval.add_argument("--trials", type=int, default=1)
    a1_eval.add_argument("--execution-horizon", type=int, default=16, choices=(8, 16))
    a1_eval.add_argument("--seed", type=int, default=7)
    a1_eval.add_argument("--output")
    a1_eval.add_argument("--no-trace-images", action="store_true")
    a1_eval.add_argument("--resume", action="store_true")
    a1_eval.add_argument("--execute", action="store_true")

    a2_check = subparsers.add_parser(
        "a2-check", help="Validate the frozen A1 checkpoint and A2 hierarchy plans"
    )
    a2_check.add_argument("--task-types", nargs="+", default=["Ideal"])
    a2_check.add_argument("--cases", nargs="*", default=[])

    a2_server = subparsers.add_parser(
        "a2-server", help="Preview or start the frozen A2 low-level policy server"
    )
    a2_server.add_argument("--seed", type=int, default=7)
    a2_server.add_argument("--execute", action="store_true")

    a2_eval = subparsers.add_parser(
        "a2-eval", help="Preview or run the fixed-anchor hierarchical A2 evaluation"
    )
    a2_eval.add_argument("--task-types", nargs="+", default=["Ideal"])
    a2_eval.add_argument("--cases", nargs="*", default=[])
    a2_eval.add_argument("--trials", type=int, default=1)
    a2_eval.add_argument("--execution-horizon", type=int, default=16, choices=(8, 16))
    a2_eval.add_argument("--seed", type=int, default=7)
    a2_eval.add_argument("--output")
    a2_eval.add_argument("--no-trace-images", action="store_true")
    a2_eval.add_argument("--resume", action="store_true")
    a2_eval.add_argument("--execute", action="store_true")

    b_check = subparsers.add_parser(
        "b-check", help="Validate B selector index, features, or frozen checkpoint"
    )
    b_check.add_argument("--stage", choices=("index", "features", "checkpoint"), default="index")

    b_prepare = subparsers.add_parser(
        "b-prepare", help="Preview or build B's audited selector boundary index"
    )
    b_prepare.add_argument("--limit", type=int)
    b_prepare.add_argument("--output")
    b_prepare.add_argument("--execute", action="store_true")

    b_features = subparsers.add_parser(
        "b-features", help="Preview or extract frozen A1 contexts and chunks for B"
    )
    b_features.add_argument("--batch-size", type=int, default=64)
    b_features.add_argument("--limit-episodes", type=int)
    b_features.add_argument("--no-resume", action="store_true")
    b_features.add_argument("--execute", action="store_true")

    b_train = subparsers.add_parser("b-train", help="Preview or train B's unified selector")
    b_train.add_argument("--steps", type=int)
    b_train.add_argument("--batch-size", type=int)
    b_train.add_argument("--no-resume", action="store_true")
    b_train.add_argument("--execute", action="store_true")

    b_server = subparsers.add_parser(
        "b-server", help="Preview or serve frozen GR00T-RC with the B selector"
    )
    b_server.add_argument("--seed", type=int, default=7)
    b_server.add_argument("--execute", action="store_true")

    b_eval = subparsers.add_parser(
        "b-eval", help="Preview or run adaptive STOP/action-prefix B evaluation"
    )
    b_eval.add_argument("--task-types", nargs="+", default=["Ideal"])
    b_eval.add_argument("--cases", nargs="*", default=[])
    b_eval.add_argument("--trials", type=int, default=1)
    b_eval.add_argument("--execution-horizon", type=int, default=16, choices=(8, 16))
    b_eval.add_argument("--seed", type=int, default=7)
    b_eval.add_argument("--output")
    b_eval.add_argument("--no-trace-images", action="store_true")
    b_eval.add_argument("--resume", action="store_true")
    b_eval.add_argument("--execute", action="store_true")

    subparsers.add_parser("b-retry-check", help="Validate frozen B assets and naive retry contract")

    b_retry_server = subparsers.add_parser(
        "b-retry-server", help="Preview or serve the unchanged frozen B assets"
    )
    b_retry_server.add_argument("--seed", type=int, default=7)
    b_retry_server.add_argument("--execute", action="store_true")

    b_retry_eval = subparsers.add_parser(
        "b-retry-eval", help="Preview or run the outcome-blind naive retry control"
    )
    b_retry_eval.add_argument("--task-types", nargs="+", default=["Ideal"])
    b_retry_eval.add_argument("--cases", nargs="*", default=[])
    b_retry_eval.add_argument("--trials", type=int, default=1)
    b_retry_eval.add_argument("--execution-horizon", type=int, default=16, choices=(8, 16))
    b_retry_eval.add_argument("--seed", type=int, default=7)
    b_retry_eval.add_argument("--output")
    b_retry_eval.add_argument("--no-trace-images", action="store_true")
    b_retry_eval.add_argument("--resume", action="store_true")
    b_retry_eval.add_argument("--execute", action="store_true")

    subparsers.add_parser(
        "ours-check",
        help="Validate the frozen Ours recovery protocol and parent B assets",
    )

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

    if args.command == "a0-check":
        contract = inspect_checkpoint(config.robocerebra.checkpoint_dir)
        cases = discover_cases(config.robocerebra.benchmark_dir, args.task_types, args.cases)
        if contract.action_horizon != config.robocerebra.action_horizon:
            raise ValueError(
                "Configured A0 action horizon does not match checkpoint contract: "
                f"H{config.robocerebra.action_horizon} != H{contract.action_horizon}"
            )
        payload = asdict(contract)
        payload["checkpoint_dir"] = str(contract.checkpoint_dir)
        payload["benchmark_cases"] = [f"{case.task_type}/{case.case_name}" for case in cases]
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "a0-server":
        inspect_checkpoint(config.robocerebra.checkpoint_dir)
        return run_or_preview(build_a0_server_command(config, args.seed), args.execute)

    if args.command == "a0-eval":
        output = (
            Path(args.output).expanduser()
            if args.output
            else config.artifact_dir / "A0" / f"H{args.execution_horizon}-seed{args.seed}"
        )
        spec = build_a0_eval_command(
            config,
            task_types=args.task_types,
            case_names=args.cases,
            trials=args.trials,
            execution_horizon=args.execution_horizon,
            seed=args.seed,
            output_dir=output,
            trace_images=not args.no_trace_images,
            resume=args.resume,
        )
        return run_or_preview(spec, args.execute)

    if args.command == "a1-check":
        a1 = config.robocerebra_posttrain
        if args.stage == "source":
            audit = audit_training_source(
                a1.raw_training_dir,
                a1.training_manifest,
                config.robocerebra.benchmark_dir,
                expected_manifest_sha256=a1.training_manifest_sha256,
                expected_manifest_rows=a1.expected_training_manifest_rows,
                expected_usable_episodes=a1.expected_training_episodes,
            )
            payload = asdict(audit) | {
                "source_root": str(audit.source_root),
                "manifest": str(audit.manifest),
                "valid": audit.valid,
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0 if audit.valid else 2
        if args.stage == "dataset":
            audit = validate_converted_dataset(
                a1.lerobot_training_dir,
                expected_episodes=a1.expected_training_episodes,
                expected_revision=a1.training_dataset_revision,
            )
            payload = asdict(audit) | {"root": str(audit.root)}
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0 if audit.valid else 2
        contract, provenance = inspect_a1_checkpoint(
            a1.checkpoint_dir,
            expected_training_revision=a1.training_dataset_revision,
        )
        payload = asdict(contract) | {
            "checkpoint_dir": str(contract.checkpoint_dir),
            "training_provenance": provenance,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if args.command == "a1-prepare":
        spec = build_a1_prepare_command(
            config,
            workers=args.workers,
            limit=args.limit,
            resume=not args.no_resume,
            output_dir=args.output,
        )
        return run_or_preview(spec, args.execute)

    if args.command == "a1-train":
        spec = build_a1_train_command(
            config,
            max_steps=args.max_steps,
            dataset_dir=args.dataset,
            checkpoint_dir=args.output,
            artifact_dir=args.artifact,
        )
        return run_or_preview(spec, args.execute)

    if args.command == "a1-server":
        inspect_a1_checkpoint(
            config.robocerebra_posttrain.checkpoint_dir,
            expected_training_revision=config.robocerebra_posttrain.training_dataset_revision,
        )
        return run_or_preview(build_a1_server_command(config, args.seed), args.execute)

    if args.command == "a1-eval":
        inspect_a1_checkpoint(
            config.robocerebra_posttrain.checkpoint_dir,
            expected_training_revision=config.robocerebra_posttrain.training_dataset_revision,
        )
        output = (
            Path(args.output).expanduser()
            if args.output
            else config.artifact_dir / "A1" / f"H{args.execution_horizon}-seed{args.seed}"
        )
        spec = build_a1_eval_command(
            config,
            task_types=args.task_types,
            case_names=args.cases,
            trials=args.trials,
            execution_horizon=args.execution_horizon,
            seed=args.seed,
            output_dir=output,
            trace_images=not args.no_trace_images,
            resume=args.resume,
        )
        return run_or_preview(spec, args.execute)

    if args.command == "a2-check":
        a1 = config.robocerebra_posttrain
        a2 = config.robocerebra_hierarchy
        contract, checkpoint_provenance = inspect_a1_checkpoint(
            a1.checkpoint_dir,
            expected_training_revision=a1.training_dataset_revision,
        )
        cases = discover_cases(config.robocerebra.benchmark_dir, args.task_types, args.cases)
        audit = audit_fixed_hierarchy(cases, a2.subgoal_horizon_steps)
        payload = {
            "experiment_id": a2.experiment_id,
            "variant": a2.variant,
            "checkpoint": str(contract.checkpoint_dir),
            "checkpoint_source_experiment": checkpoint_provenance["experiment_id"],
            "checkpoint_training_revision": checkpoint_provenance["training_dataset_revision"],
            "hierarchy": hierarchy_audit_payload(audit),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if audit.valid else 2

    if args.command == "a2-server":
        inspect_a1_checkpoint(
            config.robocerebra_posttrain.checkpoint_dir,
            expected_training_revision=config.robocerebra_posttrain.training_dataset_revision,
        )
        return run_or_preview(build_a2_server_command(config, args.seed), args.execute)

    if args.command == "a2-eval":
        inspect_a1_checkpoint(
            config.robocerebra_posttrain.checkpoint_dir,
            expected_training_revision=config.robocerebra_posttrain.training_dataset_revision,
        )
        output = (
            Path(args.output).expanduser()
            if args.output
            else config.artifact_dir / "A2" / f"H{args.execution_horizon}-seed{args.seed}"
        )
        spec = build_a2_eval_command(
            config,
            task_types=args.task_types,
            case_names=args.cases,
            trials=args.trials,
            execution_horizon=args.execution_horizon,
            seed=args.seed,
            output_dir=output,
            trace_images=not args.no_trace_images,
            resume=args.resume,
        )
        return run_or_preview(spec, args.execute)

    if args.command == "b-check":
        b = config.robocerebra_selector
        if args.stage == "checkpoint":
            contract, _ = inspect_a1_checkpoint(
                config.robocerebra_posttrain.checkpoint_dir,
                expected_training_revision=config.robocerebra_posttrain.training_dataset_revision,
            )
            audit, provenance = inspect_selector_checkpoint(
                b.checkpoint_dir,
                expected_action_horizon=b.action_horizon,
                expected_context_width=b.context_width,
            )
            verify_a1_weight_hashes(contract, provenance.get("a1_checkpoint_weight_shards_sha256"))
            payload = asdict(audit) | {
                "checkpoint_dir": str(audit.checkpoint_dir),
                "provenance": provenance,
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0
        manifest_path = b.dataset_dir / (
            "feature_manifest.json" if args.stage == "features" else "manifest.json"
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(f"B {args.stage} manifest not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.stage == "index":
            if payload.get("experiment_id") != "B":
                raise ValueError("selector index identity is not B")
            for filename in ("episodes.jsonl", "samples.jsonl"):
                key = f"{filename.removesuffix('.jsonl')}_sha256"
                if sha256_file(b.dataset_dir / filename) != payload[key]:
                    raise ValueError(f"B selector index hash mismatch: {filename}")
            if payload.get("benchmark_exact_prompt_overlaps") != 0:
                raise ValueError("B selector index overlaps exact held-out prompts")
        else:
            contract, _ = inspect_a1_checkpoint(
                config.robocerebra_posttrain.checkpoint_dir,
                expected_training_revision=config.robocerebra_posttrain.training_dataset_revision,
            )
            index_manifest_path = b.dataset_dir / "manifest.json"
            index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
            run_contract_path = b.dataset_dir / FEATURE_RUN_CONTRACT
            if (
                payload.get("experiment_id") != "B"
                or payload.get("selector_index_manifest_sha256") != sha256_file(index_manifest_path)
                or int(payload.get("samples", -1)) != int(index_manifest["samples"])
                or payload.get("feature_run_contract_sha256") != sha256_file(run_contract_path)
            ):
                raise ValueError("B feature manifest identity or parent contract is invalid")
            verify_a1_weight_hashes(contract, payload.get("checkpoint_weight_shards_sha256"))
            hashes = payload.get("feature_files_sha256")
            if not isinstance(hashes, dict) or len(hashes) != int(payload.get("feature_files", -1)):
                raise ValueError("B feature manifest is missing its complete hash inventory")
            actual_feature_files = {
                path.name for path in (b.dataset_dir / "features").glob("episode_*.npz")
            }
            if actual_feature_files != set(hashes):
                raise ValueError("B feature cache file inventory differs from its manifest")
            for filename, expected_hash in hashes.items():
                if Path(filename).name != filename:
                    raise ValueError(f"B feature manifest contains an unsafe filename: {filename}")
                path = b.dataset_dir / "features" / filename
                if not path.is_file() or sha256_file(path) != expected_hash:
                    raise ValueError(f"B feature cache hash mismatch: {filename}")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        valid = args.stage == "index" or (
            bool(payload.get("complete")) and not bool(payload.get("limited"))
        )
        return 0 if valid else 2

    if args.command == "b-prepare":
        return run_or_preview(
            build_b_prepare_command(config, limit=args.limit, output_dir=args.output), args.execute
        )

    if args.command == "b-features":
        return run_or_preview(
            build_b_features_command(
                config,
                batch_size=args.batch_size,
                limit_episodes=args.limit_episodes,
                resume=not args.no_resume,
            ),
            args.execute,
        )

    if args.command == "b-train":
        spec = build_b_train_command(config, steps=args.steps, batch_size=args.batch_size)
        if args.no_resume:
            argv = tuple(value for value in spec.argv if value != "--resume")
            spec = type(spec)(argv, spec.cwd, spec.description)
        return run_or_preview(spec, args.execute)

    if args.command == "b-server":
        inspect_a1_checkpoint(
            config.robocerebra_posttrain.checkpoint_dir,
            expected_training_revision=config.robocerebra_posttrain.training_dataset_revision,
        )
        inspect_selector_checkpoint(
            config.robocerebra_selector.checkpoint_dir,
            expected_action_horizon=config.robocerebra_selector.action_horizon,
            expected_context_width=config.robocerebra_selector.context_width,
        )
        return run_or_preview(build_b_server_command(config, args.seed), args.execute)

    if args.command == "b-eval":
        inspect_selector_checkpoint(
            config.robocerebra_selector.checkpoint_dir,
            expected_action_horizon=config.robocerebra_selector.action_horizon,
            expected_context_width=config.robocerebra_selector.context_width,
        )
        output = (
            Path(args.output).expanduser()
            if args.output
            else config.artifact_dir / "B" / f"H{args.execution_horizon}-seed{args.seed}"
        )
        return run_or_preview(
            build_b_eval_command(
                config,
                task_types=args.task_types,
                case_names=args.cases,
                trials=args.trials,
                execution_horizon=args.execution_horizon,
                seed=args.seed,
                output_dir=output,
                trace_images=not args.no_trace_images,
                resume=args.resume,
            ),
            args.execute,
        )

    if args.command in {"b-retry-check", "b-retry-server", "b-retry-eval"}:
        retry_contract = retry_contract_payload(config.robocerebra_retry)
        contract, _ = inspect_a1_checkpoint(
            config.robocerebra_posttrain.checkpoint_dir,
            expected_training_revision=config.robocerebra_posttrain.training_dataset_revision,
        )
        selector_audit, selector_provenance = inspect_selector_checkpoint(
            config.robocerebra_selector.checkpoint_dir,
            expected_action_horizon=config.robocerebra_selector.action_horizon,
            expected_context_width=config.robocerebra_selector.context_width,
        )
        verify_a1_weight_hashes(
            contract, selector_provenance.get("a1_checkpoint_weight_shards_sha256")
        )
        if args.command == "b-retry-check":
            payload = {
                "retry_contract": retry_contract,
                "parent_experiment": config.robocerebra_selector.experiment_id,
                "checkpoint": str(contract.checkpoint_dir),
                "selector_checkpoint": str(selector_audit.checkpoint_dir),
                "selector_weights_sha256": selector_audit.weights_sha256,
                "valid": True,
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return 0
        if args.command == "b-retry-server":
            return run_or_preview(build_b_retry_server_command(config, args.seed), args.execute)

        output = (
            Path(args.output).expanduser()
            if args.output
            else config.artifact_dir / "B-retry" / f"H{args.execution_horizon}-seed{args.seed}"
        )
        return run_or_preview(
            build_b_retry_eval_command(
                config,
                task_types=args.task_types,
                case_names=args.cases,
                trials=args.trials,
                execution_horizon=args.execution_horizon,
                seed=args.seed,
                output_dir=output,
                trace_images=not args.no_trace_images,
                resume=args.resume,
            ),
            args.execute,
        )

    if args.command == "ours-check":
        recovery_contract = validate_recovery_config(config.robocerebra_recovery)
        contract, _ = inspect_a1_checkpoint(
            config.robocerebra_posttrain.checkpoint_dir,
            expected_training_revision=config.robocerebra_posttrain.training_dataset_revision,
        )
        selector_audit, selector_provenance = inspect_selector_checkpoint(
            config.robocerebra_selector.checkpoint_dir,
            expected_action_horizon=config.robocerebra_selector.action_horizon,
            expected_context_width=config.robocerebra_selector.context_width,
        )
        verify_a1_weight_hashes(
            contract, selector_provenance.get("a1_checkpoint_weight_shards_sha256")
        )
        payload = recovery_contract | {
            "checkpoint": str(contract.checkpoint_dir),
            "selector_checkpoint": str(selector_audit.checkpoint_dir),
            "selector_weights_sha256": selector_audit.weights_sha256,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

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
