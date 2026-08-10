#!/usr/bin/env python3
"""Check pinned SONIC/GR00T sources and fetch-pipeline prerequisites."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require",
        choices=("source", "sim", "data", "collection-sim", "inference", "demo-sim", "train"),
        default="source",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _versions() -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in (PROJECT_ROOT / "configs/setup/versions.env").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )


def _commit(directory: Path) -> str | None:
    if not (directory / ".git").is_dir():
        return None
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _can_import(python: Path, modules: tuple[str, ...]) -> bool:
    if not python.is_file():
        return False
    statement = "; ".join(f"import {module}" for module in modules)
    try:
        result = subprocess.run(
            [str(python), "-c", statement],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def main() -> int:
    args = parse_args()
    versions = _versions()
    sonic = PROJECT_ROOT / ".deps/GR00T-WholeBodyControl"
    groot = PROJECT_ROOT / ".deps/Isaac-GR00T"
    sonic_sim_python = sonic / ".venv_sim/bin/python"
    sonic_inference_python = sonic / ".venv_inference/bin/python"
    groot_python = groot / ".venv/bin/python"
    sample_mesh = sonic / "gear_sonic/data/assets/robot_description/meshes/g1/head_link.STL"
    deploy = sonic / "gear_sonic_deploy"
    local_bin = PROJECT_ROOT / ".deps/sonic-deploy/bin"
    checks: dict[str, bool] = {
        "sonic_source_pinned": _commit(sonic) == versions["SONIC_REF"],
        "groot_source_pinned": _commit(groot) == versions["GR00T_REF"],
        "sonic_lfs_assets": sample_mesh.is_file() and sample_mesh.stat().st_size > 1_000,
        "git_lfs": shutil.which("git-lfs") is not None,
        "cmake": shutil.which("cmake") is not None,
        "tmux": shutil.which("tmux") is not None or (local_bin / "tmux").is_file(),
        "just": shutil.which("just") is not None or (local_bin / "just").is_file(),
        "sonic_sim_env": _can_import(sonic_sim_python, ("gear_sonic", "mujoco")),
        "sonic_teleop_env": (sonic / ".venv_teleop/bin/python").is_file(),
        "sonic_data_env": (sonic / ".venv_data_collection/bin/python").is_file(),
        "sonic_inference_env": _can_import(sonic_inference_python, ("gear_sonic", "msgpack_numpy", "zmq")),
        "sonic_deploy_built": (deploy / "target/release/g1_deploy_onnx_ref").is_file(),
        "sonic_v1_1_encoder": (deploy / "policy/sonic_v1_1/model_encoder.onnx").is_file(),
        "sonic_v1_1_decoder": (deploy / "policy/sonic_v1_1/model_decoder.onnx").is_file(),
        "sonic_planner": (deploy / "planner/target_vel/V2/planner_sonic.onnx").is_file(),
        "groot_env_ready": _can_import(groot_python, ("gr00t", "torch")),
    }
    sonic_models = ("sonic_v1_1_encoder", "sonic_v1_1_decoder", "sonic_planner")
    requirements = {
        "source": ("sonic_source_pinned", "groot_source_pinned", "sonic_lfs_assets", "git_lfs"),
        "sim": ("sonic_source_pinned", "sonic_lfs_assets", "sonic_sim_env"),
        "data": ("sonic_teleop_env", "sonic_data_env", "tmux", "sonic_deploy_built", *sonic_models),
        "collection-sim": (
            "sonic_sim_env",
            "sonic_teleop_env",
            "sonic_data_env",
            "tmux",
            "sonic_deploy_built",
            *sonic_models,
        ),
        "inference": (
            "sonic_inference_env",
            "tmux",
            "sonic_deploy_built",
            "groot_env_ready",
            *sonic_models,
        ),
        "demo-sim": (
            "sonic_sim_env",
            "sonic_inference_env",
            "tmux",
            "sonic_deploy_built",
            "groot_env_ready",
            *sonic_models,
        ),
        "train": ("groot_env_ready",),
    }
    failed = [name for name in requirements[args.require] if not checks[name]]
    report = {
        "mode": args.require,
        "checks": checks,
        "failed_required_checks": failed,
        "ready": not failed,
        "action_contract": {
            "embodiment": "UNITREE_G1_SONIC",
            "vla_hz": 2.5,
            "sonic_hz": 50,
            "motion_token": 64,
            "left_hand": 7,
            "right_hand": 7,
            "total": 78,
            "action_horizon": 40,
        },
    }
    print(json.dumps(report, indent=2))
    return int(args.strict and bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
