"""Environment diagnostics for mock, simulation, and real-robot profiles."""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import ProjectConfig


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _command_check(command: str, required: bool = True) -> Check:
    path = shutil.which(command)
    return Check(command, bool(path), path or "not found on PATH", required)


def _git_revision(path: Path, expected: str) -> Check:
    if not path.is_dir():
        return Check(path.name, False, f"missing checkout: {path}")
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    actual = completed.stdout.strip()
    ok = completed.returncode == 0 and (actual == expected or actual.startswith(expected))
    return Check(path.name, ok, f"revision {actual or 'unknown'}; expected {expected}")


def _gpu_check() -> Check:
    if not shutil.which("nvidia-smi"):
        return Check("NVIDIA GPU", False, "nvidia-smi not found on PATH")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    detail = completed.stdout.strip() or completed.stderr.strip() or "driver query failed"
    return Check("NVIDIA GPU", completed.returncode == 0, detail)


def _tcp_check(name: str, host: str, port: int, required: bool) -> Check:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return Check(name, True, f"reachable at {host}:{port}", required)
    except OSError as exc:
        return Check(name, False, f"unreachable at {host}:{port}: {exc}", required)


def _hf_access_check() -> Check:
    if not shutil.which("hf"):
        return Check("Hugging Face access", False, "hf CLI is not installed")
    completed = subprocess.run(
        [
            "hf",
            "download",
            "nvidia/Cosmos-Reason2-2B",
            "--include",
            "config.json",
            "--dry-run",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    detail = (
        "Cosmos-Reason2-2B accessible"
        if completed.returncode == 0
        else "request/accept the gated Cosmos-Reason2-2B license and run `hf auth login`"
    )
    return Check("Hugging Face access", completed.returncode == 0, detail)


def run_doctor(config: ProjectConfig, profile: str = "mock", online: bool = False) -> list[Check]:
    if profile not in {"mock", "sim", "real"}:
        raise ValueError("profile must be mock, sim, or real")
    checks = [
        Check("Python", sys.version_info >= (3, 11), sys.version.split()[0]),
        Check("Project config", True, str(config.root / "configs" / "project.toml")),
    ]
    if profile == "mock":
        return checks

    checks.extend(
        [
            _command_check("git"),
            _command_check("git-lfs"),
            _command_check("uv"),
            _command_check("ffmpeg"),
            _command_check("tmux"),
            _gpu_check(),
            _git_revision(config.isaac_gr00t_dir, config.upstream.isaac_gr00t_revision),
            _git_revision(config.sonic_dir, config.upstream.sonic_revision),
            Check(
                "SONIC controller",
                (
                    config.sonic_dir
                    / "gear_sonic_deploy"
                    / "policy"
                    / config.sonic.variant
                    / "model_encoder.onnx"
                ).is_file(),
                f"variant {config.sonic.variant}",
            ),
        ]
    )
    if online:
        checks.append(_hf_access_check())
        checks.append(_tcp_check("PolicyServer", config.model.host, config.model.port, False))
        if profile == "real":
            checks.append(
                _tcp_check("Robot camera", config.sonic.camera_host, config.sonic.camera_port, True)
            )
    return checks


def doctor_ok(checks: list[Check]) -> bool:
    return all(check.ok or not check.required for check in checks)
