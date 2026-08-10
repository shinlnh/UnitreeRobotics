"""Pre-flight validation for the Isaac Lab side of the project."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_ROOT", PROJECT_ROOT / ".deps" / "IsaacLab")).resolve()
EXPECTED_TASKS = {
    "Unitree-G1-Velocity-Flat-Robust",
    "Unitree-G1-Velocity-Rough-Robust",
    "Unitree-G1-GR00T-Navigation",
}


def _read_lock() -> dict[str, str]:
    lock: dict[str, str] = {}
    for raw_line in (PROJECT_ROOT / "configs" / "setup" / "versions.env").read_text().splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", maxsplit=1)
            lock[key] = value
    return lock


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _file_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _git_head(repository: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _gpu_summary() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return False, f"nvidia-smi unavailable: {exc}"
    return True, result.stdout.strip()


def _check_tasks() -> tuple[bool, str]:
    try:
        import gymnasium as gym
        import unitree_rl_groot.tasks  # noqa: F401
    except Exception as exc:
        return False, f"task import failed: {type(exc).__name__}: {exc}"
    found = {task_id for task_id in gym.registry if task_id in EXPECTED_TASKS}
    missing = EXPECTED_TASKS - found
    return (not missing, f"registered={sorted(found)}, missing={sorted(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Return non-zero when a required check fails")
    args = parser.parse_args()
    lock = _read_lock()
    checks: list[tuple[str, bool, str]] = []

    checks.append(("Python", sys.version_info[:2] == (3, 12), platform.python_version()))
    checks.append(("OS", sys.platform.startswith("linux"), platform.platform()))

    package_expectations = {
        "isaacsim": lock["ISAAC_SIM_VERSION"],
        "rsl-rl-lib": lock["RSL_RL_VERSION"],
        "unitree-rl-groot": "0.1.0",
    }
    for package, expected in package_expectations.items():
        actual = _package_version(package)
        checks.append((package, actual == expected, f"actual={actual}, expected={expected}"))

    # Isaac Lab's release version and its Python distribution version intentionally differ
    # (for example, release 3.0.0 currently ships the ``isaaclab`` distribution as 15.5.0).
    # Validate the repository's release marker and exact pinned commit instead.
    release = _file_text(ISAACLAB_ROOT / "VERSION")
    checks.append(
        (
            "Isaac Lab release",
            release == lock["ISAAC_LAB_VERSION"],
            f"actual={release}, expected={lock['ISAAC_LAB_VERSION']}",
        )
    )
    commit = _git_head(ISAACLAB_ROOT)
    checks.append(
        (
            "Isaac Lab commit",
            commit == lock["ISAAC_LAB_REF"],
            f"actual={commit}, expected={lock['ISAAC_LAB_REF']}",
        )
    )

    gpu_ok, gpu_text = _gpu_summary()
    checks.append(("NVIDIA GPU", gpu_ok, gpu_text))
    checks.append(("Tasks", *_check_tasks()))

    eula = os.environ.get("OMNI_KIT_ACCEPT_EULA")
    checks.append(("EULA env", eula == "1", f"OMNI_KIT_ACCEPT_EULA={eula!r}"))

    for label, ok, detail in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {label}: {detail}")

    failed = [label for label, ok, _ in checks if not ok]
    if failed:
        print(f"\nPre-flight failures: {', '.join(failed)}")
        return 1 if args.strict else 0
    print("\nPre-flight passed. The project is ready for smoke tests or training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
