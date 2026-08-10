#!/usr/bin/env python3
"""Report host settings that commonly bottleneck Isaac Lab GPU utilization."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _values(pattern: str) -> set[str]:
    return {path.read_text().strip() for path in Path("/").glob(pattern) if path.is_file()}


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def main() -> int:
    governors = _values("sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
    drivers = _values("sys/devices/system/cpu/cpu*/cpufreq/scaling_driver")
    preferences = _values("sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference")
    power_profile = _command_output(["powerprofilesctl", "get"])
    result: dict[str, object] = {
        "cpu_governors": sorted(governors),
        "cpu_scaling_drivers": sorted(drivers),
        "energy_performance_preferences": sorted(preferences),
        "power_profile": power_profile or "unavailable",
    }
    result["gpu"] = (
        _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,pstate,power.limit,clocks.current.graphics,clocks.max.graphics,memory.total",
                "--format=csv,noheader",
            ]
        )
        or "unavailable"
    )
    print(json.dumps(result, indent=2))

    pstate_managed = bool(drivers & {"intel_pstate", "amd_pstate"})
    performance_epp = bool(preferences) and preferences <= {"performance", "balance_performance"}
    if "powersave" in governors and not (pstate_managed and performance_epp):
        print("[WARN] CPU governor is powersave; Isaac Sim may starve the GPU during scene stepping.")
        return 1
    if power_profile and power_profile != "performance":
        print(f"[WARN] Desktop power profile is {power_profile!r}, not 'performance'.")
        return 1
    print("[OK] CPU frequency policy is configured for performance workloads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
