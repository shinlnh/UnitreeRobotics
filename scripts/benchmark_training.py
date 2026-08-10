#!/usr/bin/env python3
"""Benchmark bounded PPO runs and recommend a throughput-oriented environment count."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "rsl_rl" / "unitree_g1_flat_benchmark"


def _gpu_sample(pid: int) -> dict[str, float] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        process_memory = 0.0
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 2 and int(fields[0]) == pid:
                process_memory = float(fields[1])
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        utilization, power = [float(value.strip()) for value in gpu.stdout.splitlines()[0].split(",")]
        return {"process_vram_mib": process_memory, "gpu_utilization_percent": utilization, "power_w": power}
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _last_scalar_values(run_dir: Path, tag: str, warmup: int) -> list[float]:
    event_files = list(run_dir.glob("events.out.tfevents*"))
    if not event_files:
        return []
    accumulator = EventAccumulator(str(event_files[0]), size_guidance={"scalars": 0}).Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    values = [float(event.value) for event in accumulator.Scalars(tag)]
    return values[min(warmup, len(values)) :]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def benchmark(env_count: int, iterations: int, warmup: int, output_dir: Path) -> dict[str, Any]:
    mini_batches = max(1, env_count // 1_024)
    run_suffix = f"benchmark_{env_count}_{int(time.time())}"
    before = set(DEFAULT_LOG_ROOT.iterdir()) if DEFAULT_LOG_ROOT.exists() else set()
    log_path = output_dir / f"train_{env_count}.log"
    command = [
        str(PROJECT_ROOT / ".deps/IsaacLab/.venv/bin/python"),
        str(PROJECT_ROOT / "scripts/train.py"),
        "--rl_library",
        "rsl_rl",
        "--task",
        "Unitree-G1-Velocity-Flat-Robust",
        "--experiment_name",
        "unitree_g1_flat_benchmark",
        "--num_envs",
        str(env_count),
        "--max_iterations",
        str(iterations),
        "--run_name",
        run_suffix,
        "--viz",
        "none",
        f"agent.algorithm.num_mini_batches={mini_batches}",
    ]
    samples: list[dict[str, float]] = []
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            sample = _gpu_sample(process.pid)
            if sample is not None:
                samples.append(sample)
            time.sleep(0.5)
    after = set(DEFAULT_LOG_ROOT.iterdir()) if DEFAULT_LOG_ROOT.exists() else set()
    created = sorted(after - before, key=lambda path: path.stat().st_mtime)
    run_dir = created[-1] if created else None
    result: dict[str, Any] = {
        "num_envs": env_count,
        "num_mini_batches": mini_batches,
        "iterations": iterations,
        "exit_code": process.returncode,
        "wall_time_seconds": time.time() - started,
        "log": str(log_path),
        "run_dir": str(run_dir) if run_dir else None,
    }
    if samples:
        result.update(
            {
                "peak_process_vram_mib": max(sample["process_vram_mib"] for sample in samples),
                "mean_gpu_utilization_percent": _mean(
                    [sample["gpu_utilization_percent"] for sample in samples]
                ),
                "mean_power_w": _mean([sample["power_w"] for sample in samples]),
            }
        )
    if process.returncode == 0 and run_dir is not None:
        result.update(
            {
                "mean_total_fps": _mean(_last_scalar_values(run_dir, "Perf/total_fps", warmup)),
                "mean_collection_seconds": _mean(
                    _last_scalar_values(run_dir, "Perf/collection_time", warmup)
                ),
                "mean_learning_seconds": _mean(_last_scalar_values(run_dir, "Perf/learning_time", warmup)),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-counts", type=int, nargs="+", default=[4_096, 6_144, 8_192])
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/training_benchmark.json")
    args = parser.parse_args()
    if min(*args.env_counts, args.iterations) <= 0 or not 0 <= args.warmup < args.iterations:
        raise ValueError("counts/iterations must be positive and warmup smaller than iterations")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for env_count in args.env_counts:
        print(f"[BENCH] {env_count} envs", flush=True)
        result = benchmark(env_count, args.iterations, args.warmup, args.output.parent)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    successful = [result for result in results if result.get("mean_total_fps")]
    recommendation = (
        max(successful, key=lambda result: result["mean_total_fps"])["num_envs"] if successful else None
    )
    report = {"profiles": results, "recommended_num_envs": recommendation}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[SAVE] {args.output}")
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
