#!/usr/bin/env python3
"""Generate a reproducible whole-body fetch collection curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

from unitree_rl_groot.groot.fetch import FetchMission, FetchPhase

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/fetch/curriculum.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets/sonic/fetch_curriculum.jsonl")
    return parser.parse_args()


def _split(identifier: str, ratios: dict[str, float]) -> str:
    total = sum(ratios.values())
    if abs(total - 1.0) > 1.0e-6 or any(value <= 0.0 for value in ratios.values()):
        raise ValueError("split ratios must be positive and sum to one")
    value = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    for name, ratio in ratios.items():
        cumulative += ratio
        if value <= cumulative:
            return name
    return next(reversed(ratios))


def _profile(label: str) -> str:
    return "_".join(label.lower().split())


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    repetitions = int(config["demonstrations_per_combination"])
    if repetitions <= 0:
        raise ValueError("demonstrations_per_combination must be positive")
    records = []
    for object_name, source, destination in product(
        config["objects"], config["sources"], config["destinations"]
    ):
        mission = FetchMission(object_name, source, destination)
        for repetition in range(repetitions):
            identifier = f"{object_name}|{source}|{destination}|{repetition:03d}"
            records.append(
                {
                    "mission_id": hashlib.sha256(identifier.encode()).hexdigest()[:16],
                    "split": _split(identifier, config["split_ratios"]),
                    "object": mission.object_name,
                    "source": mission.source,
                    "destination": mission.destination,
                    "instruction": mission.instruction,
                    "phase_prompts": {phase.value: mission.phase_prompt(phase) for phase in FetchPhase},
                    "sim_profiles": {
                        "object": _profile(object_name),
                        "source": _profile(source),
                        "destination": _profile(destination),
                    },
                    "randomization": config["randomization"],
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    counts = {name: sum(record["split"] == name for record in records) for name in config["split_ratios"]}
    print(json.dumps({"missions": len(records), "splits": counts, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
