"""Merge disjoint A2 shards while preserving the hierarchy summary contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .a0_merge import merge
from .a2_eval import _build_summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge A2 benchmark shards")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--create-target", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = merge(
        args.target,
        args.shard,
        args.create_target,
        summary_builder=_build_summary,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
