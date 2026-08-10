#!/usr/bin/env python3
"""Return success when the isolated locomanipulation policy server is ready."""

from __future__ import annotations

import argparse

from unitree_rl_groot.groot.locomanipulation import GrootN15PolicyClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--timeout-ms", type=int, default=1_000)
    args = parser.parse_args()
    with GrootN15PolicyClient(args.host, args.port, timeout_ms=args.timeout_ms) as client:
        return 0 if client.ping() else 1


if __name__ == "__main__":
    raise SystemExit(main())
