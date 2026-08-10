#!/usr/bin/env python3
"""Serve the official ApplePnP LeApp graph over the GR00T ZMQ protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from applepnp_policy import DEFAULT_MODEL, ApplePnPOnnxPolicy
from gr00t.policy.server_client import PolicyServer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5550)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    policy = ApplePnPOnnxPolicy(args.model.resolve(), seed=args.seed)
    with PolicyServer(policy, host=args.host, port=args.port) as server:
        try:
            server.run()
        except KeyboardInterrupt:
            print("ApplePnP server stopped.")


if __name__ == "__main__":
    main()
