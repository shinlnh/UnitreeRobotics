#!/usr/bin/env python3
"""Fail early with an actionable message when a gated GR00T dependency is unavailable."""

from __future__ import annotations

import argparse

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    try:
        hf_hub_download(args.repo, filename="config.json")
    except (GatedRepoError, HfHubHTTPError) as error:
        print(f"Cannot access the required model {args.repo}: {error}")
        print(f"1. Accept its access terms at https://huggingface.co/{args.repo}")
        print("2. Authenticate this machine once with: hf auth login")
        return 2
    print(f"Hugging Face access verified: {args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
