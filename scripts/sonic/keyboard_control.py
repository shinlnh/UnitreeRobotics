#!/usr/bin/env python3
"""Interactive keyboard publisher for the SONIC fetch inference stack."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import zmq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "source/unitree_rl_groot"))

from unitree_rl_groot.groot.fetch import FetchEvent, FetchExecutive, FetchMission  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-name", default="apple")
    parser.add_argument("--source", default="source table")
    parser.add_argument("--destination", default="green plate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    executive = FetchExecutive(FetchMission(args.object_name, args.source, args.destination))
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://localhost:5580")
    time.sleep(0.5)
    print("Ready: k=start/stop, i=initial pose, p=pause, r=reset scene, [ ]=hands")
    print("Executive: status, e <event>, fail. Evidence is operator-confirmed in this demo.")
    print("Events:", ", ".join(event.value for event in FetchEvent))
    print(f"phase={executive.phase.value} prompt={executive.prompt}")
    try:
        while True:
            command = input().strip()
            if not command:
                continue
            if command == "status":
                print(
                    f"phase={executive.phase.value} retries={executive.retries} "
                    f"done={executive.done} prompt={executive.prompt}"
                )
                continue
            if command == "fail":
                event = FetchEvent.FAILURE
            elif command.startswith("e "):
                try:
                    event = FetchEvent(command[2:].strip())
                except ValueError as exc:
                    print(exc)
                    continue
            else:
                event = None
            if event is not None:
                try:
                    executive.advance(event)
                except (RuntimeError, ValueError) as exc:
                    print(exc)
                    continue
                print(f"phase={executive.phase.value} retries={executive.retries} done={executive.done}")
                if executive.done:
                    if executive.failed:
                        publisher.send_string("p")
                        print("mission failed; sent pause")
                    else:
                        publisher.send_string("prompt:" + executive.prompt)
                        print("mission complete; sent neutral posture prompt")
                    continue
                message = "prompt:" + executive.prompt
            else:
                message = "prompt:" + command[2:] if command.startswith("t ") else command
            publisher.send_string(message)
            print(f"sent: {message}")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        publisher.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
