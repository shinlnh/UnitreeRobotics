#!/usr/bin/env python3
"""Drive the SONIC C++ controller with a neutral planner command for smoke tests.

This deliberately replaces the PICO manager only during unattended simulation
smoke tests.  It verifies the command/proprio/action transport without creating
demonstrations suitable for training.
"""

from __future__ import annotations

import argparse
import signal
import time

import zmq
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--frequency", type=float, default=20.0)
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=1.5,
        help="PUB/SUB subscription settling time in seconds.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run duration in seconds; zero runs until interrupted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frequency <= 0.0:
        raise SystemExit("--frequency must be greater than zero")
    if args.startup_delay < 0.0 or args.duration < 0.0:
        raise SystemExit("--startup-delay and --duration cannot be negative")

    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://*:{args.port}")
    print(f"Neutral SONIC smoke manager bound to tcp://*:{args.port}", flush=True)
    print("This is a transport smoke test, not a demonstration recorder.", flush=True)
    time.sleep(args.startup_delay)

    start_message = build_command_message(start=True, stop=False, planner=True)
    planner_message = build_planner_message(
        mode=0,
        movement=[0.0, 0.0, 0.0],
        facing=[1.0, 0.0, 0.0],
        speed=-1.0,
        height=-1.0,
    )
    stop_message = build_command_message(start=False, stop=True, planner=True)

    started_at = time.monotonic()
    last_start_at = float("-inf")
    period = 1.0 / args.frequency
    next_tick = started_at
    try:
        while not stopping:
            now = time.monotonic()
            if args.duration and now - started_at >= args.duration:
                break
            # Repeat the start command so the smoke is robust to PUB/SUB's
            # slow-joiner behavior and to a controller that initializes late.
            if now - last_start_at >= 1.0:
                socket.send(start_message)
                last_start_at = now
            socket.send(planner_message)
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    finally:
        for _ in range(3):
            socket.send(stop_message)
            time.sleep(0.02)
        socket.close()
        context.term()
        print("Neutral SONIC smoke manager stopped the controller.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
