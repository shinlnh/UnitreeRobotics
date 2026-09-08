"""Evaluate B-retry's frozen outcome-blind naive retry control."""

from __future__ import annotations

import argparse
import json

from .b_eval import EvaluationIdentity
from .b_eval import _parser as _b_parser
from .b_eval import run as run_b
from .b_retry import (
    B_RETRY_DECISION_SCHEDULE,
    B_RETRY_ID,
    B_RETRY_MAX_RETRIES_PER_SUBTASK,
    B_RETRY_METHOD,
    B_RETRY_PAPER,
    B_RETRY_TRIGGER,
    B_RETRY_VARIANT,
    confirmed_stop_transition,
)

B_RETRY_EVALUATION = EvaluationIdentity(
    experiment_id=B_RETRY_ID,
    variant=B_RETRY_VARIANT,
    method=B_RETRY_METHOD,
    paper=B_RETRY_PAPER,
    retry=True,
    retry_trigger=B_RETRY_TRIGGER,
    max_retries_per_subtask=B_RETRY_MAX_RETRIES_PER_SUBTASK,
    decision_schedule=B_RETRY_DECISION_SCHEDULE,
)


def _parser() -> argparse.ArgumentParser:
    parser = _b_parser()
    parser.description = __doc__
    parser.set_defaults(
        experiment_id=B_RETRY_ID,
        variant=B_RETRY_VARIANT,
        method=B_RETRY_METHOD,
        paper=B_RETRY_PAPER,
    )
    parser.add_argument(
        "--max-retries-per-subtask",
        type=int,
        default=B_RETRY_MAX_RETRIES_PER_SUBTASK,
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.max_retries_per_subtask != B_RETRY_MAX_RETRIES_PER_SUBTASK:
        raise ValueError("B-retry is frozen to one retry per subtask")
    return run_b(
        args,
        identity=B_RETRY_EVALUATION,
        stop_transition=confirmed_stop_transition,
    )


def main() -> int:
    summary = run(_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
