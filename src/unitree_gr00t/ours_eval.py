"""Evaluate Ours temporal completion gate with selective action consensus."""

from __future__ import annotations

import argparse
import json

from .b_eval import EvaluationIdentity
from .b_eval import _parser as _b_parser
from .b_eval import run as run_b
from .b_retry import (
    B_RETRY_MAX_RETRIES_PER_SUBTASK,
    B_RETRY_TRIGGER,
    confirmed_stop_transition,
)
from .ours import OURS_ID, OURS_METHOD, OURS_VARIANT
from .ours_policy import SelectiveConsensusRecovery
from .ours_train import inspect_recovery_checkpoint

OURS_PAPER = "working-method-before-final-freeze"
OURS_DECISION_SCHEDULE = "temporal-completion-gate-onset-bounded-consensus-v4"
OURS_OPTION_DECISION_SCHEDULE = "counterfactual-value-direct-option-bounded-v6"
OURS_RESIDUAL_DECISION_SCHEDULE = "counterfactual-residual-over-b-retry-v1"
OURS_EVALUATION = EvaluationIdentity(
    experiment_id=OURS_ID,
    variant=OURS_VARIANT,
    method=OURS_METHOD,
    paper=OURS_PAPER,
    decision_schedule=OURS_DECISION_SCHEDULE,
    recovery=True,
)
OURS_RESIDUAL_EVALUATION = EvaluationIdentity(
    experiment_id=OURS_ID,
    variant=OURS_VARIANT,
    method=OURS_METHOD,
    paper=OURS_PAPER,
    retry=True,
    retry_trigger=B_RETRY_TRIGGER,
    max_retries_per_subtask=B_RETRY_MAX_RETRIES_PER_SUBTASK,
    decision_schedule=OURS_RESIDUAL_DECISION_SCHEDULE,
    recovery=True,
)


def _parser() -> argparse.ArgumentParser:
    parser = _b_parser()
    parser.description = __doc__
    parser.set_defaults(
        experiment_id=OURS_ID,
        variant=OURS_VARIANT,
        method=OURS_METHOD,
        paper=OURS_PAPER,
        seed=30007,
    )
    parser.add_argument("--recovery-checkpoint", required=True)
    parser.add_argument("--consensus-hypotheses", type=int, choices=(1, 4, 8), default=4)
    parser.add_argument(
        "--gate-signal",
        choices=("completion", "progress", "maximum"),
        default="completion",
    )
    parser.add_argument(
        "--gate-threshold-override",
        type=float,
        help="Development-search threshold; forbidden for the final held-out seed",
    )
    parser.add_argument(
        "--stagnation-boundary-steps",
        type=int,
        help="Bounded outcome-blind ADVANCE fallback searched only on development runs",
    )
    parser.add_argument(
        "--failure-threshold",
        type=float,
        help="Development-only override; otherwise use checkpoint calibration",
    )
    parser.add_argument("--max-recovery-attempts", type=int, choices=(1, 2), default=1)
    parser.add_argument("--min-recovery-elapsed-steps", type=int, default=75)
    parser.add_argument("--use-option-values", action="store_true")
    parser.add_argument("--option-value-margin", type=float, default=0.0)
    parser.add_argument(
        "--residual-retry-baseline",
        action="store_true",
        help="Abstain to frozen B-retry and learn only a confirmed-STOP override",
    )
    parser.add_argument("--consensus-cooldown-decisions", type=int, default=16)
    parser.add_argument("--capture-training-context", action="store_true")
    parser.add_argument("--collection-force-boundary-steps", type=int)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.capture_training_context and args.seed not in {10007, 11007, 12007}:
        raise ValueError("training context export is restricted to frozen train base seeds")
    if args.collection_force_boundary_steps is not None and not args.capture_training_context:
        raise ValueError("forced boundaries are restricted to training-context collection")
    if (
        args.collection_force_boundary_steps is not None
        and args.stagnation_boundary_steps is not None
    ):
        raise ValueError("collection and searched stagnation boundaries are mutually exclusive")
    if args.seed == 7 and (
        args.gate_threshold_override is not None
        or args.failure_threshold is not None
        or args.stagnation_boundary_steps is not None
    ):
        raise ValueError("held-out evaluation requires controls frozen into checkpoint provenance")
    if args.gate_threshold_override is not None and not 0.0 <= args.gate_threshold_override <= 1.0:
        raise ValueError("gate threshold override must be inside [0, 1]")
    if args.failure_threshold is not None and not 0.0 <= args.failure_threshold <= 1.0:
        raise ValueError("failure threshold must be inside [0, 1]")
    if args.consensus_cooldown_decisions < 0:
        raise ValueError("consensus cooldown cannot be negative")
    if args.min_recovery_elapsed_steps < 0:
        raise ValueError("minimum recovery elapsed steps cannot be negative")
    if args.option_value_margin < 0.0:
        raise ValueError("option value margin cannot be negative")
    if args.residual_retry_baseline and not args.use_option_values:
        raise ValueError("residual B-retry mode requires --use-option-values")
    recovery_audit, recovery_provenance = inspect_recovery_checkpoint(args.recovery_checkpoint)
    metrics = recovery_provenance["development_metrics"]
    threshold_key = {
        "completion": "completion_threshold",
        "progress": "progress_gate_threshold",
        "maximum": "maximum_gate_threshold",
    }[args.gate_signal]
    if threshold_key not in metrics:
        raise ValueError(
            f"Ours checkpoint predates {args.gate_signal} gate calibration: {recovery_audit.checkpoint_dir}"
        )
    checkpoint_gate_threshold = float(metrics[threshold_key])
    gate_threshold = (
        float(args.gate_threshold_override)
        if args.gate_threshold_override is not None
        else checkpoint_gate_threshold
    )
    checkpoint_failure_threshold = float(metrics.get("failure_threshold") or 0.5)
    failure_threshold = (
        float(args.failure_threshold)
        if args.failure_threshold is not None
        else checkpoint_failure_threshold
    )
    boundary_steps = (
        args.collection_force_boundary_steps
        if args.collection_force_boundary_steps is not None
        else args.stagnation_boundary_steps
    )
    controller = SelectiveConsensusRecovery(
        gate_signal=args.gate_signal,
        gate_threshold=gate_threshold,
        consensus_hypotheses=args.consensus_hypotheses,
        max_recovery_attempts=args.max_recovery_attempts,
        min_recovery_elapsed_steps=args.min_recovery_elapsed_steps,
        use_option_values=args.use_option_values,
        option_value_margin=args.option_value_margin,
        failure_threshold=failure_threshold,
        consensus_cooldown_decisions=args.consensus_cooldown_decisions,
        force_boundary_steps=boundary_steps,
        residual_retry_baseline=args.residual_retry_baseline,
    )
    manifest_extensions = {
        "decision_schedule": (
            OURS_RESIDUAL_DECISION_SCHEDULE
            if args.residual_retry_baseline
            else OURS_OPTION_DECISION_SCHEDULE
            if args.use_option_values
            else OURS_DECISION_SCHEDULE
        ),
        "recovery_checkpoint": str(recovery_audit.checkpoint_dir),
        "recovery_weights_sha256": recovery_audit.weights_sha256,
        "recovery_provenance_sha256": recovery_audit.provenance_sha256,
        "recovery_parameter_count": recovery_audit.parameter_count,
        "recovery_encoder": recovery_audit.encoder,
        "recovery_history_length": recovery_audit.history_length,
        "completion_threshold": recovery_audit.completion_threshold,
        "gate_signal": args.gate_signal,
        "gate_threshold": gate_threshold,
        "checkpoint_gate_threshold": checkpoint_gate_threshold,
        "gate_threshold_override": args.gate_threshold_override,
        "consensus_hypotheses": args.consensus_hypotheses,
        "max_recovery_attempts": args.max_recovery_attempts,
        "min_recovery_elapsed_steps": args.min_recovery_elapsed_steps,
        "use_option_values": args.use_option_values,
        "option_value_margin": args.option_value_margin,
        "residual_retry_baseline": args.residual_retry_baseline,
        "residual_abstention_option": (
            "RETRY_CURRENT" if args.residual_retry_baseline else None
        ),
        "failure_threshold": failure_threshold,
        "checkpoint_failure_threshold": checkpoint_failure_threshold,
        "failure_threshold_override": args.failure_threshold,
        "consensus_cooldown_decisions": args.consensus_cooldown_decisions,
        "recovery_checkpoint_stage": recovery_provenance["stage"],
        "failure_detector": True,
        "recovery_memory": True,
        "recovery_policy": True,
        "observes_goal_predicates": False,
        "observes_injection_labels": False,
        "observes_task_outcomes": False,
        "state_restoration": False,
        "capture_training_context": args.capture_training_context,
        "collection_force_boundary_steps": args.collection_force_boundary_steps,
        "stagnation_boundary_steps": args.stagnation_boundary_steps,
    }
    return run_b(
        args,
        identity=(OURS_RESIDUAL_EVALUATION if args.residual_retry_baseline else OURS_EVALUATION),
        stop_transition=(confirmed_stop_transition if args.residual_retry_baseline else None),
        proposal_transition=controller,
        manifest_extensions=manifest_extensions,
    )


def main() -> int:
    summary = run(_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
