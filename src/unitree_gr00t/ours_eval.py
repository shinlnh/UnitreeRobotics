"""Evaluate Ours temporal completion gate with selective action consensus."""

from __future__ import annotations

import argparse
import json

from .b_eval import EvaluationIdentity
from .b_eval import _parser as _b_parser
from .b_eval import run as run_b
from .ours import OURS_ID, OURS_METHOD, OURS_VARIANT
from .ours_policy import SelectiveConsensusRecovery
from .ours_train import inspect_recovery_checkpoint

OURS_PAPER = "working-method-before-final-freeze"
OURS_DECISION_SCHEDULE = "temporal-completion-gate-selective-consensus-v1"
OURS_EVALUATION = EvaluationIdentity(
    experiment_id=OURS_ID,
    variant=OURS_VARIANT,
    method=OURS_METHOD,
    paper=OURS_PAPER,
    decision_schedule=OURS_DECISION_SCHEDULE,
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
    parser.add_argument("--failure-threshold", type=float, default=0.5)
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
        args.gate_threshold_override is not None or args.stagnation_boundary_steps is not None
    ):
        raise ValueError("held-out evaluation requires controls frozen into checkpoint provenance")
    if args.gate_threshold_override is not None and not 0.0 <= args.gate_threshold_override <= 1.0:
        raise ValueError("gate threshold override must be inside [0, 1]")
    if not 0.0 <= args.failure_threshold <= 1.0:
        raise ValueError("failure threshold must be inside [0, 1]")
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
    boundary_steps = (
        args.collection_force_boundary_steps
        if args.collection_force_boundary_steps is not None
        else args.stagnation_boundary_steps
    )
    controller = SelectiveConsensusRecovery(
        gate_signal=args.gate_signal,
        gate_threshold=gate_threshold,
        consensus_hypotheses=args.consensus_hypotheses,
        failure_threshold=args.failure_threshold,
        force_boundary_steps=boundary_steps,
    )
    manifest_extensions = {
        "decision_schedule": OURS_DECISION_SCHEDULE,
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
        "failure_threshold": args.failure_threshold,
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
        identity=OURS_EVALUATION,
        proposal_transition=controller,
        manifest_extensions=manifest_extensions,
    )


def main() -> int:
    summary = run(_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
