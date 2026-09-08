# Ours R1 mechanism-pruning evidence

This report preserves development-only failures that changed the controller
before the corrected R1 matrix. Partial runs are diagnostic ablations, not
complete benchmark results and never enter the final comparison.

## Full unbounded c1 run

The first registered R1 run completed all 60 cases with forced 150-step
subgoal boundaries:

- task-macro subtask rate: 2.3782% versus 1.6200% for paired B-retry
- paired delta: +0.7475 percentage points
- paired 95% bootstrap CI: [-0.3081, +2.0809] percentage points
- false recovery: 26.31%
- policy-call ratio: 8.326x
- simulator-step ratio: 3.828x

It is Pareto-invalid despite its positive point estimate.

## Pruned diagnostic runs

| Controller failure | Episodes | Ours/base subtasks | Ours/base final | Call ratio | Step ratio | Reason pruned |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| unbounded c4 consensus | 12 | 0/0 | 0/0 | 7.865x | 2.197x | already irrecoverably above both cost caps |
| bounded but forced on low failure | 17 | 0/0 | 0/0 | 1.201x | 1.894x | low failure incorrectly caused non-STOP execution |
| selective but no 75-step onset guard | 48 | 4/8 | 2/0 | 0.797x | 1.128x | offline trigger precision could not satisfy 5% false-recovery cap |

The 48-episode run is scientifically interesting: selective recovery produced
two final successes while the paired prefix had none and remained inside both
cost caps, but it completed fewer subtasks. It is retained as motivation for
the onset-calibrated controller, not presented as a win.

## Resulting correction

The active decision schedule is
`temporal-completion-gate-onset-bounded-consensus-v4`. It changes B only when:

1. B proposes STOP;
2. the learned completion belief is below its calibrated threshold;
3. at least 75 causal control steps have elapsed in the current subgoal;
4. learned failure belief exceeds the registered threshold; and
5. the one- or two-attempt recovery budget is not exhausted.

The 75-step guard is the failure-onset boundary frozen in the train-label
contract. On the completed unbounded development trace, applying this guard to
the first eligible trigger per segment gave 100% trigger precision for failure
thresholds 0.90, 0.95, 0.99, and 1.00. This is a retrospective diagnostic used
to register the next R1 variants; it is not a held-out performance claim.
