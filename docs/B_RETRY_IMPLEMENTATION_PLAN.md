# Experiment B-retry implementation freeze

This document freezes B-retry before any B-retry held-out outcome is inspected.
It is an evaluation control, not a learned recovery method or a contribution of
the proposed system.

## Identity and parent

- Experiment: `B-retry`
- Variant: `GR00T-RC-SparkVLA-style-execution-naive-retry`
- Method: `B-plus-fixed-unconditional-single-subtask-retry`
- Parent B source: `32838a4d42af8c8e4616f12539c9a5ef4d9c618f`
- Frozen B benchmark artifact commit:
  `06ae5d385ad55415332e4f1688b7cd59f868ac96`
- Low-level checkpoint, selector checkpoint, plans, observations, actions,
  STOP confirmation, seeds, task construction, injections, post-success window,
  H8/H16 masks, and global episode budget: byte- or value-identical to B.

## Fixed naive retry law

B-retry changes exactly one controller transition. After the first confirmed
STOP for any active canonical subtask, the controller repeats that same subtask
once from the live simulator state. After the next confirmed STOP, it advances
to the next canonical subtask. The retry starts a new B selector anchor for the
unchanged instruction. A subtask therefore has exactly two attempts, indexed
`0` and `1`.

The retry is unconditional and outcome-blind. Its only inputs are the active
subtask index, its fixed attempt index, and B's confirmed STOP event. It never
reads:

- success or goal predicates;
- failure-injection type, time, or parameters;
- object state or a failure label;
- future observations or test outcomes.

Retry consumes the existing B global step budget. The simulator is never reset,
restored, teleported, or rewound. The policy and selector are not trained or
fine-tuned, and no horizon or retry count may be tuned on the 60 held-out cases.

## Explicitly absent

B-retry has no failure detector, learned or heuristic failure classification,
recovery-state memory, recovery action space, recovery policy, recovery loss,
or oracle signal. Its one integer attempt counter is only the state required to
execute the fixed control law; it stores no observation, failure, or outcome.

Because the control retries successful and unsuccessful subtasks equally, any
change relative to B estimates the effect of blind repetition under an equal
budget. It must not be described as autonomous recovery.

## Trace and evaluation contract

Every B decision field remains present. B-retry additionally records the fixed
retry trigger, attempt index before and after the decision, whether retry was
triggered, whether the subtask advanced, and retry counts per episode/subtask.
The manifest records all prohibited inputs as false and binds the reused A1 and
B selector weight hashes.

Evaluation uses the same 600 task/trial tuples for each of H16 and H8 and reports
paired deltas against frozen B. Failure-detection and learned-recovery metrics
remain not applicable; retry attempt counts and the ordinary success, stability,
efficiency, and injection-conditioned outcomes are reported.

## Delivery gates

1. **BR0 — contract freeze:** commit this rule and typed config before the pilot.
2. **BR1 — controller tests:** prove first STOP retries, second STOP advances,
   the final subtask follows the same rule, and any retry-count drift fails.
3. **BR2 — inheritance audit:** validate the frozen A1 and B selector hashes and
   prove B's default evaluator path remains retry-free.
4. **BR3 — deterministic pilot:** run one H16 `Ideal/case1` trace only to audit
   anchors, attempt transitions, seeds, actions, and the unchanged step budget.
   The outcome is retained and never used for tuning.
5. **BR4 — frozen benchmark:** run both full horizons, merge resumable shards,
   audit all traces, and generate the paired reviewer report against B.

Run the complete evaluation-only pipeline with:

```bash
scripts/run_b_retry_pipeline.sh
```
