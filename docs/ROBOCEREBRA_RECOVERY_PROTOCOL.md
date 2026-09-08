# RoboCerebra recovery evaluation protocol

This document freezes the evaluation order and prevents recovery-specific
changes from leaking into the baselines.

## Research question

Can a GR00T controller recover from execution failures in a continuous physical
timeline, after stop-aware adaptive chunking alone has been accounted for?

The frozen experiment matrix is incremental:

1. **A0 — NVIDIA GR00T N1.7 LIBERO:** official `libero_10` weights, unchanged;
   full-task instruction, no hierarchy, no stop selector, and no recovery.
2. **A1 — GR00T-RC:** one shared post-trained RoboCerebra checkpoint, with no
   hierarchy, stop selector, or recovery.
3. **A2 — GR00T-RC + fixed hierarchy:** the shared A1 checkpoint plus the fixed
   HPE anchor schedule from RoboCerebra's public evaluator, evaluated with fixed
   H8 and native H16 execution.
4. **B — GR00T-RC + SparkVLA-style execution:** A2 plus a faithful
   reimplementation of unified `STOP` versus action-prefix selection.
5. **B-retry — naive retry control:** B plus a non-learned subtask retry.
6. **Ours — learned recovery:** B plus failure detection, recovery-state memory,
   and recovery action selection.
7. **Oracle — upper bound:** Ours with an oracle failure trigger that is never
   exposed to the deployable policy.

SparkVLA's public repository contains only its README and assets at the pinned
revision. B must therefore be labelled a reimplementation, not an official
SparkVLA checkpoint. Paper-reported SparkVLA results may be shown separately but
must not be mixed with locally reproduced results.

SparkVLA's paper was published later as
[`arXiv:2608.16172v1`](https://arxiv.org/abs/2608.16172v1). B uses the paper as
the method specification while retaining the pinned repository revision as
evidence that no official code or checkpoint was available. The GR00T-RC
adaptation and frozen delivery gates are defined in
[`SPARKVLA_B_IMPLEMENTATION_PLAN.md`](SPARKVLA_B_IMPLEMENTATION_PLAN.md).

## Frozen assets

The exact repository, dataset, model revisions, and evaluation constants are in
`configs/project.toml`. The canonical A0 checkpoint is `libero_10`, matching
the checkpoint used by NVIDIA's official LIBERO examples. Only inference files
are required; optimizer and RNG states are excluded from downloads.

The A1 training snapshot is `qiukingballball/RoboCerebra` at the pinned revision.
Its 1,000-row manifest contains 995 replayable demonstrations: four rows have no
HDF5 demonstration and one HDF5 row has no matching authoritative BDDL. Those
five exclusions are frozen by source audit and reported in the converted-dataset
provenance. HDF5 `problem_info.language_instruction` and `bddl_file_name` are the
authoritative label and environment contract when manifest summaries or duplicate
BDDL files disagree.

## Two evaluation tracks

### Track A — benchmark-compatible

Use the six official RoboCerebra categories and ten trials per task. Preserve the
benchmark's task construction, success predicates, initial states, camera setup,
20 Hz control frequency, and episode budget. Report the benchmark's task success,
subtask completion, planning accuracy, and efficiency metrics.

RoboCerebra's `resume` path restores serialized simulator states at segment
boundaries. Results from this track are comparable to the benchmark paper but
must not be described as autonomous physical recovery.

### Track B — continuous recovery

Start each rollout once and never restore, teleport, or reset robot/object state
inside the rollout. A miss, slip, drop, displacement, or incorrect placement
remains in the world. Observations and policy calls continue from simulation
runtime. Success is latched for measurement, but evaluation continues for 80
steps to measure post-success reactivation.

Track B uses deterministic failure injections with the same injection step,
magnitude, task, and seed for A0, B, B-retry, Ours, and Oracle:

- grasp miss;
- object slip/drop;
- target displacement before contact;
- target displacement while carrying;
- incorrect placement;
- transient visual occlusion or observation mismatch.

## Fixed-action controls

The `libero_sim` processor contract predicts 16-step chunks. A0 is reported with both:

- `A0-H16`: execute the native full chunk;
- `A0-H8`: receding-horizon control with a fixed eight-step prefix.

B may choose `STOP` or a prefix from the same 16 actions. Reporting H8 prevents
an apparent SparkVLA-style gain from being attributed merely to executing fewer steps.
No execution horizon may be selected using test results.

### Frozen A2 hierarchy

RoboCerebra's paper describes a dynamic VLM planner with visual monitoring and
memory, but its pinned public evaluator directly consumes the canonical `Step:`
annotations and switches them at fixed 150-step anchors. A2 faithfully freezes the
released mechanism as `RoboCerebra-HPE-fixed-anchor-reimplementation`. It must not
be labelled as the complete paper HPE runtime.

The active subgoal is a pure function of the rollout control-step counter. A2 does
not inspect images, predicates, completion, or failures to select or advance a
subgoal. A chunk that crosses a fixed anchor is truncated exactly at the anchor so
the next policy call receives the next instruction; this deterministic truncation
is not an adaptive selector. A2 never restores state, retries a subtask, re-plans,
or carries recovery state. Canonical plan text, plan SHA-256, active subgoal, fixed
anchor bounds, and every executed prefix are retained in the trace.

### Frozen B-retry control

B-retry reuses B's frozen A1 and selector checkpoints, canonical plan, confirmed
STOP rule, H8/H16 candidate masks, continuous simulator state, injections, and
global episode budget. On the first confirmed STOP for each active subtask, it
unconditionally repeats the same instruction once and creates a new B subtask
anchor; the second confirmed STOP advances. This transition depends only on the
subtask index and fixed attempt counter. It cannot inspect goal predicates,
failure labels, injection metadata, object state, or rollout outcomes.

B-retry does not reset or restore the simulator and has no detector, recovery
memory, recovery action, recovery policy, recovery loss, or learned parameter.
The complete pre-outcome freeze and trace contract are in
[`B_RETRY_IMPLEMENTATION_PLAN.md`](B_RETRY_IMPLEMENTATION_PLAN.md).

## Required measurements

Every policy decision writes a machine-readable trace containing:

- benchmark revision, model revision, variant, task, episode, and seed;
- simulation step and policy-call index;
- exact agent-view and wrist-view frames supplied to the model;
- proprioception and language instruction;
- the complete predicted 16-step chunk;
- selected prefix length or `STOP` decision;
- every executed action and resulting state;
- success predicates before and after execution;
- injected-failure type, time, and parameters;
- failure detection, recovery start, and recovery completion times.

Primary benchmark metrics:

- full-task success rate;
- subtask completion rate;
- action efficiency and total executed steps.

Primary recovery metrics:

- failure detection precision, recall, and latency;
- conditional recovery success, `P(final success | injected failure)`;
- recovery attempts per failure and false-recovery rate;
- time and actions from failure to recovered state;
- post-success reactivation rate;
- steps executed after first success;
- empty-location revisit rate.

The first-success step and final-success state must both be retained. A rollout
that succeeds temporarily and then destroys that success is not equivalent to a
stable success.

## Evaluation order

1. Validate GR00T observation/action conversion on recorded data.
2. Run one deterministic task and verify frame/action traces manually.
3. Run a one-trial-per-task pilot for A0-H16 and A0-H8.
4. Run the frozen full A0 benchmark.
5. Create and freeze the one shared GR00T-RC checkpoint for A1 onward.
6. Implement A2 and B without any recovery state or recovery loss.
7. Freeze B before designing and evaluating B-retry, Ours, and Oracle.
8. Run paired statistical analysis using identical task/seed/failure tuples.

Do not inspect full-test outcomes while tuning. Development choices are made on a
separate development set, then frozen before the final evaluation.
