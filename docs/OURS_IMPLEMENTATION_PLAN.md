# Experiment Ours search and implementation freeze

This document freezes the research protocol for Ours before any Ours outcome is
measured on the final benchmark. It freezes the admissible search space rather
than pretending that the winning architecture is known in advance.

## Identity and research claim

- Experiment: `Ours`
- Working variant: `GR00T-RC-SparkVLA-counterfactual-temporal-recovery`
- Parent baseline: B at commit
  `32838a4d42af8c8e4616f12539c9a5ef4d9c618f`
- Control baseline: B-retry artifact commit
  `16ff826c7da577ec9ca9be2b311c545568ca5171`
- Frozen low-level policy: the byte-identical A1 `GR00T-RC` checkpoint
- Frozen execution selector: the byte-identical B selector checkpoint
- Runtime protocol: continuous physical timeline with no restore, teleport, or
  reset inside an episode

The hypothesis is that a compact temporal controller can learn when B has
entered a recoverable failure state and which live-state recovery option to
invoke. Privileged simulator state and goal predicates may supervise offline
data generation, but they are never controller inputs at inference time.

The working method is **counterfactual temporal recovery (CTR)**:

1. retain a causal memory of frozen GR00T context, proprioception, proposed and
   executed actions, B scores, STOP history, and previous recovery choices;
2. estimate calibrated subtask progress and failure belief from that history;
3. score a fixed set of live-state recovery options;
4. learn option preferences from counterfactual simulator branches during
   training; and
5. execute only the selected option in the uninterrupted evaluation rollout.

The final paper name is chosen only after the winning method is frozen. The
working name must not be presented as a result or novelty claim.

## Relationship to current methods

The search is informed by, but must remain distinguishable from:

- [FPC-VLA](https://arxiv.org/abs/2509.04018), which uses a VLM supervisor and
  action correction/fusion;
- [CycleVLA](https://arxiv.org/abs/2601.02295), which uses a VLM planner,
  subtask backtracking, reverse execution, and MBR sampling;
- [Failing Forward / AFIL](https://arxiv.org/abs/2605.08434), which learns a
  failure action generator for negative diffusion guidance;
- [FLARE](https://arxiv.org/abs/2608.26645), which combines retry augmentation,
  an online MLLM monitor, and a reset-skill library; and
- [SparkVLA](https://arxiv.org/abs/2608.16172), whose unified STOP/prefix
  execution is already isolated as B.

CTR's intended distinction is simulator-privileged counterfactual option
distillation into an outcome-blind temporal executive. It uses no online MLLM,
does not alter the frozen VLA or B weights, requires no state restore at
deployment, and reasons over failure, progress, and recovery completion in one
causal memory.

Published numbers from different backbones, tasks, resets, or evaluation
protocols are never mixed with local results. Superiority claims require a
same-protocol implementation or an explicitly qualified comparison.

## Frozen evaluation boundary

Ours inherits byte-for-byte from B:

- A1 and B checkpoint files and their SHA-256 values;
- canonical plans, ordering, plan hashes, and STOP confirmation window;
- H8/H16 candidate masks and action conversion;
- benchmark construction, initial states, cameras, control rate, failure
  injections, post-success window, and total episode budget;
- deterministic policy-call seed derivation; and
- trace, resume, merge, and paired-report contracts.

Ours may change only decisions made by the recovery controller. At inference it
may observe:

- the frozen GR00T context produced from current RGB views, language, and
  proprioception;
- a bounded causal history of those contexts;
- current and onset anchor contexts;
- proposed action chunks and actually executed action summaries;
- B candidate scores, selected prefix, pending/confirmed STOP state;
- subgoal index, normalized elapsed budget, and its own prior decisions.

At inference it must not observe:

- goal predicates, object poses from the simulator, or success state;
- condition, injection type, injection time, or failure label;
- future observations, branch returns, episode outcome, or oracle trigger;
- demonstration identity or held-out case/trial identity; or
- serialized simulator state.

The evaluator may compute forbidden values only after decisions for measurement
and trace auditing. No option can restore or teleport state. The total control
step budget is identical to B and B-retry.

## Recovery option set

The search is restricted to the following auditable live-state options:

1. `ACCEPT_B`: execute or commit B's current decision unchanged;
2. `REOBSERVE`: discard a pending proposal and obtain a new deterministic policy
   sample without consuming a simulator step, subject to a fixed loop guard;
3. `RETRY_CURRENT`: retain the current subtask, create a new live-state anchor,
   clear STOP confirmation, and continue;
4. `BACKTRACK_ONE`: select the immediately preceding canonical subtask, create a
   new live-state anchor, and continue without reversing or restoring state;
5. `ADVANCE`: accept subtask completion and advance exactly one canonical
   subtask; and
6. `CONSENSUS_PREFIX`: choose one of a fixed number of deterministic GR00T chunk
   hypotheses using only the learned option/value head and pairwise action
   consensus.

Invalid options are masked. In particular, `BACKTRACK_ONE` is invalid on the
first subtask, `ADVANCE` is valid only at a confirmed STOP boundary, and all
zero-step options share one finite guard. Recovery attempts remain bounded.

## Counterfactual training data

Training data has three sources:

1. the existing leakage-audited B demonstration split for successful temporal
   progress supervision;
2. new B rollouts with deterministic perturbations and seed bases reserved for
   training; and
3. counterfactual branches created from selected training-only simulator states.

For a counterfactual state, the data generator may snapshot the simulator,
evaluate each valid option from that same state, and use predicates or future
return to label reachability, progress, time-to-recovery, and option preference.
Snapshotting is a training-data operation only. The saved simulator state is
never a model input and the final evaluator has no branch/restore code path.
Every option is evaluated for the same simulator-step and policy-call horizon;
confirmed STOP transitions continue into the next subtask rather than ending a
branch early. `ADVANCE` labels require a causally pending STOP, matching the
runtime option mask.

Counterfactual option generalization is audited on a deterministic episode
holdout drawn only from training base seeds. With 60-case corpora, episode
indices congruent to 4 modulo 5 form a 12-episode option-validation split and
the other 48 episodes remain eligible for option-head fitting. This split is
distinct from the rollout development seeds used in R1--R4.

Seed partitions are immutable:

| Partition | Base seeds | Permitted use |
| --- | --- | --- |
| train | `10007`, `11007`, `12007` | fitting and counterfactual labels |
| development | `20007`, `21007`, `22007`, `23007` | disjoint staged model selection described below |
| implementation smoke | `30007` | deterministic contract checks only |
| final held-out | `7` | one frozen H16/H8 evaluation after all gates |

Each episode still derives its seed as `base_seed + trial`. No final-seed Ours
outcome, trace, frame, predicate, or metric may enter training or model
selection. Existing B/B-retry final artifacts are comparison targets, not Ours
training corpora.

All generated samples record source case, seed partition, snapshot provenance,
valid option mask, per-option rollout seed, labels, feature hashes, and the
frozen parent checkpoint hashes. Splits are checked by code, not convention.

## Model search space

The frozen search varies only the following components:

- temporal encoder: linear probe, MLP window, GRU, or causal Transformer;
- context window: `4`, `8`, or `16` policy decisions;
- option head: behavior classification or counterfactual pairwise value ranking;
- detector loss: weighted BCE or focal loss with an auxiliary time-to-failure
  target;
- memory objectives: progress, failure onset, recovery completion, and temporal
  consistency;
- consensus hypotheses: `1`, `4`, or `8`, enabled only after a failure trigger;
- bounded recovery attempts: `1` or `2`; and
- thresholds selected by development calibration under the constraints below.

The frozen default neural budget is context width 2,048 projected to width 256,
at most four temporal layers, eight attention heads, and at most 15 million new
parameters. The A1 and B parameters remain frozen. Mixed precision and cached
float16 contexts are required on the 16 GiB GPU.

After the corrected single-seed v3 search exposed episode memorization, the
multi-seed v5 R0 search is pre-registered as six lower-capacity models: linear
H4/W32, linear H8/W64, MLP H4/W32, MLP H8/W64, GRU H8/W64, and Transformer
H8/W64. They use fusion dropout 0.25--0.50, weight decay 0.05--0.10, and
1,000--1,500 steps. The option objective combines strict-winner classification
(weight 1.0), pairwise ranking (0.25), and normalized return regression (0.10),
so absolute branch-return scale cannot dominate the discrete decision. All
three train seeds are pooled, while the modulo-5
episode holdout from every seed remains excluded from fitting. A recovery
margin is calibrated on that holdout at false-recovery rate at most 5%, and the
failure detector uses the same 5% false-positive constraint on those train-seed
held-out episodes. R0 ranks models by beneficial recovery rate at that
operating point, then recovery
recall, decision regret, balanced option recall, and registered id. This matrix
is fixed before any v5 result is observed. Because the bounded controller emits
far fewer redundant STOPs than the retired controller, seeds `11007` and
`12007` use STOP stride 1 and at most eight states per episode; the corrected
seed-`10007` corpus uses its already frozen stride 32 / eight-state sampling.
This final density choice was made after seed-`11007` pilots at strides 32 and 4
yielded only 34 and 42 labeled states, respectively, and before any v5 model was
produced. Both superseded pilots remain negative sampling evidence.
Training batches balance the Cartesian seed/winning-option strata, so the
pathological high-STOP seed `10007` cannot dominate merely by contributing more
states. Non-option live rows are balanced across train seeds as well.

## Trial-and-error protocol

Search uses deterministic successive halving, as amended prospectively after
the original CTR family closed R1 without a valid candidate:

1. **R0 — offline sanity:** reject models that fail split, replay, calibration,
   or synthetic temporal tests.
2. **R1 — original CTR breadth (closed):** at most 12 registered variants, one
   development trial over all 60 cases at base seed `20007`.
3. **R1b — residual breadth:** seven prospectively registered variants, one
   development trial over all 60 cases at base seed `22007`.
4. **R2b — residual mechanism:** top four Pareto-valid variants, three trials
   over all 60 cases at base seed `23007`.
5. **R3b — confirmation:** top two variants, all ten trials over all 60 cases
   at base seed `21007`.
6. **R4 — freeze:** select one architecture, checkpoint, thresholds, option
   masks, retry bounds, and hypothesis count. Exact ties choose the earlier
   registered variant and earlier checkpoint.
7. **R5 — held-out:** run the frozen method once for 600 H16 and 600 H8 episodes
   at final base seed `7`.

A variant is Pareto-valid only if it improves development task-macro subtask
success over B-retry, keeps false recovery at or below 5%, and uses no more than
20% extra mean policy calls or simulator steps. Ranking is lexicographic:

1. largest lower bound of the paired 95% bootstrap interval for task-macro
   subtask-success delta against B-retry;
2. conditional recovery success;
3. strict final success;
4. lower false-recovery rate; and
5. lower executed steps and policy latency.

If no candidate passes, Ours is not run on the final split. The failure is
reported and the search protocol must be amended in a new, explicit commit;
final outcomes can never justify a retroactive amendment.

R1 slots `r1-00` through `r1-04` are already consumed by the recorded initial
and corrected-option attempts, including the interrupted `r1-04` negative run.
The seven remaining slots are fixed before v5 R0 results: offline ranks 1--4
each receive one C1 run; ranks 1 and 2 also receive C4; and rank 1 receives one
C1 run with its calibrated option margin increased by 0.05. All use the
checkpoint-calibrated completion and failure thresholds, one bounded recovery
attempt, and minimum elapsed time 75. The rank labels are determined solely by
the train-seed option audit and cannot be reordered using R1 outcomes.

## Prospective residual-over-B-retry amendment

The original CTR family completed all seven remaining R1 slots at seed `20007`.
Every model completed 5/537 pooled subtasks with task-macro rate 0.992657%,
versus 9/537 and 1.619991% for paired B-retry.  Their common paired delta was
-0.595960 percentage points; all bootstrap intervals excluded zero below.
False-recovery rates ranged from 0% to 20%, and none was Pareto-valid.  These
results close the original R1 family: no candidate advances to its planned R2.

Trace comparison exposed a structural cause.  CTR's abstention path inherited
B's confirmed-STOP advance, so it silently discarded B-retry's one bounded
retry.  Varying architecture, option margin, and hypothesis count could not
repair a controller whose default policy was weaker than its control.

Before using any newly added development seed, the revised working method is
frozen as **counterfactual residual recovery over B-retry**:

1. B-retry remains the exact default policy.  Until the second consecutive STOP,
   and after B-retry's bounded retry is spent, the learned controller abstains.
2. At the first confirmed STOP after at least 75 causal simulator steps, the
   controller compares B-retry's `RETRY_CURRENT` value with four physically
   distinct overrides: `REOBSERVE`, `BACKTRACK_ONE`, `ADVANCE`, and
   `CONSENSUS_PREFIX`.
3. `ACCEPT_B` is excluded from residual override training because at a confirmed
   STOP it is physically identical to `ADVANCE`; retaining both would create
   duplicate actions with inconsistent shaped targets.
4. An override is permitted only when its learned advantage over
   `RETRY_CURRENT` clears an episode-held-out margin calibrated to at most 5%
   false override.  Otherwise execution is byte-equivalent to B-retry.
5. Re-observation and consensus consume the retry opportunity they replace;
   no override restores simulator state or extends the global step budget.

Residual labels use only the already frozen training rollouts at seeds `10007`,
`11007`, and `12007`.  Every eligible confirmed STOP is sampled with stride 1,
at most eight states per episode, producing a pre-audited upper bound of 426
states (363 + 41 + 22).  Each valid option gets the same 75-step/24-call
continuation budget.  The six low-capacity R0 architectures and all optimizer
settings remain identical to v5; models are reranked only by held-out residual
advantage against `RETRY_CURRENT`.

The seed roles are now fixed as follows.  Seed `20007` is retired after the
original-family diagnosis.  Seed `22007` is used once for R1b breadth with a new
paired B-retry control.  Seed `23007` is reserved for three-trial R2b mechanism
tests.  Seed `21007` remains unseen and is reserved for the ten-trial R3b
confirmation.  Seed `30007` remains implementation-smoke-only and seed `7`
remains final-held-out-only.

The seven R1b identities are frozen before seed `22007` is consumed: residual
offline ranks 1--4 with option-only gating and C4; rank 1 additionally receives
the checkpoint failure gate at C4, a C1 consensus ablation, and a C4 margin
increased by 0.05.  C4 is primary because its rollout matches the four-hypothesis
counterfactual consensus label.  If no R1b candidate improves paired task-macro
success while satisfying both safety and overhead caps, the residual family
stops without touching R2b, R3b, or the final seed.

## Training objectives and calibration

The learned heads predict:

- failure probability and time-to-failure;
- monotonic subtask progress and completion probability;
- recovery-active and recovery-complete probabilities; and
- masked value for every valid recovery option.

Losses include calibrated classification, ordinal progress, pairwise option
ranking, value regression, and temporal-consistency regularization. Failure
classes and tasks are balanced at the episode level. Model selection uses the
frozen development partition and minimum registered selection objective;
earliest checkpoint breaks ties.

Thresholds are selected on development data with temperature scaling. Failure
triggering requires hysteresis/confirmation, recovery completion has a separate
threshold, and a cooldown prevents repeated triggers from one event. Exact
values and calibration curves are checkpoint provenance.

## Required comparisons and ablations

The final report includes paired comparisons against A2, B, and B-retry, plus:

- detector only with no recovery action;
- selective retry without counterfactual option ranking;
- option ranking without temporal memory;
- no progress auxiliary loss;
- no recovery-completion head;
- no consensus sampling;
- behavior-classification versus counterfactual value targets; and
- Oracle trigger with the same frozen recovery policy, reported only after Ours.

Primary recovery metrics are those in
`ROBOCEREBRA_RECOVERY_PROTOCOL.md`. The report additionally includes parameter
count, peak VRAM, policy latency, calibration error, per-failure-type results,
paired confidence intervals, and compute/data-generation cost.

## Delivery gates

1. **O0 — evidence/search freeze:** this document, current-paper audit, parent
   SHAs, seed partitions, and hardware budget.
2. **O1 — typed contract:** config, prohibited-signal validation, option masks,
   deterministic seeds, and tests.
3. **O2 — compact trace corpus:** expose and hash frozen contexts without using
   final Ours outcomes.
4. **O3 — counterfactual generator:** training-only snapshot branching with
   replay and leakage audits.
5. **O4 — model/training:** temporal detector, memory, option/value heads,
   calibration, resumable checkpoints, and provenance.
6. **O5 — runtime:** frozen B server plus bounded recovery decisions and full
   machine-readable state transitions.
7. **O6 — development search:** R0–R4 with a complete trial registry, including
   rejected variants and negative results.
8. **O7 — final evaluation:** outcome-blind deterministic smoke, then the single
   frozen H16/H8 run and reviewer report.
9. **O8 — publication:** GitHub reviewer bundle plus model/dataset branches and
   verified SHA-256 inventories.

Before O7, tests, lint, source and split audits, checkpoint hashes, deterministic
replay, prohibited-signal tests, and continuous-state tests must all pass.
