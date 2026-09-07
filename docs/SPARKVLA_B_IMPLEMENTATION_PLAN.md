# Experiment B implementation freeze

This document freezes the implementation boundary for experiment B before its
selector is trained or any held-out benchmark outcome is inspected.

## Identity and evidence

- Experiment: `B`
- Variant: `GR00T-RC-SparkVLA-style-execution`
- Parent baseline: A2 at commit `c9930fd0aa9ffec8d4fde817981680955ff9281f`
- Low-level checkpoint: the exact frozen A1 `GR00T-RC` checkpoint
- Hierarchy source: the exact canonical A2 subgoal plans and plan hashes
- SparkVLA repository: `huhuhushou/SparkVLA` at
  `ae90ec94ccc77a1e6745e4f5bcc94e05502d8cbf`
- Method specification: [SparkVLA arXiv:2608.16172v1](https://arxiv.org/abs/2608.16172v1)

The pinned repository contains no training code, evaluation code, or checkpoint.
The paper was published after that repository revision. B is therefore a
GR00T-RC adaptation and reimplementation of the published method, not an
official SparkVLA reproduction. Paper results must never be mixed with local B
results.

## Frozen scientific boundary

B changes one factor relative to A2: fixed H8/H16 execution and time-based
subgoal advancement are replaced by a learned unified decision over
`[STOP, prefix-1, ..., prefix-H]` at each policy boundary.

Everything else remains frozen:

- A1 model weights, LIBERO action contract, observations, task construction,
  initial states, success predicates, failure injections, control rate, rollout
  budget, post-success window, and test cases;
- A2 canonical plan text, order, source, and SHA-256 values;
- continuous execution with no in-rollout restore, teleport, or reset;
- no retry, failure detector, recovery state, recovery action, recovery loss, or
  oracle signal;
- no success predicate, goal state, injection label, or future observation is
  visible to the selector at evaluation time.

`STOP` advances to the next canonical subgoal. A selected positive candidate
executes exactly that many leading actions from the predicted chunk and then
re-observes. H8 masks candidates 9 through 16; H16 exposes all 16 prefixes. The
global A2 step budget remains the hard termination bound, but its 150-step
anchors no longer advance the active subgoal.

A confirmed STOP on the final subgoal terminates an unsuccessful rollout. If the
ordered goal has already been reached, the evaluator preserves the frozen
80-step reactivation window using zero-Cartesian controller holds with the last
gripper command. Those holds are traced separately, are not selector actions or
policy calls, and never feed success information back into selector masks. The
Random Disturbance/Mix injection schedule and related-object segment remain a
function of the original fixed control-step clock, not adaptive subgoal state.

## GR00T-RC adaptation

The paper uses separately parameterized pi-0.5 planner and executor branches.
B instead keeps the frozen GR00T-RC executor and adapts the selector interface:

- the canonical A2 subgoal replaces autoregressive subgoal generation;
- the frozen GR00T N1.7 `backbone_features` (width 2,048) provide current visual,
  language, and proprioceptive context;
- cached contexts and H16 proposals use float16 storage; online values take the
  same float16 round-trip before the float32 selector computation;
- the subgoal-onset context and ordered completed-subgoal history form the cached
  history-aware anchor;
- each action-prefix candidate is represented by its terminal action plus a
  learned length embedding;
- a learned STOP token, the anchor, current context, and all valid prefix tokens
  are scored jointly by a full-self-attention selector;
- only selector/adaptation parameters are trainable; the A1 checkpoint stays
  byte-identical.

Any departure forced by the GR00T backbone or the unpublished SparkVLA code is
recorded in the training manifest and reviewer report.

## Training contract

Ordinal labels follow the paper's Algorithm 1:

- candidate set `0..H`, with `0 = STOP`;
- successful-boundary jitter `k = 2`;
- unsuccessful-rollout ranking weight `0.1`;
- STOP-positive and near-boundary STOP-negative weight `3.0`;
- unsuccessful-rollout STOP weight `1.5`;
- pairwise ranking plus STOP-aware log-sum-exp loss with
  `lambda_stop = 1`;
- train seed `7`.

The available source supports successful demonstrations only. It does not expose
an authoritative goal-predicate contract for generating paper-equivalent failed
policy-rollout labels. B therefore trains on demonstrations and records
`policy_rollouts_included=false`; it does not synthesize failure labels. This is
an explicit adaptation/limitation, not an unreported substitute for the paper's
rollout corpus. No held-out B evaluation frames, actions, predicates, episode
outcomes, or selector traces may enter training. Demonstration counts, split
hashes, boundary provenance, feature-extractor revision, and checkpoint hashes
are required artifacts.

Source demonstrations with malformed, non-contiguous, or empty subgoal ranges
are excluded from selector supervision rather than repaired heuristically. Their
source indices, cases, and rejection reasons are retained in the dataset
manifest. They remain part of the already-frozen A1 low-level training history;
the exclusion applies only to B boundary supervision.

The paper mentions a short STOP confirmation window but does not publish its
length. B conservatively freezes it a priori at two consecutive decisions—the
smallest non-trivial window—records that choice as an adaptation, and never
tunes it on the 60 held-out cases.

## B0–B10 delivery sequence and gates

1. **B0 — evidence freeze:** pin the paper, repository revision, parent A2
   commit, and clearly label the GR00T-RC adaptation.
2. **B1 — experiment contract:** add typed configuration and freeze identity,
   candidate order, H8/H16 masks, STOP confirmation, and prohibited signals.
3. **B2 — source/leakage audit:** replay the A1 source and converted-dataset
   audits and reject exact held-out prompt overlap.
4. **B3 — boundary index:** strictly parse canonical annotations, replay A1's
   no-op filter, map boundaries, exclude malformed rows, and freeze train/dev
   splits. The resulting index has 967 usable episodes, 28 audited exclusions,
   and 203,410 samples; train/dev episode counts are 871/96.
5. **B4 — ordinal supervision:** implement deterministic Algorithm 1 target
   construction, valid-candidate masks, boundary jitter, and replay payloads.
6. **B5 — unified scorer:** implement anchor history, terminal-action plus length
   candidates, joint STOP/prefix attention, pairwise ranking loss, and STOP loss.
7. **B6 — frozen-A1 feature cache:** capture the exact last-valid 2,048-wide
   backbone token and proposed H16 chunk with per-episode atomic, resumable files
   bound to immutable index, A1-weight, seed, device, and batch-size hashes.
8. **B7 — selector training/freeze:** train only the selector, validate on the
   frozen development split, and write safetensors plus byte-level provenance.
9. **B8 — inference service:** combine byte-validated A1 and selector checkpoints
   behind the official GR00T wire protocol, seed diffusion independently for
   every episode, and return auditable decisions.
10. **B9 — continuous evaluator:** replace time anchors with confirmed STOP and
    learned prefixes; retain complete traces, resume/merge, and explicit
    `retry=false`, `recovery=false`.
11. **B10 — pilot and benchmark:** replay one deterministic trace, run the
    held-out pilot, freeze all settings, then run paired H8/H16 shards and build
    the reviewer report.

Before any held-out pilot, `make test`, `make lint`, the B contract audit, split
leakage audit, checkpoint provenance audit, and deterministic selector replay
must all pass.
