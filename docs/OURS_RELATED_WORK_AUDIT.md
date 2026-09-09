# Ours related-work and claim audit

Audit date: 2026-09-09. This note fixes the nearest-method comparison before
the final Ours benchmark. It is a claim boundary, not a leaderboard claim.

## Pinned primary sources

- [RoboCerebra v2](https://arxiv.org/abs/2506.06677v2) defines the long-horizon
  benchmark, hierarchical System-1/System-2 setting, and paper-style subtask
  success metric used as context for this project.
- [SparkVLA v1](https://arxiv.org/abs/2608.16172v1) unifies STOP and action-prefix
  selection. Its paper reports 47.12% on its own RoboCerebra protocol. The local
  B branch is an auditable adaptation because the paper's complete runtime and
  checkpoint were unavailable; local continuous/no-restore outcomes are not an
  official SparkVLA reproduction.
- [CoRe v1](https://arxiv.org/abs/2608.14822v1) is training-free at inference:
  it detects deviation, synthesizes a continuation from an earlier viable
  state, probes restoration subsets, and physically realigns the robot/scene.
- [Dream2Fix v1](https://arxiv.org/abs/2603.13528v1) generates more than 120,000
  paired failure/correction samples with a learned world model and fine-tunes a
  VLM to emit recovery trajectories.
- [SAFE v2](https://arxiv.org/abs/2506.09937v2) learns a lightweight failure
  probe from frozen VLA internal features and calibrates time-varying alerts
  with functional conformal prediction. It detects failures but does not learn
  or execute the recovery option selected after an alert.
- [Hide-and-Seek v1](https://arxiv.org/abs/2605.30834v1) localizes failure
  signals from rollout-level labels using inter- and intra-trajectory
  contrastive objectives. Its LIBERO, VLABench, and real-robot results concern
  detection accuracy and timeliness rather than closed-loop recovery return.
- [SAFECAST v1](https://arxiv.org/abs/2608.04246v1) improves hidden-state
  failure probes under visual and language distribution shift using contrast
  sets and calibrated detection. CTR instead holds the benchmark distribution
  fixed and learns the value of bounded physical recovery actions.
- [FailSafe v1](https://arxiv.org/abs/2510.01642v1) uses a 7B VLM and generated
  failure-action data for diagnosis and recovery in ManiSkill. Its reported
  gains use different policies, tasks, and recovery machinery and therefore
  are not direct numerical baselines here.
- [RepairVLA](https://openreview.net/pdf?id=b388d15b4ca665e9e03ce7c84942760c9dfcb044.pdf)
  trains a high-level VLM to decompose, check, and synthesize action-primitive
  repairs from generated failures. Its anonymous-review results remain context,
  not an authoritative or protocol-matched leaderboard target.
- [LIBERO-RECOVER v1](https://arxiv.org/abs/2609.05178v1) is a newly released
  benchmark of 2,178 naturally occurring failure scenarios across action retry,
  action adaptation, object-state recovery, and environmental recovery.  It
  reports recovery success, degradation, and cross-task consistency.  Its
  initial-state, task, time-limit, and recovery-scenario protocol differs from
  this repository's RoboCerebra continuous/no-restore protocol, so its model
  numbers are context rather than comparable baselines.
- [Manipulation Benchmark Audit v1](https://arxiv.org/abs/2606.04233v1)
  identifies shortcut solvability, statistical insignificance, creeping
  overfitting, and data-source dependence as common failure modes in robot
  manipulation claims. Its diagnosis motivates this project's paired seeds,
  task-stratified confidence intervals, immutable negative variants, and strict
  separation of train, development, and final outcomes.

## Nearest value, chunk, and recovery methods

- [V-VLAPS v3](https://arxiv.org/abs/2601.00969v3) trains a value head over a
  frozen VLA's latent representation from offline Monte Carlo returns and uses
  it to guide MCTS at inference. This is the closest value-guided VLA method.
  BranchQ must therefore distinguish direct candidate ranking and execution,
  SMDP TD credit, its joint subgoal/hypothesis/prefix lattice, and absence of
  runtime simulator search; “a value function for a frozen VLA” is not novel.
- [Q-chunking](https://arxiv.org/abs/2507.07969) supplies unbiased n-step TD
  formulations for temporally extended action chunks in offline-to-online RL.
  BranchQ's claim cannot be Q-learning over variable action prefixes alone.
- [Conservative Q-Learning](https://arxiv.org/abs/2006.04779) regularizes
  offline value estimates against overestimating out-of-distribution actions.
  Its conservative objective is an ingredient, not a BranchQ contribution.
- [SPIBB](https://arxiv.org/abs/1712.06924) formalizes safe improvement by
  constraining poorly supported decisions to a baseline policy. BranchQ's
  exact-B candidate and minimum-support fallback are related to this principle;
  confidence fallback must not be described as the first safe policy
  improvement method.
- [CycleVLA](https://arxiv.org/abs/2601.02295) adds progress prediction,
  planner-driven backtracking, and model-based refinement to VLA execution.
  BranchQ differs only if recovery sequences emerge from learned continuation
  values rather than a separately defined planner/state machine.
- [FLARE](https://arxiv.org/abs/2608.26645) separates Retry and Reset responses
  under an online multimodal monitor. It is a direct fixed-recovery taxonomy
  comparator for the CTR ablation.
- [Bellman-Guided Retrials](https://arxiv.org/abs/2406.15917) changes strategy
  across repeated attempts using Bellman-guided value reasoning. Its setting is
  not the same physical VLA SMDP, but it prevents a broad claim that Bellman
  values have never been used to guide retry behavior.

## Distinction of the retired CTR method

CTR keeps the A1 VLA and B selector frozen. Simulator snapshots are used only
to label same-state, common-random-number option returns on reserved train
seeds. The outcome-first target ranks final success, completed subtasks, and
goal-predicate progress without step/call bonuses, while every branch is capped
by the source episode's remaining global step budget. At runtime a small causal
temporal controller selects among six fixed, auditable live-state options. It
receives no predicates, failure injections, future observations, serialized
simulator state, world model, or state-restoration operation.

The current technical distinction is therefore not “counterfactual recovery”
alone. It is the combination of:

1. same-state, equal-budget counterfactual supervision for a fixed option set;
2. direct online option execution without rewind or physical scene restoration;
3. a frozen VLA plus a small temporal value/classification head; and
4. safety-constrained selective intervention, with failure and recovery margins
   calibrated under a 5% false-intervention ceiling.

CTR v9 and all seven seed-22007 R1b variants are now negative ablations. The
best variant changes the outcome by only +1/537 completed subtasks, its paired
task-macro confidence interval includes zero, and it achieves 0/60 final
success. It must not be presented as the working method.

## Closest methods to the active program-induction hypothesis

- [RecoveryChaining](https://arxiv.org/abs/2410.13979) uses hierarchical RL to
  learn a separate recovery policy with nominal controllers as temporally
  extended handoff options. A two-policy architecture, failure-conditioned
  activation, and learned return-to-nominal behavior are therefore not new.
- [Self-Improving VLAs via Residual RL](https://arxiv.org/abs/2511.00091)
  freezes a VLA, trains lightweight residual RL specialists on base-policy
  failure regions, and distills their successful trajectories. Residual action
  learning and base-policy probing are required baselines, not contributions.
- [RePO-VLA](https://arxiv.org/abs/2605.09410) learns from success, failure, and
  corrective trajectories using a progress-aware semantic value and value-
  conditioned refinement. Recovery-driven VLA optimization is not new.
- [Policy-Conditioned Counterfactual Credit](https://arxiv.org/abs/2606.05263)
  uses deletion and other interventions for counterfactual credit in long-
  horizon language-agent RL. Broad claims about first deletion-based policy
  credit are invalid even though its setting and estimator differ.

- [VLA-ATTC](https://arxiv.org/abs/2605.01194) learns a relative action critic
  for pairwise selection among inference-time candidates. MOSAIC-VLA must beat a
  relative-critic baseline and cannot claim that relative action comparison is
  new. VLA-ATTC does not, from its published description, estimate the mixed
  physical effect of ordered interventions against all factorial ablations.
- [Selected Diffusion Noise](https://arxiv.org/abs/2606.14084) treats diffusion
  noise as a test-time control variable and selects separated, smooth action
  candidates. Therefore noise steering alone is not a contribution. MOSAIC uses
  it only as a continuous proposal family for adaptive programs.
- [TTT-VLA](https://arxiv.org/abs/2606.03127) optimizes a latent prompt at test
  time using an auxiliary self-supervised task. MOSAIC neither updates the VLA
  nor uses a proxy task at deployment; its labels are paired physical outcomes.
- [Mostly Harmless VLA Steering](https://arxiv.org/abs/2606.12299) uses
  conformalized improvement prediction for closed-loop language steering. A
  confidence fallback is therefore only a safety implementation detail here,
  not a novelty point.
- [Null Counterfactual Factor Interactions](https://arxiv.org/abs/2505.03172)
  defines object interaction through null counterfactual dynamics and improves
  hindsight relabeling. It makes a broad “first counterfactual interaction in
  robot learning” claim invalid.
- [Factorial causal effects](https://academic.oup.com/jrsssb/article/77/4/727/7040593),
  [dynamic treatment effects](https://arxiv.org/abs/1805.09397), and
  [Q/A-learning for dynamic regimes](https://arxiv.org/abs/1202.4177) establish
  that factorial interaction contrasts and sequential treatment regimes are
  classical. The paired cross-temporal curvature quartet is therefore only a
  mechanism diagnostic; neither its formula nor the interaction concept is a
  MOSAIC novelty claim.

## Active RESOLVE-VLA claim boundary

BranchQ remains a history-complete value-learning baseline. The factorial
curvature quartet remains a diagnostic. MOSAIC is the deletion-minimal credit
and certification layer. The active hypothesis is a dedicated residual recovery
transformer trained with a three-world reachability Bellman system: learned
recovery continuation `R`, current-macro deletion `D`, and pure frozen baseline
`B`. Its counterfactual rescue bottleneck is positive only when `R` beats pure B
and the current adaptive macro is necessary relative to D. Final programs are
then audited with MOSAIC's `d+2` paired arms.

This is a candidate contribution, not a novelty assertion. It is rejected
unless the coupled Bellman operator survives theoretical comparison with causal
policy-gradient/dynamic-regime methods and its compute-matched recovery policy
beats PLD-style residual SAC, RecoveryChaining-style handoff, RePO-style
refinement, standard actor-critic, BranchQ, and sequence-only MOSAIC on held-out
physical outcomes.

## Claim rules

- “Better than A2, B, or B-retry” requires paired results under this repository's
  identical continuous/no-restore evaluator and frozen seed protocol.
- “Better than a paper” is permitted only for a genuinely matched public
  protocol. Otherwise paper numbers are contextual and the report must state
  the mismatch.
- LIBERO-RECOVER's recovery success/degradation/consistency metrics may be
  reproduced on this repository's outcomes, but their names do not make the
  resulting values cross-benchmark comparable.
- A large point estimate is insufficient: the primary comparison uses the
  paired task-macro delta and its 95% bootstrap interval, plus recovery safety,
  compute, and stability metrics.
- Negative variants, interrupted pilots, model/data hashes, and seed usage stay
  in the artifact trail; they may not be removed to improve the narrative.
