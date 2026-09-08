# Ours related-work and claim audit

Audit date: 2026-09-08. This note fixes the nearest-method comparison before
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

## Distinction of the working method

CTR keeps the A1 VLA and B selector frozen. Simulator snapshots are used only
to label equal-horizon option returns on reserved train seeds. At runtime a
small causal temporal controller selects among six fixed, auditable live-state
options. It receives no predicates, failure injections, future observations,
serialized simulator state, world model, or state-restoration operation.

The current technical distinction is therefore not “counterfactual recovery”
alone. It is the combination of:

1. same-state, equal-budget counterfactual supervision for a fixed option set;
2. direct online option execution without rewind or physical scene restoration;
3. a frozen VLA plus a small temporal value/classification head; and
4. safety-constrained selective intervention, with failure and recovery margins
   calibrated under a 5% false-intervention ceiling.

## Claim rules

- “Better than A2, B, or B-retry” requires paired results under this repository's
  identical continuous/no-restore evaluator and frozen seed protocol.
- “Better than a paper” is permitted only for a genuinely matched public
  protocol. Otherwise paper numbers are contextual and the report must state
  the mismatch.
- A large point estimate is insufficient: the primary comparison uses the
  paired task-macro delta and its 95% bootstrap interval, plus recovery safety,
  compute, and stability metrics.
- Negative variants, interrupted pilots, model/data hashes, and seed usage stay
  in the artifact trail; they may not be removed to improve the narrative.
