# Ours R1 option-gate matrix amendment

This amendment was recorded after completing `r1-03` and before executing any
later option-aware R1 variant. It does not use final seed `7`.

## Observed registered variant

`r1-03-cf-v00-c1-m000` used R0 checkpoint `v00`, one hypothesis,
option-value margin 0, failure threshold 0.90, one recovery attempt, and the
75-step causal onset guard. Across the 60 development episodes at base seed
`20007`:

- task-macro subtask success: 1.3524% versus 1.6200% for paired B-retry;
- paired delta: -0.3056 percentage points;
- paired bootstrap CI95: [-0.9143, +0.1639] percentage points;
- false-recovery rate: 0/15 (0%);
- mean policy-call ratio: 0.6153x;
- mean simulator-step ratio: 0.7680x.

The candidate is Pareto-invalid because its primary metric is lower than the
control. The 0.90 failure pre-gate admitted only 15 recovery decisions, while
the counterfactual head itself had already been trained to compare accept and
recovery values. The safety and efficiency results support testing whether this
pre-gate is unnecessarily conservative; they do not support a performance
claim.

## Prospective remaining matrix

The eight remaining R1 slots test failure thresholds 0 and 0.50, option margins
0, 0.025, and 0.050, one versus four hypotheses, and all four registered R0
architectures. All other protocol values stay fixed. The script and exact
variant IDs are committed before those runs begin. The R1 total remains capped
at 12, including the three earlier registered controller variants.

## Subsequent generator audit

Before `r1-04` completed, source inspection found that the counterfactual
`ACCEPT_B` branch terminated on confirmed STOP while continuing recovery
options used the remaining branch horizon. This rewarded inaction through
lower step/call penalties and invalidated the option-value comparison. The
302/320 `ACCEPT_B` winner collapse was therefore causal evidence of a labeling
bug, not merely class imbalance.

`r1-04` was stopped at 18/60 episodes and is not a completed R1 candidate. Its
partial files are preserved locally with run-manifest, episode, and decision
SHA-256 values `2e994567...`, `3e3224e0...`, and `6a543720...`. The completed
`r1-03` files have corresponding hashes `6f54997d...`, `de516cfb...`, and
`bab5aab1...`. No final-seed data were inspected. Later R1 slots may use only a
regenerated corpus in which every option follows identical confirmed-STOP
continuation semantics.
