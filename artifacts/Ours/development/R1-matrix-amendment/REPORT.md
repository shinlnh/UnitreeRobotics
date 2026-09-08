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
