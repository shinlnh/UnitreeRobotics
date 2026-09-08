# Ours counterfactual R0 pilot

This artifact records the first train-only same-state option-branching pilot.
It is a rejected search result, not benchmark evidence and not a final result.

## Inputs and isolation

- train base seed: `10007`
- source rollout: `train-seed10007-full-c1-H16`
- source rows: 25,369 across 60 episodes
- selected B STOP states: 60
- counterfactual branches: 180
- final/development seeds consumed: none
- simulator restore at runtime: false

The generator rejected two earlier strict replay attempts before producing this
corpus. MuJoCo physical restoration is checked at absolute tolerance `1e-12`.
Delayed evaluator predicate snapshots are never used as labels when their replay
does not match: 175 of 25,369 rows (0.6898%) were skipped, below the frozen 5%
abort threshold.

## Result

The structural corpus/hash audit passed. The option-diversity audit failed:

- labeled states: 60
- strict preferences: 60/60
- distinct winning options: 1
- winner: `CONSENSUS_PREFIX` in 60/60 states
- `ACCEPT_B` mean return: -8.0
- `ADVANCE` mean return: -8.0
- `CONSENSUS_PREFIX` mean return: 0.1465

This pilot is intentionally rejected because it does not contain a learnable
state-dependent choice among recovery options. It must not be scaled or used to
support a paper claim. The next generator iteration must roll each high-level
option forward from the same live state, so `ADVANCE`, `RETRY_CURRENT`,
`BACKTRACK_ONE`, and consensus can receive distinct future returns.
