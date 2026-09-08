# Ours closed-loop counterfactual R0 corpus

This is train-only counterfactual supervision, not a development or held-out
benchmark result.

## Protocol

- base seed: `10007`
- source: completed 60-episode live H16 rollout
- candidate states: B STOP proposals at or after 75 causal subgoal steps
- maximum selected states: eight per episode, separated by a 32-decision stride
- branch horizon: 75 simulator steps, 24 policy calls
- consensus: four deterministic GR00T+B hypotheses
- all valid options start from the same MuJoCo snapshot
- restoration tolerance: absolute `1e-12`
- final/development seeds consumed: none

Predicate replay mismatches are rejected before branching: 176/25,369 source
rows (0.6938%), below the frozen 5% abort limit.

## Corpus result

- corpus files: 60
- source samples: 25,369
- labeled states: 322
- valid option targets: 1,868
- strict preferences: 320/322 (99.38%)
- ties: 2
- mean best-option margin: 0.21784
- manifest SHA-256: `1f8d2040ee2ba9913608b6ad2003a6e4cfa681b62b31a57a4d6e3f9505ab075b`
- branch-log SHA-256: `373f494d8435f8a61e3a05a256276af9a1bb63a92f878f37b1bedf54fa74b399`

Strict winners:

| Option | States won |
| --- | ---: |
| `ACCEPT_B` | 302 |
| `ADVANCE` | 7 |
| `BACKTRACK_ONE` | 4 |
| `CONSENSUS_PREFIX` | 3 |
| `RETRY_CURRENT` | 3 |
| `REOBSERVE` | 1 |

All six frozen options win at least one state, so the option-diversity gate
passes. The naturally observed distribution is intentionally retained. Its
strong `ACCEPT_B` imbalance is addressed only in the training sampler and is a
reason to treat the option-head fit audit as diagnostic rather than evidence of
generalization.
