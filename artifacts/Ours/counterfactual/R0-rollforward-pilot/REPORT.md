# Ours closed-loop counterfactual R0 pilot

This is train-only data-quality evidence, not a development or held-out
benchmark result.

## Protocol

- base seed: `10007`
- source: completed 60-episode live H16 rollout
- candidate: B STOP proposal at or after 75 causal subgoal steps
- maximum selected states: one per episode
- branch horizon: 75 simulator steps, 24 policy calls
- consensus: four deterministic GR00T+B hypotheses
- runtime restore: false
- final/development seeds consumed: none

All options for a selected state begin from the same training-only simulator
snapshot. MuJoCo restoration uses absolute tolerance `1e-12`. Predicate replay
mismatches are excluded before branching: 176/25,369 rows (0.6938%), below the
frozen 5% abort limit.

## Data result

- corpus files: 60
- labeled states: 59
- valid option targets: 304
- strict preferences: 57/59 (96.61%)
- ties: 2
- mean best-option margin: 0.7175
- branch SHA-256: `1b539c88035ac46618e60106846f6adf08c75aa7435ff999e913369cf68478de`

Strict winners:

| Option | States won |
| --- | ---: |
| `ACCEPT_B` | 44 |
| `ADVANCE` | 4 |
| `BACKTRACK_ONE` | 3 |
| `RETRY_CURRENT` | 3 |
| `CONSENSUS_PREFIX` | 2 |
| `REOBSERVE` | 1 |

The diversity gate passes: all six frozen recovery options are optimal in at
least one sampled state. The dominant `ACCEPT_B` class is retained rather than
downsampled because learning when not to intervene is part of the claimed
controller behavior. The full generator therefore scales this exact protocol
without changing its reward or option definitions.
