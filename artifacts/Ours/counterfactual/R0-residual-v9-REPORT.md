# R0 residual-v9 outcome-first report

Status: complete train-seed counterfactual collection and offline search. No
rollout development seed or final seed was used in this report.

## Corpus

Residual-v9 branches only from the first confirmed B-retry STOP at or after 75
elapsed subgoal steps. Alternatives share common random numbers, cannot restore
state at runtime, continue under one global episode budget, and are labeled by
physical outcome before any action-cost term. All three corpus audits pass,
including source abstention, confirmed-STOP, branch hash, common-random-number,
global-budget, replay, and outcome-first physical-alignment checks.

| Base seed | States | Branches | Strict winners | Physical-benefit states | Replay mismatch | Budget-truncated | Train/validation episodes | Positive episodes (validation) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10007 | 62 | 303 | 5 | 6 | 1.2632% | 0 | 48 / 12 | 5 (1) |
| 11007 | 72 | 353 | 5 | 5 | 2.6862% | 3 | 48 / 12 | 5 (1) |
| 12007 | 55 | 273 | 7 | 8 | 4.8687% | 1 | 47 / 13 | 7 (2) |
| **Total** | **189** | **929** | **17** | **19** | — | **4** | **143 / 37** | **17 (4)** |

The 19/189 physical-benefit rate is 10.05%. Residual-v8 exposed 8/189
(4.23%), so v9 exposes 2.375x the signal; because v9 changed both target order
and continuation horizon, this is not a causal attribution to either change.
The deterministic episode split is SHA-256 ranked within positive/non-positive
strata. It was frozen before v9 model training.

Final corpus bindings after the split amendment:

| Seed | Manifest SHA-256 | Branch inventory SHA-256 |
|---:|---|---|
| 10007 | `0ff983b6fdca787a1055220ef3c8e1fb152cf9ed8e41e31fc9451f14fb4032ac` | `ed991cecacac85d857ec4ad948736280e2359923d0c90a4d21a775506509cc65` |
| 11007 | `3206d44b1e3eb53935cdd0668318271bd0105cba63d5ac92816ddef6013732d3` | `bf2ab1dbed3f78f192c0d5a959f4f969453fc99a3ead2e1cdcae802c775a360f` |
| 12007 | `8a4e374e2872e4f20bc1b2c855961cb85d21f028bf3f78717456318242c96146` | `2c3736b63af5a8eba14113e7b085fdc6d0a1b79258aca2aa5ca1f4d6a7d61511` |

## Trial-and-error record

The positive-only sampler was rejected because raw false overrides were
96--100%. Balancing override and retry/tie examples reduced but did not solve
that failure (72--100%). Adding a binary intervention loss at weight 1.0 then
overcorrected: raw false overrides fell to 0--4%, but all six models suppressed
every positive. These checkpoints and logs remain retained as negative results.

The preregistered binary-weight sweep crossed `{0.05, 0.10, 0.25, 0.50}` with
six fixed architectures, producing 24 complete checkpoints. With all four
override actions enabled, none made a beneficial held-out decision at its safe
margin. A preregistered exhaustive audit then evaluated all 15 non-empty fixed
override libraries per checkpoint (360 configurations), without fitting more
parameters or observing a rollout development seed.

## Offline selection

The selected configuration is
`w005-v03-mlp-h8-w64-d025-advance+consensus`: MLP history 8, binary weight
0.05, allowed overrides `{ADVANCE, CONSENSUS_PREFIX}`, and calibrated margin
0.3125. Among 29 eligible held-out states it makes one beneficial decision,
one false decision (1/27 = 3.70%), and recovers 1/2 allowed strict positives.
This is only a sparse positive candidate, not evidence of task-level gain.

The top four registered configurations are carried unchanged into R1b on the
single development base seed 22007. R1b includes a paired B-retry control and
three preregistered rank-1 mechanism/safety ablations. If no candidate improves
paired task-macro success within safety and overhead caps, this family stops.

Registry bindings:

- exhaustive option-library registry SHA-256:
  `91aed949983b631532cf764fc6da92d44dea4cfc5819731b5128d584a895eb2f`
- 24-checkpoint aggregate weight registry SHA-256:
  `1c8fddc600f373ca7251d94d11f8a251024f883396c67cd08ebc78c545785fbe`
