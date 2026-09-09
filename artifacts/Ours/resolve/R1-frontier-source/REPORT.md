# RESOLVE R1 executable-frontier source

Status: completed train-only source and rejected supervised warm-start,
2026-09-09. This artifact does not make a novelty or benchmark claim.

## Registered source

- Source: exact demonstration XML/state, executable expert continuation, and
  exact frozen B from the same anchor.
- Selection: train split, anchor offset 32, SHA-256 ordering seed 89007.
- Runtime: GR00T-RC revision
  `5d2e1e361bf65aabbe4d18179515f5a10936cc96` with registered B selector.
- Selected anchors: 599.
- Valid paired arms: 597 across 416 source-demonstration groups.
- Exclusions: 2. In both, executable reset post-processing already satisfied
  the registered predicate; neither was used as a treatment outcome.
- Corpus manifest SHA-256:
  `1edf406bbe0aac1355450b5f0bd6f54670fe5132d37dd790e42ade1490c2b890`.

| Paired physical outcome | Anchors |
| --- | ---: |
| Expert succeeds, B fails | 75 |
| Expert fails, B succeeds | 101 |
| Both succeed | 74 |
| Both fail (unresolved) | 347 |

The 75 positive rescues cover 70 source groups: 53 coffee-table anchors, 13
kitchen-table anchors, and 9 study-table anchors. The complete valid source is
467/92/38 anchors in those scenes respectively.

## Throughput diagnosis

A single simulator feeds the GPU intermittently because MuJoCo execution,
observation construction, and policy inference alternate. Direct timing on the
same 16-anchor probe gave 31 seconds with four workers and 20 seconds with eight
workers; the eight-worker run reached a sampled 94% GPU utilization and about
10.1 GB allocated VRAM. Eight workers were therefore retained. The runner now
uses 16 dynamically scheduled micro-shards by default, preventing long-tail
episodes in one static shard from idling the other CPU workers and the GPU.

## Supervised warm-start falsification

The preregistered model used a width-256, four-layer recovery transformer,
4,000 optimizer steps, batch size 32, and seed 84007. Only expert-only pairs
were recovery targets. B-success pairs targeted exact handoff, and all 347
both-failure pairs were excluded rather than mislabeled as negative recovery.
The split was group-disjoint: 204 train / 46 development decisive pairs, with
14 positive development pairs.

The warm-start failed its fixed selection gate:

| Development metric | Required | Step 4,000 |
| --- | ---: | ---: |
| Positive intervention rate | >= 0.50 | 0.50 |
| False intervention rate | <= 0.10 | 0.28125 |
| Positive action MAE / B MAE | < 1.00 | 1.23106 |

No trained checkpoint was accepted; `best_step=0`, which is exact B. Increasing
the positive source five-fold therefore did not rescue behavior cloning. This
is a negative result and closes threshold/architecture tuning for this stage.
The next admissible experiment is paired physical R/D/B reinforcement learning,
not another classifier sweep.
