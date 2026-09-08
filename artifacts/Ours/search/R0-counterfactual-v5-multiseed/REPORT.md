# Ours R0 v5 multi-seed counterfactual search

This round tests whether additional train-seed coverage and stronger
regularization repair the single-seed option-generalization failure recorded
in R0 v3. Development seed `20007` and final seed `7` were not consumed.

## Data and protocol

The frozen VLA and selector were rolled out on base seeds `10007`, `11007`,
and `12007`. Candidate recovery actions were labeled from the same simulator
state using an equal continuation budget of 75 simulator steps and 24 policy
calls. Invalid `ADVANCE` actions were masked. The expanded corpus contains 508
labeled states, of which 467 have a strict winner:

| Seed | Labeled | Strict | ACCEPT | BACKTRACK | CONSENSUS | REOBSERVE | RETRY |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10007 | 322 | 300 | 170 | 67 | 15 | 25 | 23 |
| 11007 | 122 | 106 | 54 | 11 | 10 | 7 | 24 |
| 12007 | 64 | 61 | 37 | 6 | 3 | 11 | 4 |
| Total | 508 | 467 | 261 | 84 | 28 | 43 | 51 |

The `11007` and `12007` live corpora add 3,762 causal decision samples and
1,017 failure-positive samples. Their absence of completion positives is
retained rather than hidden: primary successful demonstrations supply the
completion classes, while these live rollouts contribute failure and option
coverage. Sampling is balanced across the Cartesian product of rollout seed
and winning recovery option so that the much longer seed-`10007` rollout
cannot dominate training.

All option validation is split by episode (`modulus=5`, `remainder=4`). Across
the three seeds this yields 395 labeled/360 strict training states and 113
labeled/107 strict validation states. Six low-capacity variants and all
hyperparameters were registered before their metrics were read.

## Held-out option audit

The selection rule is beneficial-recovery recall at calibrated false recovery
at or below 5%, then recovery recall, regret, balanced recall, and registered
identifier.

| Rank | Variant | Step | Top-1 | Balanced recall | Pairwise | Beneficial recovery | Recovery recall | False recovery | Regret | Margin |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | linear H8 W64 | 750 | 40.19% | 20.74% | 62.91% | 11.48% | 13.11% | 4.00% | 0.3091 | 1.380859 |
| 2 | MLP H4 W32 | 750 | 37.38% | 21.28% | 61.49% | 9.84% | 9.84% | 2.00% | 0.3083 | 1.914062 |
| 3 | MLP H8 W64 | 1250 | 42.99% | 19.97% | 64.33% | 4.92% | 8.20% | 4.00% | 0.3103 | 2.398438 |
| 4 | Transformer H8 W64 | 1000 | 51.40% | 32.19% | 63.77% | 4.92% | 6.56% | 2.00% | 0.3104 | 3.515625 |
| 5 | linear H4 W32 | 750 | 42.99% | 26.53% | 61.12% | 4.92% | 6.56% | 4.00% | 0.4522 | 1.986328 |
| 6 | GRU H8 W64 | 1500 | 41.12% | 25.98% | 63.01% | 1.64% | 6.56% | 4.00% | 0.3114 | 2.859375 |

The primary completion heads span 20.85--69.88% true-positive rate at
approximately 4.8--5.0% false-positive rate. Completion rank and option rank
are intentionally kept separate because good STOP calibration does not imply
good recovery-action selection.

## Decision

Multi-seed stratification materially lowers false recovery compared with the
rejected single-seed models, but the best safe beneficial-recovery recall is
only 11.48%. This is sufficient to advance the preregistered top four variants
to the bounded R1 online development test, not sufficient for a paper claim.
No R0 model is called a winner until it beats the matched B-retry control in
continuous, no-restore rollouts.

Integrity anchors:

- Combined training dataset manifest: `7c158bdc007cb5ac25995f119894d7457cd4f7ed9bf7bdc7aa850aed41865a1a`.
- Completion registry: `2019f1e4ad9be9630d7353f8bd1b3210cde0d61840e33a7f4825d2155f316b76`.
- Option registry: `d9485f55df75a89d1cb463c4a9c141b39fe9374e32c1d3bae21273c03f6b9e19`.
- Seed-`11007` corpus manifest/branch log: `1d90c921e16fc0bf1e0668d70f0b9e193f12198ecea853e3f9a6ca8ea75f6080` / `a135774302ca4917d9cb33c7231903e0e4dce98f071fd28fa3e88fc08437f786`.
- Seed-`12007` corpus manifest/branch log: `7de2858d3f4171abd43767a2978a9de9fc82d34665233a977647d90ab55d8ba3` / `c68aaa3c039e4c68df0d2f27eac48d8ab4ef96d316bd0c827ef974d2ace85094`.
