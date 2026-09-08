# Ours R0 counterfactual-v3 single-seed search

Four pre-registered temporal encoders were trained with the corrected v3
counterfactual corpus from train seed `10007`. Model selection here is an
offline sanity check only; neither rollout development nor final seeds were
consumed.

All four checkpoints fit the 251 option-training states very strongly
(95.30--97.01% strict top-1 accuracy). Their primary completion heads also met
the frozen false-positive operating point: 80.91--84.55% TPR at approximately
4.9% FPR.

## Held-out option labels

The decisive audit uses 71 labeled states from 12 held-out episodes of the
same train seed:

| Variant | Strict top-1 | Balanced recall | Pairwise | Accept/recover | False recovery | Mean regret |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v00 MLP H16 | 30.30% | 19.90% | 66.07% | 48.57% | 43.33% | 0.2380 |
| v01 MLP H16 | 36.36% | 24.95% | 68.19% | 48.57% | 30.00% | 0.2385 |
| v02 GRU H16 | 39.39% | 24.10% | 63.13% | 61.43% | 36.67% | 0.2614 |
| v03 Transformer H8 | 27.27% | 23.33% | 66.23% | 47.14% | 50.00% | 0.1144 |

## Decision

The train-to-holdout gap is too large for a paper claim or an R1 candidate.
Every checkpoint is rejected for online selection despite its strong fit and
completion-gate metrics. This is retained as negative evidence of
episode-level memorization in a high-dimensional option-value head. The next
iteration expands coverage with base seeds `11007` and `12007`, then evaluates
regularized, lower-capacity selectors against the same train-only holdout.

Registry SHA-256:
`2782375d5ebfdf2c6c661c283026cc8caa052cce7239ac737b27a05ea2ed6b2b`.
Option-audit SHA-256:
`3672054fea1fe03588419842e76679a72effd36b98c4b9c24fc67adf1369be8f`.
