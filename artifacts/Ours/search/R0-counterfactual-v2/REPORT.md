# Ours R0 counterfactual option distillation

This round trains compact temporal completion, failure, progress, and option
value heads. Model selection in R0 uses the pre-registered completion
calibration objective only. Option metrics below are measured on the train-only
counterfactual states and therefore establish fit, not rollout efficacy.

## Offline calibration

| Variant | Encoder | Selected step | TPR at FPR <= 5% | Brier | Progress MAE |
| --- | --- | ---: | ---: | ---: | ---: |
| `v00` | MLP H16 | 1,750 | 85.32% | 0.04756 | 0.07687 |
| `v01` | MLP H16 | 2,000 | 84.32% | 0.04932 | 0.07647 |
| `v02` | GRU H16 | 1,750 | 82.66% | 0.05164 | 0.07800 |
| `v03` | Transformer H8 | 1,500 | 81.27% | 0.05622 | 0.08327 |

## Train-only option fit

| Variant | Strict top-1 | Balanced recall | Pairwise rank | Accept/recover | False recovery | Missed recovery |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `v00` | 98.75% | 99.78% | 98.99% | 98.75% | 1.29% | 0.00% |
| `v01` | 98.12% | 99.67% | 99.32% | 98.13% | 1.94% | 0.00% |
| `v02` | 97.19% | 99.50% | 99.19% | 97.20% | 2.91% | 0.00% |
| `v03` | 94.69% | 99.06% | 98.45% | 94.70% | 5.50% | 0.00% |

The audit contains 322 labeled states, 320 strict multiclass preferences, and
321 strict accept-versus-recover preferences. Only 12 strict states favor a
recovery option, so rare-option recall is not interpreted as a generalization
claim. `v00` is the first rollout candidate because it ranks first under the
registered calibration rule and has the lowest train-only decision regret.

Registry SHA-256:
`3f5ada492bcbc41bb0f236726f76f28be7676247098eaf511faa3905c5cb6132`.
Option-fit audit SHA-256:
`ef70037faf8365147fc2665c6b680c66450b917588d9a7c403badf0440e9d563`.
