# Ours R1 v5 multi-seed development result

This report closes the seven remaining registered R1 slots for the original
counterfactual temporal recovery (CTR) controller.  Every candidate was run on
all 60 cases at development base seed `20007`; final base seed `7` was not
consumed.

## Matched result

The paired B-retry control completed 9/537 subtasks, with a task-macro subtask
rate of 1.619991%, 3,634 policy calls, and 21,315 simulator steps.  Every CTR v5
candidate completed the same 5/537 subtasks and had a task-macro rate of
0.992657%.  Thus every candidate's paired task-macro delta was -0.595960
percentage points and every 95% bootstrap interval excluded zero on the
negative side.

| Variant | Triggers | False recovery | Calls | Steps | Paired CI95 (pp) | Pareto-valid |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| rank 1, C1 | 5 | 20% | 2,328 | 18,173 | [-1.1793, -0.1389] | no |
| rank 1, C4 | 5 | 20% | 2,328 | 18,173 | [-1.1641, -0.1389] | no |
| rank 1, C1, margin +0.05 | 5 | 20% | 2,323 | 18,108 | [-1.1793, -0.1389] | no |
| rank 2, C1 | 2 | 0% | 2,272 | 17,278 | [-1.1768, -0.1389] | no |
| rank 2, C4 | 2 | 0% | 2,317 | 18,036 | [-1.1919, -0.1389] | no |
| rank 3, C1 | 1 | 0% | 2,312 | 17,924 | [-1.1795, -0.1389] | no |
| rank 4, C1 | 1 | 0% | 2,312 | 17,963 | [-1.1768, -0.1389] | no |

All seven candidates produced two strict final successes versus one for
B-retry, but this secondary metric does not rescue a candidate that is worse
on the preregistered primary task-macro metric.  No v5 candidate is eligible
for R2.

## Mechanistic diagnosis

The identical 5/537 outcome across architectures, margins, and hypothesis
counts isolates the controller boundary rather than model capacity.  When CTR
abstains, its evaluator follows B's confirmed-STOP transition and advances;
it does not preserve B-retry's first confirmed-STOP retry.  The learned
controller therefore removed the four additional subtasks delivered by the
strong control even on almost all decisions where it made no intervention.

The next method revision must be residual over B-retry: abstention executes
the control exactly, while a learned action may replace the first retry only
after its predicted counterfactual advantage clears a held-out safety margin.
This is a new prospective hypothesis, not a reinterpretation of these failed
results.

Integrity anchor: registry SHA-256
`00a5ed6495af8d83b8f0da425425f9ff0bac927acf1be4a6fad84365f4dffbed`.
Full decision traces remain in ignored local storage; Git records the compact
manifests, episode outcomes, summaries, audits, logs, and registry.
