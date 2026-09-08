# Ours corrected closed-loop counterfactual R0 pilot

This is train-only data-quality evidence, not a development or held-out result.
It replaces the retired v2 label generator.

## Protocol corrections

- every option continues through confirmed STOP transitions for the same
  75-simulator-step/24-policy-call branch budget;
- `ADVANCE` is valid only when a prior STOP is pending;
- `ACCEPT_B` applies the current STOP using the live confirmation state;
- all valid options still start from one simulator snapshot restored to
  absolute tolerance `1e-12`;
- base seed is `10007`; no development or final seed is consumed.

## Pilot result

- corpus files: 60
- source samples: 25,369
- labeled states: 59
- valid option targets: 245
- strict preferences: 48/59 (81.36%)
- ties: 11
- distinct winning options: 5
- mean best-option margin: 0.74759
- manifest SHA-256: `cc55ef2bca81208bb6b16ab4f0a002f078557df400c4b9481a131803e0432d36`
- branch-log SHA-256: `8cfb019a35c540852aea70f33a04500f3a1daaf573616ef8bbc267f71d362bd1`

| Option | Strict wins | Mean return | Valid states |
| --- | ---: | ---: | ---: |
| `ACCEPT_B` | 21 | 0.10766 | 59 |
| `RETRY_CURRENT` | 10 | 0.47986 | 59 |
| `REOBSERVE` | 8 | 0.07624 | 59 |
| `CONSENSUS_PREFIX` | 8 | 0.08407 | 59 |
| `BACKTRACK_ONE` | 1 | -0.25267 | 9 |
| `ADVANCE` | 0 | n/a | 0 |

The pilot passes the pre-registered minimum-state, winner-diversity, and strict
preference-rate gates. Zero `ADVANCE` examples are expected in this one-state-
per-episode pilot because none of the selected source states had a pending
STOP. The full sampler contains six such candidate states and retains the
runtime mask rather than fabricating invalid labels.
