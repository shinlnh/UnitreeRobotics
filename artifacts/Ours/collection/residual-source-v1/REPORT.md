# Residual source v1: exact B-retry behavior

This artifact freezes the on-policy training sources for residual
counterfactual recovery.  All three registered train seeds ran the full 60-case
H16 protocol with the residual controller forced to abstain by an option margin
of `1,000,000`.  The source contract requires B-retry's one retry per confirmed
STOP, captured causal training context, no forced/stagnation boundary, and zero
learned recovery triggers.

| Seed | Final success | Subtasks | Calls | Steps | B-retries | Eligible confirmed STOP states |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10007 | 1/60 | 7/537 | 3,404 | 18,036 | 554 | 63 |
| 11007 | 0/60 | 8/537 | 3,760 | 23,410 | 554 | 74 |
| 12007 | 2/60 | 9/537 | 3,389 | 18,028 | 547 | 55 |
| Total | 3/180 | 24/1,611 | 10,553 | 59,474 | 1,655 | 192 |

Every source recorded exactly zero learned recovery triggers.  Eligible states
are second-consecutive STOP proposals at causal subtask elapsed step 75 or
later, sampled at stride one with an eight-state episode cap; no episode reached
that cap.  The 192 states cover all six task families and are partitioned by
whole source episode downstream.

## Integrity anchors

| Seed | Run manifest SHA-256 | Prepared corpus manifest SHA-256 |
| ---: | --- | --- |
| 10007 | `75f56ea1ee3aac27fbabfa3f8edd01d8525d966d1d6f8d877748b99c48ad05b4` | `6ab7125ce1a01c0f61e42f515abf538f189844f68e97022f72cf1b987031e64c` |
| 11007 | `11051ede1ee393aba3e8d7e0eb5b556a90b556c5d467ea73e36c2e79ab1fa700` | `9caec47d450c7363c884aaa13836a2ac972713ff32dfd4ba16265548d9ba016b` |
| 12007 | `2f5915f3359d5d310d4aca47cefb423fdc52a21ad22966f0f8ce419713c97e39` | `580078386523b45cd11326ecd5076b0c1b6747fb35a27e71704dce919f879b72` |

The collection runner and fail-closed source validator were frozen in Git
commit `aa27820`.  Raw decisions and feature corpora remain in ignored local
storage for the counterfactual generator and Hugging Face mirror; Git records
the compact manifests, summaries, episode outcomes, audits, logs, and this
report.
