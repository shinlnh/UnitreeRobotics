# B preflight

This reviewer-safe preflight records the frozen selector data, training, and
runtime contracts before the full B outcome was used for any conclusion.

- Branch: `feat(B)/implement-GR00T-RC-SparkVLA-style-execution`
- Variant: `GR00T-RC-SparkVLA-style-execution`
- Parent policy: frozen A1 `GR00T-RC`
- Method: GR00T adaptation of the SparkVLA unified STOP/action-prefix selector
- Training source: 967 successful demonstration episodes, 203,410 samples
- Development selection: minimum frozen-development loss, earliest exact tie
- Selected checkpoint: step 6,000 of the scheduled 20,000 steps
- Stop rule: two consecutive STOP proposals on an unchanged observation
- Outcome signals, re-planning, retry, state restoration, recovery: disabled

The one-episode H16 pilot on `Ideal/case1`, seed 7 was a plumbing and trace
check only. It verified all 14 request-local diffusion seeds, selector/A1
runtime identities, 17-candidate masks, growing anchor histories, and six
two-request STOP confirmations. It executed 16 actions and did not succeed.
That held-out outcome was retained without tuning the model or protocol.

The subsequent full benchmark completed 600 episodes for each of H16 and H8.
See `../full-benchmark/reviewer/REVIEWER_REPORT.md` for the final results and
limitations.
