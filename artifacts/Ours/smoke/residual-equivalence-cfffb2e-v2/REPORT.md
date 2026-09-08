# Residual abstention equivalence smoke

The residual controller was forced to abstain with option margin `1000000` and
compared with frozen B-retry on `Ideal/case1`, implementation-smoke seed
`30007`.  Both runs used independently loaded servers with identical frozen VLA
and selector weights.

The audit matched all 42 policy decisions and the episode result after removing
only method identity, recovery-head diagnostics, and wall-clock latency.  It
compared selector contexts/scores/candidates, predicted chunks, chosen prefixes,
STOP and retry state, simulator transitions, predicates, subgoal indices, step
counts, and outcomes.  The projections were exactly equal.  Both runs executed
236 steps, made 42 policy calls, committed 12 STOPs, and used six B-retry
attempts; the learned path triggered zero recoveries.

The raw decision traces remain in ignored local storage.  Their SHA-256 values
are `8b7b8244e420efc69f2a33064ba12cc82a211ccda0455a736a65c49d6d6519b4`
for B-retry and
`177ad2be10be5caf414621743249646aabe20c8b8be11941402e9edd89d39afb`
for residual mode.  They differ as files because the latter includes recovery
diagnostics; the audited behavior projection is identical.
