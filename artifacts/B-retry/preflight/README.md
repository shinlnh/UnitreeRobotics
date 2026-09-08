# B-retry preflight

This preflight records the frozen evaluation-only naive retry contract before
the full B-retry benchmark outcome is inspected.

- Branch: `feat(B-retry)/implement-naive-retry-control`
- Variant: `GR00T-RC-SparkVLA-style-execution-naive-retry`
- Parent: frozen B source and its unchanged A1 plus selector checkpoints
- Retry: one unconditional repeat after the first confirmed STOP per subtask
- Budget: unchanged from B; retries consume the existing episode steps
- Learned parameters added: zero
- Failure detector, recovery memory, recovery policy, oracle, restore: disabled

The one-episode H16 pilot on `Ideal/case1`, seed 7 was used only for plumbing
and trace audit. All 27 request-local seeds and selector candidates replayed.
The trace contains 12 confirmed STOP events, split into exactly six retries and
six subtask advances, with 12 sequential onset anchors and one retry for each of
the six canonical subtasks. It executed 48 actions, stayed inside the unchanged
900-step budget, and did not succeed. No setting was changed after that outcome.

The full benchmark must use `scripts/run_b_retry_pipeline.sh`, retain the same
600 episodes per H16/H8 horizon, and compare against frozen B.
