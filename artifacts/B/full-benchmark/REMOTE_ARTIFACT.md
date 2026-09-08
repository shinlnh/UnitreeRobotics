# B remote artifact bundle

This directory contains the Git-hosted, reviewer-facing evidence for the frozen
B benchmark. It retains all 1,200 episode records (600 each for H16 and H8), run
manifests and summaries, aggregate metrics with confidence intervals,
per-condition/per-case tables, A2-paired ablations, provenance hashes, and the
report figure.

The local raw artifact additionally contains 861,568,800 bytes of merged
decision JSONL and 5,695,897,230 bytes across 35,415 merged frame bundles, plus
resumable worker shards. The frozen feature cache contains 967 per-episode files
for 203,410 samples, and the selected selector weights contain 176,494,396
bytes. These large files are excluded from GitHub; exact core-file and frame-tree
SHA-256 digests are recorded in `reviewer/artifact_inventory.json`, while the
feature manifest records every feature-file digest.

The excluded large artifacts are assigned to the Hugging Face Space
`shin0412/UnitreeRobotic`. They are uploaded only after this B result is frozen
and pushed, on a Space branch corresponding to this Git branch with commit
messages that identify the source Git commit. A later documentation-only commit
will record the resolved Space revision after upload verification.

Headline frozen results:

- H16: 600 episodes; task-macro predicate SR 0.012805897805897807;
  terminal goal-state SR 0.01; 21,476 selector decisions.
- H8: 600 episodes; task-macro predicate SR 0.00522060347060347;
  terminal goal-state SR 0.0016666666666666668; 13,939 selector decisions.
- Paired H16-H8 task-macro SR delta: +0.007585294335294335 with 95% CI
  [0.00541212814962815, 0.009825567025567025].
- Against A2, task-macro SR delta is -0.015185120435120434 for H16 and
  -0.021300574425574426 for H8; both confidence intervals exclude zero.

The negative A2 comparison and aggressive STOP behavior are retained without
tuning on held-out outcomes. See `reviewer/REVIEWER_REPORT.md` for definitions,
confidence intervals, limitations, and the frozen protocol.
