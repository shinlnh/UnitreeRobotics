# B-retry remote artifact bundle

This directory contains the Git-hosted, reviewer-facing evidence for the frozen
B-retry benchmark. It retains all 1,200 episode records (600 each for H16 and
H8), run manifests and summaries, aggregate metrics with confidence intervals,
per-condition/per-case tables, B-paired ablations, provenance hashes, and the
report figure.

The local raw artifact additionally contains 1,402,326,594 bytes across 61,742
merged decision records and 9,979,903,443 bytes across 61,742 merged frame
bundles, plus resumable worker shards. These large files are excluded from
GitHub. Exact core-file and frame-tree SHA-256 digests are recorded in
`reviewer/artifact_inventory.json`.

The large decision logs and frame bundles are intended for the canonical
Hugging Face dataset repository documented in `docs/HUGGING_FACE_ARTIFACTS.md`,
on a branch corresponding to the Git feature branch. Upload and remote revision
verification are a separate publication step; no remote revision is claimed by
this local benchmark commit.

Headline frozen results:

- H16: 600 episodes; task-macro predicate SR 0.010691021941021941;
  terminal goal-state SR 0.005; 35,541 selector decisions; 5,557 retries.
- H8: 600 episodes; task-macro predicate SR 0.005836284086284086;
  terminal goal-state SR 0.0016666666666666668; 26,201 selector decisions;
  5,630 retries.
- Against B, task-macro SR delta is -0.002114875864875865 for H16 with 95% CI
  [-0.004657510776260776, 0.0003741883116883103], and
  +0.0006156806156806157 for H8 with 95% CI
  [-0.0006424573112073111, 0.001906084656084656]. Both intervals include zero.
- Mean executed-step deltas versus B are +73.1983 for H16 and +10.7233 for H8;
  mean policy-call deltas are +23.4417 and +20.4367, respectively.

The absence of a significant benefit and the increased execution cost are
retained without tuning. B-retry is a blind repetition control, not learned or
autonomous recovery. See `reviewer/REVIEWER_REPORT.md` for definitions,
confidence intervals, limitations, and the frozen protocol.
